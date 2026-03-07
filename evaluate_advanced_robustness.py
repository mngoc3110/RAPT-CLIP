import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from models.Generate_Model import GenerateModel
from models.clip import clip
from utils.builders import get_class_info, build_dataloaders

class Args:
    pass

def setup_args(checkpoint_path):
    args = Args()
    args.dataset = "RAER"
    args.text_type = "prompt_ensemble"
    args.class_names_with_context = "False"
    args.temporal_layers = 1
    args.contexts_number = 8
    args.class_token_position = "end"
    args.class_specific_contexts = "True"
    args.load_and_tune_prompt_learner = "True"
    args.clip_path = "ViT-B/16"
    args.drop_path_rate = 0.0
    
    if torch.backends.mps.is_available():
        args.device = torch.device("mps")
    elif torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
        
    args.freeze_image_encoder = False
    
    args.batch_size = 8
    args.workers = 4
    args.root_dir = os.path.abspath("./")
    args.train_annotation = os.path.abspath("RAER/annotation/train.txt")
    args.val_annotation = os.path.abspath("RAER/annotation/test.txt")
    args.test_annotation = os.path.abspath("RAER/annotation/test.txt")
    args.bounding_box_face = os.path.abspath("RAER/bounding_box/face.json")
    args.bounding_box_body = os.path.abspath("RAER/bounding_box/body.json")
    args.num_segments = 16
    args.duration = 1
    args.image_size = 224
    args.crop_body = True
    args.use_moco = False
    args.use_weighted_sampler = False
    args.temperature = 0.07
    
    return args

def load_model(args, checkpoint_path):
    clip_model, _ = clip.load(args.clip_path, device=args.device)
    clip_model.float() 
    class_names, input_text = get_class_info(args)
    
    model = GenerateModel(input_text, clip_model, args)
    model.to(args.device)
    model.float() 
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        
    model.eval()
    return model, class_names

# --- Perturbation Functions ---

def add_gaussian_noise(tensor, std=0.1):
    noise = torch.randn_like(tensor) * std
    return torch.clamp(tensor + noise, 0.0, 1.0) # Assuming tensor is normalized, clamp to prevent weird values. Usually CLIP inputs are normalized with specific mean/std, but this is an approximation.

def apply_random_erasing(tensor, p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
    """
    Applies Random Erasing to a batch of videos.
    tensor shape: (B, T, C, H, W)
    """
    if random.random() > p:
        return tensor
        
    erased_tensor = tensor.clone()
    b, t, c, h, w = tensor.shape
    area = h * w
    
    for i in range(b):
        target_area = random.uniform(scale[0], scale[1]) * area
        aspect_ratio = random.uniform(ratio[0], ratio[1])
        
        h_e = int(round((target_area * aspect_ratio) ** 0.5))
        w_e = int(round((target_area / aspect_ratio) ** 0.5))
        
        if h_e < h and w_e < w:
            y1 = random.randint(0, h - h_e)
            x1 = random.randint(0, w - w_e)
            # Apply same mask across all frames for consistency in action
            erased_tensor[i, :, :, y1:y1+h_e, x1:x1+w_e] = value
            
    return erased_tensor

def shuffle_temporal_frames(tensor):
    """
    Shuffles frames along the temporal dimension for each video in the batch.
    tensor shape: (B, T, C, H, W)
    """
    b, t, c, h, w = tensor.shape
    shuffled = torch.empty_like(tensor)
    for i in range(b):
        indices = torch.randperm(t)
        shuffled[i] = tensor[i, indices]
    return shuffled


def run_robustness_tests(model, dataloader, args, out_dir):
    print("--- Starting Advanced Robustness Testing ---")
    
    # 1. Noise Levels
    noise_levels = [0.0, 0.1, 0.3, 0.5]
    noise_results = {std: {"correct": 0, "total": 0} for std in noise_levels}
    
    # 2. Temporal & Spatial
    scenarios = {
        "Normal": {"correct": 0, "total": 0},
        "Temporal Shuffling": {"correct": 0, "total": 0},
        "Random Erasing (Body)": {"correct": 0, "total": 0}
    }
    
    with torch.no_grad():
        for img_f, img_b, labels in tqdm(dataloader, desc="Evaluating"):
            img_f = img_f.to(args.device)
            img_b = img_b.to(args.device)
            labels = labels.to(args.device).squeeze()
            if labels.dim() == 0: labels = labels.unsqueeze(0)
            bs = labels.size(0)
            
            # --- 1. Noise Injection (on Context/Body) ---
            for std in noise_levels:
                if std == 0.0:
                     out, _, _, _ = model(img_f, img_b)
                else:
                     noisy_b = add_gaussian_noise(img_b, std)
                     out, _, _, _ = model(img_f, noisy_b)
                pred = torch.argmax(out, dim=-1)
                noise_results[std]["correct"] += (pred == labels).sum().item()
                noise_results[std]["total"] += bs
                
            # Use normal prediction for scenarios base
            scenarios["Normal"]["correct"] = noise_results[0.0]["correct"]
            scenarios["Normal"]["total"] = noise_results[0.0]["total"]
            
            # --- 2. Temporal Shuffling ---
            shuffled_f = shuffle_temporal_frames(img_f)
            shuffled_b = shuffle_temporal_frames(img_b)
            out_shuffled, _, _, _ = model(shuffled_f, shuffled_b)
            pred_shuffled = torch.argmax(out_shuffled, dim=-1)
            scenarios["Temporal Shuffling"]["correct"] += (pred_shuffled == labels).sum().item()
            scenarios["Temporal Shuffling"]["total"] += bs
            
            # --- 3. Random Erasing on Body/Context ---
            # Simulating occlusions in the classroom
            erased_b = apply_random_erasing(img_b, p=1.0)
            out_erased, _, _, _ = model(img_f, erased_b)
            pred_erased = torch.argmax(out_erased, dim=-1)
            scenarios["Random Erasing (Body)"]["correct"] += (pred_erased == labels).sum().item()
            scenarios["Random Erasing (Body)"]["total"] += bs

    # --- Process and Save Results ---
    print("\n[RESULTS] Gaussian Noise Robustness (Context):")
    noise_accs = []
    for std in noise_levels:
        acc = noise_results[std]["correct"] / max(noise_results[std]["total"], 1) * 100
        noise_accs.append(acc)
        print(f"  Std {std:.1f}: {acc:.2f}%")
        
    print("\n[RESULTS] Structural Robustness:")
    bar_names = []
    bar_accs = []
    for name, res in scenarios.items():
        acc = res["correct"] / max(res["total"], 1) * 100
        bar_names.append(name)
        bar_accs.append(acc)
        print(f"  {name:25s}: {acc:.2f}%")
        
    # --- Plotting ---
    # 1. Noise Line Plot
    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, noise_accs, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    plt.title('Robustness to Context Camera Noise', fontsize=14)
    plt.xlabel('Gaussian Noise Standard Deviation', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(noise_levels)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'robustness_noise.png'), dpi=300)
    plt.close()
    
    # 2. Structural Bar Chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(bar_names, bar_accs, color=['#2ca02c', '#d62728', '#ff7f0e'])
    plt.title('Impact of Temporal & Spatial Disturbances', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'robustness_structure.png'), dpi=300)
    plt.close()
    
    # Save raw text
    with open(os.path.join(out_dir, 'advanced_robustness_results.txt'), 'w') as f:
        f.write("Noise Injection Results:\n")
        for std, acc in zip(noise_levels, noise_accs):
            f.write(f"Std {std:.1f}: {acc:.2f}%\n")
        f.write("\nStructural Disturbances:\n")
        for name, acc in zip(bar_names, bar_accs):
            f.write(f"{name}: {acc:.2f}%\n")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/thesis_assets/robustness")
    args_cmd = parser.parse_args()
    
    if not os.path.exists(args_cmd.out_dir):
        os.makedirs(args_cmd.out_dir)
        
    args = setup_args(args_cmd.checkpoint)
    model, class_names = load_model(args, args_cmd.checkpoint)
    _, val_loader, _ = build_dataloaders(args)
    
    run_robustness_tests(model, val_loader, args, args_cmd.out_dir)
