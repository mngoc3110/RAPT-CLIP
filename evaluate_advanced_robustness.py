import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from sklearn.metrics import recall_score

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


def forward_drop_stream(model, img_f, img_b, drop='face'):
    """
    Custom forward that zeros out face or body features
    at the concat point (before project_fc), not at image input.
    This properly ablates one stream at the architecture level.
    """
    dtype = model.dtype
    
    # --- Face stream ---
    n, t, c, h, w = img_f.shape
    image_face_reshaped = img_f.contiguous().view(-1, c, h, w)
    image_face_features = model.image_encoder(image_face_reshaped.type(dtype))
    image_face_features = model.face_adapter(image_face_features)
    image_face_features = image_face_features.contiguous().view(n, t, -1)
    video_face_features = model.temporal_net(image_face_features)
    
    # --- Body stream ---
    n, t, c, h, w = img_b.shape
    image_body_reshaped = img_b.contiguous().view(-1, c, h, w)
    image_body_features = model.image_encoder(image_body_reshaped.type(dtype))
    image_body_features = image_body_features.contiguous().view(n, t, -1)
    video_body_features = model.temporal_net_body(image_body_features)
    
    # --- Zero out one stream at concat point ---
    if drop == 'face':
        video_face_features = torch.zeros_like(video_face_features)
    elif drop == 'body':
        video_body_features = torch.zeros_like(video_body_features)
    
    # --- Continue normal forward from concat ---
    video_features = torch.cat((video_face_features, video_body_features), dim=-1)
    video_features = model.project_fc(video_features)
    video_features = video_features / (video_features.norm(dim=-1, keepdim=True) + 1e-6)
    
    # Text
    prompts = model.prompt_learner()
    with torch.cuda.amp.autocast(enabled=False):
        text_features = model.text_encoder(prompts, model.tokenized_prompts).float()
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
    
    # Classification
    if model.is_ensemble:
        text_features_view = text_features.view(model.num_classes, model.num_prompts_per_class, -1)
        text_features_view = text_features_view / (text_features_view.norm(dim=-1, keepdim=True) + 1e-6)
        logits = torch.einsum('bd,cpd->bcp', video_features, text_features_view)
        output = torch.mean(logits, dim=2) / model.args.temperature
    else:
        output = video_features @ text_features.t() / model.args.temperature
    
    return output


def run_robustness_tests(model, dataloader, args, out_dir):
    print("--- Starting Advanced Robustness Testing ---")
    
    # 1. Noise Levels
    noise_levels = [0.0, 0.1, 0.3, 0.5]
    noise_results = {std: {"preds": [], "labels": []} for std in noise_levels}
    
    # 2. Temporal & Spatial
    scenarios = {
        "Normal": {"preds": [], "labels": []},
        "Temporal Shuffling": {"preds": [], "labels": []},
        "Random Erasing (Face)": {"preds": [], "labels": []},
        "Random Erasing (Body)": {"preds": [], "labels": []},
        "Drop Face Stream": {"preds": [], "labels": []},
        "Drop Body Stream": {"preds": [], "labels": []}
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if len(batch) == 4:
                img_f, img_b, _, labels = batch
            else:
                img_f, img_b, labels = batch
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
                noise_results[std]["preds"].extend(pred.cpu().tolist())
                noise_results[std]["labels"].extend(labels.cpu().tolist())
                
            # Use normal prediction for scenarios base
            scenarios["Normal"]["preds"].extend(noise_results[0.0]["preds"][-bs:])
            scenarios["Normal"]["labels"].extend(noise_results[0.0]["labels"][-bs:])
            
            # --- 2. Temporal Shuffling ---
            shuffled_f = shuffle_temporal_frames(img_f)
            shuffled_b = shuffle_temporal_frames(img_b)
            out_shuffled, _, _, _ = model(shuffled_f, shuffled_b)
            pred_shuffled = torch.argmax(out_shuffled, dim=-1)
            scenarios["Temporal Shuffling"]["preds"].extend(pred_shuffled.cpu().tolist())
            scenarios["Temporal Shuffling"]["labels"].extend(labels.cpu().tolist())
            
            # --- 3. Random Erasing on Face ---
            # Simulating face occlusion (hand covering, mask, turned away)
            erased_f = apply_random_erasing(img_f, p=1.0)
            out_erased_f, _, _, _ = model(erased_f, img_b)
            pred_erased_f = torch.argmax(out_erased_f, dim=-1)
            scenarios["Random Erasing (Face)"]["preds"].extend(pred_erased_f.cpu().tolist())
            scenarios["Random Erasing (Face)"]["labels"].extend(labels.cpu().tolist())
            
            # --- 4. Random Erasing on Body/Context ---
            # Simulating body/context occlusions in the classroom
            erased_b = apply_random_erasing(img_b, p=1.0)
            out_erased_b, _, _, _ = model(img_f, erased_b)
            pred_erased_b = torch.argmax(out_erased_b, dim=-1)
            scenarios["Random Erasing (Body)"]["preds"].extend(pred_erased_b.cpu().tolist())
            scenarios["Random Erasing (Body)"]["labels"].extend(labels.cpu().tolist())
            
            # --- 5. Drop Face Stream (zero face features at concat) ---
            out_no_face = forward_drop_stream(model, img_f, img_b, drop='face')
            pred_no_face = torch.argmax(out_no_face, dim=-1)
            scenarios["Drop Face Stream"]["preds"].extend(pred_no_face.cpu().tolist())
            scenarios["Drop Face Stream"]["labels"].extend(labels.cpu().tolist())
            
            # --- 6. Drop Body Stream (zero body features at concat) ---
            out_no_body = forward_drop_stream(model, img_f, img_b, drop='body')
            pred_no_body = torch.argmax(out_no_body, dim=-1)
            scenarios["Drop Body Stream"]["preds"].extend(pred_no_body.cpu().tolist())
            scenarios["Drop Body Stream"]["labels"].extend(labels.cpu().tolist())

    # --- Process and Save Results ---
    def calc_uar(preds, labels):
        return recall_score(labels, preds, average='macro', zero_division=0) * 100.0

    print("\n[RESULTS] Gaussian Noise Robustness (Context):")
    noise_accs = []
    for std in noise_levels:
        acc = calc_uar(noise_results[std]["preds"], noise_results[std]["labels"])
        noise_accs.append(acc)
        print(f"  Std {std:.1f}: {acc:.2f}% (UAR)")
        
    print("\n[RESULTS] Structural Robustness:")
    bar_names = []
    bar_accs = []
    for name, res in scenarios.items():
        acc = calc_uar(res["preds"], res["labels"])
        bar_names.append(name)
        bar_accs.append(acc)
        print(f"  {name:25s}: {acc:.2f}% (UAR)")
        
    # --- Plotting ---
    # 1. Noise Line Plot
    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, noise_accs, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    plt.title('Robustness to Context Camera Noise (UAR)', fontsize=14)
    plt.xlabel('Gaussian Noise Standard Deviation', fontsize=12)
    plt.ylabel('UAR Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(noise_levels)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'robustness_noise.png'), dpi=300)
    plt.close()
    
    # 2. Structural Bar Chart
    plt.figure(figsize=(8, 6))
    colors = ['#2ca02c', '#d62728', '#9467bd', '#ff7f0e'][:len(bar_names)]
    bars = plt.bar(bar_names, bar_accs, color=colors)
    plt.title('Impact of Temporal & Spatial Disturbances (UAR)', fontsize=14)
    plt.ylabel('UAR Accuracy (%)', fontsize=12)
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
