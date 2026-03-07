import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from copy import deepcopy

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
    
    args.batch_size = 4
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
        print(f"Loaded checkpoint from {checkpoint_path}")
        
    model.eval()
    return model, class_names

def run_tsne(model, dataloader, class_names, args, out_dir):
    print("--- Running t-SNE Analysis on Visual Features ---")
    features = []
    labels_list = []
    
    with torch.no_grad():
        for img_f, img_b, labels in tqdm(dataloader, desc="Extracting features"):
            img_f = img_f.to(args.device)
            img_b = img_b.to(args.device)
            
            n, t, c, h, w = img_f.shape
            img_f_reshaped = img_f.contiguous().view(-1, c, h, w)
            img_b_reshaped = img_b.contiguous().view(-1, c, h, w)
            
            # Forward pass up to visual feature projection (before text matching)
            f_feat = model.image_encoder(img_f_reshaped.type(model.dtype))
            f_feat = model.face_adapter(f_feat)
            f_feat = f_feat.contiguous().view(n, t, -1)
            f_feat = model.temporal_net(f_feat)
            
            b_feat = model.image_encoder(img_b_reshaped.type(model.dtype))
            b_feat = b_feat.contiguous().view(n, t, -1)
            b_feat = model.temporal_net_body(b_feat)
            
            v_feat = torch.cat((f_feat, b_feat), dim=-1)
            v_feat = model.project_fc(v_feat)
            # Normalize
            if model.project_fc.weight.dtype == torch.float16 and v_feat.device.type == 'mps':
                v_feat = v_feat.float()
                v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)
                v_feat = v_feat.half()
            else:
                v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)
                
            features.append(v_feat.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            
    features = np.concatenate(features, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    
    print("Running t-SNE algorithm (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels_list, cmap='tab10', alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=class_names, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('t-SNE Visualization of Pure Visual Features')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'tsne_visual_features.png'), dpi=300)
    plt.close()
    print(f"Saved t-SNE plot to {os.path.join(out_dir, 'tsne_visual_features.png')}")

def run_modality_ablation(model, dataloader, args, out_dir):
    print("--- Running Missing Modality / Occlusion Study ---")
    scenarios = ["Normal (Both Face & Body)", "Masked Face (Zeros)", "Masked Body (Zeros)"]
    results = {s: {"correct": 0, "total": 0} for s in scenarios}
    
    with torch.no_grad():
        for img_f, img_b, labels in tqdm(dataloader, desc="Testing Modalities"):
            img_f = img_f.to(args.device)
            img_b = img_b.to(args.device)
            labels = labels.to(args.device).squeeze()
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            
            # Normal
            out_norm, _, _, _ = model(img_f, img_b)
            pred_norm = torch.argmax(out_norm, dim=-1)
            results["Normal (Both Face & Body)"]["correct"] += (pred_norm == labels).sum().item()
            results["Normal (Both Face & Body)"]["total"] += labels.size(0)
            
            # Masked Face
            zeros_f = torch.zeros_like(img_f)
            out_mf, _, _, _ = model(zeros_f, img_b)
            pred_mf = torch.argmax(out_mf, dim=-1)
            results["Masked Face (Zeros)"]["correct"] += (pred_mf == labels).sum().item()
            results["Masked Face (Zeros)"]["total"] += labels.size(0)
            
            # Masked Body
            zeros_b = torch.zeros_like(img_b)
            out_mb, _, _, _ = model(img_f, zeros_b)
            pred_mb = torch.argmax(out_mb, dim=-1)
            results["Masked Body (Zeros)"]["correct"] += (pred_mb == labels).sum().item()
            results["Masked Body (Zeros)"]["total"] += labels.size(0)

    print("\nModality Ablation Results:")
    with open(os.path.join(out_dir, 'modality_ablation.txt'), 'w') as f:
        for s in scenarios:
            acc = results[s]["correct"] / results[s]["total"] * 100
            msg = f"{s:30s}: {acc:.2f}%"
            print(msg)
            f.write(msg + "\n")

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
    
    run_tsne(model, val_loader, class_names, args, args_cmd.out_dir)
    run_modality_ablation(model, val_loader, args, args_cmd.out_dir)
    
    print("\nAll robustness tests completed successfully!")
