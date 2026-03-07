"""
t-SNE visualization of RAER model features AFTER fine-tuning.
Loads the trained checkpoint, extracts features through the full pipeline
(CLIP ViT → Adapter → Temporal Transformer → Projection), then runs t-SNE.
"""
import os, sys, argparse, numpy as np
import torch
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(__file__))
from models.clip import clip
from models.Generate_Model import GenerateModel
from dataloader.video_dataloader import test_data_loader
from utils.builders import get_class_info

CLASS_NAMES_RAER = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']
COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']


def extract_finetuned_features(model, dataset, device, max_samples=528):
    """Extract features through the full fine-tuned model pipeline."""
    model.eval()
    
    face_feats, body_feats, fused_feats, labels = [], [], [], []
    
    with torch.no_grad():
        for i in range(min(len(dataset), max_samples)):
            img_f, img_b, label = dataset[i]
            
            # Add batch dim: (T, C, H, W) -> (1, T, C, H, W)
            img_f = img_f.unsqueeze(0).to(device).float()
            img_b = img_b.unsqueeze(0).to(device).float()
            
            # --- Face stream ---
            n, t, c, h, w = img_f.shape
            face_reshaped = img_f.contiguous().view(-1, c, h, w)
            face_img_feat = model.image_encoder(face_reshaped.type(model.dtype))
            face_img_feat = model.face_adapter(face_img_feat)
            face_img_feat = face_img_feat.contiguous().view(n, t, -1)
            face_temporal = model.temporal_net(face_img_feat)  # (1, 512)
            
            # --- Body stream ---
            n, t, c, h, w = img_b.shape
            body_reshaped = img_b.contiguous().view(-1, c, h, w)
            body_img_feat = model.image_encoder(body_reshaped.type(model.dtype))
            body_img_feat = body_img_feat.contiguous().view(n, t, -1)
            body_temporal = model.temporal_net_body(body_img_feat)  # (1, 512)
            
            # --- Fusion ---
            concat = torch.cat((face_temporal, body_temporal), dim=-1)
            projected = model.project_fc(concat)
            projected = projected / (projected.norm(dim=-1, keepdim=True) + 1e-6)
            
            face_feats.append(face_temporal.cpu().numpy())
            body_feats.append(body_temporal.cpu().numpy())
            fused_feats.append(projected.cpu().numpy())
            labels.append(label)
            
            if (i + 1) % 100 == 0:
                print(f"  Extracted {i+1}/{min(len(dataset), max_samples)}")
    
    face_feats = np.concatenate(face_feats, axis=0)
    body_feats = np.concatenate(body_feats, axis=0)
    fused_feats = np.concatenate(fused_feats, axis=0)
    labels = np.array(labels)
    return face_feats, body_feats, fused_feats, labels


def plot_tsne(features, labels, class_names, colors, title, save_path):
    """Run t-SNE and plot."""
    print(f"  Running t-SNE for {title}...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                max_iter=1000, learning_rate='auto', init='pca')
    embeddings = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, cls_name in enumerate(class_names):
        mask = labels == idx
        if mask.sum() > 0:
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                      c=colors[idx], label=cls_name, alpha=0.6, s=25,
                      edgecolors='white', linewidth=0.3)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.set_xlabel('t-SNE dim 1', fontsize=12)
    ax.set_ylabel('t-SNE dim 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='outputs/RAER-ramp-up/model_best.pth')
    parser.add_argument('--out_dir', type=str, default='outputs/tsne_finetuned')
    parser.add_argument('--max_samples', type=int, default=528)
    args_cmd = parser.parse_args()

    os.makedirs(args_cmd.out_dir, exist_ok=True)

    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load CLIP
    clip_model, _ = clip.load('ViT-B/16', device='cpu')
    clip_model.float()

    # Build model with same config as training
    class Args:
        dataset = 'RAER'
        contexts_number = 8
        class_specific_contexts = 'True'
        class_token_position = 'end'
        load_and_tune_prompt_learner = 'True'
        freeze_image_encoder = False
        temporal_layers = 1
        temperature = 0.07
        text_type = 'prompt_ensemble'
        use_moco = False

    args = Args()
    input_text, _ = get_class_info(args)

    model = GenerateModel(input_text, clip_model, args)
    
    # Load checkpoint
    ckpt = torch.load(args_cmd.checkpoint, map_location='cpu')
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    # Remove 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Filter out keys with shape mismatch
    model_dict = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered[k] = v
        elif k in model_dict:
            print(f"  Skipping {k}: checkpoint {v.shape} vs model {model_dict[k].shape}")
    
    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    model = model.to(device).float()
    model.eval()
    print(f"Loaded checkpoint from {args_cmd.checkpoint}")

    # Load dataset
    val_data = test_data_loader(
        root_dir=os.path.abspath('./'),
        list_file=os.path.abspath('./RAER/annotation/test.txt'),
        num_segments=16, duration=1, image_size=224,
        bounding_box_face=os.path.abspath('./RAER/bounding_box/face.json'),
        bounding_box_body=os.path.abspath('./RAER/bounding_box/body.json'),
        crop_body=True, num_classes=5
    )
    print(f"Dataset: {len(val_data)} samples")

    # Extract or load cached features
    cache_path = os.path.join(args_cmd.out_dir, 'features_cache_finetuned.npz')
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        data = np.load(cache_path)
        face_feats = data['face']
        body_feats = data['body']
        fused_feats = data['fused']
        labels = data['labels']
    else:
        print("Extracting fine-tuned features...")
        face_feats, body_feats, fused_feats, labels = extract_finetuned_features(
            model, val_data, device, args_cmd.max_samples)
        np.savez(cache_path, face=face_feats, body=body_feats,
                 fused=fused_feats, labels=labels)
        print(f"Cached features to {cache_path}")

    print(f"Features shape: face={face_feats.shape}, body={body_feats.shape}, fused={fused_feats.shape}")

    # Plot t-SNE
    plot_tsne(face_feats, labels, CLASS_NAMES_RAER, COLORS,
             't-SNE of Fine-tuned Features (Face stream)',
             os.path.join(args_cmd.out_dir, 'tsne_finetuned_face.png'))

    plot_tsne(body_feats, labels, CLASS_NAMES_RAER, COLORS,
             't-SNE of Fine-tuned Features (Body stream)',
             os.path.join(args_cmd.out_dir, 'tsne_finetuned_body.png'))

    plot_tsne(fused_feats, labels, CLASS_NAMES_RAER, COLORS,
             't-SNE of Fine-tuned Features (Fused: Face + Body)',
             os.path.join(args_cmd.out_dir, 'tsne_finetuned_fused.png'))

    print("Done!")


if __name__ == '__main__':
    main()
