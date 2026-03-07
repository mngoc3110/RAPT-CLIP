"""
t-SNE visualization of CLIP features BEFORE fine-tuning.
Extracts features from pretrained CLIP ViT-B/16 on raw face+body crops,
then visualises with t-SNE coloured by emotion class.
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
from dataloader.video_dataloader import test_data_loader

CLASS_NAMES_RAER = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']
COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']


def extract_features(clip_model, loader, device, max_samples=500):
    """Extract CLIP features for middle frame of each video."""
    clip_model.eval()
    dataset = loader.dataset

    face_feats, body_feats, labels = [], [], []

    with torch.no_grad():
        for i in range(min(len(dataset), max_samples)):
            img_f, img_b, label = dataset[i]
            # Middle frame
            t = img_f.shape[0]
            mid = t // 2
            
            face_frame = img_f[mid].unsqueeze(0).to(device).float()
            body_frame = img_b[mid].unsqueeze(0).to(device).float()

            f_feat = clip_model.visual(face_frame)
            b_feat = clip_model.visual(body_frame)

            face_feats.append(f_feat.cpu().numpy())
            body_feats.append(b_feat.cpu().numpy())
            labels.append(label)

            if (i + 1) % 100 == 0:
                print(f"  Extracted {i+1}/{min(len(dataset), max_samples)}")

    face_feats = np.concatenate(face_feats, axis=0)
    body_feats = np.concatenate(body_feats, axis=0)
    labels = np.array(labels)
    return face_feats, body_feats, labels


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
                      c=colors[idx], label=cls_name, alpha=0.6, s=25, edgecolors='white', linewidth=0.3)

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
    parser.add_argument('--out_dir', type=str, default='outputs/tsne_pretrained')
    parser.add_argument('--max_samples', type=int, default=528)
    args_cmd = parser.parse_args()

    os.makedirs(args_cmd.out_dir, exist_ok=True)

    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load pretrained CLIP (NO fine-tuning)
    clip_model, _ = clip.load('ViT-B/16', device=device)
    clip_model.float()
    clip_model.eval()
    print("Loaded pretrained CLIP ViT-B/16")

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
    cache_path = os.path.join(args_cmd.out_dir, 'features_cache.npz')
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        data = np.load(cache_path)
        face_feats, body_feats, labels = data['face'], data['body'], data['labels']
    else:
        print("Extracting pretrained CLIP features...")
        loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False)
        face_feats, body_feats, labels = extract_features(clip_model, loader, device, args_cmd.max_samples)
        np.savez(cache_path, face=face_feats, body=body_feats, labels=labels)
        print(f"Cached features to {cache_path}")
    print(f"Features shape: face={face_feats.shape}, body={body_feats.shape}")

    # Concatenated features (face + body)
    concat_feats = np.concatenate([face_feats, body_feats], axis=1)

    # Plot t-SNE
    plot_tsne(face_feats, labels, CLASS_NAMES_RAER, COLORS,
             't-SNE of Pretrained CLIP Features (Face)',
             os.path.join(args_cmd.out_dir, 'tsne_pretrained_face.png'))

    plot_tsne(body_feats, labels, CLASS_NAMES_RAER, COLORS,
             't-SNE of Pretrained CLIP Features (Body)',
             os.path.join(args_cmd.out_dir, 'tsne_pretrained_body.png'))

    plot_tsne(concat_feats, labels, CLASS_NAMES_RAER, COLORS,
             't-SNE of Pretrained CLIP Features (Face + Body)',
             os.path.join(args_cmd.out_dir, 'tsne_pretrained_concat.png'))

    print("Done!")


if __name__ == '__main__':
    main()
