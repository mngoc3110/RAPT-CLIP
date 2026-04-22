"""
t-SNE visualization for RAER ramp-down checkpoint.
Side-by-side: Pretrained CLIP features vs Fine-tuned features.
Larger dots, high-contrast colors.
"""
import os, sys, argparse, numpy as np
import torch
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(__file__))
from models.clip import clip
from models.Generate_Model import GenerateModel
from dataloader.video_dataloader import test_data_loader
from utils.builders import get_class_info

CLASS_NAMES_RAER = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']

# ── High-contrast palette ─────────────────────────────────────────────────────
COLORS = [
    '#2196F3',   # Neutrality  – vivid blue
    '#FF5722',   # Enjoyment   – deep orange
    '#4CAF50',   # Confusion   – green
    '#9C27B0',   # Fatigue     – purple
    '#FFD600',   # Distraction – vivid yellow
]
MARKERS = ['o', 's', '^', 'D', 'P']   # different shapes per class


# ── Feature extraction: pretrained CLIP (middle frame) ───────────────────────
def extract_pretrained_features(clip_model, dataset, device, max_samples=528):
    clip_model.eval()
    face_feats, body_feats, labels = [], [], []
    with torch.no_grad():
        for i in range(min(len(dataset), max_samples)):
            img_f, img_b, label = dataset[i]
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
                print(f"  [Pretrained] Extracted {i+1}/{min(len(dataset), max_samples)}")
    face_feats = np.concatenate(face_feats, axis=0)
    body_feats = np.concatenate(body_feats, axis=0)
    labels = np.array(labels)
    return face_feats, body_feats, labels


# ── Feature extraction: fine-tuned model ─────────────────────────────────────
def extract_finetuned_features(model, dataset, device, max_samples=528):
    model.eval()
    face_feats, body_feats, fused_feats, labels = [], [], [], []
    with torch.no_grad():
        for i in range(min(len(dataset), max_samples)):
            img_f, img_b, label = dataset[i]
            img_f = img_f.unsqueeze(0).to(device).float()
            img_b = img_b.unsqueeze(0).to(device).float()

            n, t, c, h, w = img_f.shape
            face_reshaped = img_f.contiguous().view(-1, c, h, w)
            face_img_feat = model.image_encoder(face_reshaped.type(model.dtype))
            face_img_feat = model.face_adapter(face_img_feat)
            face_img_feat = face_img_feat.contiguous().view(n, t, -1)
            face_temporal = model.temporal_net(face_img_feat)

            n, t, c, h, w = img_b.shape
            body_reshaped = img_b.contiguous().view(-1, c, h, w)
            body_img_feat = model.image_encoder(body_reshaped.type(model.dtype))
            body_img_feat = body_img_feat.contiguous().view(n, t, -1)
            body_temporal = model.temporal_net_body(body_img_feat)

            concat = torch.cat((face_temporal, body_temporal), dim=-1)
            projected = model.project_fc(concat)
            projected = projected / (projected.norm(dim=-1, keepdim=True) + 1e-6)

            face_feats.append(face_temporal.cpu().numpy())
            body_feats.append(body_temporal.cpu().numpy())
            fused_feats.append(projected.cpu().numpy())
            labels.append(label)

            if (i + 1) % 100 == 0:
                print(f"  [Fine-tuned] Extracted {i+1}/{min(len(dataset), max_samples)}")

    face_feats  = np.concatenate(face_feats,  axis=0)
    body_feats  = np.concatenate(body_feats,  axis=0)
    fused_feats = np.concatenate(fused_feats, axis=0)
    labels      = np.array(labels)
    return face_feats, body_feats, fused_feats, labels


# ── t-SNE + plot helper ───────────────────────────────────────────────────────
def run_tsne(features):
    tsne = TSNE(n_components=2, perplexity=35, random_state=42,
                max_iter=1200, learning_rate='auto', init='pca')
    return tsne.fit_transform(features)


def scatter_on_ax(ax, embeddings, labels, class_names, colors, markers, title):
    """Plot one t-SNE panel with large, high-contrast dots."""
    for idx, cls_name in enumerate(class_names):
        mask = labels == idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=colors[idx],
            marker=markers[idx],
            label=cls_name,
            alpha=0.88,
            s=90,                          # ← bigger dots
            edgecolors='#1a1a1a',          # ← dark edge for contrast
            linewidth=0.6,
            zorder=3,
        )

    ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
    leg = ax.legend(fontsize=10, loc='best', framealpha=0.92,
                    markerscale=1.4, edgecolor='#cccccc')
    ax.set_xlabel('t-SNE dim 1', fontsize=11)
    ax.set_ylabel('t-SNE dim 2', fontsize=11)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')


# ── Side-by-side combined figure ─────────────────────────────────────────────
def plot_combined(pre_embeddings, fine_embeddings, labels,
                  class_names, colors, markers, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor('white')

    scatter_on_ax(axes[0], pre_embeddings,  labels, class_names, colors, markers,
                  't-SNE of Pretrained CLIP Features (Face + Body)')
    scatter_on_ax(axes[1], fine_embeddings, labels, class_names, colors, markers,
                  't-SNE of Fine-tuned Features (Fused: Face + Body)\n[RAER Ramp-Down]')

    plt.tight_layout(pad=3.0)
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved combined figure → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',   type=str,
                        default='outputs/RAER-ramp-down/model_best.pth')
    parser.add_argument('--out_dir',      type=str,
                        default='outputs/tsne_rampdown')
    parser.add_argument('--max_samples',  type=int, default=528)
    args_cmd = parser.parse_args()

    os.makedirs(args_cmd.out_dir, exist_ok=True)

    device = torch.device('mps'  if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    val_data = test_data_loader(
        root_dir=os.path.abspath('./'),
        list_file=os.path.abspath('./RAER/annotation/test.txt'),
        num_segments=16, duration=1, image_size=224,
        bounding_box_face=os.path.abspath('./RAER/bounding_box/face.json'),
        bounding_box_body=os.path.abspath('./RAER/bounding_box/body.json'),
        crop_body=True, num_classes=5
    )
    print(f"Dataset: {len(val_data)} samples")

    # ── 1. Pretrained features ────────────────────────────────────────────────
    pre_cache = os.path.join(args_cmd.out_dir, 'cache_pretrained.npz')
    if os.path.exists(pre_cache):
        print(f"Loading cached pretrained features from {pre_cache}")
        d = np.load(pre_cache)
        pre_concat, labels = d['concat'], d['labels']
    else:
        clip_model, _ = clip.load('ViT-B/16', device=device)
        clip_model.float(); clip_model.eval()
        print("Extracting pretrained CLIP features...")
        face_pre, body_pre, labels = extract_pretrained_features(
            clip_model, val_data, device, args_cmd.max_samples)
        pre_concat = np.concatenate([face_pre, body_pre], axis=1)
        np.savez(pre_cache, concat=pre_concat, labels=labels)
        print(f"Cached pretrained features → {pre_cache}")
        del clip_model

    # ── 2. Fine-tuned features (ramp-down checkpoint) ─────────────────────────
    fine_cache = os.path.join(args_cmd.out_dir, 'cache_finetuned_rampdown.npz')
    if os.path.exists(fine_cache):
        print(f"Loading cached fine-tuned features from {fine_cache}")
        d = np.load(fine_cache)
        fused_feats = d['fused']
    else:
        clip_model2, _ = clip.load('ViT-B/16', device='cpu')
        clip_model2.float()

        class Args:
            dataset                    = 'RAER'
            contexts_number            = 8
            class_specific_contexts    = 'True'
            class_token_position       = 'end'
            load_and_tune_prompt_learner = 'True'
            freeze_image_encoder       = False
            temporal_layers            = 1
            temperature                = 0.07
            text_type                  = 'prompt_ensemble'
            use_moco                   = False

        args = Args()
        input_text, _ = get_class_info(args)
        model = GenerateModel(input_text, clip_model2, args)

        ckpt = torch.load(args_cmd.checkpoint, map_location='cpu')
        state_dict = (ckpt.get('model_state_dict')
                      or ckpt.get('state_dict')
                      or ckpt)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model_dict = model.state_dict()
        filtered   = {k: v for k, v in state_dict.items()
                      if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        model = model.to(device).float()
        model.eval()
        print(f"Loaded checkpoint: {args_cmd.checkpoint}")

        print("Extracting fine-tuned features...")
        _, _, fused_feats, labels2 = extract_finetuned_features(
            model, val_data, device, args_cmd.max_samples)
        np.savez(fine_cache, fused=fused_feats, labels=labels2)
        print(f"Cached fine-tuned features → {fine_cache}")
        labels = labels2

    # ── 3. Run t-SNE ─────────────────────────────────────────────────────────
    print("Running t-SNE on pretrained features...")
    pre_emb  = run_tsne(pre_concat)

    print("Running t-SNE on fine-tuned features...")
    fine_emb = run_tsne(fused_feats)

    # ── 4. Individual plots ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8)); fig.patch.set_facecolor('white')
    scatter_on_ax(ax, pre_emb, labels, CLASS_NAMES_RAER, COLORS, MARKERS,
                  't-SNE of Pretrained CLIP Features (Face + Body)')
    plt.tight_layout()
    plt.savefig(os.path.join(args_cmd.out_dir, 'tsne_pretrained.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8)); fig.patch.set_facecolor('white')
    scatter_on_ax(ax, fine_emb, labels, CLASS_NAMES_RAER, COLORS, MARKERS,
                  't-SNE of Fine-tuned Features (Fused: Face + Body)\n[RAER Ramp-Down]')
    plt.tight_layout()
    plt.savefig(os.path.join(args_cmd.out_dir, 'tsne_finetuned_fused.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    # ── 5. Combined side-by-side ──────────────────────────────────────────────
    plot_combined(pre_emb, fine_emb, labels,
                  CLASS_NAMES_RAER, COLORS, MARKERS,
                  os.path.join(args_cmd.out_dir, 'tsne_combined.png'))

    print("Done! All plots saved to:", args_cmd.out_dir)


if __name__ == '__main__':
    main()
