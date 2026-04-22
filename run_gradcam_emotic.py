"""
GradCAM (Token Similarity) Visualization for EMOTIC Dataset
============================================================
Paste this script into your Colab/Kaggle environment and run after training.

Usage:
    python run_gradcam_emotic.py \
        --checkpoint outputs/EMOTIC-Full-ASL-Cosine-xxx/model_best.pth \
        --samples 10 \
        --out_dir outputs/gradcam_emotic

The script:
1. Loads best checkpoint
2. For each sample, computes CLIP token-level similarity maps for Face and Body streams
3. Overlays heatmaps on original images
4. Saves 2-row grids: [Original Face | Original Body] / [Attention Face | Attention Body]
   with multi-label predictions shown at bottom
"""

import argparse
import os
import sys
import json
import torch
import cv2
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from models.Generate_Model import GenerateModel
from models.clip import clip
from dataloader.video_dataloader import test_data_loader


# ============================================================
# EMOTIC 26 Emotion Categories (same order as annotation)
# ============================================================
EMOTIC_CLASSES = [
    'Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion',
    'Confidence', 'Disapproval', 'Disconnection', 'Disquietment', 'Doubt/Confusion',
    'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue',
    'Fear', 'Happiness', 'Pain', 'Peace', 'Pleasure',
    'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning'
]

# Prompt ensemble for EMOTIC (7 prompts per class for rich representation)
EMOTIC_PROMPT_ENSEMBLE = [
    [  # Affection
        "A person showing warm affection toward someone.",
        "A photo of a caring and loving interaction.",
        "A person expressing fondness and tenderness."
    ],
    [  # Anger
        "A person showing anger and frustration.",
        "An angry facial expression with tense body.",
        "A person expressing rage or irritation."
    ],
    [  # Annoyance
        "A person looking annoyed and bothered.",
        "A photo showing mild irritation and displeasure.",
        "A person appearing frustrated by something."
    ],
    [  # Anticipation
        "A person looking forward with anticipation.",
        "A photo of someone eagerly expecting something.",
        "A person showing excitement about what's coming."
    ],
    [  # Aversion
        "A person showing disgust or aversion.",
        "A photo of someone recoiling or turning away.",
        "A person expressing strong dislike."
    ],
    [  # Confidence
        "A person looking confident and self-assured.",
        "A photo of someone standing tall with confidence.",
        "A person showing poise and assurance."
    ],
    [  # Disapproval
        "A person showing disapproval or disagreement.",
        "A photo of someone frowning with disapproval.",
        "A person expressing criticism or rejection."
    ],
    [  # Disconnection
        "A person appearing disconnected and distant.",
        "A photo of someone withdrawn and zoned out.",
        "A person showing emotional detachment."
    ],
    [  # Disquietment
        "A person showing unease and discomfort.",
        "A photo of someone looking anxious and unsettled.",
        "A person appearing worried and restless."
    ],
    [  # Doubt/Confusion
        "A person looking doubtful and confused.",
        "A photo of someone scratching their head in confusion.",
        "A person appearing uncertain and puzzled."
    ],
    [  # Embarrassment
        "A person looking embarrassed and self-conscious.",
        "A photo of someone blushing or hiding their face.",
        "A person showing shame or awkwardness."
    ],
    [  # Engagement
        "A person looking engaged and interested.",
        "A photo of someone paying close attention.",
        "A person showing active participation."
    ],
    [  # Esteem
        "A person showing respect and admiration.",
        "A photo of someone expressing high regard.",
        "A person looking up to someone with esteem."
    ],
    [  # Excitement
        "A person showing excitement and enthusiasm.",
        "A photo of someone jumping or cheering with joy.",
        "A person expressing thrilling excitement."
    ],
    [  # Fatigue
        "A person looking tired and fatigued.",
        "A photo of someone yawning or drooping with exhaustion.",
        "A person showing signs of weariness."
    ],
    [  # Fear
        "A person showing fear and terror.",
        "A photo of someone with wide eyes in fright.",
        "A person expressing panic or alarm."
    ],
    [  # Happiness
        "A person showing happiness and joy.",
        "A photo of someone smiling brightly.",
        "A person expressing delight and contentment."
    ],
    [  # Pain
        "A person showing pain and distress.",
        "A photo of someone wincing in discomfort.",
        "A person expressing physical or emotional pain."
    ],
    [  # Peace
        "A person looking peaceful and calm.",
        "A photo of someone in a serene state.",
        "A person expressing tranquility and relaxation."
    ],
    [  # Pleasure
        "A person showing pleasure and satisfaction.",
        "A photo of someone enjoying a pleasant moment.",
        "A person expressing contentment and delight."
    ],
    [  # Sadness
        "A person looking sad and sorrowful.",
        "A photo of someone with a downcast expression.",
        "A person expressing grief or melancholy."
    ],
    [  # Sensitivity
        "A person showing emotional sensitivity.",
        "A photo of someone appearing touched or moved.",
        "A person expressing vulnerability."
    ],
    [  # Suffering
        "A person showing suffering and anguish.",
        "A photo of someone in deep emotional pain.",
        "A person expressing torment or agony."
    ],
    [  # Surprise
        "A person showing surprise and astonishment.",
        "A photo of someone with a shocked expression.",
        "A person expressing unexpected amazement."
    ],
    [  # Sympathy
        "A person showing sympathy and compassion.",
        "A photo of someone comforting another.",
        "A person expressing empathy and care."
    ],
    [  # Yearning
        "A person showing yearning and longing.",
        "A photo of someone gazing wistfully.",
        "A person expressing deep desire or nostalgia."
    ],
]

# Hand-crafted descriptors for MI Loss
EMOTIC_DESCRIPTORS = [
    "A person expressing warm affection and care.",
    "A person showing anger with a tense expression.",
    "A person looking annoyed and mildly frustrated.",
    "A person showing eager anticipation.",
    "A person recoiling with aversion and disgust.",
    "A person standing with confidence and poise.",
    "A person frowning with disapproval.",
    "A person appearing emotionally disconnected.",
    "A person looking uneasy and anxious.",
    "A person looking confused and uncertain.",
    "A person appearing embarrassed and awkward.",
    "A person actively engaged and attentive.",
    "A person showing respect and admiration.",
    "A person expressing excitement and enthusiasm.",
    "A person looking tired and exhausted.",
    "A person showing fear with wide eyes.",
    "A person smiling with happiness.",
    "A person wincing in pain.",
    "A person looking peaceful and serene.",
    "A person expressing pleasure and satisfaction.",
    "A person looking sad with a downcast face.",
    "A person appearing emotionally touched.",
    "A person showing deep suffering.",
    "A person with a surprised expression.",
    "A person showing sympathy and compassion.",
    "A person gazing with yearning and longing.",
]


# ============================================================
# Token Similarity Map
# ============================================================
def compute_token_similarity(model, face_input, body_input, target_class, stream='face'):
    """
    Compute spatial similarity map using CLIP patch tokens.
    For multi-label EMOTIC, target_class is the index of the
    emotion we want to visualize attention for.
    """
    n, t, c, h, w = face_input.shape
    mid_idx = t // 2

    with torch.no_grad():
        prompts = model.prompt_learner()
        text_features = model.text_encoder(prompts, model.tokenized_prompts).float()
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)

        if model.is_ensemble:
            text_features = text_features.view(model.num_classes, model.num_prompts_per_class, -1)
            all_class_text = text_features.mean(dim=1)
        else:
            all_class_text = text_features
        all_class_text = all_class_text / (all_class_text.norm(dim=-1, keepdim=True) + 1e-6)

    # Hook transformer output
    transformer_output = {}
    def hook_fn(module, input, output):
        transformer_output['out'] = output.detach()
    handle = model.image_encoder.transformer.register_forward_hook(hook_fn)

    with torch.no_grad():
        target_4d = (face_input if stream == 'face' else body_input).contiguous().view(-1, c, h, w)
        _ = model.image_encoder(target_4d)
    handle.remove()

    if 'out' not in transformer_output:
        return np.zeros((14, 14), dtype=np.float32)

    with torch.no_grad():
        feat = transformer_output['out'].permute(1, 0, 2)
        spatial = feat[mid_idx, 1:]  # (196, 768) — skip CLS token
        spatial_n = model.image_encoder.ln_post(spatial)
        if model.image_encoder.proj is not None:
            spatial_p = spatial_n @ model.image_encoder.proj
        else:
            spatial_p = spatial_n
        spatial_p = spatial_p / (spatial_p.norm(dim=-1, keepdim=True) + 1e-6)

        # Similarity with all classes
        all_sim = spatial_p @ all_class_text.t()  # (196, 26)
        target_sim = all_sim[:, target_class]
        mean_sim = all_sim.mean(dim=1)
        disc_sim = (target_sim - mean_sim).cpu().numpy().reshape(14, 14)

        cam = np.maximum(disc_sim, 0)
        cam_max = cam.max()
        if cam_max > 1e-7:
            cam = cam / cam_max
        return cam


# ============================================================
# Visualization Helpers
# ============================================================
def overlay_heatmap(img_bgr, cam, alpha=0.5):
    h, w = img_bgr.shape[:2]
    cam_smooth = cv2.GaussianBlur(cam.astype(np.float32), (5, 5), sigmaX=2.0)
    cam_resized = cv2.resize(cam_smooth, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0, None)
    if cam_resized.max() > 1e-7:
        cam_resized = cam_resized / cam_resized.max()
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    blend_w = np.clip(cam_resized * 1.5, 0, 1) * alpha
    blend_3ch = np.stack([blend_w]*3, axis=-1)
    result = img_bgr.astype(np.float32) * (1 - blend_3ch) + heatmap.astype(np.float32) * blend_3ch
    return np.uint8(np.clip(result, 0, 255))


def unnormalize_tensor(tensor, device='cpu'):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(device)
    t = tensor.detach() * std + mean
    t = t.clamp(0, 1).cpu().numpy()
    img_hwc = np.transpose(t, (1, 2, 0))
    return cv2.cvtColor((img_hwc * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def create_emotic_grid(orig_f, orig_b, att_f, att_b, true_labels, pred_labels, top_emotion):
    """Create 2x2 grid with multi-label info at bottom."""
    s = 224
    imgs = [cv2.resize(x, (s, s)) for x in [orig_f, orig_b, att_f, att_b]]

    pad_top = 40
    pad_bottom = 80  # space for multi-label text
    gap = 4
    grid_w = s * 2 + gap
    grid_h = s * 2 + pad_top + gap + pad_bottom

    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Column headers
    for col, title in enumerate(["Face Stream", "Body/Context Stream"]):
        x = col * (s + gap)
        tw = cv2.getTextSize(title, font, 0.55, 2)[0][0]
        cv2.putText(canvas, title, (x + (s - tw) // 2, 25), font, 0.55, (0, 0, 0), 2)

    # Row 1: originals
    canvas[pad_top:pad_top+s, 0:s] = imgs[0]
    canvas[pad_top:pad_top+s, s+gap:s*2+gap] = imgs[1]

    # Row 2: attention overlays
    y2 = pad_top + s + gap
    canvas[y2:y2+s, 0:s] = imgs[2]
    canvas[y2:y2+s, s+gap:s*2+gap] = imgs[3]

    # Bottom: multi-label info
    y_text = y2 + s + 16

    # Attention target (highlighted emotion)
    cv2.putText(canvas, f"Attention: {top_emotion}", (6, y_text), font, 0.5, (220, 50, 50), 2)

    # True labels
    true_str = "True: " + ", ".join(true_labels[:5])
    if len(true_labels) > 5:
        true_str += f" +{len(true_labels)-5}"
    cv2.putText(canvas, true_str, (6, y_text + 22), font, 0.38, (0, 120, 0), 1)

    # Predicted labels
    pred_str = "Pred: " + ", ".join(pred_labels[:5])
    if len(pred_labels) > 5:
        pred_str += f" +{len(pred_labels)-5}"
    cv2.putText(canvas, pred_str, (6, y_text + 42), font, 0.38, (0, 0, 180), 1)

    return canvas


# ============================================================
# Main GradCAM Generation
# ============================================================
def generate_gradcam_emotic(model, dataset, class_names, args, out_dir, num_samples=10):
    os.makedirs(out_dir, exist_ok=True)
    count = 0

    indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)

    for i in indices:
        if count >= num_samples:
            break

        try:
            img_f, img_b, label = dataset[i]
        except Exception as e:
            print(f"  Skipping {i}: {e}")
            continue

        # Parse multi-hot label
        if isinstance(label, (int, np.integer)):
            # Single-label fallback
            true_indices = [label]
        elif isinstance(label, torch.Tensor):
            true_indices = torch.where(label > 0)[0].tolist()
        elif isinstance(label, np.ndarray):
            true_indices = np.where(label > 0)[0].tolist()
        else:
            true_indices = [label]

        true_labels = [class_names[j] for j in true_indices if j < len(class_names)]
        if not true_labels:
            continue

        img_f_input = img_f.unsqueeze(0).to(args.device).float()
        img_b_input = img_b.unsqueeze(0).to(args.device).float()
        n, t, c, h, w = img_f_input.shape
        mid_seg = t // 2

        # 1. Prediction
        with torch.no_grad():
            output, _, _, _ = model(img_f_input, img_b_input)
            probs = torch.sigmoid(output[0])  # Multi-label → sigmoid
            pred_indices = torch.where(probs > 0.5)[0].tolist()
            if not pred_indices:
                # Take top-3 if none pass threshold
                pred_indices = torch.topk(probs, k=min(3, len(probs)))[1].tolist()

        pred_labels = [class_names[j] for j in pred_indices if j < len(class_names)]

        # Choose which emotion to visualize attention for (highest confidence true emotion)
        best_target = true_indices[0]
        best_prob = 0
        for ti in true_indices:
            if ti < len(probs) and probs[ti] > best_prob:
                best_prob = probs[ti]
                best_target = ti
        top_emotion = class_names[best_target] if best_target < len(class_names) else f"class_{best_target}"

        # 2. Compute attention maps
        cam_face = compute_token_similarity(model, img_f_input, img_b_input, best_target, stream='face')
        cam_body = compute_token_similarity(model, img_f_input, img_b_input, best_target, stream='body')

        # 3. Display images
        orig_f_bgr = unnormalize_tensor(img_f[mid_seg], device='cpu')
        orig_b_bgr = unnormalize_tensor(img_b[mid_seg], device='cpu')

        # 4. Overlay
        att_f = overlay_heatmap(orig_f_bgr, cam_face)
        att_b = overlay_heatmap(orig_b_bgr, cam_body)

        # 5. Grid
        grid = create_emotic_grid(orig_f_bgr, orig_b_bgr, att_f, att_b,
                                   true_labels, pred_labels, top_emotion)

        safe_emotion = top_emotion.replace("/", "-").replace(" ", "_")
        prefix = f"emotic_{count:02d}_{safe_emotion}"
        save_path = os.path.join(out_dir, f"{prefix}.jpg")
        cv2.imwrite(save_path, grid)
        print(f"[{count+1}/{num_samples}] Attention:{top_emotion} | True:{true_labels[:3]} | Pred:{pred_labels[:3]} → {save_path}")

        count += 1

    print(f"\nDone! {count} GradCAM images saved to {out_dir}/")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GradCAM for EMOTIC (Colab/Kaggle)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model_best.pth (e.g., outputs/EMOTIC-xxx/model_best.pth)")
    parser.add_argument("--out_dir", type=str, default="outputs/gradcam_emotic")
    parser.add_argument("--samples", type=int, default=10)
    # Data paths (defaults match EMOTIC training config from log)
    parser.add_argument("--root_dir", type=str, default="emotic_dataset/cvpr_emotic")
    parser.add_argument("--test_annotation", type=str, default="emotic_dataset/test_bbox.txt")
    parser.add_argument("--face_bbox", type=str, default="emotic_dataset/emotic_face_bboxes_mtcnn.json")
    parser.add_argument("--body_bbox", type=str, default="emotic_dataset/emotic_body_bboxes.json")
    args_cmd = parser.parse_args()

    # ---- Setup model args ----
    class Args:
        pass
    args = Args()
    args.dataset = "EMOTIC"
    args.text_type = "prompt_ensemble"
    args.temporal_layers = 1
    args.contexts_number = 16  # Match training config
    args.class_token_position = "end"
    args.class_specific_contexts = "True"
    args.load_and_tune_prompt_learner = "True"
    args.clip_path = "ViT-B/16"
    args.drop_path_rate = 0.0
    args.freeze_image_encoder = False
    args.batch_size = 1
    args.workers = 1
    args.duration = 1
    args.image_size = 224
    args.crop_body = True
    args.use_moco = False
    args.use_weighted_sampler = False
    args.temperature = 0.07
    args.num_segments = 1  # EMOTIC = static images
    args.num_classes = 26

    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        args.device = torch.device("mps")
    else:
        args.device = torch.device("cpu")
    print(f"Using device: {args.device}")

    # Data paths
    args.root_dir = args_cmd.root_dir
    args.val_annotation = args_cmd.test_annotation
    args.bounding_box_face = args_cmd.face_bbox
    args.bounding_box_body = args_cmd.body_bbox

    # ---- Load model ----
    print(f"==> Loading CLIP ViT-B/16...")
    clip_model, _ = clip.load(args.clip_path, device=args.device)
    clip_model.float()

    # Use EMOTIC prompts
    input_text = EMOTIC_PROMPT_ENSEMBLE
    class_names = EMOTIC_CLASSES

    print(f"==> Building model with {len(class_names)} classes...")
    model = GenerateModel(input_text, clip_model, args)
    model.to(args.device)
    model.float()

    # Load checkpoint
    print(f"==> Loading checkpoint: {args_cmd.checkpoint}")
    checkpoint = torch.load(args_cmd.checkpoint, map_location=args.device, weights_only=False)
    msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
    print(f"    Load result: {msg}")
    model.eval()

    # ---- Load test data ----
    print(f"==> Loading test data from: {args.val_annotation}")
    val_data = test_data_loader(
        root_dir=args.root_dir,
        list_file=args.val_annotation,
        num_segments=args.num_segments,
        duration=args.duration,
        image_size=args.image_size,
        bounding_box_face=args.bounding_box_face,
        bounding_box_body=args.bounding_box_body,
        crop_body=args.crop_body,
        num_classes=args.num_classes
    )
    print(f"    Test dataset: {len(val_data)} samples")

    # ---- Run GradCAM ----
    generate_gradcam_emotic(model, val_data, class_names, args,
                            args_cmd.out_dir, args_cmd.samples)
