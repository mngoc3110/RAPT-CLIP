"""
Gradient-weighted Attention Rollout for RAER dual-stream CLIP-ViT model.

Standard GradCAM CANNOT work on CLIP ViT because the image_encoder only 
propagates the CLS token through subsequent layers → gradient = 0 for all 
spatial tokens (1-196). Instead, we use Attention Rollout weighted by 
CLS-token gradients to create class-specific spatial attention maps.

Approach:
1. Forward pass through the target stream, capturing attention weights 
   from each transformer layer's MultiheadAttention.
2. Backward pass to get CLS token gradients at each layer.
3. Weight each layer's attention map by the CLS gradient magnitude.
4. Rollout: multiply attention matrices across layers to propagate 
   information from CLS → spatial tokens.
5. Extract CLS row → 14x14 spatial map.
"""
import argparse
import os
import torch
import cv2
import numpy as np

from models.Generate_Model import GenerateModel
from models.clip import clip
from utils.builders import get_class_info
from dataloader.video_dataloader import test_data_loader
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# Token-level CLIP Similarity Map
# ============================================================
def compute_token_similarity(model, face_input, body_input, target_class, stream='face'):
    """
    Compute spatial similarity map by projecting EACH spatial patch token 
    through CLIP's ln_post + proj (same as CLS), then computing cosine 
    similarity with the target class text embedding.
    
    Returns:
        cam: (14, 14) numpy array, normalized [0, 1]
    """
    n, t, c, h, w = face_input.shape
    mid_idx = t // 2
    
    # Get text embeddings for ALL classes (for discriminative similarity)
    with torch.no_grad():
        prompts = model.prompt_learner()
        text_features = model.text_encoder(prompts, model.tokenized_prompts).float()
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
        
        if model.is_ensemble:
            text_features = text_features.view(model.num_classes, model.num_prompts_per_class, -1)
            # Average across prompts per class → (num_classes, embed_dim)
            all_class_text = text_features.mean(dim=1)
        else:
            all_class_text = text_features  # (num_classes, embed_dim)
        
        all_class_text = all_class_text / (all_class_text.norm(dim=-1, keepdim=True) + 1e-6)
    
    # Hook to capture transformer output
    transformer_output = {}
    def hook_fn(module, input, output):
        transformer_output['out'] = output.detach()
    
    handle = model.image_encoder.transformer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        if stream == 'face':
            target_4d = face_input.contiguous().view(-1, c, h, w)
        else:
            target_4d = body_input.contiguous().view(-1, c, h, w)
        _ = model.image_encoder(target_4d)
    
    handle.remove()
    
    if 'out' not in transformer_output:
        return np.zeros((14, 14), dtype=np.float32)
    
    with torch.no_grad():
        feat = transformer_output['out'].permute(1, 0, 2)  # (B*T, 197, 768)
        spatial = feat[mid_idx, 1:]  # (196, 768)
        
        spatial_n = model.image_encoder.ln_post(spatial)
        if model.image_encoder.proj is not None:
            spatial_p = spatial_n @ model.image_encoder.proj
        else:
            spatial_p = spatial_n
        spatial_p = spatial_p / (spatial_p.norm(dim=-1, keepdim=True) + 1e-6)
        
        # Similarity with ALL classes: (196, num_classes)
        all_sim = spatial_p @ all_class_text.t()
        
        # Class-discriminative: target_sim - mean(other_sims)
        target_sim = all_sim[:, target_class]  # (196,)
        mean_sim = all_sim.mean(dim=1)         # (196,) — mean across all classes
        disc_sim = (target_sim - mean_sim).cpu().numpy().reshape(14, 14)
        
        # Keep only positive discriminative values
        cam = np.maximum(disc_sim, 0)
        
        cam_max = cam.max()
        if cam_max > 1e-7:
            cam = cam / cam_max
        
        return cam


# ============================================================  
# Visualization helpers
# ============================================================
def overlay_heatmap(img_bgr, cam, alpha=0.5):
    """Intensity-weighted alpha blending: zero-attention areas stay original."""
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


def create_grid_2x3(orig_f, orig_b, orig_c, att_f, att_b, att_c, true_cls, pred_cls):
    s = 224
    imgs = [cv2.resize(x, (s, s)) for x in [orig_f, orig_b, orig_c, att_f, att_b, att_c]]
    
    pad_top = 40; gap = 4
    grid_w = s * 3 + gap * 2
    grid_h = s * 2 + pad_top * 2 + gap
    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    titles = ["Face", "Body (Cropped)", "Context (Full)"]
    for col, title in enumerate(titles):
        x = col * (s + gap)
        tw = cv2.getTextSize(title, font, 0.55, 2)[0][0]
        cv2.putText(canvas, title, (x + (s - tw) // 2, 25), font, 0.55, (0, 0, 0), 2)
    
    for col in range(3):
        x = col * (s + gap)
        canvas[pad_top:pad_top+s, x:x+s] = imgs[col]
    
    for col in range(3):
        x = col * (s + gap)
        y_label = pad_top + s + gap
        tw = cv2.getTextSize("Attention", font, 0.4, 1)[0][0]
        cv2.putText(canvas, "Attention", (x + (s - tw) // 2, y_label + 10), font, 0.4, (80, 80, 80), 1)
    
    y_bottom = pad_top * 2 + s + gap
    for col in range(3):
        x = col * (s + gap)
        canvas[y_bottom:y_bottom+s, x:x+s] = imgs[3 + col]
    
    color = (0, 180, 0) if true_cls == pred_cls else (0, 0, 220)
    cv2.putText(canvas, f"True: {true_cls} | Pred: {pred_cls}", (6, grid_h - 6), font, 0.5, color, 2)
    
    return canvas


# ============================================================
# Model Setup
# ============================================================
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
    args.batch_size = 1
    args.workers = 1
    args.root_dir = os.path.abspath("./")
    args.val_annotation = os.path.abspath("RAER/annotation/test.txt")
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
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    model.eval()
    return model, class_names


# ============================================================
# Main
# ============================================================
def generate_thesis_assets(model, loader, class_names, args, out_dir, num_samples=5):
    os.makedirs(out_dir, exist_ok=True)
    dataset = loader.dataset
    count = 0
    
    for i in range(len(dataset)):
        if count >= num_samples:
            break
        
        img_f, img_b, label = dataset[i]
        record = dataset.video_list[i]
        
        indices = dataset._get_test_indices(record)
        mid_seg = len(indices) // 2
        p = int(indices[mid_seg])
        
        img_f_input = img_f.unsqueeze(0).to(args.device).float()
        img_b_input = img_b.unsqueeze(0).to(args.device).float()
        n, t, c, h, w = img_f_input.shape
        
        # 1. Get prediction
        with torch.no_grad():
            output, _, _, _ = model(img_f_input, img_b_input)
            pred_idx = torch.argmax(output, dim=-1).item()
        
        true_cls = class_names[label]
        pred_cls = class_names[pred_idx]
        
        # 2. Compute attention for face
        cam_face = compute_token_similarity(model, img_f_input, img_b_input, pred_idx, stream='face')
        print(f"  [face] cam min={cam_face.min():.4f} max={cam_face.max():.4f} mean={cam_face.mean():.4f}")
        
        # 3. Compute attention for body
        cam_body_raw = compute_token_similarity(model, img_f_input, img_b_input, pred_idx, stream='body')
        cam_body = cam_body_raw.copy()
        
        print(f"  [body] cam min={cam_body.min():.4f} max={cam_body.max():.4f} mean={cam_body.mean():.4f}")
        
        # 4. Load RAW frame (no augmentation/transform)
        orig_ctx_bgr = None
        raw_frame_path = None
        if os.path.isdir(record.path):
            import glob
            frames = sorted(glob.glob(os.path.join(record.path, '*')))
            fp = min(p, len(frames) - 1)
            raw_frame_path = frames[fp]
            try:
                from PIL import Image
                orig_ctx_bgr = cv2.cvtColor(np.array(Image.open(frames[fp]).convert('RGB')), cv2.COLOR_RGB2BGR)
            except:
                pass
        else:
            cap = cv2.VideoCapture(record.path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(p, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
            ret, frame = cap.read()
            if ret: orig_ctx_bgr = frame
            cap.release()
        
        if orig_ctx_bgr is None:
            orig_ctx_bgr = unnormalize_tensor(img_b[mid_seg], device='cpu')
        
        # Get face and body bounding boxes for context masking
        rel_path = record.path.replace('./', '').replace('\\', '/')
        video_key_full = os.path.splitext(rel_path)[0]
        video_key_alt = video_key_full.replace('train', 'images').replace('test', 'images').replace('val', 'images')
        
        # Find matching key in body_boxes
        matched_video_key = None
        if hasattr(dataset, 'body_boxes'):
            for potential_key in [video_key_full, video_key_alt]:
                if potential_key in dataset.body_boxes:
                    matched_video_key = potential_key
                    break
                parts = potential_key.split('/')
                for idx in range(1, len(parts)):
                    sub_key = '/'.join(parts[idx:])
                    if sub_key in dataset.body_boxes:
                        matched_video_key = sub_key
                        break
                if matched_video_key: break
        
        # Get body box
        body_box = None
        if matched_video_key and matched_video_key in dataset.body_boxes:
            box_data = dataset.body_boxes[matched_video_key]
            frame_key = f"{p}.jpg" if not os.path.isdir(record.path) else os.path.basename(frames[fp])
            if isinstance(box_data, dict) and frame_key in box_data:
                body_box = box_data[frame_key]
            elif isinstance(box_data, list):
                if len(box_data) == 4 and all(isinstance(x, (int, float)) for x in box_data):
                    body_box = box_data
                elif len(box_data) > 0 and isinstance(box_data[0], list):
                    body_box = box_data[min(p, len(box_data)-1)]
        
        # Get face box 
        face_box = None
        if hasattr(dataset, 'boxs'):
            # Same key lookup as body
            for potential_key in [video_key_full, video_key_alt]:
                if potential_key in dataset.boxs:
                    fb_data = dataset.boxs[potential_key]
                    frame_key = f"{p}.jpg" if not os.path.isdir(record.path) else os.path.basename(frames[fp])
                    if isinstance(fb_data, dict) and frame_key in fb_data:
                        face_box = fb_data[frame_key]
                    elif isinstance(fb_data, list):
                        if len(fb_data) == 4 and all(isinstance(x, (int, float)) for x in fb_data):
                            face_box = fb_data
                        elif len(fb_data) > 0 and isinstance(fb_data[0], list):
                            face_box = fb_data[min(p, len(fb_data)-1)]
                    break
                parts = potential_key.split('/')
                for idx_k in range(1, len(parts)):
                    sub_key = '/'.join(parts[idx_k:])
                    if sub_key in dataset.boxs:
                        fb_data = dataset.boxs[sub_key]
                        frame_key = f"{p}.jpg" if not os.path.isdir(record.path) else os.path.basename(frames[fp])
                        if isinstance(fb_data, dict) and frame_key in fb_data:
                            face_box = fb_data[frame_key]
                        elif isinstance(fb_data, list):
                            if len(fb_data) == 4 and all(isinstance(x, (int, float)) for x in fb_data):
                                face_box = fb_data
                            elif len(fb_data) > 0 and isinstance(fb_data[0], list):
                                face_box = fb_data[min(p, len(fb_data)-1)]
                        break
                if face_box: break
        
        # 5. Crop display images
        ch, cw = orig_ctx_bgr.shape[:2]
        
        # Face: always use model's face input (correct face from detector)
        orig_f_bgr = unnormalize_tensor(img_f[mid_seg], device='cpu')
        
        # Body: use raw crop from frame (no padding)
        if body_box is not None:
            bl, bu, br, blo = [int(v) for v in body_box]
            bl, bu = max(0, bl), max(0, bu)
            br, blo = min(cw, br), min(ch, blo)
            if br > bl and blo > bu:
                orig_b_bgr = orig_ctx_bgr[bu:blo, bl:br].copy()
            else:
                orig_b_bgr = unnormalize_tensor(img_b[mid_seg], device='cpu')
        else:
            orig_b_bgr = unnormalize_tensor(img_b[mid_seg], device='cpu')
        
        # 6. Overlay face
        att_f = overlay_heatmap(orig_f_bgr, cam_face)
        
        # Mask face region from body cam (face bbox relative to body crop → 14x14 grid)
        if face_box is not None and body_box is not None:
            bl_f, bu_f, br_f, blo_f = [float(v) for v in body_box]
            bw_orig = br_f - bl_f
            bh_orig = blo_f - bu_f
            if bw_orig > 0 and bh_orig > 0:
                fl, fu, fr, flo = [float(v) for v in face_box]
                # Face coords relative to body crop
                rel_fl = (fl - bl_f) / bw_orig
                rel_fu = (fu - bu_f) / bh_orig 
                rel_fr = (fr - bl_f) / bw_orig
                rel_flo = (flo - bu_f) / bh_orig
                # Map to 14x14 grid
                g_l = max(0, int(rel_fl * 14))
                g_u = max(0, int(rel_fu * 14))
                g_r = min(14, int(np.ceil(rel_fr * 14)))
                g_lo = min(14, int(np.ceil(rel_flo * 14)))
                if g_r > g_l and g_lo > g_u:
                    cam_body[g_u:g_lo, g_l:g_r] = 0
                    # Renormalize
                    cm = cam_body.max()
                    if cm > 1e-7:
                        cam_body = cam_body / cm
        
        # Overlay body (after face masking)
        att_b = overlay_heatmap(orig_b_bgr, cam_body)
        
        # Context heatmap: paste body cam into body bbox, then mask OUT face+body regions
        ch, cw = orig_ctx_bgr.shape[:2]
        cam_ctx = np.zeros((ch, cw), dtype=np.float32)
        
        # First, paste body cam into body region (gives a base heatmap)
        if body_box is not None:
            left, upper, right, lower = [int(v) for v in body_box]
            left, upper = max(0, left), max(0, upper)
            right, lower = min(cw, right), min(ch, lower)
            bw, bh = right - left, lower - upper
            if bw > 0 and bh > 0:
                cam_ctx[upper:lower, left:right] = cv2.resize(cam_body_raw, (bw, bh), interpolation=cv2.INTER_CUBIC)
        
        # Mask OUT face and body regions → only context remains
        if face_box is not None:
            fl, fu, fr, flo = [int(v) for v in face_box]
            fl, fu = max(0, fl), max(0, fu)
            fr, flo = min(cw, fr), min(ch, flo)
            cam_ctx[fu:flo, fl:fr] = 0
        
        if body_box is not None:
            left, upper, right, lower = [int(v) for v in body_box]
            left, upper = max(0, left), max(0, upper)
            right, lower = min(cw, right), min(ch, lower)
            cam_ctx[upper:lower, left:right] = 0
        
        att_ctx = overlay_heatmap(orig_ctx_bgr, cam_ctx)
        
        # 7. Save
        prefix = f"sample_{count}_true_{true_cls}_pred_{pred_cls}"
        grid = create_grid_2x3(orig_f_bgr, orig_b_bgr, orig_ctx_bgr, att_f, att_b, att_ctx, true_cls, pred_cls)
        cv2.imwrite(f"{out_dir}/{prefix}_GRID.jpg", grid)
        
        print(f"[{count+1}/{num_samples}] T:{true_cls} P:{pred_cls}")
        count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/thesis_assets")
    parser.add_argument("--samples", type=int, default=5)
    args_cmd = parser.parse_args()
    
    args = setup_args(args_cmd.checkpoint)
    model, class_names = load_model(args, args_cmd.checkpoint)
    
    class_names_list, _ = get_class_info(args)
    num_classes = len(class_names_list)
    
    val_data = test_data_loader(
        root_dir=args.root_dir, list_file=args.val_annotation,
        num_segments=args.num_segments, duration=args.duration,
        image_size=args.image_size, bounding_box_face=args.bounding_box_face,
        bounding_box_body=args.bounding_box_body, crop_body=args.crop_body,
        num_classes=num_classes
    )
    
    import torch.utils.data
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False)
    
    print(f"Val dataset: {len(val_data)} samples")
    generate_thesis_assets(model, val_loader, class_names, args, args_cmd.out_dir, args_cmd.samples)
    print("Done!")
