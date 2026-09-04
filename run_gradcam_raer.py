import argparse
import os
import sys
import torch
import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from models.Generate_Model import GenerateModel
from models.clip import clip

# RAER 5 Emotion Categories
RAER_CLASSES = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']

def compute_token_similarity(model, face_input, body_input, target_class, stream='face', **kwargs):
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

    transformer_output = {}
    def hook_fn(module, input, output):
        transformer_output['out'] = output.detach()
    handle = model.image_encoder.transformer.register_forward_hook(hook_fn)

    with torch.no_grad():
        if stream == 'face':
            target_4d = face_input
        elif stream == 'body':
            target_4d = body_input
        else:
            target_4d = kwargs.get('context_input')
        
        target_4d = target_4d.contiguous().view(-1, c, h, w)
        _ = model.image_encoder(target_4d.type(model.dtype))
    handle.remove()

    if 'out' not in transformer_output:
        return np.zeros((14, 14), dtype=np.float32)

    with torch.no_grad():
        feat = transformer_output['out'].permute(1, 0, 2)
        spatial = feat[mid_idx, 1:]  # (196, 768)
        spatial_n = model.image_encoder.ln_post(spatial)
        if model.image_encoder.proj is not None:
            spatial_p = spatial_n @ model.image_encoder.proj
        else:
            spatial_p = spatial_n
        spatial_p = spatial_p / (spatial_p.norm(dim=-1, keepdim=True) + 1e-6)

        all_sim = spatial_p @ all_class_text.t()  
        target_sim = all_sim[:, target_class]
        mean_sim = all_sim.mean(dim=1)
        disc_sim = (target_sim - mean_sim).cpu().numpy().reshape(14, 14)

        cam = np.maximum(disc_sim, 0)
        cam_max = cam.max()
        if cam_max > 1e-7:
            cam = cam / cam_max
        return cam

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

def create_raer_grid(orig_f, orig_b, orig_c, att_f, att_b, att_c, true_label, top_emotion):
    pad = 10
    h, w = orig_f.shape[:2]
    
    grid_w = w * 3 + pad * 4
    grid_h = h * 2 + pad * 3 + 80  
    
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    
    # Text headers
    cv2.putText(grid, "Face", (pad + w//2 - 20, pad + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(grid, "Body", (pad*2 + w + w//2 - 20, pad + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(grid, "Context", (pad*3 + w*2 + w//2 - 35, pad + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    top_offset = pad + 30
    grid[top_offset:top_offset+h, pad:pad+w] = orig_f
    grid[top_offset:top_offset+h, pad*2+w:pad*2+w*2] = orig_b
    grid[top_offset:top_offset+h, pad*3+w*2:pad*3+w*3] = orig_c
    
    grid[top_offset+pad+h:top_offset+pad+h*2, pad:pad+w] = att_f
    grid[top_offset+pad+h:top_offset+pad+h*2, pad*2+w:pad*2+w*2] = att_b
    grid[top_offset+pad+h:top_offset+pad+h*2, pad*3+w*2:pad*3+w*3] = att_c
    
    text_y = top_offset + pad + h*2 + 30
    cv2.putText(grid, f"True: {RAER_CLASSES[true_label]} | Pred: {RAER_CLASSES[top_emotion]}", 
                (pad, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return grid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='RAER')
    parser.add_argument('--text-type', default='prompt_ensemble')
    parser.add_argument('--class-names-with-context', default='False')
    parser.add_argument('--temporal-layers', default=1, type=int)
    parser.add_argument('--contexts-number', default=8, type=int)
    parser.add_argument('--class-token-position', default='end')
    parser.add_argument('--class-specific-contexts', default='True')
    parser.add_argument('--load-and-tune-prompt-learner', default='True')
    parser.add_argument('--clip-path', default='ViT-B/16')
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--num-workers', default=1, type=int)
    parser.add_argument('--use-context', action='store_true')
    parser.add_argument('--fusion-type', default='cmaf')
    parser.add_argument('--is-ensemble', action='store_true')
    parser.add_argument('--use-cocoop', action='store_true')
    parser.add_argument('--use-label-gcn', action='store_true')
    parser.add_argument('--drop_path_rate', default=0.0, type=float)
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    args.freeze_image_encoder = False
    args.lr_image_encoder = 0
    args.use_ldl = False
    args.ldl_warmup = 5
    args.moco_dim = 512
    args.moco_k = 65536
    args.moco_m = 0.999
    args.moco_t = 0.07
    args.use_moco = False
    args.use_context = True
    args.temperature = 0.07
    
    args.train_annotation = "RAER/annotation/train.txt"
    args.val_annotation = "RAER/annotation/val.txt"
    args.test_annotation = "RAER/annotation/test.txt"
    args.bounding_box_face = "RAER/bounding_box/face.json"
    args.bounding_box_body = "RAER/bounding_box/body.json"
    args.data_root = "RAER"
    args.crop_body = True
    
    from utils.builders import get_class_info, build_model
    class_names, input_text = get_class_info(args)
    model = build_model(args, input_text).to(args.device)

    ckpt_path = "outputs/RAER-FrameLevelFusion-[06-07]-[03:06]/model_best.pth"
    print(f"=> Loading {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    from dataloader.video_dataloader import test_data_loader
    from torch.utils.data import DataLoader
    test_dataset = test_data_loader(
        root_dir="", list_file="RAER/annotation/test.txt", num_segments=16, duration=1,
        image_size=224, bounding_box_face="RAER/bounding_box/face.json", bounding_box_body="RAER/bounding_box/body.json",
        crop_body=True, num_classes=5
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    out_dir = "outputs/gradcam_raer_16f"
    os.makedirs(out_dir, exist_ok=True)

    for i, batch in enumerate(test_loader):
        if len(batch) == 4:
            img_f, img_b, img_c, target = batch
        else:
            img_f, img_b, target = batch
            img_c = None

        img_f = img_f.to(args.device)
        img_b = img_b.to(args.device)
        if img_c is not None: img_c = img_c.to(args.device)
        target = target.item()

        with torch.no_grad():
            if img_c is not None:
                output_tuple = model(img_f, img_b, img_c)
            else:
                output_tuple = model(img_f, img_b)
            output = output_tuple[0]
            pred_class = output.argmax(dim=1).item()

        cam_face = compute_token_similarity(model, img_f, img_b, pred_class, stream='face')
        cam_body = compute_token_similarity(model, img_f, img_b, pred_class, stream='body')
        cam_context = compute_token_similarity(model, img_f, img_b, pred_class, stream='context', context_input=img_c)

        t = img_f.shape[1]
        mid = t // 2
        orig_f_bgr = unnormalize_tensor(img_f[0, mid], args.device)
        orig_b_bgr = unnormalize_tensor(img_b[0, mid], args.device)
        orig_c_bgr = unnormalize_tensor(img_c[0, mid], args.device) if img_c is not None else orig_b_bgr

        att_f_bgr = overlay_heatmap(orig_f_bgr, cam_face)
        att_b_bgr = overlay_heatmap(orig_b_bgr, cam_body)
        att_c_bgr = overlay_heatmap(orig_c_bgr, cam_context)

        grid = create_raer_grid(orig_f_bgr, orig_b_bgr, orig_c_bgr, att_f_bgr, att_b_bgr, att_c_bgr, target, pred_class)
        save_path = os.path.join(out_dir, f"raer_sample_{i}_true_{RAER_CLASSES[target]}_pred_{RAER_CLASSES[pred_class]}.png")
        cv2.imwrite(save_path, grid)
        print(f"Saved {save_path}")

if __name__ == '__main__':
    main()
