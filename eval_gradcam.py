import argparse
import os
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models.Generate_Model import GenerateModel
from models.clip import clip
from utils.builders import get_class_info, build_dataloaders
from gradcam import GradCAM, overlay_cam_on_image

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
    
    # Setup device
    if torch.backends.mps.is_available():
        args.device = torch.device("mps")
    elif torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
        
    args.freeze_image_encoder = False
    
    # Dataloader args
    args.batch_size = 1
    args.workers = 1
    args.root_dir = os.path.abspath("./")
    args.train_annotation = os.path.abspath("RAER/annotation/train.txt")
    args.val_annotation = os.path.abspath("RAER/annotation/test.txt")
    args.test_annotation = os.path.abspath("RAER/annotation/test.txt")
    args.bounding_box_face = os.path.abspath("RAER/bounding_box/face.json")
    args.bounding_box_body = os.path.abspath("RAER/bounding_box/body.json")
    # MUST match training config for correct inference
    args.num_segments = 16
    args.duration = 1
    args.image_size = 224
    args.crop_body = True  # MUST match training config
    args.use_moco = False
    args.use_weighted_sampler = False
    args.temperature = 0.07
    
    return args

def load_model(args, checkpoint_path):
    clip_model, _ = clip.load(args.clip_path, device=args.device)
    clip_model.float() # Force float32 for GradCAM stability on MPS
    class_names, input_text = get_class_info(args)
    
    model = GenerateModel(input_text, clip_model, args)
    model.to(args.device)
    model.float() # Force float32
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using untrained weights.")
        
    model.eval()
    return model, class_names

def process_batch_gradcam(model, val_loader, output_dir, class_names, args, num_samples=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Get target layer: moving back to block -6 for higher spatial fidelity
    layer_idx = -6
    target_layer = model.image_encoder.transformer.resblocks[layer_idx]
    cam = GradCAM(model, target_layer)
    
    count = 0
    
    # We need to un-normalize images for visualization
    # CLIP normalization values
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(args.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(args.device)
    
    print(f"Starting GradCAM evaluation on {num_samples} samples using block {layer_idx}...")
    
    for i, (image_face, image_body, labels) in enumerate(val_loader):
        if count >= num_samples:
            break
            
        image_face = image_face.to(args.device)
        image_face.requires_grad_(True)
        
        image_body = image_body.to(args.device)
        image_body.requires_grad_(True)
        
        labels = labels.to(args.device)
        
        # Calculate GradCAM
        mask, target_idx = cam(image_face, image_body, target_class=labels[0].item())
        
        if mask is None:
            print("GradCAM returned None. Skipping.")
            continue
            
        # Prepare original image for overlay
        # image_face shape: [batch, time, channel, H, W] -> get first batch, middle time step
        mid_t = image_face.shape[1] // 2
        img_tensor = image_face[0, mid_t] 
        
        # Un-normalize
        img_unnorm = img_tensor.detach() * std + mean
        img_unnorm = img_unnorm.clamp(0, 1).cpu().numpy()
        
        # Convert to HWC for OpenCV
        img_hwc = np.transpose(img_unnorm, (1, 2, 0))
        img_bgr = cv2.cvtColor((img_hwc * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Resize mask
        mask_resized = cv2.resize(mask, (img_hwc.shape[1], img_hwc.shape[0]))
        
        # Create overlay
        result = overlay_cam_on_image(img_bgr, mask_resized)
        result = np.ascontiguousarray(result) # Fix OpenCV putText error
        
        # True and Pred classes
        true_class = class_names[labels[0].item()]
        pred_class = class_names[target_idx]
        
        # Setup display text
        text = f"T: {true_class} | P: {pred_class}"
        color = (0, 255, 0) if true_class == pred_class else (0, 0, 255)
        
        cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Save output
        v_name = f"sample_{count}"
        out_path = os.path.join(output_dir, f"cam_{v_name}_layer{abs(layer_idx)}.jpg")
        cv2.imwrite(out_path, result)
        print(f"[{count+1}/{num_samples}] Saved GradCAM for {v_name} to {out_path}")
        
        count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--out_dir", type=str, default="outputs/grad_cam_results", help="Output directory")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "val", "test"])
    
    args_cmd = parser.parse_args()
    
    args = setup_args(args_cmd.checkpoint)
    model, class_names = load_model(args, args_cmd.checkpoint)
    
    _, val_loader, test_loader = build_dataloaders(args)
    
    loader = test_loader if args_cmd.data_split == "test" else val_loader
    
    # Note: dataset needs to be present in RAER/ path for this to work
    try:
        process_batch_gradcam(model, loader, args_cmd.out_dir, class_names, args, num_samples=args_cmd.samples)
        print("GradCAM evaluation completed successfully.")
    except Exception as e:
        print(f"Error during GradCAM evaluation: {e}")
        print("Make sure your dataset paths in setup_args() are correct and accessible.")
