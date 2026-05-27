import argparse
import os
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms.functional as TF
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from utils.builders import build_model, get_class_info, build_dataloaders
from utils.utils import AverageMeter

# Argument Parser (Simplified for Eval)
parser = argparse.ArgumentParser(description='RAER TTA Evaluation')
parser.add_argument('--eval-checkpoint', type=str, required=True, help='Path to checkpoint')
parser.add_argument('--dataset', type=str, default='RAER')
parser.add_argument('--root-dir', type=str, default='./dataset/RAER')
parser.add_argument('--test-annotation', type=str, default='RAER/annotation/test.txt')
# Dummy paths for builder compatibility
parser.add_argument('--train-annotation', type=str, default='RAER/annotation/train_80.txt') 
parser.add_argument('--val-annotation', type=str, default='RAER/annotation/val_20.txt')
parser.add_argument('--clip-path', type=str, default='ViT-B/16')
parser.add_argument('--bounding-box-face', type=str, default='RAER/bounding_box/face.json')
parser.add_argument('--bounding-box-body', type=str, default='RAER/bounding_box/body.json')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--use-amp', action='store_true')
parser.add_argument('--text-type', default='prompt_ensemble')
parser.add_argument('--temporal-layers', type=int, default=3) # Must match training!
parser.add_argument('--contexts-number', type=int, default=8)
parser.add_argument('--class-token-position', type=str, default="end")
parser.add_argument('--class-specific-contexts', type=str, default='True')
parser.add_argument('--load_and_tune_prompt_learner', type=str, default='True')
parser.add_argument('--num-segments', type=int, default=16)
parser.add_argument('--duration', type=int, default=1)
parser.add_argument('--image-size', type=int, default=224)
parser.add_argument('--crop-body', action='store_true')
parser.add_argument('--temperature', type=float, default=0.07)
# Add MoCo args just to satisfy builder (even if not used in eval)
parser.add_argument('--use-moco', action='store_true')
parser.add_argument('--moco-k', type=int, default=2048)
parser.add_argument('--moco-m', type=float, default=0.999)
parser.add_argument('--moco-t', type=float, default=0.07)
parser.add_argument('--lr-image-encoder', type=float, default=0) # Dummy
parser.add_argument('--use-weighted-sampler', action='store_true', help='Dummy arg for builder compatibility')
# New Bias Argument
parser.add_argument('--confusion-bias', type=float, default=1.0, help='Multiplier for Confusion class probability')
parser.add_argument('--drop_path_rate', default=0.0, type=float, help='Drop path rate for ViT')

def main():
    args = parser.parse_args()
    
    # Device setup
    if args.gpu == 'mps':
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    args.device = device
    print(f"Using device: {device}")

    # Build Model
    class_names, input_text = get_class_info(args)
    model = build_model(args, input_text)
    model = model.to(device)
    model.eval()

    # Load Checkpoint
    print(f"=> Loading checkpoint: {args.eval_checkpoint}")
    checkpoint = torch.load(args.eval_checkpoint, map_location=device, weights_only=False)
    # Use strict=False for robustness
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # Build DataLoader (Only Test)
    print("=> Building Test DataLoader...")
    # We call build_dataloaders but only use the last one
    _, _, test_loader = build_dataloaders(args)
    
    print(f"=> Starting TTA Evaluation (Standard + Horizontal Flip) with Confusion Bias={args.confusion_bias}...")
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="TTA Eval")):
            if len(batch) == 4:
                images_face, images_body, images_context, target = batch
                images_context = images_context.to(device)
            else:
                images_face, images_body, target = batch
                images_context = None

            images_face = images_face.to(device)
            images_body = images_body.to(device)
            target = target.to(device)

            # 1. Standard Forward
            if images_context is not None:
                output_std, _, _, _ = model(images_face, images_body, images_context)
            else:
                output_std, _, _, _ = model(images_face, images_body)
            prob_std = torch.softmax(output_std, dim=1)

            # 2. Flipped Forward (TTA)
            images_face_flip = torch.flip(images_face, dims=[-1])
            images_body_flip = torch.flip(images_body, dims=[-1])
            
            if images_context is not None:
                images_context_flip = torch.flip(images_context, dims=[-1])
                output_flip, _, _, _ = model(images_face_flip, images_body_flip, images_context_flip)
            else:
                output_flip, _, _, _ = model(images_face_flip, images_body_flip)
            prob_flip = torch.softmax(output_flip, dim=1)

            # 3. Ensemble (Average Probabilities)
            avg_prob = (prob_std + prob_flip) / 2.0
            
            # 4. Apply Confusion Bias (Index 2)
            if args.confusion_bias != 1.0:
                avg_prob[:, 2] *= args.confusion_bias
            
            preds = avg_prob.argmax(dim=1)
            
            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())

    # Calculate Metrics
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    cm = confusion_matrix(all_targets, all_preds)
    acc = np.trace(cm) / np.sum(cm) * 100
    class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6) * 100
    uar = np.nanmean(class_acc)

    print("\n" + "="*40)
    print("      TTA + BIAS EVALUATION RESULTS")
    print("="*40)
    print(f"Confusion Matrix:\n{cm}")
    print("-" * 40)
    print("Per-Class Accuracy (%):")
    for i, name in enumerate(class_names):
        print(f"  {name:<12}: {class_acc[i]:.2f}")
    print("-" * 40)
    print(f"WAR (Accuracy): {acc:.2f}%")
    print(f"UAR (Avg Recall): {uar:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()
