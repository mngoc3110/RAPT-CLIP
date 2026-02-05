# test_tta.py
# Script đánh giá model với Test-Time Augmentation (TTA)
# Kỹ thuật: Five-Crop + Horizontal Flip (Hoặc đơn giản là Center + Flip)

import argparse
import os
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from clip import clip
import numpy as np
import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from models.Generate_Model import GenerateModel
from models.Text import *
from utils.utils import plot_confusion_matrix
from utils.builders import get_class_info
from dataloader.video_dataloader import test_data_loader
import torchvision.transforms.functional as TF

def parse_args():
    parser = argparse.ArgumentParser(description='TTA Evaluation for RAER')
    parser.add_argument('--dataset', type=str, default='RAER')
    parser.add_argument('--root-dir', type=str, default='./')
    parser.add_argument('--test-annotation', type=str, required=True)
    parser.add_argument('--eval-checkpoint', type=str, required=True)
    parser.add_argument('--clip-path', type=str, default='ViT-B/16')
    parser.add_argument('--bounding-box-face', type=str)
    parser.add_argument('--bounding-box-body', type=str)
    parser.add_argument('--num-segments', type=int, default=16)
    parser.add_argument('--duration', type=int, default=1)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--crop-body', action='store_true')
    parser.add_argument('--text-type', default='prompt_ensemble')
    parser.add_argument('--temporal-type', default='attn_pool')
    parser.add_argument('--temporal-layers', type=int, default=1)
    parser.add_argument('--use-adapter', type=str, default='True')
    parser.add_argument('--contexts-number', type=int, default=8)
    parser.add_argument('--class-token-position', type=str, default="end")
    parser.add_argument('--class-specific-contexts', type=str, default='True')
    parser.add_argument('--load_and_tune_prompt_learner', type=str, default='True')
    parser.add_argument('--temperature', type=float, default=0.07)
    return parser.parse_args()

def run_tta(args):
    # Setup Device
    if args.gpu == 'mps':
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f="=> Using device: {device}")

    # Build Model
    print("=> Building model...")
    class_names, input_text = get_class_info(args)
    CLIP_model, _ = clip.load(args.clip_path, device='cpu')
    model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)
    model = model.to(device)
    model.eval()

    # Load Checkpoint
    print(f="=> Loading checkpoint: {args.eval_checkpoint}")
    checkpoint = torch.load(args.eval_checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)

    # Build DataLoader
    print("=> Loading test data...")
    test_data = test_data_loader(
        root_dir=args.root_dir, list_file=args.test_annotation, 
        num_segments=args.num_segments, duration=args.duration, 
        image_size=args.image_size, bounding_box_face=args.bounding_box_face,
        bounding_box_body=args.bounding_box_body, crop_body=args.crop_body,
        num_classes=len(class_names)
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # TTA Loop
    print("=> Starting TTA Evaluation (Original + Horizontal Flip)...")
    all_predicted = []
    all_targets = []

    with torch.no_grad():
        for i, (images_face, images_body, target) in enumerate(tqdm.tqdm(test_loader)):
            images_face = images_face.to(device) # (B, T, C, H, W)
            images_body = images_body.to(device)
            target = target.to(device)

            # 1. Forward Original
            output_orig, _, _, _ = model(images_face, images_body)
            
            # 2. Forward Flipped (TTA)
            # Flip face and body images across the width dimension (last dim)
            # Images are tensors, so we can use torch.flip or TF.hflip
            # Shape is (B, T, C, H, W). We want to flip W.
            images_face_flip = torch.flip(images_face, dims=[-1])
            images_body_flip = torch.flip(images_body, dims=[-1])
            
            output_flip, _, _, _ = model(images_face_flip, images_body_flip)

            # 3. Average Logits
            output_avg = (output_orig + output_flip) / 2.0
            
            predicted = output_avg.argmax(dim=1)
            all_predicted.append(predicted.cpu())
            all_targets.append(target.cpu())

    # Calculate Metrics
    all_predicted = torch.cat(all_predicted, 0)
    all_targets = torch.cat(all_targets, 0)

    correct = (all_predicted == all_targets).sum().item()
    war = 100. * correct / len(test_loader.dataset)

    cm = confusion_matrix(all_targets.numpy(), all_predicted.numpy())
    class_recall = cm.diagonal() / cm.sum(axis=1)
    class_recall[np.isnan(class_recall)] = 0 
    uar = np.mean(class_recall) * 100.0

    print("\n**************** TTA RESULTS ****************")
    print(f"Confusion Matrix Diag (%): {np.diag(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100)}")
    print(f"UAR (Test): {uar:.2f}%")
    print(f"WAR (Accuracy): {war:.2f}%")
    print("*********************************************\n")

if __name__ == '__main__':
    args = parse_args()
    run_tta(args)
