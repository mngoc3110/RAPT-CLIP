# train_distill.py
import argparse
import datetime
import os
import random
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from utils.builders import build_model, get_class_info, build_dataloaders
from utils.checkpoint_utils import load_slim_checkpoint
from utils.utils import AverageMeter, RecorderMeter, plot_confusion_matrix, computer_uar_war
from models.Student_Model import StudentModel

# Suppress OpenCV and FFmpeg warnings
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_DEBUG_LOG_LEVEL"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)

# ==================== Argument Parser ====================
parser = argparse.ArgumentParser(
    description='Knowledge Distillation pipeline for RAPT-CLIP Student Model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# --- Basic Experiment ---
parser.add_argument('--exper-name', type=str, default='Student-Distill', help='Experiment output folder name.')
parser.add_argument('--dataset', type=str, default='RAER', help='Name of the dataset to use.')
parser.add_argument('--gpu', type=str, default='mps', help='ID of GPU or "mps" for Apple Silicon.')
parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

# --- Distillation Settings ---
parser.add_argument('--teacher-checkpoint', type=str, required=True, help='Path to the trained RAPT-CLIP teacher slim checkpoint (e.g. outputs/Ablation-.../model_best_slim.pth).')
parser.add_argument('--alpha-distill', type=float, default=0.5, help='Weight for distillation loss vs hard target loss.')
parser.add_argument('--beta-feature', type=float, default=1.0, help='Weight for visual feature alignment MSE loss.')
parser.add_argument('--distill-temp', type=float, default=3.0, help='Temperature for smoothing logit distributions in KL Divergence.')

# --- Data Paths ---
parser.add_argument('--root-dir', type=str, default='./', help='Root directory of the dataset.')
parser.add_argument('--train-annotation', type=str, default='RAER/annotation/train_80.txt', help='Training annotation file.')
parser.add_argument('--val-annotation', type=str, default='RAER/annotation/val_20.txt', help='Validation annotation file.')
parser.add_argument('--test-annotation', type=str, default='RAER/annotation/test.txt', help='Testing annotation file.')
parser.add_argument('--bounding-box-face', type=str, default='RAER/bounding_box/face.json', help='Face bounding box JSON.')
parser.add_argument('--bounding-box-body', type=str, default='RAER/bounding_box/body.json', help='Body bounding box JSON.')

# --- Training Control ---
parser.add_argument('--epochs', type=int, default=30, help='Total number of epochs for student training.')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training and validation.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate for the Student model.')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay.')
parser.add_argument('--milestones', nargs='+', type=int, default=[15, 25], help='Epochs to decay learning rate by 0.1.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR decay factor.')
parser.add_argument('--use-amp', action='store_true', help='Use Automatic Mixed Precision.')
parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping threshold.')

# --- Model Specifics ---
parser.add_argument('--num-segments', type=int, default=16, help='Number of segments sampled per video.')
parser.add_argument('--duration', type=int, default=1, help='Duration of each segment.')
parser.add_argument('--image-size', type=int, default=224, help='Input image size.')
parser.add_argument('--crop-body', action='store_true', help='Crop body from the input images.')
parser.add_argument('--use-weighted-sampler', action='store_true', help='Use WeightedRandomSampler for training.')

# --- Teacher Mock Args (needed for build_model factory) ---
parser.add_argument('--clip-path', type=str, default='ViT-B/16', help='Path to the pretrained CLIP model.')
parser.add_argument('--lr-image-encoder', type=float, default=0.0, help='Placeholder for teacher.')
parser.add_argument('--freeze-image-encoder', action='store_true', default=True, help='Placeholder for teacher.')
parser.add_argument('--lr-prompt-learner', type=float, default=0.0, help='Placeholder for teacher.')
parser.add_argument('--lr-adapter', type=float, default=0.0, help='Placeholder for teacher.')
parser.add_argument('--text-type', default='prompt_ensemble', help='Placeholder for teacher.')
parser.add_argument('--temporal-layers', type=int, default=1, help='Placeholder for teacher.')
parser.add_argument('--contexts-number', type=int, default=8, help='Placeholder for teacher.')
parser.add_argument('--class-token-position', type=str, default="end", help='Placeholder for teacher.')
parser.add_argument('--class-specific-contexts', type=str, default='True', help='Placeholder for teacher.')
parser.add_argument('--load_and_tune_prompt_learner', type=str, default='True', help='Placeholder for teacher.')
parser.add_argument('--temperature', type=float, default=0.07, help='Placeholder for teacher.')
parser.add_argument('--drop-path-rate', type=float, default=0.0, help='Placeholder for teacher.')

def setup_environment(args):
    if args.gpu == 'mps':
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    
    args.device = device
    if device.type == 'cpu':
        args.use_amp = False
        print("=> Device is CPU. Disabling AMP.")
    print(f"Using device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    return args

def setup_paths(args):
    now = datetime.datetime.now()
    time_str = now.strftime("-[%m-%d]-[%H:%M]")
    args.name = args.exper_name + time_str
    args.output_path = os.path.join("outputs", args.name)
    os.makedirs(args.output_path, exist_ok=True)
    
    print('************************')
    print("Running Student Distillation with configuration:")
    for k, v in vars(args).items():
        print(f'{k} = {v}')
    print('************************')
    
    log_txt_path = os.path.join(args.output_path, 'log.txt')
    with open(log_txt_path, 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k} = {v}\n')
        f.write('*'*50 + '\n\n')
    args.log_txt_path = log_txt_path
    return args

# ==================== Distillation Loss ====================
def get_distillation_loss(student_logits, teacher_logits, targets, alpha=0.5, T=3.0):
    """
    Logit-level Knowledge Distillation Loss.
    Combines standard CrossEntropy (hard targets) and KL-Divergence (soft targets).
    """
    # 1. Hard target loss
    hard_loss = F.cross_entropy(student_logits, targets)
    
    # 2. Soft target loss (KL-Divergence)
    p_s = F.log_softmax(student_logits / T, dim=-1)
    p_t = F.softmax(teacher_logits / T, dim=-1)
    soft_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)
    
    # Combined loss
    loss = (1 - alpha) * hard_loss + alpha * soft_loss
    return loss, hard_loss, soft_loss

def main():
    args = parser.parse_args()
    args = setup_environment(args)
    args = setup_paths(args)

    class_names, input_text = get_class_info(args)
    args.num_classes = len(class_names)

    # 1. Load Teacher Model (frozen)
    print("=> Loading Teacher RAPT-CLIP Model...")
    teacher = build_model(args, input_text)
    load_slim_checkpoint(teacher, args.teacher_checkpoint, device=args.device)
    teacher = teacher.to(args.device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print("=> Teacher model loaded and frozen successfully.")

    # 2. Instantiate Student Model
    print("=> Instantiating Student Model (MobileNetV3 Backbone)...")
    student = StudentModel(num_classes=args.num_classes, num_segments=args.num_segments)
    student = student.to(args.device)
    
    # Size check
    student_size_mb = sum(p.numel() * 4 for p in student.parameters()) / (1024**2)
    print(f"=> Student model instantiated successfully. Estimated size: {student_size_mb:.2f} MB")

    # 3. Load Dataloaders
    print("=> Building dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders(args)
    print("=> Dataloaders built successfully.")

    # Optimizer & Scheduler for Student
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    
    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    recorder = RecorderMeter(args.epochs)
    log_curve_path = os.path.join(args.output_path, 'log.png')
    log_confusion_matrix_path = os.path.join(args.output_path, 'confusion_matrix.png')
    best_student_checkpoint = os.path.join(args.output_path, 'student_best.pth')
    
    best_val_uar = 0.0
    best_val_war = 0.0

    # Main Training Loop
    for epoch in range(args.epochs):
        epoch_str = f"Epoch {epoch}/{args.epochs}"
        print(f"\n******************** {epoch_str} ********************")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.1e}")

        # Training Phase
        student.train()
        train_losses = AverageMeter('Loss')
        train_hard_losses = AverageMeter('Hard Loss')
        train_soft_losses = AverageMeter('Soft Loss')
        train_feat_losses = AverageMeter('Feat Loss')
        train_war_meter = AverageMeter('WAR')
        
        train_preds_list = []
        train_targets_list = []

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", file=os.sys.stdout)
        for i, batch in enumerate(pbar):
            images_face, images_body, target = batch[:3]
            images_face = images_face.to(args.device)
            images_body = images_body.to(args.device)
            target = target.to(args.device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                # 1. Forward Teacher (frozen)
                with torch.no_grad():
                    teacher_logits, _, _, _ = teacher(images_face, images_body)
                    # Retrieve the stored video features using our Python attribute trick!
                    teacher_features = teacher.last_video_features

                # 2. Forward Student (trainable)
                student_logits, student_features = student(images_face, images_body)

                # 3. Calculate Distillation Losses
                kd_loss, hard_loss, soft_loss = get_distillation_loss(
                    student_logits, teacher_logits, target,
                    alpha=args.alpha_distill, T=args.distill_temp
                )
                
                # Feature alignment loss (Cosine similarity or MSE)
                feat_loss = F.mse_loss(student_features, teacher_features)
                
                # Combined total loss
                total_loss = kd_loss + args.beta_feature * feat_loss

            if args.use_amp:
                scaler.scale(total_loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                optimizer.step()

            # Record training metrics
            preds = student_logits.argmax(dim=1)
            acc = (preds.eq(target).sum().item() / target.size(0)) * 100.0
            
            train_losses.update(total_loss.item(), target.size(0))
            train_hard_losses.update(hard_loss.item(), target.size(0))
            train_soft_losses.update(soft_loss.item(), target.size(0))
            train_feat_losses.update(feat_loss.item(), target.size(0))
            train_war_meter.update(acc, target.size(0))
            
            train_preds_list.append(preds.cpu())
            train_targets_list.append(target.cpu())

            pbar.set_postfix({
                'Loss': f"{train_losses.avg:.4f}",
                'Feat': f"{train_feat_losses.avg:.4f}",
                'WAR': f"{train_war_meter.avg:.2f}%"
            })

        # Calculate epoch-level train UAR
        train_preds = torch.cat(train_preds_list).numpy()
        train_targets = torch.cat(train_targets_list).numpy()
        train_cm = confusion_matrix(train_targets, train_preds)
        train_class_acc = train_cm.diagonal() / (train_cm.sum(axis=1) + 1e-6)
        train_uar = np.nanmean(train_class_acc) * 100
        train_war = train_war_meter.avg

        # Validation Phase
        student.eval()
        val_losses = AverageMeter('Loss')
        val_war_meter = AverageMeter('WAR')
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                images_face, images_body, target = batch[:3]
                images_face = images_face.to(args.device)
                images_body = images_body.to(args.device)
                target = target.to(args.device)

                student_logits, _ = student(images_face, images_body)
                val_loss = F.cross_entropy(student_logits, target)
                
                preds = student_logits.argmax(dim=1)
                acc = (preds.eq(target).sum().item() / target.size(0)) * 100.0

                val_losses.update(val_loss.item(), target.size(0))
                val_war_meter.update(acc, target.size(0))
                val_preds_list.append(preds.cpu())
                val_targets_list.append(target.cpu())

        scheduler.step()

        # Calculate epoch-level validation metrics
        val_preds = torch.cat(val_preds_list).numpy()
        val_targets = torch.cat(val_targets_list).numpy()
        val_cm = confusion_matrix(val_targets, val_preds)
        val_class_acc = val_cm.diagonal() / (val_cm.sum(axis=1) + 1e-6)
        val_uar = np.nanmean(val_class_acc) * 100
        val_war = val_war_meter.avg

        is_best = val_uar > best_val_uar
        best_val_uar = max(val_uar, best_val_uar)
        best_val_war = max(val_war, best_val_war)

        # Save student weights
        if is_best:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': student.state_dict(),
                'best_uar': best_val_uar,
                'best_war': best_val_war
            }, best_student_checkpoint)
            print(f"=> Saved Best Student checkpoint with UAR {best_val_uar:.2f}% to {best_student_checkpoint}")

        epoch_log = (
            f"\n--- Epoch {epoch} Distillation Summary ---\n"
            f"Train Loss: {train_losses.avg:.4f} | Hard Loss: {train_hard_losses.avg:.4f} | Soft Loss: {train_soft_losses.avg:.4f} | Feat Loss: {train_feat_losses.avg:.4f}\n"
            f"Train WAR: {train_war:.2f}% | Train UAR: {train_uar:.2f}%\n"
            f"Valid WAR: {val_war:.2f}% | Valid UAR: {val_uar:.2f}%\n"
            f"Best Valid UAR so far: {best_val_uar:.2f}%\n"
            f"Valid Confusion Matrix:\n{val_cm}\n"
            f"-----------------------------------------\n"
        )
        print(epoch_log)
        with open(args.log_txt_path, 'a') as f:
            f.write(epoch_log + '\n')

        # Update Recorder Meter
        recorder.update(epoch, train_losses.avg, train_war, train_uar, val_losses.avg, val_war, val_uar)
        recorder.plot_curve(log_curve_path)

    # ==================== Final Evaluation on Test Set ====================
    print("\n=> Final evaluation of Student Model on test set...")
    checkpoint = torch.load(best_student_checkpoint, map_location=args.device)
    student.load_state_dict(checkpoint['state_dict'])
    student.eval()

    test_preds_list = []
    test_targets_list = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Student"):
            images_face, images_body, target = batch[:3]
            images_face = images_face.to(args.device)
            images_body = images_body.to(args.device)
            target = target.to(args.device)

            student_logits, _ = student(images_face, images_body)
            preds = student_logits.argmax(dim=1)
            test_preds_list.append(preds.cpu())
            test_targets_list.append(target.cpu())

    test_preds = torch.cat(test_preds_list).numpy()
    test_targets = torch.cat(test_targets_list).numpy()
    
    test_cm = confusion_matrix(test_targets, test_preds)
    test_class_acc = test_cm.diagonal() / (test_cm.sum(axis=1) + 1e-6)
    test_uar = np.nanmean(test_class_acc) * 100
    test_war = (test_preds == test_targets).sum() / len(test_targets) * 100.0

    normalized_cm = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis]
    normalized_cm_percent = normalized_cm * 100
    list_diag_percent = np.diag(normalized_cm_percent)

    results_msg = (
        f"\n****************************************\n"
        f"STUDENT DISTILLATION FINAL TEST RESULTS:\n"
        f"Confusion Matrix Diag (%):\n{list_diag_percent}\n"
        f"UAR (Unweighted Average Recall): {test_uar:.2f}%\n"
        f"WAR (Weighted Average Recall/Accuracy): {test_war:.2f}%\n"
        f"Student Weight Checkpoint Size: {os.path.getsize(best_student_checkpoint) / 1024**2:.2f} MB\n"
        f"****************************************\n"
    )
    print(results_msg)
    with open(args.log_txt_path, 'a') as f:
        f.write(results_msg + '\n')

    # Draw test confusion matrix
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(
        normalized_cm_percent,
        classes=class_names,
        normalize=True,
        title=f"Student Model Confusion Matrix on {args.dataset} Test Set"
    )
    plt.savefig(log_confusion_matrix_path)
    plt.close()
    
    print(f"=> Distillation complete. Outputs saved in: {args.output_path}")

if __name__ == '__main__':
    main()
