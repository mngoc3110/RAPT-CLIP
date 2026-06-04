# ==================== Imports ====================
import argparse
import datetime
import os
import random
import shutil
import time

# Suppress OpenCV and FFmpeg warnings
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_DEBUG_LOG_LEVEL"] = "0"

import matplotlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import warnings
from models.clip import clip
from dataloader.video_dataloader import train_data_loader, test_data_loader
from models.Generate_Model import GenerateModel
from models.Text import *
from trainer import Trainer
from utils.loss import *
from utils.utils import *
from utils.builders import *
from utils.checkpoint_utils import save_slim_checkpoint, load_slim_checkpoint

# Ignore specific warnings (for cleaner output)
warnings.filterwarnings("ignore", category=UserWarning)
# Use 'Agg' backend for matplotlib (no GUI required)
matplotlib.use('Agg')

# ==================== Argument Parser ====================
parser = argparse.ArgumentParser(
    description='A highly configurable training script for RAER Dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# --- Experiment and Environment ---
exp_group = parser.add_argument_group('Experiment & Environment', 'Basic settings for the experiment')
exp_group.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                       help="Execution mode: 'train' for a full training run, 'eval' for evaluation only.")
exp_group.add_argument('--eval-checkpoint', type=str,
                       help="Path to the model checkpoint for evaluation mode (e.g., outputs/exp_name/model_best.pth).")
exp_group.add_argument('--resume', type=str,
                       help="Path to the model checkpoint to resume training from (e.g., outputs/exp_name/model.pth).")
exp_group.add_argument('--exper-name', type=str, default='Train', help='A name for the experiment to create a unique output folder.')
exp_group.add_argument('--dataset', type=str, default='RAER', help='Name of the dataset to use.')
exp_group.add_argument('--gpu', type=str, default='mps', help='ID of the GPU to use (e.g., 0, 1) or "mps" for Apple Silicon.')
exp_group.add_argument('--workers', type=int, default=4, help='Number of data loading workers.')
exp_group.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

# --- Data & Path ---
path_group = parser.add_argument_group('Data & Path', 'Paths to datasets and pretrained models')
path_group.add_argument('--root-dir', type=str, default='./', help='Root directory of the dataset. E.g., /kaggle/input/raer-video-emotion-dataset/RAER')
path_group.add_argument('--train-annotation', type=str, default='RAER/annotation/train_80.txt', help='Absolute path to training annotation file. E.g., /kaggle/input/raer-annot/annotation/train_abs.txt')
path_group.add_argument('--val-annotation', type=str, default='RAER/annotation/val_20.txt', help='Absolute path to validation annotation file. E.g., /kaggle/input/raer-annot/annotation/val_20.txt')
path_group.add_argument('--test-annotation', type=str, default='RAER/annotation/test.txt', help='Absolute path to testing annotation file. E.g., /kaggle/input/raer-annot/annotation/test_abs.txt')
path_group.add_argument('--clip-path', type=str, default='ViT-B/16', help='Path to the pretrained CLIP model.')
path_group.add_argument('--bounding-box-face', type=str, default='RAER/bounding_box/face.json', help='Absolute path to face bounding box JSON. E.g., /kaggle/input/raer-annot/annotation/bounding_box/face_abs.json')
path_group.add_argument('--bounding-box-body', type=str, default='RAER/bounding_box/body.json', help='Absolute path to body bounding box JSON. E.g., /kaggle/input/raer-annot/annotation/bounding_box/body_abs.json')

# --- Training Control ---
train_group = parser.add_argument_group('Training Control', 'Parameters to control the training process')
train_group.add_argument('--epochs', type=int, default=20, help='Total number of training epochs.')
train_group.add_argument('--batch-size', type=int, default=4, help='Batch size for training and validation.')
train_group.add_argument('--print-freq', type=int, default=10, help='Frequency of printing training logs.')
train_group.add_argument('--use-amp', action='store_true', help='Use Automatic Mixed Precision.')
train_group.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping value.')

# --- Optimizer & Learning Rate ---
optim_group = parser.add_argument_group('Optimizer & LR', 'Hyperparameters for the optimizer and scheduler')
optim_group.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'AdamW'], help='The optimizer to use (SGD or AdamW).')
optim_group.add_argument('--lr', type=float, default=2e-5, help='Initial learning rate for main modules (temporal, project_fc).')
optim_group.add_argument('--lr-image-encoder', type=float, default=1e-6, help='Learning rate for the image encoder part (set to 0 to freeze).')
optim_group.add_argument('--lr-prompt-learner', type=float, default=2e-4, help='Learning rate for the prompt learner.')
optim_group.add_argument('--lr-adapter', type=float, default=1e-4, help='Learning rate for the adapter.')
optim_group.add_argument('--weight-decay', type=float, default=0.005, help='Weight decay for the optimizer.')
optim_group.add_argument('--momentum', type=float, default=0.9, help='Momentum for the SGD optimizer.')
optim_group.add_argument('--milestones', nargs='+', type=int, default=[10, 15], help='Epochs at which to decay the learning rate.')
optim_group.add_argument('--gamma', type=float, default=0.1, help='Factor for learning rate decay.')

# --- Loss & Imbalance Handling ---
loss_group = parser.add_argument_group('Loss & Imbalance Handling', 'Parameters for loss functions and imbalance handling')
loss_group.add_argument('--loss-type', type=str, default='ce', choices=['ce', 'ldl', 'ldam'], help='Type of primary classification loss (ce, ldl, ldam).')
loss_group.add_argument('--lambda_mi', type=float, default=0.1, help='Weight for the Mutual Information loss.')
loss_group.add_argument('--lambda_dc', type=float, default=0.1, help='Weight for the Decorrelation loss.')
loss_group.add_argument('--mi-warmup', type=int, default=5, help='Warmup epochs for MI loss.')
loss_group.add_argument('--mi-ramp', type=int, default=10, help='Ramp-up epochs for MI loss.')
loss_group.add_argument('--mi-ramp-type', type=str, default='ramp_up', choices=['ramp_up', 'ramp_down'], help='Type of ramp for MI loss weight (ramp_up or ramp_down).')
loss_group.add_argument('--dc-warmup', type=int, default=5, help='Warmup epochs for DC loss.')
loss_group.add_argument('--dc-ramp', type=int, default=10, help='Ramp-up epochs for DC loss.')
loss_group.add_argument('--use-weighted-sampler', action='store_true', help='Use WeightedRandomSampler.')
loss_group.add_argument('--label-smoothing', type=float, default=0.05, help='Label smoothing factor.')
loss_group.add_argument('--use-ldl', action='store_true', help='Use Semantic Label Distribution Learning (LDL) Loss.')
loss_group.add_argument('--ldl-temperature', type=float, default=1.0, help='Temperature for LDL target distribution.')
loss_group.add_argument('--ldl-warmup', type=int, default=5, help='Warmup epochs for LDL loss (during warmup, use CE).')
loss_group.add_argument('--mixup-alpha', type=float, default=0.0, help='Alpha value for Mixup data augmentation. Set to 0.0 to disable. NOTE: Mixup is incompatible with LDAM (hard-label margin), keep 0.0 when using loss-type=ldam.')
# NEW LDAM ARGS
loss_group.add_argument('--ldam-max-m', type=float, default=0.5, help='Max margin for LDAM Loss.')
loss_group.add_argument('--ldam-s', type=float, default=30.0, help='Scaling factor for LDAM Loss. s=30 works well with CLIP cosine-sim outputs (proven: 73.76%% UAR on RAER). Lower values (e.g. s=3) produce weak gradients.')

# --- Model & Input ---
model_group = parser.add_argument_group('Model & Input', 'Parameters for model architecture and data handling')
model_group.add_argument('--text-type', default='prompt_ensemble', choices=['class_names', 'class_names_with_context', 'class_descriptor', 'prompt_ensemble', 'au_guided_prompts'], help='Type of text prompts to use.')
model_group.add_argument('--temporal-layers', type=int, default=1, help='Number of layers in the temporal modeling part.')
model_group.add_argument('--contexts-number', type=int, default=8, help='Number of context vectors in the prompt learner.')
model_group.add_argument('--class-token-position', type=str, default="end", help='Position of the class token in the prompt.')
model_group.add_argument('--class-specific-contexts', type=str, default='True', choices=['True', 'False'], help='Whether to use class-specific context prompts.')
model_group.add_argument('--load_and_tune_prompt_learner', type=str, default='True', choices=['True', 'False'], help='Whether to load and fine-tune the prompt learner.')
model_group.add_argument('--num-segments', type=int, default=16, help='Number of segments to sample from each video.')
model_group.add_argument('--duration', type=int, default=1, help='Duration of each segment.')
model_group.add_argument('--image-size', type=int, default=224, help='Size to resize input images to.')
model_group.add_argument('--temperature', type=float, default=0.07, help='Temperature for the classification layer.')
model_group.add_argument('--crop-body', action='store_true', help='Crop body from the input images.')
model_group.add_argument('--use-moco', action='store_true', help='Use MoCoRank for training.')
model_group.add_argument('--moco-k', type=int, default=4096, help='Queue size for MoCo.')
model_group.add_argument('--moco-m', type=float, default=0.99, help='Momentum for MoCo.')
model_group.add_argument('--moco-t', type=float, default=0.07, help='Temperature for MoCo.')
model_group.add_argument('--drop-path-rate', type=float, default=0.0, help='Drop Path rate for Stochastic Depth.')
model_group.add_argument('--freeze-image-encoder', action='store_true', help='Freeze the image encoder.')
model_group.add_argument('--ablation-no-text', action='store_true', help='Use Visual-Only ablation architecture.')
model_group.add_argument('--use-v2', action='store_true', help='Use V2 Triple-Stream Architecture.')
model_group.add_argument('--modality-dropout', type=float, default=0.3, help='Modality Dropout probability.')

# ==================== Helper Functions ====================
def setup_environment(args: argparse.Namespace) -> argparse.Namespace:
    if args.gpu == 'mps':
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print("MPS device not found, falling back to CPU.")
            device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        print("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")
    
    args.device = device
    if device.type == 'cpu':
        args.use_amp = False
        print("=> Device is CPU. Disabling AMP (use_amp=False) to prevent GradScaler crash.")
    print(f"Using device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    
    print("Environment and random seeds set successfully.")
    return args


def setup_paths_and_logging(args: argparse.Namespace) -> argparse.Namespace:
    now = datetime.datetime.now()
    time_str = now.strftime("-[%m-%d]-[%H:%M]")
    
    args.name = args.exper_name + time_str
        
    args.output_path = os.path.join("outputs", args.name)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    print('************************')
    print("Running with the following configuration:")
    for k, v in vars(args).items():
        print(f'{k} = {v}')
    print('************************')
    
    log_txt_path = os.path.join(args.output_path, 'log.txt')
    with open(log_txt_path, 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k} = {v}\n')
        f.write('*'*50 + '\n\n')
        
    return args

# ==================== Training Function ====================
def run_training(args: argparse.Namespace) -> None:
    # Paths for logging and saving
    log_txt_path = os.path.join(args.output_path, 'log.txt')
    log_curve_path = os.path.join(args.output_path, 'log.png')
    log_confusion_matrix_path = os.path.join(args.output_path, 'confusion_matrix.png')
    checkpoint_path = os.path.join(args.output_path, 'model.pth')
    best_checkpoint_path = os.path.join(args.output_path, 'model_best.pth')        
    best_train_uar = 0.0
    best_train_war = 0.0
    best_val_uar = 0.0
    best_val_war = 0.0
    start_epoch = 0
    
    # Build model
    print("=> Building model...")
    class_names, input_text = get_class_info(args)
    args.num_classes = len(class_names)
    model = build_model(args, input_text)
    model = model.to(args.device)
    print("=> Model built and moved to device successfully.")

    # Load data
    print("=> Building dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders(args)
    print("=> Dataloaders built successfully.")

    # Calculate cls_num_list for LDAM or other imbalance handling
    cls_num_list = [0] * len(class_names)
    # Check if dataset has video_list (standard VideoDataset)
    if hasattr(train_loader.dataset, 'video_list'):
        print(f"=> Calculating class distribution from video_list...")
        for record in train_loader.dataset.video_list:
            # Labels in RAER/CAER annotations are typically 1-based (e.g., 1..8)
            # VideoDataset.__getitem__ returns label-1.
            # So we map record.label (1-based) to 0-based index.
            label_idx = record.label - 1
            if 0 <= label_idx < len(cls_num_list):
                cls_num_list[label_idx] += 1
    else:
        # Fallback or warning if dataset structure is different
        print("=> Warning: Could not calculate class distribution directly from dataset. Using uniform distribution placeholder if needed.")
        # Attempt to infer from simple iteration if small, but likely too slow. 
        # For now, just warn. LDAM might fail or perform poorly if this is zero.
        pass
    
    print(f"=> Class distribution (Training): {cls_num_list}")

    # Loss and optimizer
    if args.use_ldl:
        print(f"=> Using SemanticLDLLoss (LDL) with temperature {args.ldl_temperature}")
        criterion = SemanticLDLLoss(temperature=args.ldl_temperature).to(args.device)
    elif args.loss_type == 'ldam':
        if sum(cls_num_list) > 0:
            print(f"=> Using LDAM Loss with s={args.ldam_s}, max_m={args.ldam_max_m}")
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=args.ldam_max_m, s=args.ldam_s).to(args.device)
        else:
            print("=> Error: cls_num_list is empty/zero. Cannot use LDAM. Falling back to CrossEntropy.")
            criterion = nn.CrossEntropyLoss().to(args.device)
    elif args.label_smoothing > 0:
        criterion = LSR2(e=args.label_smoothing, label_mode='class_descriptor').to(args.device)
    else:
        criterion = nn.CrossEntropyLoss().to(args.device)

    mi_criterion = MILoss().to(args.device) if args.lambda_mi > 0 else None
    dc_criterion = DCLoss().to(args.device) if args.lambda_dc > 0 else None

    recorder = RecorderMeter(args.epochs)
    
    if hasattr(args, 'ablation_no_text') and args.ablation_no_text:
        optimizer_grouped_parameters = [
            {"params": model.temporal_net.parameters(), "lr": args.lr},
            {"params": model.temporal_net_body.parameters(), "lr": args.lr},
            {"params": model.image_encoder.parameters(), "lr": args.lr_image_encoder},
            {"params": model.project_fc.parameters(), "lr": args.lr},
            {"params": model.face_adapter.parameters(), "lr": args.lr_adapter},
            {"params": model.classifier.parameters(), "lr": args.lr}
        ]
    else:
        if hasattr(model, 'temporal_net_face'):
            # V2 Architecture
            optimizer_grouped_parameters = [
                {"params": model.temporal_net_face.parameters(), "lr": args.lr},
                {"params": model.temporal_net_body.parameters(), "lr": args.lr},
                {"params": model.temporal_net_context.parameters(), "lr": args.lr},
                {"params": model.image_encoder.parameters(), "lr": args.lr_image_encoder},
                {"params": model.prompt_learner.parameters(), "lr": args.lr_prompt_learner},
                {"params": model.project_fc.parameters(), "lr": args.lr},
                {"params": model.face_adapter.parameters(), "lr": args.lr_adapter},
                {"params": model.cross_attn_fb.parameters(), "lr": args.lr},
                {"params": model.cross_attn_fbc.parameters(), "lr": args.lr}
            ]
        else:
            # V1 Architecture
            optimizer_grouped_parameters = [
                {"params": model.temporal_net.parameters(), "lr": args.lr},
                {"params": model.temporal_net_body.parameters(), "lr": args.lr},
                {"params": model.image_encoder.parameters(), "lr": args.lr_image_encoder},
                {"params": model.prompt_learner.parameters(), "lr": args.lr_prompt_learner},
                {"params": model.project_fc.parameters(), "lr": args.lr},
                {"params": model.face_adapter.parameters(), "lr": args.lr_adapter},
                {"params": model.cmaf.parameters(), "lr": args.lr}
            ]

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported.")

    # Resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
            start_epoch = checkpoint['epoch']
            best_val_uar = checkpoint.get('best_acc', 0.0)
            
            # Use strict=False to allow loading older checkpoints into the new model (e.g., when adding MoCo)
            msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(f"=> Load result: {msg}")
            
            if 'optimizer' in checkpoint and not args.use_moco: # Skip optimizer resume if architecture changed significantly
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                except:
                    print("=> Warning: Could not resume optimizer state.")
            
            if 'recorder' in checkpoint:
                recorder = checkpoint['recorder']
            print(f"=> Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            print(f"=> No checkpoint found at '{args.resume}'")

    for group in optimizer.param_groups:
        if 'initial_lr' not in group:
            group['initial_lr'] = group['lr']

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma, last_epoch=start_epoch - 1)

    if args.resume and os.path.isfile(args.resume):
        if 'scheduler' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
                print("=> Loaded scheduler state dict from checkpoint.")
            except Exception as e:
                print(f"=> Warning: Could not resume scheduler state dict: {e}")

    trainer = Trainer(model, criterion, optimizer, scheduler, args.device, log_txt_path, 
                    mi_criterion=mi_criterion, lambda_mi=args.lambda_mi,
                    dc_criterion=dc_criterion, lambda_dc=args.lambda_dc,
                    mi_warmup=args.mi_warmup, mi_ramp=args.mi_ramp, mi_ramp_type=args.mi_ramp_type,
                    dc_warmup=args.dc_warmup, dc_ramp=args.dc_ramp, 
                    use_amp=args.use_amp, grad_clip=args.grad_clip, mixup_alpha=args.mixup_alpha,
                    use_ldl=args.use_ldl, ldl_warmup=args.ldl_warmup)
    
    for epoch in range(start_epoch, args.epochs):
        inf = f'******************** Epoch: {epoch} ********************'
        start_time = time.time()
        print(inf)
        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')

        # Log current learning rates
        current_lrs = [param_group['lr'] for param_group in trainer.optimizer.param_groups]
        lr_str = ' '.join([f'{lr:.1e}' for lr in current_lrs])
        log_msg = f'Current learning rates: {lr_str}'
        with open(log_txt_path, 'a') as f:
            f.write(log_msg + '\n')
        print(log_msg)

        # Train & Validate
        train_war, train_uar, train_los, train_cm = trainer.train_epoch(train_loader, epoch)
        val_war, val_uar, val_los, val_cm = trainer.validate(val_loader, str(epoch))
        trainer.scheduler.step()

        # Save checkpoint
        is_best = val_uar > best_val_uar
        best_val_uar = max(val_uar, best_val_uar)
        best_val_war = max(val_war, best_val_war)
        best_train_uar = max(train_uar, best_train_uar)
        best_train_war = max(train_war, best_train_war)

        # 1. Save full checkpoint (with optimizer) → model.pth, used for --resume
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': trainer.model.state_dict(),
            'best_acc': best_val_uar,
            'optimizer': trainer.optimizer.state_dict(),
            'scheduler': trainer.scheduler.state_dict(),
            'recorder': recorder
        }, False, checkpoint_path, checkpoint_path)  # always overwrite model.pth

        # 2. Save slim checkpoint (fine-tuned layers only, no optimizer) → model_best.pth
        # This reduces model_best.pth from ~1.3GB → ~20MB, with ZERO accuracy change.
        if is_best:
            if hasattr(trainer, 'ema') and trainer.ema is not None:
                trainer.ema.apply(trainer.model)
            
            save_slim_checkpoint(
                model=trainer.model,
                path=best_checkpoint_path,
                meta={'epoch': epoch + 1, 'best_acc': best_val_uar}
            )
            
            if hasattr(trainer, 'ema') and trainer.ema is not None:
                trainer.ema.restore(trainer.model)

        # Record metrics
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, train_war, train_uar, val_los, val_war, val_uar)
        recorder.plot_curve(log_curve_path)
        
        log_msg = (
                   f'\n'
                   f'--- Epoch {epoch} Summary ---\n'
                   f'Train WAR: {train_war:.2f}% | Train UAR: {train_uar:.2f}%\n'
                   f'Valid WAR: {val_war:.2f}% | Valid UAR: {val_uar:.2f}%\n'
                   f'Best Valid UAR so far: {best_val_uar:.2f}%\n'
                   f'Time: {epoch_time:.2f}s\n'
                   f'Train Confusion Matrix:\n{train_cm}\n'
                   f'Validation Confusion Matrix:\n{val_cm}\n'
                   f'--- End of Epoch {epoch} ---\n'
                   )
        print(log_msg)
        with open(log_txt_path, 'a') as f:
            f.write(log_msg + '\n\n')

    # Final evaluation with best model (load slim checkpoint)
    print("=> Final evaluation on test set...")
    load_slim_checkpoint(model, best_checkpoint_path, device=args.device)
    computer_uar_war(
        val_loader=test_loader,
        model=model,
        device=args.device,
        class_names=class_names,
        log_confusion_matrix_path=log_confusion_matrix_path,
        log_txt_path=log_txt_path,
        title=f"Confusion Matrix on {args.dataset} Test Set"
    )

def run_eval(args: argparse.Namespace) -> None:
    print("=> Starting evaluation mode...")
    log_txt_path = os.path.join(args.output_path, 'log.txt')
    log_confusion_matrix_path = os.path.join(args.output_path, 'confusion_matrix.png')

    class_names, input_text = get_class_info(args)
    args.num_classes = len(class_names)
    model = build_model(args, input_text)
    model = model.to(args.device)

    # Load pretrained weights (supports both slim and full checkpoints)
    load_slim_checkpoint(model, args.eval_checkpoint, device=args.device)

    # Load data
    _, _, test_loader = build_dataloaders(args)

    # Run evaluation
    computer_uar_war(
        val_loader=test_loader,
        model=model,
        device=args.device,
        class_names=class_names,
        log_confusion_matrix_path=log_confusion_matrix_path,
        log_txt_path=log_txt_path,
        title=f"Confusion Matrix on {args.dataset}"
    )
    print("=> Evaluation complete.")


# ==================== Entry Point ====================
if __name__ == '__main__':
    args = parser.parse_args()
    args = setup_environment(args)
    args = setup_paths_and_logging(args)
    
    if args.mode == 'eval':
        run_eval(args)
    else:
        run_training(args)
