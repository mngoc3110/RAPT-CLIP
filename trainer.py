# trainer.py
import logging
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import os
import torchvision
import sys

from utils.utils import AverageMeter, get_loss_weight, get_loss_weight_rampdown
from utils.loss import SemanticLDLLoss

class Trainer:
    """A class that encapsulates the training and validation logic."""
    def __init__(self, model, criterion, optimizer, scheduler, device,log_txt_path, 
                 mi_criterion=None, lambda_mi=0, 
                 dc_criterion=None, lambda_dc=0,
                 mi_warmup=0, mi_ramp=0, mi_ramp_type='ramp_up',
                 dc_warmup=0, dc_ramp=0, use_amp=False, grad_clip=1.0, mixup_alpha=0.0,
                 use_ldl=False, ldl_warmup=0):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.print_freq = 10 
        self.log_txt_path = log_txt_path
        self.mi_criterion = mi_criterion
        self.lambda_mi = lambda_mi
        self.dc_criterion = dc_criterion
        self.lambda_dc = lambda_dc
        self.mi_warmup = mi_warmup
        self.mi_ramp = mi_ramp
        self.mi_ramp_type = mi_ramp_type
        self.dc_warmup = dc_warmup
        self.dc_ramp = dc_ramp
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        self.mixup_alpha = mixup_alpha
        self.use_ldl = use_ldl
        self.ldl_warmup = ldl_warmup
        print(f"DEBUG: Trainer initialized with use_ldl={use_ldl}, ldl_warmup={ldl_warmup}")
        
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Create directory for saving debug prediction images
        self.debug_predictions_path = 'debug_predictions'
        os.makedirs(self.debug_predictions_path, exist_ok=True)

    def _save_debug_image(self, tensor, prediction, target, epoch_str, batch_idx, img_idx):
        """Saves a single image tensor for debugging, with prediction and target in the filename."""
        # Un-normalize the image
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)

        # Create a directory for the current epoch if it doesn't exist
        epoch_debug_path = os.path.join(self.debug_predictions_path, f"epoch_{epoch_str}")
        os.makedirs(epoch_debug_path, exist_ok=True)
        
        # Construct filename
        filename = f"batch_{batch_idx}_img_{img_idx}_pred_{prediction}_true_{target}.png"
        filepath = os.path.join(epoch_debug_path, filename)
        
        # Save the image
        torchvision.utils.save_image(tensor, filepath)

    def mixup_data(self, x1, x2, alpha=1.0):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x1.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
        mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
        return mixed_x1, mixed_x2, index, lam

    def _run_one_epoch(self, loader, epoch_str, is_train=True):
        """Runs one epoch of training or validation."""
        if is_train:
            self.model.train()
            mode_str = "Train"
        else:
            self.model.eval()
            mode_str = "Valid"

        losses = AverageMeter('Loss', ':.4e')
        mi_losses = AverageMeter('MI Loss', ':.4e')
        dc_losses = AverageMeter('DC Loss', ':.4e')
        moco_losses = AverageMeter('MoCo Loss', ':.4e')
        war_meter = AverageMeter('WAR', ':6.2f')
        
        # Lists to store predictions for UAR calculation
        all_preds_list = []
        all_targets_list = []
        
        saved_images_count = 0

        # Print weights at the start of training epoch
        if is_train:
            if self.mi_ramp_type == 'ramp_up':
                mi_weight = get_loss_weight(int(epoch_str), self.mi_warmup, self.mi_ramp, self.lambda_mi)
            elif self.mi_ramp_type == 'ramp_down':
                mi_weight = get_loss_weight_rampdown(int(epoch_str), self.mi_warmup, self.mi_ramp, self.lambda_mi)
            else:
                mi_weight = self.lambda_mi # Fallback to the final weight

            dc_weight = get_loss_weight(int(epoch_str), self.dc_warmup, self.dc_ramp, self.lambda_dc)
            
            # Determine effective LDL weight (warmup)
            ldl_weight = 1.0
            if self.use_ldl and int(epoch_str) < self.ldl_warmup:
                ldl_weight = 0.0 # Disable LDL during warmup
            
            # MoCo weight display (typically fixed at 1.0 if enabled)
            moco_weight = 0.0
            if hasattr(self.model, 'args') and hasattr(self.model.args, 'use_moco') and self.model.args.use_moco:
                moco_weight = 1.0
                
            weight_msg = f"--- Epoch {epoch_str}: MI={mi_weight:.4f}, DC={dc_weight:.4f}, LDL_Wt={ldl_weight:.1f}, MoCo={moco_weight:.1f} ---"
            print(weight_msg)
            with open(self.log_txt_path, 'a') as f:
                f.write(weight_msg + '\n')

        context = torch.enable_grad() if is_train else torch.no_grad()
        
        # Use tqdm for progress bar
        pbar = tqdm(loader, desc=f"{mode_str} Epoch {epoch_str}", file=sys.stdout)
        
        with context:
            for i, (images_face, images_body, target) in enumerate(pbar):
                # DEBUG: Check for NaN in inputs
                if torch.isnan(images_face).any() or torch.isinf(images_face).any():
                    print(f"\n[CRITICAL ERROR] NaN/Inf detected in images_face at batch {i}!")
                    # raise ValueError("Input images_face contains NaN")
                
                images_face = images_face.to(self.device)
                images_body = images_body.to(self.device)
                target = target.to(self.device)
                
                # Apply Mixup
                if is_train and self.mixup_alpha > 0:
                    images_face, images_body, target_b, lam = self.mixup_data(images_face, images_body, self.mixup_alpha)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # Forward pass
                    output, learnable_text_features, hand_crafted_text_features, moco_logits = self.model(images_face, images_body)
                    
                    # DEBUG: Check model output for NaN
                    if torch.isnan(output).any():
                        print(f"\n[CRITICAL ERROR] Model output contains NaN at batch {i}!")
                        print(f"  Input Min/Max: {images_face.min().item():.4f} / {images_face.max().item():.4f}")
                        # Check intermediates if possible or just break
                        
                    # For MI and DC losses, if using prompt ensembling, average the learnable_text_features
                    processed_learnable_text_features = learnable_text_features
                    if hasattr(self.model, 'is_ensemble') and self.model.is_ensemble:
                        num_classes = self.model.num_classes
                        num_prompts_per_class = self.model.num_prompts_per_class
                        # Reshape from (C*P, D) to (C, P, D) and then average over P
                        processed_learnable_text_features = learnable_text_features.view(num_classes, num_prompts_per_class, -1).mean(dim=1)

                    # Calculate loss
                    # Check if we should use LDL (after warmup) or fallback to CE
                    current_criterion = self.criterion
                    if self.use_ldl and int(epoch_str) < self.ldl_warmup:
                         # Fallback to standard CE during warmup if using LDL wrapper
                         # But self.criterion is SemanticLDLLoss. We need a simple CE.
                         # Assuming we can just compute CE here or use a separate criterion.
                         # Simpler: SemanticLDLLoss already handles temperature. If we want HARD labels,
                         # we can just use F.cross_entropy.
                         current_criterion = torch.nn.CrossEntropyLoss()
                    
                    if isinstance(current_criterion, SemanticLDLLoss):
                        if is_train and self.mixup_alpha > 0:
                            classification_loss = lam * current_criterion(output, target, processed_learnable_text_features) + \
                                                  (1 - lam) * current_criterion(output, target_b, processed_learnable_text_features)
                        else:
                            classification_loss = current_criterion(output, target, processed_learnable_text_features)
                    else:
                        # Standard CE or LSR
                        if is_train and self.mixup_alpha > 0:
                            classification_loss = lam * current_criterion(output, target) + (1 - lam) * current_criterion(output, target_b)
                        else:
                            classification_loss = current_criterion(output, target)
                    
                    # DEBUG: Print details for the first batch of the first epoch
                    if is_train and int(epoch_str) == 0 and i == 0:
                        print(f"\n[DEBUG] Batch 0 Check:")
                        print(f"  Logits Shape: {output.shape}")
                        print(f"  Target Shape: {target.shape}")
                        print(f"  Target Min/Max: {target.min().item()} / {target.max().item()}")
                        # Handle NaN print safely
                        logits_np = output[:2].detach().cpu().numpy()
                        print(f"  Logits (first 2): {logits_np}")
                        print(f"  Targets (first 2): {target[:2].detach().cpu().numpy()}")
                        print(f"  CE/LDL Loss: {classification_loss.item():.6f}")
                        if hasattr(self.model, 'args') and hasattr(self.model.args, 'temperature'):
                             print(f"  Model Temperature: {self.model.args.temperature}")

                    loss = classification_loss

                    if is_train and self.mi_criterion is not None:
                        mi_loss = self.mi_criterion(processed_learnable_text_features, hand_crafted_text_features)
                        loss += mi_weight * mi_loss
                        mi_losses.update(mi_loss.item(), target.size(0))

                    if is_train and self.dc_criterion is not None:
                        dc_loss = self.dc_criterion(processed_learnable_text_features)
                        loss += dc_weight * dc_loss
                        dc_losses.update(dc_loss.item(), target.size(0))

                    if is_train and moco_logits is not None:
                         moco_target = torch.zeros(moco_logits.size(0), dtype=torch.long).to(self.device)
                         moco_loss = torch.nn.CrossEntropyLoss()(moco_logits, moco_target)
                         loss += moco_loss
                         moco_losses.update(moco_loss.item(), target.size(0))

                if is_train:
                    self.optimizer.zero_grad()
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        if self.grad_clip > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        if self.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()

                # Record metrics
                preds = output.argmax(dim=1)
                correct_preds = preds.eq(target).sum().item()
                acc = (correct_preds / target.size(0)) * 100.0

                losses.update(loss.item(), target.size(0))
                war_meter.update(acc, target.size(0))

                # Collect preds for UAR
                all_preds_list.append(preds.cpu())
                all_targets_list.append(target.cpu())

                if not is_train and saved_images_count < 32:
                    for img_idx in range(images_face.size(0)):
                        if saved_images_count < 32:
                            self._save_debug_image(
                                images_face[img_idx].cpu(),
                                preds[img_idx].item(),
                                target[img_idx].item(),
                                epoch_str,
                                i,
                                img_idx
                            )
                            saved_images_count += 1
                        else:
                            break
                
                # Update progress bar with Running UAR
                running_uar = 0.0
                if len(all_preds_list) > 0:
                    curr_preds = torch.cat(all_preds_list).numpy()
                    curr_targets = torch.cat(all_targets_list).numpy()
                    # Only calc UAR every 10 batches to save CPU time
                    if i % 10 == 0: 
                        try:
                            cm = confusion_matrix(curr_targets, curr_preds, labels=range(output.shape[1]))
                            class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6)
                            running_uar = np.nanmean(class_acc) * 100
                        except:
                            pass
                
                pbar.set_postfix({
                    'Loss': f"{losses.avg:.4f}",
                    'WAR': f"{war_meter.avg:.2f}%",
                    'UAR': f"{running_uar:.2f}%"
                })
        
        # Calculate epoch-level metrics
        all_preds = torch.cat(all_preds_list)
        all_targets = torch.cat(all_targets_list)
        
        cm = confusion_matrix(all_targets.numpy(), all_preds.numpy())
        war = war_meter.avg 
        
        class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6)
        uar = np.nanmean(class_acc) * 100

        prefix = f"{mode_str} Epoch: [{epoch_str}]"
        logging.info(f"{prefix} * WAR: {war:.3f} | UAR: {uar:.3f}")
        with open(self.log_txt_path, 'a') as f:
            f.write('Current WAR: {war:.3f}'.format(war=war) + '\n')
            f.write('Current UAR: {uar:.3f}'.format(uar=uar) + '\n')
        return war, uar, losses.avg, cm
        
    def train_epoch(self, train_loader, epoch_num):
        """Executes one full training epoch."""
        res = self._run_one_epoch(train_loader, str(epoch_num), is_train=True)
        torch.cuda.empty_cache()
        return res
    
    def validate(self, val_loader, epoch_num_str="Final"):
        """Executes one full validation run."""
        res = self._run_one_epoch(val_loader, epoch_num_str, is_train=False)
        torch.cuda.empty_cache()
        return res