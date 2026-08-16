# trainer.py
import logging
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, average_precision_score
from tqdm import tqdm
import os
import torchvision
import sys

from utils.utils import AverageMeter, get_loss_weight, get_loss_weight_rampdown
from utils.loss import SemanticLDLLoss

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters() if p.requires_grad}
        self.decay = decay
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply(self, model):
        self.backup = {name: p.clone() for name, p in model.named_parameters() if name in self.shadow}
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}  # Clear backup to free GPU memory immediately

def self_attention_pooling(perspective_features, tau=0.07):
    """
    Multi-Perspective Self-Attention Pooling from PromptCAD (TCSVT 2026, Eq. 12-14).
    Dynamically aggregates features across multiple views/perspectives via learned attention.
    
    Args:
        perspective_features: (M, B, D) or list of M tensors (B, D)
        tau (float): Temperature for attention sharpness (Default: 0.07)
    Returns:
        f_agg: (B, D) aggregated feature
    """
    if isinstance(perspective_features, list):
        perspective_features = torch.stack(perspective_features, dim=0)  # (M, B, D)
    
    M, B, D = perspective_features.shape
    feats_norm = perspective_features / (perspective_features.norm(dim=-1, keepdim=True) + 1e-6)
    
    # 1. Global context query q = 1/M \sum f_i
    q = feats_norm.mean(dim=0)  # (B, D)
    q_norm = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
    
    # 2. Attention weights \alpha_i = softmax( \tau * cos(f_i, q) )
    cos_sim = torch.einsum('mbd,bd->mb', feats_norm, q_norm) / tau
    alpha = F.softmax(cos_sim, dim=0).unsqueeze(-1)  # (M, B, 1)
    
    # 3. Aggregated feature f_agg = \sum \alpha_i * f_i
    f_agg = (alpha * feats_norm).sum(dim=0)
    f_agg = f_agg / (f_agg.norm(dim=-1, keepdim=True) + 1e-6)
    return f_agg


class Trainer:
    """A class that encapsulates the training and validation logic."""
    def __init__(self, model, criterion, optimizer, scheduler, device, log_txt_path, 
                 mi_criterion=None, lambda_mi=0, 
                 dc_criterion=None, lambda_dc=0,
                 cad_criterion=None, lambda_cad=0,
                 text_distill_criterion=None, lambda_text=0,
                 concept_prototypes=None,
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
        self.cad_criterion = cad_criterion
        self.lambda_cad = lambda_cad
        self.text_distill_criterion = text_distill_criterion
        self.lambda_text = lambda_text
        self.concept_prototypes = concept_prototypes.to(device) if concept_prototypes is not None else None
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
        print(f"DEBUG: Trainer initialized with use_ldl={use_ldl}, ldl_warmup={ldl_warmup}, lambda_cad={lambda_cad}, lambda_text={lambda_text}")
        
        # Initialize ModelEMA
        self.ema = ModelEMA(self.model, decay=0.999)
        
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
        cad_losses = AverageMeter('CAD Loss', ':.4e')
        text_losses = AverageMeter('TextDistill Loss', ':.4e')
        war_meter = AverageMeter('WAR', ':6.2f')
        
        # Lists to store predictions for UAR calculation
        all_preds_list = []
        all_targets_list = []
        
        running_uar = 0.0
        running_map = 0.0
        
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
                
            weight_msg = f"--- Epoch {epoch_str}: MI={mi_weight:.4f}, DC={dc_weight:.4f}, LDL_Wt={ldl_weight:.1f}, MoCo={moco_weight:.1f}, CAD={self.lambda_cad:.2f}, TextDistill={self.lambda_text:.2f} ---"
            print(weight_msg)
            with open(self.log_txt_path, 'a') as f:
                f.write(weight_msg + '\n')

        context = torch.enable_grad() if is_train else torch.no_grad()
        
        # Use tqdm for progress bar
        pbar = tqdm(loader, desc=f"{mode_str} Epoch {epoch_str}", file=sys.stdout)
        
        with context:
            for i, data in enumerate(pbar):
                # Handle both 2-stream (Face, Body) and 3-stream (Face, Body, Context)
                if len(data) == 4:
                    images_face, images_body, images_context, target = data
                    images_context = images_context.to(self.device)
                else:
                    images_face, images_body, target = data
                    images_context = None

                # DEBUG: Check for NaN in inputs
                if torch.isnan(images_face).any() or torch.isinf(images_face).any():
                    print(f"\n[CRITICAL ERROR] NaN/Inf detected in images_face at batch {i}!")
                
                images_face = images_face.to(self.device)
                images_body = images_body.to(self.device)
                target = target.to(self.device)

                # --- Guard: validate target labels are in-bounds ---
                num_classes_expected = None
                if hasattr(self.model, 'num_classes'):
                    num_classes_expected = self.model.num_classes
                elif hasattr(self.model, 'args') and hasattr(self.model.args, 'num_classes'):
                    num_classes_expected = self.model.args.num_classes
                # Ensure target is float for BCE/ASL, long for CE/LDL
                if hasattr(self.criterion, 'gamma_neg') or isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                    target = target.float()
                else:
                    target = target.long()

                if num_classes_expected is not None and target.dtype == torch.long:
                    invalid_mask = (target < 0) | (target >= num_classes_expected)
                    if invalid_mask.any():
                        bad_vals = target[invalid_mask].cpu().tolist()
                        print(f"\n[CRITICAL] Batch {i}: target labels out of bounds [0, {num_classes_expected-1}]: {bad_vals}")
                        print(f"  -> Clamping to valid range to avoid CUDA crash. CHECK YOUR DATALOADER LABELS!")
                        target = target.clamp(0, num_classes_expected - 1)
                
                # Apply Mixup (Skip for multi-label / float targets)
                if is_train and self.mixup_alpha > 0 and target.dtype == torch.long:
                    images_face, images_body, index, lam = self.mixup_data(images_face, images_body, self.mixup_alpha)
                    target_b = target[index]
                else:
                    self.mixup_alpha = 0  # Temporarily disable mixup for this batch if float target

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # Forward pass
                    if images_context is not None:
                        res_model = self.model(images_face, images_body, images_context)
                    else:
                        res_model = self.model(images_face, images_body)
                    
                    if len(res_model) == 5:
                        output, learnable_text_features, hand_crafted_text_features, moco_logits, patch_features = res_model
                    else:
                        output, learnable_text_features, hand_crafted_text_features, moco_logits = res_model
                        patch_features = None
                    
                    # DEBUG: Check model output for NaN
                    if torch.isnan(output).any():
                        print(f"\n[CRITICAL ERROR] Model output contains NaN at batch {i}!")
                        print(f"  Input Min/Max: {images_face.min().item():.4f} / {images_face.max().item():.4f}")
                        
                    # For MI and DC losses, if using prompt ensembling, average the learnable_text_features
                    processed_learnable_text_features = learnable_text_features
                    if hasattr(self.model, 'is_ensemble') and self.model.is_ensemble:
                        num_classes = self.model.num_classes
                        num_prompts_per_class = self.model.num_prompts_per_class
                        # Reshape from (C*P, D) to (C, P, D) and then average over P
                        processed_learnable_text_features = learnable_text_features.view(num_classes, num_prompts_per_class, -1).mean(dim=1)

                    # Calculate loss
                    current_criterion = self.criterion
                    if self.use_ldl and int(epoch_str) < self.ldl_warmup:
                          current_criterion = torch.nn.CrossEntropyLoss()
                    
                    if isinstance(current_criterion, SemanticLDLLoss):
                        if is_train and self.mixup_alpha > 0:
                            classification_loss = lam * current_criterion(output, target, processed_learnable_text_features) + \
                                                  (1 - lam) * current_criterion(output, target_b, processed_learnable_text_features)
                        else:
                            classification_loss = current_criterion(output, target, processed_learnable_text_features)
                    else:
                        if is_train and self.mixup_alpha > 0:
                            classification_loss = lam * current_criterion(output, target) + (1 - lam) * current_criterion(output, target_b)
                        else:
                            classification_loss = current_criterion(output, target)
                    
                    # DEBUG: Print details for the first batch of the first epoch
                    if is_train and int(epoch_str) == 0 and i == 0:
                        print(f"\n[DEBUG] Batch 0 Check:")
                        print(f"  Logits Shape: {output.shape}")
                        print(f"  Target Shape: {target.shape}")
                        target_cpu = target.detach().cpu()
                        print(f"  Target Min/Max: {target_cpu.min().item()} / {target_cpu.max().item()}")
                        print(f"  Unique Targets: {target_cpu.unique().tolist()}")
                        logits_np = output[:2].detach().cpu().numpy()
                        print(f"  Logits (first 2): {logits_np}")
                        print(f"  Targets (first 2): {target_cpu[:2].numpy()}")
                        print(f"  Classification Loss: {classification_loss.item():.6f}")
                        if hasattr(self.model, 'args') and hasattr(self.model.args, 'temperature'):
                             print(f"  Model Temperature: {self.model.args.temperature}")

                    loss = classification_loss

                    # MI Loss
                    if is_train and self.mi_criterion is not None and hand_crafted_text_features is not None:
                        mi_loss = self.mi_criterion(processed_learnable_text_features, hand_crafted_text_features)
                        loss += mi_weight * mi_loss
                        mi_losses.update(mi_loss.item(), target.size(0))

                    # DC Loss
                    if is_train and self.dc_criterion is not None:
                        dc_loss = self.dc_criterion(processed_learnable_text_features)
                        loss += dc_weight * dc_loss
                        dc_losses.update(dc_loss.item(), target.size(0))

                    # MoCo Loss
                    if is_train and moco_logits is not None:
                         moco_target = torch.zeros(moco_logits.size(0), dtype=torch.long).to(self.device)
                         moco_loss = torch.nn.CrossEntropyLoss()(moco_logits, moco_target)
                         loss += moco_loss
                         moco_losses.update(moco_loss.item(), target.size(0))

                    # Concept-guided Attention Distillation (CAD) Loss
                    if is_train and self.cad_criterion is not None and patch_features is not None:
                        ref_concepts = self.concept_prototypes if self.concept_prototypes is not None else hand_crafted_text_features
                        if ref_concepts is not None:
                            cad_loss = self.cad_criterion(patch_features, processed_learnable_text_features, ref_concepts, target)
                            loss += self.lambda_cad * cad_loss
                            cad_losses.update(cad_loss.item(), target.size(0))

                    # Text Prototype Distillation Loss
                    if is_train and self.text_distill_criterion is not None:
                        ref_concepts = self.concept_prototypes if self.concept_prototypes is not None else hand_crafted_text_features
                        if ref_concepts is not None:
                            text_loss = self.text_distill_criterion(processed_learnable_text_features, ref_concepts)
                            loss += self.lambda_text * text_loss
                            text_losses.update(text_loss.item(), target.size(0))

                if is_train:
                    self.optimizer.zero_grad()
                    loss = loss.to(self.device) # Đảm bảo loss luôn ở trên CUDA
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
                    
                    # Update EMA shadow weights
                    self.ema.update(self.model)

                # Record metrics
                is_multilabel = target.dtype == torch.float32
                if is_multilabel:
                    preds = torch.sigmoid(output)
                    # For multi-label, we don't have a single correct class
                    acc = 0.0
                else:
                    preds = output.argmax(dim=1)
                    correct_preds = preds.eq(target).sum().item()
                    acc = (correct_preds / target.size(0)) * 100.0

                losses.update(loss.item(), target.size(0))
                war_meter.update(acc, target.size(0))

                # Collect preds for UAR/mAP
                all_preds_list.append(preds.cpu().detach())
                all_targets_list.append(target.cpu().detach())

                if not is_train and not is_multilabel and saved_images_count < 32:
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
                
                # Update progress bar
                if len(all_preds_list) > 0:
                    curr_preds = torch.cat(all_preds_list).numpy()
                    curr_targets = torch.cat(all_targets_list).numpy()
                    # Only calc metrics every 10 batches to save CPU time
                    if i % 10 == 0: 
                        try:
                            if is_multilabel:
                                running_map = average_precision_score(curr_targets, curr_preds, average='macro') * 100
                            else:
                                cm = confusion_matrix(curr_targets, curr_preds, labels=range(output.shape[1]))
                                class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6)
                                running_uar = np.nanmean(class_acc) * 100
                        except:
                            pass
                
                if is_multilabel:
                    pbar.set_postfix({
                        'Loss': f"{losses.avg:.4f}",
                        'mAP': f"{running_map:.2f}%"
                    })
                else:
                    pbar.set_postfix({
                        'Loss': f"{losses.avg:.4f}",
                        'WAR': f"{war_meter.avg:.2f}%",
                        'UAR': f"{running_uar:.2f}%"
                    })
        
        # Calculate epoch-level metrics
        all_preds = torch.cat(all_preds_list)
        all_targets = torch.cat(all_targets_list)
        
        # Get class names for dataset
        from utils.metrics_logger import format_multilabel_matrix_report, format_confusion_matrix_report, get_dataset_class_names
        dataset_name = getattr(self.model.args, 'dataset', 'Unknown') if hasattr(self.model, 'args') else 'Unknown'
        num_classes = all_targets.shape[1] if all_targets.dim() > 1 else (int(all_targets.max().item() + 1))
        class_names = get_dataset_class_names(dataset_name, num_classes=num_classes)

        prefix = f"{mode_str} Epoch: [{epoch_str}]"
        if all_targets.dtype == torch.float32: # Multi-label
            report_str, metrics_dict = format_multilabel_matrix_report(all_targets, all_preds, class_names=class_names)
            map_score = metrics_dict['macro_map']
            
            logging.info(f"{prefix} * mAP: {map_score:.3f}")
            print(f"\n{report_str}")
            with open(self.log_txt_path, 'a') as f:
                f.write('Current mAP: {map_score:.3f}'.format(map_score=map_score) + '\n')
                f.write(report_str + '\n')
            return map_score, map_score, losses.avg, None
        else: # Single-label
            report_str, metrics_dict = format_confusion_matrix_report(all_targets, all_preds, class_names=class_names)
            war = metrics_dict['war']
            uar = metrics_dict['uar']
            cm = metrics_dict['confusion_matrix']

            logging.info(f"{prefix} * WAR: {war:.3f} | UAR: {uar:.3f}")
            print(f"\n{report_str}")
            with open(self.log_txt_path, 'a') as f:
                f.write('Current WAR: {war:.3f}'.format(war=war) + '\n')
                f.write('Current UAR: {uar:.3f}'.format(uar=uar) + '\n')
                f.write(report_str + '\n')
            return war, uar, losses.avg, cm
        
    def train_epoch(self, train_loader, epoch_num):
        """Executes one full training epoch."""
        res = self._run_one_epoch(train_loader, str(epoch_num), is_train=True)
        torch.cuda.empty_cache()
        return res
    
    @torch.no_grad()
    def validate(self, val_loader, epoch_num_str="Final"):
        """Executes one full validation run."""
        has_ema = hasattr(self, 'ema') and self.ema is not None
        if has_ema:
            self.ema.apply(self.model)
            
        res = self._run_one_epoch(val_loader, epoch_num_str, is_train=False)
        
        if has_ema:
            self.ema.restore(self.model)
            
        torch.cuda.empty_cache()
        return res