#!/usr/bin/env python3
"""
Local diagnostic trainer for EMOTIC on M2 Max (MPS).
Runs N_BATCHES batches and reports loss + mAP trend.
Usage: python local_train_diag.py
"""
import sys
import os
import torch
import argparse
import numpy as np
from sklearn.metrics import average_precision_score

# --- Cấu hình ---
ROOT_DIR   = 'emotic_dataset/cvpr_emotic'
TRAIN_TXT  = 'emotic_dataset/train.txt'
VAL_TXT    = 'emotic_dataset/val.txt'
FACE_JSON  = 'emotic_dataset/emotic_face_bboxes_mtcnn.json'
BODY_JSON  = 'emotic_dataset/emotic_body_bboxes.json'
N_BATCHES  = 100
BATCH_SIZE = 4
IMAGE_SIZE = 224

# M2 Max: ưu tiên MPS, fallback CPU
if torch.backends.mps.is_available():
    DEVICE_STR = 'mps'
elif torch.cuda.is_available():
    DEVICE_STR = 'cuda'
else:
    DEVICE_STR = 'cpu'

print(f"Device: {DEVICE_STR}")
print(f"PyTorch: {torch.__version__}")

# ----- Build args -----
args = argparse.Namespace(
    dataset='EMOTIC',
    mode='train',
    gpu=0,
    seed=42,
    clip_path='ViT-B/16',
    bounding_box_face=FACE_JSON,
    bounding_box_body=BODY_JSON,
    root_dir=ROOT_DIR,
    train_annotation=TRAIN_TXT,
    val_annotation=VAL_TXT,
    test_annotation=VAL_TXT,
    num_segments=1,
    duration=1,
    image_size=IMAGE_SIZE,
    temperature=0.07, # RESTORED: Need 1/0.07 = 14.28 scale to overcome class_bias (-4.90)!
    crop_body=True,
    mask_context_body=True,
    use_context=True,
    fusion_type='cmaf',
    temporal_layers=1,
    contexts_number=16,
    class_token_position='end',
    class_specific_contexts='True',
    load_and_tune_prompt_learner='True',
    text_type='prompt_ensemble',
    use_moco=False,
    moco_k=4096,
    moco_m=0.99,
    moco_t=0.07,
    drop_path_rate=0.0,
    freeze_image_encoder=False,
    ablation_no_text=False,
    use_v2=False,
    modality_dropout=0.0,
    use_q2l=False,
    use_cgr=False,
    use_mpi=False,
    lambda_cad=0.0,
    lambda_text=0.0,
    num_classes=26,
    lr_image_encoder=1e-6,
    device=torch.device(DEVICE_STR),
    workers=0,
)

# ----- Build model -----
print("\n[1] Building model...")
from utils.builders import build_model, get_class_info

class_names, input_text = get_class_info(args)
model = build_model(args, input_text)   # build_model(args, input_text) - correct signature
model = model.to(args.device)

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {n_trainable:,}")

# ----- Build dataloader -----
print("\n[2] Building dataloader...")
from dataloader.video_dataloader import train_data_loader

train_data = train_data_loader(
    root_dir=ROOT_DIR,
    list_file=TRAIN_TXT,
    num_segments=1,
    duration=1,
    image_size=IMAGE_SIZE,
    dataset_name='EMOTIC',
    bounding_box_face=FACE_JSON,
    bounding_box_body=BODY_JSON,
    crop_body=True,
    mask_context_body=True,
    num_classes=26,
)
loader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=False, drop_last=True,
)
print(f"Dataset: {len(train_data)} images")

# ----- Compute class prior -----
print("\n[3] Computing class prior...")
all_labels = []
for record in train_data.video_list:
    if isinstance(record.label, torch.Tensor):
        all_labels.append(record.label.numpy())

if all_labels:
    all_labels_np = np.stack(all_labels)  # (N, 26)
    class_freq = all_labels_np.mean(axis=0)
    freq_tensor = torch.from_numpy(class_freq).float()
    
    # Set co-occurrence
    cooccur = all_labels_np.T @ all_labels_np
    row_sums = cooccur.diagonal().copy(); row_sums[row_sums == 0] = 1
    cooccur = cooccur / row_sums[:, None]
    model.set_label_cooccurrence(torch.from_numpy(cooccur).float())
    
    # Set class_bias
    if hasattr(model, 'set_class_prior'):
        model.set_class_prior(freq_tensor.to(args.device))
    
    # Make class_bias learnable to absorb ASL intercept offset
    if hasattr(model, 'class_bias'):
        model.class_bias.requires_grad_(True)
        print(f"=> class_bias is LEARNABLE to absorb ASL intercept offset")
    print(f"Class frequencies: min={class_freq.min():.3f}, max={class_freq.max():.3f}")

# ----- Optimizer -----
print("\n[4] Building optimizer...")
optimizer_params = [
    {"params": model.prompt_learner.parameters(), "lr": 1e-5}, # ADDED: Essential for text alignment!
    {"params": model.unified_temporal_net.parameters(), "lr": 2e-5},
    {"params": model.image_encoder.parameters(), "lr": 1e-6},
    {"params": model.project_fc.parameters(), "lr": 2e-5},
    {"params": model.face_adapter.parameters(), "lr": 3e-5},
    {"params": model.cmaf.parameters(), "lr": 2e-5},
    {"params": [model.class_bias], "lr": 1e-3}, # MUST be learnable to absorb ASL bias offset
]
optimizer = torch.optim.AdamW(optimizer_params, weight_decay=0.01)

# ----- Loss -----
from utils.loss import AsymmetricLoss
criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)

# ----- Training loop -----
print(f"\n[5] Training {N_BATCHES} batches on {DEVICE_STR}...\n")
print(f"{'Batch':>6} | {'Loss(avg)':>10} | {'mAP':>8} | {'Trend'}")
print("-" * 50)

all_losses = []
all_maps = []
window_preds = []
window_targets = []

model.train()
for i, data in enumerate(loader):
    if i >= N_BATCHES:
        break

    images_face, images_body, images_context, target = data
    images_face    = images_face.to(args.device)
    images_body    = images_body.to(args.device)
    images_context = images_context.to(args.device)
    target = target.float().to(args.device)

    optimizer.zero_grad()
    
    res = model(images_face, images_body, images_context)
    output = res[0]

    if torch.isnan(output).any():
        print(f"  ⚠️  NaN in output at batch {i}!")
        break

    loss = criterion(output, target)
    
    if torch.isnan(loss):
        print(f"  ⚠️  NaN in loss at batch {i}!")
        break
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    loss_val = loss.item()
    all_losses.append(loss_val)

    preds_np   = torch.sigmoid(output).detach().float().cpu().numpy()
    targets_np = target.detach().float().cpu().numpy()
    window_preds.append(preds_np)
    window_targets.append(targets_np)

    if (i + 1) % 10 == 0:
        curr_preds   = np.concatenate(window_preds, axis=0)
        curr_targets = np.concatenate(window_targets, axis=0)
        try:
            mAP = average_precision_score(curr_targets, curr_preds, average='macro') * 100
        except Exception as e:
            mAP = 0.0
        all_maps.append(mAP)
        avg_loss = np.mean(all_losses[-10:])
        
        trend = "📈" if len(all_maps) < 2 else ("📈" if all_maps[-1] >= all_maps[-2] else "📉")
        print(f"{i+1:>6} | {avg_loss:>10.4f} | {mAP:>7.2f}% | {trend}")

print("\n" + "=" * 50)
print("CHẨN ĐOÁN KẾT QUẢ")
print("=" * 50)
if len(all_losses) >= 2:
    loss_delta = all_losses[-1] - all_losses[0]
    loss_ok = loss_delta < 0
    print(f"Loss:  {all_losses[0]:.4f} → {all_losses[-1]:.4f}  {'✅ Giảm tốt' if loss_ok else '❌ Tăng - DIVERGING'}")
if len(all_maps) >= 2:
    map_delta = all_maps[-1] - all_maps[0]
    map_ok = map_delta > 0
    print(f"mAP:   {all_maps[0]:.2f}% → {all_maps[-1]:.2f}%   {'✅ Tăng - ĐANG HỌC' if map_ok else '❌ Giảm - KHÔNG HỌC ĐƯỢC'}")

print(f"\nLoss mỗi 10 batch: {[f'{l:.3f}' for l in all_losses[::10]]}")
print(f"mAP mỗi 10 batch:  {[f'{m:.2f}%' for m in all_maps]}")

# Kiểm tra NaN trong weights
print("\n[6] Kiểm tra NaN trong weights...")
nan_params = [(n, p) for n, p in model.named_parameters() if torch.isnan(p).any()]
if nan_params:
    print(f"❌ NaN trong: {[n for n, _ in nan_params]}")
else:
    print("✅ Không có NaN trong weights")
