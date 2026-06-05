#!/bin/bash
# ============================================================================
# RAER Training Script — V1-Restored + GFI (Gated Feature Integration)
# ============================================================================
# Dựa trên config V1 (73.76% UAR) + giữ Gated Feature Fusion (gate_fc)
#
# Các thay đổi so với Teacher-Optimized (bị regression):
#   ldam_s     : 3.0  → 30.0  (fix gradient yếu - nguyên nhân chính)
#   weight_decay: 0.0005 → 0.005 (tăng regularization)
#   mixup_alpha : 0.2  → 0.0  (LDAM + Mixup xung đột)
#
# Cải tiến giữ lại so với V1:
#   + gate_fc (Gated Feature Integration) — đã có trong model
#   + EMA trong validation
#   + drop_path_rate=0.1
# ============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

python main.py \
  --mode train \
  --exper-name RAER-GFI \
  --fusion-type gfi \
  --dataset RAER \
  --gpu 0 \
  --epochs 20 \
  --batch-size 4 \
  --print-freq 50 \
  --seed 42 \
  \
  --root-dir ./ \
  --train-annotation RAER/annotation/train.txt \
  --val-annotation RAER/annotation/test.txt \
  --test-annotation RAER/annotation/test.txt \
  --clip-path ViT-B/16 \
  --bounding-box-face RAER/bounding_box/face.json \
  --bounding-box-body RAER/bounding_box/body.json \
  \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 3e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.005 \
  --milestones 10 15 \
  --gamma 0.1 \
  \
  --loss-type ldam \
  --ldam-s 30.0 \
  --ldam-max-m 0.5 \
  --mixup-alpha 0.0 \
  --lambda_mi 0.1 \
  --lambda_dc 0.1 \
  --mi-warmup 5 \
  --mi-ramp 10 \
  --dc-warmup 5 \
  --dc-ramp 10 \
  --use-weighted-sampler \
  --grad-clip 1.0 \
  \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --temperature 0.07 \
  --crop-body \
  --drop-path-rate 0.1
