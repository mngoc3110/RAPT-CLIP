#!/bin/bash
# ============================================================================
# RAER Training Script — Frame-Level Fusion + Semantic LDL + Cosine Scheduler
# ============================================================================
#
# This script uses the new optimized Semantic LDL (Label Distribution Learning)
# to solve the "Double-Scaling Loss Trap" from LDAM and handles ambiguous 
# expressions perfectly.
# ============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

python main.py \
  --mode train \
  --exper-name RAER-FrameLevel-LDL \
  --fusion-type cmaf \
  --use-context \
  --dataset RAER \
  --gpu 0 \
  --epochs 20 \
  --batch-size 2 \
  --print-freq 50 \
  --seed 42 \
  --workers 2 \
  --root-dir /kaggle/input/datasets/bearmn/raer-video-emotion-dataset \
  --train-annotation /kaggle/input/datasets/bearmn/raer-annot/annotation/train.txt \
  --val-annotation /kaggle/input/datasets/bearmn/raer-annot/annotation/test.txt \
  --test-annotation /kaggle/input/datasets/bearmn/raer-annot/annotation/test.txt \
  --clip-path ViT-B/16 \
  --bounding-box-face /kaggle/input/datasets/bearmn/raer-video-emotion-dataset/RAER/bounding_box/face.json \
  --bounding-box-body /kaggle/input/datasets/bearmn/raer-video-emotion-dataset/RAER/bounding_box/body.json \
  \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 3e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.005 \
  --scheduler cosine \
  \
  --loss-type ldl \
  --use-ldl \
  --ldl-temperature 1.0 \
  --ldl-warmup 0 \
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
  --drop-path-rate 0.1 \
  --use-amp

# ============================================================================
# HOW TO RESUME TRAINING:
# To resume training from a checkpoint (e.g. model.pth), add a backslash '\'
# to the line above and append the resume flag:
#   --resume /path/to/checkpoint/model.pth
# ============================================================================
