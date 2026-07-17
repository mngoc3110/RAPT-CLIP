#!/bin/bash
# ============================================================================
# RAER Training Script — Frame-Level Fusion (CMAF) + LDAM Loss
# ============================================================================
#
# This script reproduces the best experiment: RAER-FrameLevelFusion-[06-07]-[03:06]
# Best Valid UAR: 77.06% (Epoch 12)
# ============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

python main.py \
  --mode train \
  --exper-name RAER-FrameLevelFusion \
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
  --momentum 0.9 \
  --milestones 10 15 \
  --gamma 0.1 \
  --grad-clip 1.0 \
  --use-amp \
  \
  --loss-type ldam \
  --ldam-max-m 0.5 \
  --ldam-s 30.0 \
  --label-smoothing 0.05 \
  --lambda_mi 0.1 \
  --lambda_dc 0.1 \
  --mi-warmup 5 \
  --mi-ramp 10 \
  --mi-ramp-type ramp_up \
  --dc-warmup 5 \
  --dc-ramp 10 \
  --use-weighted-sampler \
  --mixup-alpha 0.0 \
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
  --modality-dropout 0.3 \
  "$@"

# ============================================================================
# HOW TO RESUME TRAINING:
#   !bash train_sh/raer_framelevel_ldl.sh \
#     --resume /kaggle/input/models/bearmn/rapt-clip-framelevelfusion/tensorflow2/default/1/model_best.pth
# ============================================================================
