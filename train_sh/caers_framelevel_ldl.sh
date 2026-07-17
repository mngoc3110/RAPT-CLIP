#!/bin/bash
# CAER-S Training Script - FrameLevelFusion
# Target Accuracy: > 91%

# ==========================================
# HYPERPARAMETER OPTIMIZATION FOR CAER-S
# ==========================================
# Dataset: CAER (Context-Aware Emotion Recognition)
# Features:
#   - CMAF Fusion (Face + Scene/Context)
#   - LDAM Loss (s=30, max_m=0.5)
#   - ViT-B/16 Fine-tuned (lr=1e-6)
#   - Prompt Ensemble (detailed LLM lexicons)
#   - CoCoOp (8 context tokens)
#   - Modality Dropout (p=0.3)
# ==========================================

export CUDA_VISIBLE_DEVICES=0

python main.py \
    --mode train \
    --dataset CAER \
    --exper-name CAER-FrameLevelFusion-Optimal \
    --root-dir /kaggle/input/caer-dataset \
    --train-annotation /kaggle/input/caer-dataset/train.txt \
    --val-annotation /kaggle/input/caer-dataset/val.txt \
    --test-annotation /kaggle/input/caer-dataset/test.txt \
    --clip-path ViT-B/16 \
    --bounding-box-face /kaggle/input/caer-dataset/face_bboxes.json \
    --bounding-box-body /kaggle/input/caer-dataset/body_bboxes.json \
    --epochs 30 \
    --batch-size 32 \
    --print-freq 50 \
    --use-amp \
    --grad-clip 1.0 \
    --optimizer AdamW \
    --lr 2e-5 \
    --lr-image-encoder 1e-6 \
    --lr-prompt-learner 3e-4 \
    --lr-adapter 1e-4 \
    --weight-decay 0.005 \
    --momentum 0.9 \
    --milestones 10 20 \
    --gamma 0.1 \
    --loss-type ldam \
    --ldam-s 30.0 \
    --ldam-max-m 0.5 \
    --lambda_mi 0.1 \
    --lambda_dc 0.1 \
    --label-smoothing 0.05 \
    --use-weighted-sampler \
    --text-type prompt_ensemble \
    --num-segments 1 \
    --temporal-layers 1 \
    --fusion-type cmaf \
    --use-context \
    --modality-dropout 0.3 \
    --duration 1 \
    --image-size 224
