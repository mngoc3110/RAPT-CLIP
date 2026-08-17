#!/bin/bash
# EMOTIC Training Script - Multi-label FrameLevelFusion
# Target mAP: > 30%

# ==========================================
# HYPERPARAMETER OPTIMIZATION FOR EMOTIC
# ==========================================
# Dataset: EMOTIC (26 classes, multi-label)
# Features:
#   - CMAF Fusion (Face + Body + Context)
#   - ASL Loss (Multi-label)
#   - ViT-B/16 Fine-tuned (lr=1e-6)
#   - Prompt Ensemble (detailed LLM lexicons)
#   - Modality Dropout (p=0.1, reduced from 0.3)
#   - Fixed class bias from training data prior
#
# FIXES applied vs original config:
#   - lr_prompt_learner: 2e-4 → 1e-5
#     (was causing catastrophic forgetting of CLIP text-image alignment)
#   - modality_dropout: 0.3 → 0.1 (was too aggressive, destabilizing training)
#   - scheduler: multistep → cosine (smoother LR decay without sharp drops)
#   - drop_path_rate: 0.15 → 0.10 (slightly less aggressive stochasticity)
# ==========================================

export CUDA_VISIBLE_DEVICES=0

python main.py \
    --mode train \
    --dataset EMOTIC \
    --gpu 0 \
    --exper-name EMOTIC-FrameLevelFusion \
    --root-dir /kaggle/input/datasets/bearmn/emotic-dataset-rapt-clip-bearmn/cvpr_emotic \
    --train-annotation /kaggle/input/datasets/bearmn/emotic-dataset-rapt-clip-bearmn/train.txt \
    --val-annotation /kaggle/input/datasets/bearmn/emotic-dataset-rapt-clip-bearmn/val.txt \
    --test-annotation /kaggle/input/datasets/bearmn/emotic-dataset-rapt-clip-bearmn/test.txt \
    --clip-path ViT-B/16 \
    --bounding-box-face /kaggle/input/datasets/bearmn/emotic-dataset-rapt-clip-bearmn/emotic_face_bboxes_mtcnn.json \
    --bounding-box-body /kaggle/input/datasets/bearmn/emotic-dataset-rapt-clip-bearmn/emotic_body_bboxes.json \
    --epochs 40 \
    --batch-size 8 \
    --print-freq 50 \
    --use-amp \
    --grad-clip 1.0 \
    --optimizer AdamW \
    --loss-type asl \
    --lr 2e-5 \
    --lr-image-encoder 1e-6 \
    --lr-prompt-learner 1e-5 \
    --lr-adapter 3e-5 \
    --weight-decay 0.01 \
    --momentum 0.9 \
    --scheduler cosine \
    --lambda_mi 0.0 \
    --lambda_dc 0.0 \
    --use-cgr \
    --contexts-number 16 \
    --text-type prompt_ensemble \
    --num-segments 1 \
    --temporal-layers 1 \
    --fusion-type cmaf \
    --use-context \
    --crop-body \
    --mask-context-body \
    --modality-dropout 0.1 \
    --drop-path-rate 0.00 \
    --duration 1 \
    --image-size 224 \
    --temperature 1.0
