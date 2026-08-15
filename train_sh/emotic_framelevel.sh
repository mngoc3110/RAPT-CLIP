#!/bin/bash
# EMOTIC Training Script - Multi-label FrameLevelFusion
# Target mAP: > 30%

# ==========================================
# HYPERPARAMETER OPTIMIZATION FOR EMOTIC
# ==========================================
# Dataset: EMOTIC (26 classes, multi-label)
# Features:
#   - CMAF Fusion (Face + Body + Context)
#   - BCE Loss / ASL Loss (Multi-label)
#   - ViT-B/16 Fine-tuned (lr=1e-6)
#   - Prompt Ensemble (detailed LLM lexicons)
#   - Modality Dropout (p=0.3)
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
    --batch-size 32 \
    --print-freq 50 \
    --use-amp \
    --grad-clip 1.0 \
    --optimizer AdamW \
    --lr 2e-5 \
    --lr-image-encoder 1e-6 \
    --lr-prompt-learner 2e-4 \
    --lr-adapter 1e-4 \
    --weight-decay 0.005 \
    --momentum 0.9 \
    --milestones 15 30 \
    --gamma 0.1 \
    --loss-type masked_asl \
    --mask-ratio 0.3 \
    --lambda_mi 0.1 \
    --lambda_dc 0.0 \
    --text-type prompt_ensemble \
    --num-segments 1 \
    --temporal-layers 1 \
    --fusion-type cmaf \
    --use-context \
    --modality-dropout 0.3 \
    --drop-path-rate 0.15 \
    --duration 1 \
    --image-size 224
