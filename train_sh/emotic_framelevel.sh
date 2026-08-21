#!/bin/bash
# EMOTIC Training Script - Multi-label FrameLevelFusion
# Target mAP: > 35%

# ==========================================
# HYPERPARAMETER OPTIMIZATION FOR EMOTIC
# ==========================================
# Dataset: EMOTIC (26 classes, multi-label)
# Features:
#   - CMAF Fusion (Face + Body + Context)
#   - ASL Loss (Multi-label)
#   - ViT-B/16 FROZEN (backbone not updated)
#   - Prompt Ensemble (detailed LLM lexicons)
#   - Modality Dropout (p=0.1)
#   - Fixed class bias from training data prior
#
# KEY CHANGES vs prev run (stuck at 31%):
#
# 1. FREEZE IMAGE ENCODER (--freeze-image-encoder)
#    Root cause of overfitting: ViT-B/16 = 87M/112M params.
#    With 16K images (7000:1 ratio), ViT update drifts away from
#    CLIP pretrained features. CLIP zero-shot gives 28.6% at epoch 0,
#    we only got +2.3% before overfit. Freezing ViT frees optimizer
#    budget for classifier heads that actually need to learn.
#
# 2. DISABLE MI/DC LOSS (lambda_mi=0, lambda_dc=0)
#    emotic-new.txt evidence: valid mAP dropped 30.89%→29.18%→28.72%
#    immediately when MI/DC activated at epoch 8. Confirmed harmful.
#
# 3. INCREASE EPOCHS 20 → 40
#    With frozen encoder, overfitting is much slower (~25M trainable
#    params instead of 112M). Can train much longer safely.
#
# 4. STRONGER MIXUP (0.2 → 0.4)
#    More aggressive augmentation to reduce remaining overfit.
#
# 5. LOWER WEIGHT DECAY (0.05 → 0.01)
#    Frozen encoder params = smaller heads that need less L2.
# ==========================================

export CUDA_VISIBLE_DEVICES=0

python main.py \
    --mode train \
    --dataset EMOTIC \
    --gpu 0 \
    --exper-name EMOTIC-FrameLevelFusion \
    --root-dir /kaggle/input/datasets/bearmn/emotic-dataset-rapt-clip-bearmn/cvpr_emotic/cvpr_emotic \
    --train-annotation /kaggle/input/datasets/bearmn/emotic-dataset-rapt-clip-bearmn/train.txt \
    --val-annotation /kaggle/input/datasets/bearmn/emotic-dataset-rapt-clip-bearmn/val.txt \
    --test-annotation /kaggle/input/datasets/bearmn/emotic-dataset-rapt-clip-bearmn/test.txt \
    --clip-path ViT-B/16 \
    --bounding-box-face /kaggle/input/datasets/bearmn/emotic-dataset-rapt-clip-bearmn/emotic_face_bboxes_mtcnn.json \
    --bounding-box-body /kaggle/input/datasets/bearmn/emotic-dataset-rapt-clip-bearmn/emotic_body_bboxes.json \
    --epochs 40 \
    --batch-size 8 \
    --print-freq 50 \
    --grad-clip 1.0 \
    --optimizer AdamW \
    --loss-type asl \
    --lr 2e-5 \
    --lr-image-encoder 0 \
    --freeze-image-encoder \
    --lr-prompt-learner 1e-5 \
    --lr-adapter 3e-5 \
    --weight-decay 0.01 \
    --momentum 0.9 \
    --scheduler cosine \
    --lambda-cad 0.1 \
    --lambda-text 0.1 \
    --use-mpi \
    --use-cgr \
    --contexts-number 8 \
    --text-type prompt_ensemble \
    --num-segments 1 \
    --temporal-layers 1 \
    --fusion-type cmaf \
    --use-context \
    --crop-body \
    --mask-context-body \
    --modality-dropout 0.1 \
    --mixup-alpha 0.4 \
    --drop-path-rate 0.0 \
    --duration 1 \
    --image-size 224 \
    --temperature 0.07
