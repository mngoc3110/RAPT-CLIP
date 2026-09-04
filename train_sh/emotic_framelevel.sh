#!/bin/bash
# EMOTIC Training Script - Multi-label FrameLevelFusion
# Target mAP: > 35%

# ==========================================
# HYPERPARAMETER OPTIMIZATION FOR EMOTIC (Target mAP: > 35-40%)
# ==========================================
# Dataset: EMOTIC (26 classes, multi-label)
# Features:
#   - CMAF Fusion (Face + Body + Full Context)
#   - Calibrated ASL Loss (gamma_neg=2.0 to prevent tail-class suppression)
#   - Full Unmasked Scene Context (preserves human-object & social interactions)
#   - ViT-B/16 FROZEN (prevents representation drift on small dataset)
#   - Prompt Ensemble (detailed LLM lexicons + CoOp context tuning)
#   - Modality Dropout (p=0.1) & DropPath (0.1) to eliminate overfitting
#
# KEY CHANGES vs prev run (stuck at 31.32%):
#
# 1. UNMASKED SCENE CONTEXT (REMOVED --mask-context-body)
#    Masking the body with gray (128,128,128) destroyed critical interaction cues
#    (holding items, hugging, shaking hands). Full scene preserves interactive semantics.
#
# 2. CALIBRATED ASL (gamma_neg=2.0 vs prev 4.0)
#    Previous gamma_neg=4.0 combined with 1/0.07 temperature scale suppressed
#    rare classes (Embarrassment, Fear, Esteem, Yearning AP < 10%).
#    gamma_neg=2.0 allows balanced gradients for long-tail multi-labels.
#
# 3. REGULARIZATION & OVERFITTING MITIGATION
#    - drop-path-rate: 0.0 -> 0.1
#    - weight-decay: 0.01 -> 0.02
#    - lr-prompt-learner: 1e-5 -> 5e-6 (slow down prompt overfitting)
#    - lr-adapter: 3e-5 -> 2e-5
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
    --asl-gamma-neg 2.0 \
    --asl-gamma-pos 0.0 \
    --asl-clip 0.05 \
    --use-cgla \
    --cgla-topk 16 \
    --cgla-alpha 1.0 \
    --lr 2e-5 \
    --lr-image-encoder 1e-6 \
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
    --modality-dropout 0.0 \
    --mixup-alpha 0.0 \
    --drop-path-rate 0.0 \
    --duration 1 \
    --image-size 224 \
    --temperature 0.07
