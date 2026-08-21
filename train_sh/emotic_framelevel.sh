#!/bin/bash
# EMOTIC Training Script - Multi-label FrameLevelFusion
# Target mAP: > 35%

# ==============================================================================
# 🔥 [LATEST UPDATE - 2026-08-21] EMOTIC PROMPTCAD TRAINING SCRIPT
# Target mAP: > 35%
# ==============================================================================
# 
# 🌟 MAJOR UPGRADES & ARCHITECTURAL FIXES:
# ------------------------------------------------------------------------------
# 1. FULL PROMPTCAD FRAMEWORK (TCSVT 2026):
#    - Concept Generation & Refinement (--use-cgr): Generates LLM concept anchors (μ_c)
#    - Concept Attention Distillation (--lambda-cad 0.1): Distills spatial attention
#      maps onto 196 context/body scene patches (where emotion cues reside)
#    - Text Prototype Distillation (--lambda-text 0.1): L1 regularizer prevents prompt
#      drift WITHOUT penalizing natural multi-label emotion co-occurrences
#    - Multi-Perspective Inference (--use-mpi): Aggregates multi-prompt viewpoints
#
# 2. LEGACY MI/DC LOSSES COMPLETELY REMOVED:
#    - Legacy CrossEntropy MI & MSE DC forced 26 classes to be orthogonal/mutually exclusive
#    - Caused valid mAP to collapse from 30.89% down to 28.72% in emotic-new.txt
#    - Now 100% eliminated for clean multi-label learning
#
# 3. FROZEN ViT-B/16 BACKBONE (--freeze-image-encoder + --lr-image-encoder 0):
#    - 87M ViT parameters locked to preserve 400M-image CLIP zero-shot features
#    - Prevents catastrophic forgetting & overfitting on 16K EMOTIC samples
#    - Trainable parameters reduced from 112M down to ~25M
#
# 4. EMOTIC-OPTIMIZED MODALITY INITIALIZATION (CMAF):
#    - Init weights: Face=12%, Body=33%, Context=55% (was Face=29% - too high)
#    - Dedicated LR=1e-3 for modality_importance to allow fast adaptation
#
# 5. REGULARIZATION & SCHEDULING:
#    - Stronger Mixup: alpha=0.4 (mixes labels effectively for float32 targets)
#    - Epochs: 40 (safe from early overfitting due to frozen backbone)
#    - Weight decay: 0.01 | Cosine LR scheduler
# ==============================================================================

echo "=========================================================================="
echo "🚀 EMOTIC PROMPTCAD FRAMEWORK — LATEST UPDATE (2026-08-21)"
echo "   - Backbone: ViT-B/16 FROZEN (87M params locked, zero-shot preserved)"
echo "   - Distillation: PromptCAD (CAD=0.1 on Context Patches, TextDistill=0.1)"
echo "   - Losses: Multi-label ASL (normalized B*C) + Fixed Data Prior Bias"
echo "   - Legacy MI/DC: COMPLETELY REMOVED (No orthogonal multi-label penalty)"
echo "   - Modality Init: Face=12%, Body=33%, Context=55% (lr=1e-3)"
echo "   - Target mAP: > 35%"
echo "=========================================================================="

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
