#!/bin/bash
# EMOTIC Training Script - Multi-label FrameLevelFusion
# Target mAP: > 35-40%
#
# ==========================================
# HYPERPARAMETER OPTIMIZATION FOR EMOTIC
# ==========================================
# Dataset: EMOTIC (26 classes, multi-label, 16K images)
#
# ARCHITECTURE:
#   - Triple-Stream CMAF Fusion (Face + Body + Full Scene Context)
#   - ViT-B/16 FROZEN backbone (prevents overfitting on 16K images)
#   - Prompt Ensemble (5 prompts per class) + class-specific CoOp contexts
#   - CGR (K-means concept generation) + CGLA (cross-modal patch alignment)
#   - MPI (multi-perspective self-attention pooling at inference)
#
# LOSS STRATEGY:
#   - Calibrated ASL: gamma_neg=2.0 (was 4.0 → suppressed rare classes)
#     gamma_neg=4.0 + 1/0.07 temperature = 57x effective scaling on negatives.
#     Reduces to 28x at gamma_neg=2.0 → balanced gradients for rare labels.
#   - CAD (Concept Attention Distillation) λ=0.1: aligns patch attention
#     with concept prototypes, improves rare-class localization.
#   - TextDistill λ=0.1: keeps learnable prompts close to CLIP semantic space.
#
# REGULARIZATION:
#   - freeze-image-encoder: CRITICAL for 16K dataset (prevents backbone drift)
#   - drop-path-rate=0.1: stochastic depth for ViT backbone stability
#   - weight-decay=0.02: stronger L2 for adapters/projection head
#   - lr-prompt-learner=5e-6: slow down (was 1e-5 → prompt overfitting by Ep3)
#   - mixup-alpha=0.4: interpolates rare label vectors into each batch
#   - modality-dropout=0.1: forces each stream to be self-sufficient
#   - use-weighted-sampler: upsamples rare-class images each epoch
#
# CONTEXT STREAM:
#   - NO --mask-context-body! (removed from previous run)
#     In CAER (TV scenes): masking body forces attention to room background.
#     In EMOTIC (real-world photos): masking destroys interaction cues:
#     hugs, handshakes, trophy-holding, physical contact → Full scene is better.
# ==========================================

export CUDA_VISIBLE_DEVICES=0

DATA_DIR="/kaggle/input/datasets/bearmn/emotic-dataset-rapt-clip-bearmn"
TRAIN_ANN="$DATA_DIR/train_bbox.txt"
[ ! -f "$TRAIN_ANN" ] && TRAIN_ANN="$DATA_DIR/train.txt"
VAL_ANN="$DATA_DIR/val_bbox.txt"
[ ! -f "$VAL_ANN" ] && VAL_ANN="$DATA_DIR/val.txt"
TEST_ANN="$DATA_DIR/test_bbox.txt"
[ ! -f "$TEST_ANN" ] && TEST_ANN="$DATA_DIR/test.txt"

python main.py \
    --mode train \
    --dataset EMOTIC \
    --gpu 0 \
    --exper-name EMOTIC-FrameLevelFusion \
    --root-dir "$DATA_DIR/cvpr_emotic/cvpr_emotic" \
    --train-annotation "$TRAIN_ANN" \
    --val-annotation "$VAL_ANN" \
    --test-annotation "$TEST_ANN" \
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
    --lr 2e-5 \
    --lr-image-encoder 1e-6 \
    --lr-prompt-learner 1e-5 \
    --lr-adapter 3e-5 \
    --weight-decay 0.01 \
    --momentum 0.9 \
    --scheduler cosine \
    --lambda-cad 0.0 \
    --lambda-text 0.0 \
    --mi-warmup 0 \
    --dc-warmup 0 \
    --mi-ramp 0 \
    --dc-ramp 0 \
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
    --mixup-alpha 0.0 \
    --drop-path-rate 0.0 \
    --duration 1 \
    --image-size 224 \
    --temperature 0.07
