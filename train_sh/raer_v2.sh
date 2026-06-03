#!/bin/bash
# ============================================================================
# RAPT-CLIP v2 Training Script (Kaggle)
# ============================================================================
# Architecture: AdaptiveRegionEncoder + KAN-Fusion + EIM + DualHead
# Training fixes: CosineAnnealing+Warmup, EMA, aligned MI/DC schedule
#
# References:
#   - KAN: Liu et al., 2024 (https://arxiv.org/abs/2404.19756)
#   - MER-CLIP: Li et al., 2025 (https://arxiv.org/abs/2505.05937)
#   - DETR queries: Carion et al., ECCV 2020
#   - EMA: Tarvainen & Valpola, NeurIPS 2017
# ============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Thư mục gốc chứa dataset RAER trên Kaggle
ROOT_DIR="/kaggle/input/datasets/bearmn/raer-video-emotion-dataset"

# Files cấu trúc annotations trên Kaggle
TRAIN_ANNO="/kaggle/input/datasets/bearmn/raer-annot/annotation/train.txt"
VAL_ANNO="/kaggle/input/datasets/bearmn/raer-annot/annotation/test.txt"
TEST_ANNO="/kaggle/input/datasets/bearmn/raer-annot/annotation/test.txt"

# Bounding boxes trên Kaggle
FACE_BOXES="/kaggle/input/datasets/bearmn/raer-video-emotion-dataset/RAER/bounding_box/face.json"
BODY_BOXES="/kaggle/input/datasets/bearmn/raer-video-emotion-dataset/RAER/bounding_box/body.json"

# Clip model path
CLIP_PATH="ViT-B/16"

# Hyperparameters mới
EXPER_NAME="RAPT-CLIP-v2_ARE-KAN-EIM-DualHead"
DATASET="RAER"
EPOCHS=50
BATCH_SIZE=4
ACCUMULATION_STEPS=8
SCHEDULER="cosine"
WARMUP_EPOCHS=3
USE_EMA="true"
EMA_DECAY=0.999
OPTIMIZER="AdamW"
SEED=42
TEMPORAL_LAYERS=1
PRINT_FREQ=50

# Learning Rates
LR=2e-5
LR_IMAGE_ENCODER=1e-6
LR_PROMPT_LEARNER=3e-4
LR_ARE=1e-4
LR_KAN=1e-4
LR_DUALHEAD=1e-4
WEIGHT_DECAY=0.005

# Loss parameters
LOSS_TYPE="ldam"
LDAM_S=10.0
LDAM_MAX_M=0.8
DRW_START_EPOCH=25

# MI and DC Losses
LAMBDA_MI=0.1
LAMBDA_DC=0.1
MI_WARMUP=2
MI_RAMP=5
DC_WARMUP=2
DC_RAMP=5

# Semantic LDL (Label Distribution Learning)
USE_LDL="true"
LDL_WARMUP=5
LDL_TEMPERATURE=0.5
LAMBDA_LDL=1.0

# Supervised Contrastive Loss (SupCon)
USE_SUPCON="true"
LAMBDA_SUPCON=0.05
SUPCON_TEMPERATURE=0.07

# Input configs
NUM_SEGMENTS=16
DURATION=1
IMAGE_SIZE=224
TEXT_TYPE="prompt_ensemble"
CONTEXTS_NUMBER=8
CLASS_TOKEN_POSITION="end"
CLASS_SPECIFIC_CONTEXTS="True"
LOAD_AND_TUNE_PROMPT_LEARNER="True"
GRAD_CLIP=1.0
MIXUP_ALPHA=0.0

echo ">>> Bắt đầu huấn luyện RAPT-CLIP v2 (ARE-KAN-EIM-DualHead) trên Kaggle..."

python main.py \
  --mode train \
  --exper-name ${EXPER_NAME} \
  --dataset ${DATASET} \
  --gpu 0 \
  --use-v2 \
  --legacy-dual-input \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --accumulation-steps ${ACCUMULATION_STEPS} \
  --scheduler ${SCHEDULER} \
  --warmup-epochs ${WARMUP_EPOCHS} \
  --use-ema \
  --ema-decay ${EMA_DECAY} \
  --optimizer ${OPTIMIZER} \
  --lr ${LR} \
  --lr-image-encoder ${LR_IMAGE_ENCODER} \
  --lr-prompt-learner ${LR_PROMPT_LEARNER} \
  --lr-are ${LR_ARE} \
  --lr-kan ${LR_KAN} \
  --lr-dualhead ${LR_DUALHEAD} \
  --weight-decay ${WEIGHT_DECAY} \
  --loss-type ${LOSS_TYPE} \
  --ldam-s ${LDAM_S} \
  --ldam-max-m ${LDAM_MAX_M} \
  --drw-start-epoch ${DRW_START_EPOCH} \
  --lambda_mi ${LAMBDA_MI} \
  --lambda_dc ${LAMBDA_DC} \
  --mi-warmup ${MI_WARMUP} \
  --mi-ramp ${MI_RAMP} \
  --dc-warmup ${DC_WARMUP} \
  --dc-ramp ${DC_RAMP} \
  --num-segments ${NUM_SEGMENTS} \
  --duration ${DURATION} \
  --image-size ${IMAGE_SIZE} \
  --seed ${SEED} \
  --temporal-layers ${TEMPORAL_LAYERS} \
  --print-freq ${PRINT_FREQ} \
  --root-dir ${ROOT_DIR} \
  --train-annotation ${TRAIN_ANNO} \
  --val-annotation ${VAL_ANNO} \
  --test-annotation ${TEST_ANNO} \
  --clip-path ${CLIP_PATH} \
  --bounding-box-face ${FACE_BOXES} \
  --bounding-box-body ${BODY_BOXES} \
  --crop-body \
  --text-type ${TEXT_TYPE} \
  --contexts-number ${CONTEXTS_NUMBER} \
  --class-token-position ${CLASS_TOKEN_POSITION} \
  --class-specific-contexts ${CLASS_SPECIFIC_CONTEXTS} \
  --load_and_tune_prompt_learner ${LOAD_AND_TUNE_PROMPT_LEARNER} \
  --use-amp \
  --use-weighted-sampler \
  --grad-clip ${GRAD_CLIP} \
  --mixup-alpha ${MIXUP_ALPHA} \
  --use-ldl \
  --ldl-warmup ${LDL_WARMUP} \
  --ldl-temperature ${LDL_TEMPERATURE} \
  --lambda-ldl ${LAMBDA_LDL} \
  --use-supcon \
  --lambda-supcon ${LAMBDA_SUPCON} \
  --supcon-temperature ${SUPCON_TEMPERATURE}

