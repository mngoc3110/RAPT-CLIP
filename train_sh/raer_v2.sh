#!/bin/bash
# ============================================================================
# RAPT-CLIP v2 (Đột Phá SOTA - Triple-Stream + AU Prompts + Modality Dropout)
# ============================================================================

export CUDA_VISIBLE_DEVICES=0

# Thư mục gốc chứa dataset RAER
ROOT_DIR="/kaggle/input/datasets/bearmn/raer-video-emotion-dataset"

# Files cấu trúc annotations
TRAIN_ANNO="/kaggle/input/datasets/bearmn/raer-annot/annotation/train.txt"
VAL_ANNO="/kaggle/input/datasets/bearmn/raer-annot/annotation/test.txt"
TEST_ANNO="/kaggle/input/datasets/bearmn/raer-annot/annotation/test.txt"

# Bounding boxes
FACE_BOXES="/kaggle/input/datasets/bearmn/raer-video-emotion-dataset/RAER/bounding_box/face.json"
BODY_BOXES="/kaggle/input/datasets/bearmn/raer-video-emotion-dataset/RAER/bounding_box/body.json"

# Mô hình CLIP pretrain
CLIP_PATH="ViT-B/16"
# Nếu bạn chạy offline hoặc tải model về sẵn, thay CLIP_PATH="/path/to/ViT-B-16.pt"

# Các Hyperparameters V2
DATASET="RAER"
EXPER_NAME="RAPT-CLIP-RAER-V2"
TEXT_TYPE="au_guided_prompts" # Sử dụng AU prompts

EPOCHS=50
BATCH_SIZE=4
WORKERS=4
SEED=42

# Tốc độ học (Learning Rates)
LR=1e-4
LR_IMAGE_ENCODER=1e-5
LR_PROMPT_LEARNER=1e-4
LR_ADAPTER=1e-4
WEIGHT_DECAY=0.01

# Loss Functions & Modality Dropout
LDAM_S=30.0
LDAM_MAX_M=0.5
MODALITY_DROPOUT=0.3 # Xác suất rơi Face/Body

# Kiến trúc
TEMPORAL_LAYERS=1
NUM_SEGMENTS=16
DURATION=1

# ============================================================================
# TRAINING COMMAND
# ============================================================================
echo ">>> Bắt đầu huấn luyện RAPT-CLIP V2 trên tập RAER..."

python main.py \
    --mode train \
    --gpu 0 \
    --dataset ${DATASET} \
    --exper-name ${EXPER_NAME} \
    --root-dir ${ROOT_DIR} \
    --train-annotation ${TRAIN_ANNO} \
    --val-annotation ${VAL_ANNO} \
    --test-annotation ${TEST_ANNO} \
    --bounding-box-face ${FACE_BOXES} \
    --bounding-box-body ${BODY_BOXES} \
    --clip-path ${CLIP_PATH} \
    --use-v2 \
    --text-type ${TEXT_TYPE} \
    --modality-dropout ${MODALITY_DROPOUT} \
    --crop-body \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --workers ${WORKERS} \
    --seed ${SEED} \
    --lr ${LR} \
    --lr-image-encoder ${LR_IMAGE_ENCODER} \
    --lr-prompt-learner ${LR_PROMPT_LEARNER} \
    --lr-adapter ${LR_ADAPTER} \
    --weight-decay ${WEIGHT_DECAY} \
    --ldam-s ${LDAM_S} \
    --ldam-max-m ${LDAM_MAX_M} \
    --temporal-layers ${TEMPORAL_LAYERS} \
    --num-segments ${NUM_SEGMENTS} \
    --duration ${DURATION} \
    --use-amp
