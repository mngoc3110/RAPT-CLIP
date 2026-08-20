#!/bin/bash
# EMOTIC Training Script - Local MPS (M2 Max)
# Target mAP: > 30%

export CUDA_VISIBLE_DEVICES=0

python main.py \
    --mode train \
    --dataset EMOTIC \
    --gpu mps \
    --exper-name EMOTIC-Local-MPS \
    --root-dir emotic_dataset/cvpr_emotic \
    --train-annotation emotic_dataset/train.txt \
    --val-annotation emotic_dataset/val.txt \
    --test-annotation emotic_dataset/test.txt \
    --clip-path ViT-B/16 \
    --bounding-box-face emotic_dataset/emotic_face_bboxes_mtcnn.json \
    --bounding-box-body emotic_dataset/emotic_body_bboxes.json \
    --epochs 40 \
    --batch-size 4 \
    --print-freq 5 \
    --grad-clip 1.0 \
    --optimizer AdamW \
    --loss-type asl \
    --use-q2l \
    --lambda-cad 0.1 \
    --lambda-text 0.1 \
    --lr 2e-5 \
    --lr-image-encoder 1e-6 \
    --lr-prompt-learner 1e-5 \
    --lr-adapter 3e-5 \
    --weight-decay 0.01 \
    --momentum 0.9 \
    --scheduler cosine \
    --lambda_mi 0.1 \
    --lambda_dc 0.1 \
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
    --drop-path-rate 0.00 \
    --duration 1 \
    --image-size 224 \
    --temperature 0.07
