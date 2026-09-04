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
    --asl-gamma-neg 2.0 \
    --asl-gamma-pos 0.0 \
    --asl-clip 0.05 \
    --use-cgla \
    --cgla-topk 16 \
    --cgla-alpha 1.0 \
    --freeze-image-encoder \
    --lambda-cad 0.1 \
    --lambda-text 0.1 \
    --lr 2e-5 \
    --lr-image-encoder 0 \
    --lr-prompt-learner 0.0 \
    --lr-adapter 2e-5 \
    --weight-decay 0.02 \
    --momentum 0.9 \
    --scheduler cosine \
    --lambda_mi 0.0 \
    --lambda_dc 0.0 \
    --use-mpi \
    --use-cgr \
    --contexts-number 8 \
    --text-type prompt_ensemble \
    --num-segments 1 \
    --temporal-layers 1 \
    --fusion-type cmaf \
    --use-context \
    --crop-body \
    --modality-dropout 0.1 \
    --drop-path-rate 0.1 \
    --duration 1 \
    --image-size 224 \
    --temperature 0.07
