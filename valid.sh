#!/bin/bash
# Script for evaluating trained models locally
# [LUỒNG 1: KHỞI ĐỘNG EVAL]
# Sử dụng cấu hình kiến trúc Y HỆT như file training (train_final_fix.sh)
# Chỉ thay đổi --mode thành 'eval' và chỉ định checkpoint cần kiểm tra.

python main.py \
    --mode eval \
    --gpu mps \
    --exper-name eval_Final_Best \
    --eval-checkpoint output/full_batch_ao/model_best.pth \
    --root-dir ./ \
    --train-annotation RAER/annotation/train_80.txt \
    --val-annotation RAER/annotation/val_20.txt \
    --test-annotation RAER/annotation/test.txt \
    --clip-path ViT-B/16 \
    --bounding-box-face RAER/bounding_box/face.json \
    --bounding-box-body RAER/bounding_box/body.json \
    --text-type prompt_ensemble \
    --temporal-type attn_pool \
    --use-adapter True \
    --contexts-number 8 \
    --class-token-position end \
    --class-specific-contexts True \
    --load_and_tune_prompt_learner True \
    --temporal-layers 1 \
    --num-segments 16 \
    --duration 1 \
    --image-size 224 \
    --seed 42 \
    --temperature 0.07 \
    --crop-body