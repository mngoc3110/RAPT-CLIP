#!/bin/bash

python main.py \
    --mode eval \
    --gpu mps \
    --exper-name eval_final_fix \
    --eval-checkpoint best/model_best.pth \
    --root-dir ./ \
    --train-annotation RAER/annotation/train_80.txt \
    --val-annotation RAER/annotation/val_20.txt \
    --test-annotation RAER/annotation/test.txt \
    --clip-path ViT-B/16 \
    --bounding-box-face RAER/bounding_box/face.json \
    --bounding-box-body RAER/bounding_box/body.json \
    --text-type prompt_ensemble \
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