#!/bin/bash
# Exp 3b: CLS Token (Baseline Temporal)

cd ..
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
  --mode train \
  --exper-name ablation_cls_token \
  --gpu 0 \
  --epochs 20 \
  --batch-size 4 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.0005 \
  --milestones 10 20 30 40 50 \
  --gamma 0.1 \
  --temporal-layers 1 \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 10 \
  --root-dir /kaggle/input/raer-video-emotion-dataset \
  --train-annotation /kaggle/input/raer-annot/annotation/train_80.txt \
  --val-annotation /kaggle/input/raer-annot/annotation/val_20.txt \
  --test-annotation /kaggle/input/raer-annot/annotation/test.txt \
  --clip-path ViT-B/16 \
  --bounding-box-face /kaggle/input/raer-video-emotion-dataset/RAER/bounding_box/face.json \
  --bounding-box-body /kaggle/input/raer-video-emotion-dataset/RAER/bounding_box/body.json \
  --text-type prompt_ensemble \
  --temporal-type cls \
  --use-adapter True \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --lambda_dc 0.1 \
  --dc-warmup 5 \
  --dc-ramp 10 \
  --lambda_mi 0.1 \
  --mi-warmup 5 \
  --mi-ramp 10 \
  --slerp-weight 0.0 \
  --temperature 0.07 \
  --use-ldl \
  --ldl-temperature 1.0 \
  --use-moco \
  --moco-k 4096 \
  --moco-m 0.99 \
  --lambda_moco 0.1 \
  --moco-warmup 5 \
  --moco-ramp 10 \
  --use-amp \
  --use-weighted-sampler \
  --crop-body \
  --grad-clip 1.0 \
  --mixup-alpha 0.2