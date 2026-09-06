#!/bin/bash
# ============================================================================
# RAER Training Script — Frame-Level Fusion (CMAF) + LDAM Loss [OPTIMIZED]
# ============================================================================
#
# Base result: 77.06% UAR (Epoch 12)
# Target:      >79% UAR
#
# IMPROVEMENTS vs baseline:
#   1. freeze-image-encoder: CLIP ViT-B/16 frozen (prevents overfitting on
#      video data, adapters handle domain adaptation)
#   2. lr-prompt-learner: 3e-4 → 5e-6 (was 60x too high → prompt overfitting
#      by epoch 2-3, CoOp paper uses ~2e-3 for 16 tokens, but with ensemble
#      5 prompts × 8 classes we need much slower lr)
#   3. scheduler: multistep → cosine (smoother decay, better CLIP alignment)
#   4. use-cgr: Concept Generation & Refinement (K-means concept prototypes)
#   5. use-mpi: Multi-Perspective Self-Attention Pooling at inference
#   6. lambda-text=0.1: Text Distillation keeps prompt close to CLIP semantics
#   7. lambda-cad=0.05: Concept Attention Distillation on body patches
#      (light weight for LDAM regime)
#   8. body_adapter: now present in model code (symmetric with face adapter)
#      ratio 0.2→0.5 for better frozen backbone adaptation
#   9. EMA decay: 0.99→0.9999 (proper 10k-step smoothing window)
#   10. epochs: 20→25 (frozen encoder converges slower but more stable)
#
# KEPT FROM BASELINE (proven):
#   - LDAM + s=30.0 (proven best for RAER 8-class imbalance)
#   - mixup-alpha=0.0 (LDAM requires hard labels)
#   - lambda_mi=0.1, lambda_dc=0.1 (MI/DC regularization)
#   - use-weighted-sampler (rare class oversampling)
#   - num-segments=16 (temporal modeling)
# ============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

python main.py \
  --mode train \
  --exper-name RAER-FrameLevelFusion-Optimized \
  --fusion-type cmaf \
  --use-context \
  --dataset RAER \
  --gpu 0 \
  --epochs 25 \
  --batch-size 2 \
  --print-freq 50 \
  --seed 42 \
  --workers 2 \
  --root-dir /kaggle/input/datasets/bearmn/raer-video-emotion-dataset \
  --train-annotation /kaggle/input/datasets/bearmn/raer-annot/annotation/train.txt \
  --val-annotation /kaggle/input/datasets/bearmn/raer-annot/annotation/test.txt \
  --test-annotation /kaggle/input/datasets/bearmn/raer-annot/annotation/test.txt \
  --clip-path ViT-B/16 \
  --bounding-box-face /kaggle/input/datasets/bearmn/raer-video-emotion-dataset/RAER/bounding_box/face.json \
  --bounding-box-body /kaggle/input/datasets/bearmn/raer-video-emotion-dataset/RAER/bounding_box/body.json \
  \
  --freeze-image-encoder \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 0 \
  --lr-prompt-learner 5e-6 \
  --lr-adapter 3e-5 \
  --weight-decay 0.005 \
  --momentum 0.9 \
  --scheduler cosine \
  --grad-clip 1.0 \
  --use-amp \
  \
  --loss-type ldam \
  --ldam-max-m 0.5 \
  --ldam-s 30.0 \
  --label-smoothing 0.05 \
  --lambda_mi 0.1 \
  --lambda_dc 0.1 \
  --mi-warmup 3 \
  --mi-ramp 5 \
  --mi-ramp-type ramp_up \
  --dc-warmup 3 \
  --dc-ramp 5 \
  --lambda-text 0.1 \
  --lambda-cad 0.05 \
  --use-weighted-sampler \
  --mixup-alpha 0.0 \
  \
  --use-cgr \
  --use-mpi \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --num-segments 16 \
  --temporal-layers 1 \
  --duration 1 \
  --image-size 224 \
  --temperature 0.07 \
  --crop-body \
  --drop-path-rate 0.1 \
  --modality-dropout 0.2 \
  "$@"

# ============================================================================
# HOW TO RESUME TRAINING:
#   !bash train_sh/raer_framelevel_ldl.sh \
#     --resume /kaggle/input/models/bearmn/rapt-clip-framelevelfusion/tensorflow2/default/1/model_best.pth
# ============================================================================
