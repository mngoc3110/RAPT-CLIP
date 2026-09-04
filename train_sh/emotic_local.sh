#!/bin/bash
# EMOTIC Training Script - Local MPS (M2 Max)
# Train thử local để quan sát mAP tăng trước khi chạy full trên Kaggle.
#
# So với emotic_framelevel.sh (Kaggle):
#   --gpu mps            : Apple Silicon MPS
#   --epochs 20          : đủ để thấy convergence (~10h trên M2 Max)
#   --workers 0          : MPS không hỗ trợ multiprocessing dataloader
#   --clip-path ViT-B/32 : nhẹ hơn ViT-B/16, phù hợp RAM local
#   Không dùng --use-cgla: CGLA cần thêm 1 ViT forward pass/batch → 2x chậm hơn
#   Classifier: CLIP Cosine Similarity + learnable class_bias (per-class threshold calibration)

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
    --clip-path ViT-B/32 \
    --bounding-box-face emotic_dataset/emotic_face_bboxes_mtcnn.json \
    --bounding-box-body emotic_dataset/emotic_body_bboxes.json \
    --epochs 20 \
    --batch-size 16 \
    --workers 0 \
    --print-freq 1000 \
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
    --lambda_mi 0.0 \
    --lambda_dc 0.0 \
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
