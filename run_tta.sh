set -e
export OMP_NUM_THREADS=4

# Checkpoint path
checkpoint="outputs/RAER-ramp-down/model_best.pth"

echo "Running TTA with Confusion Bias on checkpoint: ${checkpoint}"

python eval_tta.py \
    --eval-checkpoint "${checkpoint}" \
    --gpu mps \
    --temporal-layers 1 \
    --batch-size 4 \
    --crop-body \
    --clip-path "ViT-B/16" \
    --confusion-bias 1.5 \
    --root-dir ./ \
    --drop_path_rate 0.0 \
    --train-annotation RAER/annotation/train.txt \
    --val-annotation RAER/annotation/test.txt \
    --test-annotation RAER/annotation/test.txt \
    --bounding-box-face RAER/bounding_box/face.json \
    --bounding-box-body RAER/bounding_box/body.json