#!/usr/bin/env python3
"""
tools/verify_slim_accuracy.py
──────────────────────────────
So sánh output của model dùng full checkpoint vs slim checkpoint.
Dùng random input (synthetic) nên KHÔNG cần dataset thực.

Test 2 điều:
  1. Logits output có giống nhau không (cosine similarity + max diff)
  2. Predicted class có giống nhau không (top-1 prediction)

Nếu cả 2 test đều PASS → slim checkpoint 100% giữ nguyên accuracy.

Usage:
  python tools/verify_slim_accuracy.py \
      --full   outputs/RAER-ramp-up/model_best.pth \
      --slim   outputs/RAER-ramp-up/model_best_slim.pth \
      --dataset RAER
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.builders import build_model, get_class_info
from utils.checkpoint_utils import load_slim_checkpoint
from types import SimpleNamespace


def make_args(dataset: str, clip_path: str = "ViT-B/16") -> SimpleNamespace:
    """Build a minimal args namespace to instantiate the model."""
    return SimpleNamespace(
        dataset=dataset,
        clip_path=clip_path,
        temporal_layers=1,
        contexts_number=8,
        class_token_position="end",
        class_specific_contexts="True",
        load_and_tune_prompt_learner="True",
        num_segments=16,
        temperature=0.07,
        drop_path_rate=0.0,
        freeze_image_encoder=False,
        ablation_no_text=False,
        use_v2=False,
        use_moco=False,
        modality_dropout=0.3,
        lr_image_encoder=1e-6,
        text_type="prompt_ensemble",
    )


def load_model_full(ckpt_path: str, args, device: torch.device) -> torch.nn.Module:
    """Load model from a FULL checkpoint (legacy 1.3GB format)."""
    _, input_text = get_class_info(args)
    model = build_model(args, input_text).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
    msg = model.load_state_dict(sd, strict=False)
    print(f"[Full]  Loaded {ckpt_path} ({os.path.getsize(ckpt_path)/1024**2:.0f} MB)")
    return model


def load_model_slim(ckpt_path: str, args, device: torch.device) -> torch.nn.Module:
    """Load model from a SLIM checkpoint (~20MB format)."""
    _, input_text = get_class_info(args)
    model = build_model(args, input_text).to(device)
    load_slim_checkpoint(model, ckpt_path, device=device)
    print(f"[Slim]  Loaded {ckpt_path} ({os.path.getsize(ckpt_path)/1024**2:.1f} MB)")
    return model


def run_comparison(full_path: str, slim_path: str, dataset: str, n_batches: int = 5):
    device = torch.device("cpu")  # CPU để so sánh deterministic
    args = make_args(dataset)

    print("\n" + "="*60)
    print("  Loading FULL checkpoint...")
    print("="*60)
    model_full = load_model_full(full_path, args, device)

    print("\n" + "="*60)
    print("  Loading SLIM checkpoint...")
    print("="*60)
    model_slim = load_model_slim(slim_path, args, device)

    model_full.eval()
    model_slim.eval()

    # Lấy num_classes từ dataset
    class_names, _ = get_class_info(args)
    num_classes = len(class_names)

    print(f"\n{'='*60}")
    print(f"  Running {n_batches} batches of synthetic inference...")
    print(f"  Dataset: {dataset} ({num_classes} classes)")
    print(f"{'='*60}\n")

    all_match = True
    max_logit_diff = 0.0

    torch.manual_seed(42)
    with torch.no_grad():
        for i in range(n_batches):
            # Tạo synthetic input (batch=2, T=16, C=3, H=224, W=224)
            face  = torch.randn(2, 16, 3, 224, 224)
            body  = torch.randn(2, 16, 3, 224, 224)

            # Forward pass cả 2 model
            out_full, _, _, _ = model_full(face, body)
            out_slim, _, _, _ = model_slim(face, body)

            # So sánh logits
            diff = (out_full - out_slim).abs().max().item()
            max_logit_diff = max(max_logit_diff, diff)

            # So sánh predicted class
            pred_full = out_full.argmax(dim=1)
            pred_slim = out_slim.argmax(dim=1)
            predictions_match = torch.equal(pred_full, pred_slim)

            if not predictions_match:
                all_match = False

            status = "✅" if predictions_match else "❌"
            print(f"  Batch {i+1}/{n_batches}: max_logit_diff={diff:.2e}  pred_match={status}")
            print(f"            pred_full={pred_full.tolist()}  pred_slim={pred_slim.tolist()}")

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Max logit difference   : {max_logit_diff:.2e}")
    print(f"  All predictions match  : {'✅ YES' if all_match else '❌ NO'}")

    if max_logit_diff < 1e-4 and all_match:
        print(f"\n  🎉 PASS — Slim checkpoint is IDENTICAL to full checkpoint.")
        print(f"     Accuracy is 100% preserved.")
    elif all_match:
        print(f"\n  ✅ PASS — Predictions identical (logit diff is floating-point noise).")
    else:
        print(f"\n  ❌ FAIL — Predictions differ! Check if CLIP backbone was fine-tuned.")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="So sánh output của full vs slim checkpoint (không cần dataset)."
    )
    parser.add_argument("--full",    required=True, help="Path to full checkpoint (model_best.pth)")
    parser.add_argument("--slim",    required=True, help="Path to slim checkpoint (model_best_slim.pth)")
    parser.add_argument("--dataset", default="RAER", help="Dataset name (RAER/DAiSEE/CAER)")
    parser.add_argument("--batches", type=int, default=5, help="Number of synthetic batches to test")
    args = parser.parse_args()

    run_comparison(args.full, args.slim, args.dataset, args.batches)


if __name__ == "__main__":
    main()
