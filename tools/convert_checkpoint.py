#!/usr/bin/env python3
"""
tools/convert_checkpoint.py
────────────────────────────
Convert existing RAPT-CLIP checkpoints from full (~1.3GB) → slim (~20MB).

Usage examples:
  # Convert a single checkpoint
  python tools/convert_checkpoint.py \
      --input outputs/RAER-ramp-up/model_best.pth \
      --output outputs/RAER-ramp-up/model_best_slim.pth

  # Convert ALL model_best.pth files under outputs/ in one pass
  python tools/convert_checkpoint.py --convert-all

The slim checkpoint contains only fine-tuned layers:
  temporal_net, temporal_net_body, prompt_learner, project_fc, face_adapter,
  hand_crafted_prompt_embeddings.

CLIP backbone layers (image_encoder, text_encoder, clip_model_) are excluded
because they can always be reloaded from the original CLIP weights at inference.

Accuracy is 100% preserved — only the file format changes.
"""

import os
import sys
import argparse
import glob
import torch

# ── Make sure project root is on sys.path ────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.checkpoint_utils import CLIP_BACKBONE_PREFIXES


def convert_checkpoint(input_path: str, output_path: str) -> None:
    """Strip CLIP backbone + optimizer from a checkpoint, save slim version."""
    print(f"\n[Convert] Reading  : {input_path}")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    # ── Analyse original size ─────────────────────────────────────────────
    orig_size_mb = os.path.getsize(input_path) / 1024**2

    sd = ckpt.get("state_dict", ckpt)  # handle both formats
    total_params = sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))

    # ── Filter: keep only non-CLIP-backbone keys ──────────────────────────
    slim_sd = {
        k: v for k, v in sd.items()
        if not any(k.startswith(p) for p in CLIP_BACKBONE_PREFIXES)
    }
    slim_params = sum(v.numel() for v in slim_sd.values() if isinstance(v, torch.Tensor))

    # Print what we're removing
    removed_keys = [k for k in sd if k not in slim_sd]
    removed_mb = sum(
        v.numel() * v.element_size()
        for k, v in sd.items()
        if k not in slim_sd and isinstance(v, torch.Tensor)
    ) / 1024**2
    print(f"           Removing {len(removed_keys)} CLIP-backbone tensors ({removed_mb:.1f} MB):")
    for k in removed_keys[:10]:
        print(f"             - {k}")
    if len(removed_keys) > 10:
        print(f"             ... and {len(removed_keys) - 10} more")

    # ── Build slim checkpoint ─────────────────────────────────────────────
    slim_ckpt = {
        "state_dict": slim_sd,
        "epoch":      ckpt.get("epoch", 0),
        "best_acc":   ckpt.get("best_acc", 0.0),
        # Intentionally NOT saving optimizer state in slim checkpoint
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(slim_ckpt, output_path)

    slim_size_mb = os.path.getsize(output_path) / 1024**2
    reduction_pct = (1 - slim_size_mb / orig_size_mb) * 100
    print(f"[Convert] Saved to : {output_path}")
    print(f"           {orig_size_mb:.0f} MB  →  {slim_size_mb:.1f} MB  "
          f"({reduction_pct:.0f}% reduction)")
    print(f"           Params: {total_params:,} → {slim_params:,} (fine-tuned only)")


def convert_all(outputs_dir: str) -> None:
    """Find and convert all model_best.pth files under outputs_dir."""
    pattern = os.path.join(outputs_dir, "**", "model_best.pth")
    paths = glob.glob(pattern, recursive=True)

    # Skip already-slim files (< 50 MB)
    full_paths = [p for p in paths if os.path.getsize(p) / 1024**2 > 50]
    if not full_paths:
        print("No large checkpoints found (all may already be slim).")
        return

    print(f"Found {len(full_paths)} full checkpoint(s) to convert:\n")
    for p in full_paths:
        mb = os.path.getsize(p) / 1024**2
        print(f"  {mb:.0f} MB  {p}")

    print()
    for p in full_paths:
        # Save slim version alongside original (keep original as backup)
        slim_path = p.replace("model_best.pth", "model_best_slim.pth")
        convert_checkpoint(p, slim_path)

    print("\n✅  All checkpoints converted.")
    print("   Original files are kept as backup (model_best.pth).")
    print("   Use model_best_slim.pth for inference / deployment.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert RAPT-CLIP full checkpoints to slim inference checkpoints."
    )
    parser.add_argument("--input",  type=str, help="Path to the input checkpoint (.pth).")
    parser.add_argument("--output", type=str, help="Path to save the slim checkpoint.")
    parser.add_argument(
        "--convert-all", action="store_true",
        help="Convert ALL model_best.pth files under ./outputs/ in one pass."
    )
    parser.add_argument(
        "--outputs-dir", type=str, default="outputs",
        help="Root directory to scan when using --convert-all (default: ./outputs)."
    )
    args = parser.parse_args()

    if args.convert_all:
        outputs_dir = os.path.join(ROOT, args.outputs_dir)
        convert_all(outputs_dir)
    elif args.input and args.output:
        convert_checkpoint(args.input, args.output)
    else:
        parser.error("Provide either --input + --output, or --convert-all.")


if __name__ == "__main__":
    main()
