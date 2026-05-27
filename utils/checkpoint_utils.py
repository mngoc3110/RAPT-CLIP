# utils/checkpoint_utils.py
"""
Checkpoint utilities for RAPT-CLIP.

Strategy:
  - model.pth       → Full checkpoint (state_dict + optimizer) for resuming training.
  - model_best.pth  → Slim checkpoint (no optimizer, no redundant CLIP text encoder)
                       Reduces file size from ~1.3GB → ~350MB.

What we EXCLUDE from slim checkpoint (safe — never fine-tuned):
  text_encoder.*   → CLIP text transformer (frozen, reloaded from official CLIP)
  clip_model_.*    → Redundant full-model copy (was stored by mistake in old arch)

What we INCLUDE in slim checkpoint (MUST save — fine-tuned during training):
  image_encoder.*  → ViT-B/16 visual encoder (fine-tuned with lr_image_encoder=1e-6)
  temporal_net.*   → Temporal transformer
  temporal_net_body.* → Temporal transformer (body stream)
  prompt_learner.* → Learnable CLIP prompts
  project_fc.*     → Fusion projection head
  face_adapter.*   → EAA adapter
  hand_crafted_prompt_embeddings → Buffer for MI loss

Size after fix:
  Full:  ~1.3 GB (state_dict 1069 MB + optimizer 699 MB)
  Slim:  ~350 MB (image_encoder 329 MB + fine-tuned layers 24 MB)
  Reduction: ~75% (and 100% accuracy preserved)
"""

import torch
import os

# Keys excluded from slim checkpoint: these are NEVER fine-tuned and can be
# reloaded from the original CLIP weights at inference time.
# NOTE: image_encoder.* is intentionally NOT here — it IS fine-tuned.
CLIP_BACKBONE_PREFIXES = (
    "text_encoder.",         # CLIP text transformer — frozen, reloaded from CLIP
    "clip_model_.",          # Legacy redundant full-model copy (removed in new arch)
    "image_encoder_m.",      # MoCo momentum encoder (derived from image_encoder)
    "face_adapter_m.",       # MoCo momentum adapter
    "temporal_net_m.",       # MoCo momentum temporal
    "temporal_net_body_m.",  # MoCo momentum temporal body
    "project_fc_m.",         # MoCo momentum project_fc
    "queue",                 # MoCo queue buffer (transient)
    "queue_ptr",             # MoCo queue pointer
)


def get_slim_state_dict(model: torch.nn.Module) -> dict:
    """
    Extract only the fine-tuned (non-CLIP-backbone) parameters from a model.
    Returns a dict suitable for a slim inference checkpoint.
    """
    slim_sd = {}
    for k, v in model.state_dict().items():
        if not any(k.startswith(prefix) for prefix in CLIP_BACKBONE_PREFIXES):
            slim_sd[k] = v
    return slim_sd


def save_slim_checkpoint(model: torch.nn.Module, path: str, meta: dict = None):
    """
    Save a slim checkpoint containing ONLY fine-tuned layers.
    Resulting file is typically ~20MB vs ~1.3GB for the full checkpoint.

    Args:
        model:  The model to save.
        path:   Output file path (e.g. outputs/exp/model_best.pth).
        meta:   Optional dict of extra metadata (epoch, best_acc, etc.).
    """
    slim_sd = get_slim_state_dict(model)
    state = {"state_dict": slim_sd}
    if meta:
        state.update(meta)
    torch.save(state, path)

    size_mb = os.path.getsize(path) / 1024**2
    saved_params = sum(v.numel() for v in slim_sd.values())
    print(f"[SlimCkpt] Saved slim checkpoint → {path}")
    print(f"           Size: {size_mb:.1f} MB | Params: {saved_params:,}")


def load_slim_checkpoint(model: torch.nn.Module, path: str, device: torch.device = None) -> dict:
    """
    Load a slim checkpoint into a fully-initialized model.
    The CLIP backbone weights (already loaded during model.__init__) are preserved.
    Only fine-tuned layers are overwritten by the slim checkpoint.

    Args:
        model:   A fully-initialized GenerateModel (with CLIP backbone already loaded).
        path:    Path to the slim checkpoint file.
        device:  Device to load tensors onto.

    Returns:
        The checkpoint dict (for accessing meta like epoch, best_acc, etc.).
    """
    if device is None:
        device = next(model.parameters()).device

    ckpt = torch.load(path, map_location=device, weights_only=False)
    slim_sd = ckpt["state_dict"]

    # strict=False: CLIP backbone keys are missing from slim_sd — that's intentional.
    msg = model.load_state_dict(slim_sd, strict=False)

    missing = [k for k in msg.missing_keys if not any(k.startswith(p) for p in CLIP_BACKBONE_PREFIXES)]
    unexpected = msg.unexpected_keys

    if missing:
        print(f"[SlimCkpt] WARNING: Truly missing keys (not CLIP backbone): {missing}")
    if unexpected:
        print(f"[SlimCkpt] WARNING: Unexpected keys in checkpoint: {unexpected}")

    size_mb = os.path.getsize(path) / 1024**2
    print(f"[SlimCkpt] Loaded slim checkpoint ← {path} ({size_mb:.1f} MB)")
    return ckpt
