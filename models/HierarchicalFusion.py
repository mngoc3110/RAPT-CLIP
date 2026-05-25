# HierarchicalFusion.py
# ============================================================================
# Hierarchical Cross-Attention Fusion for RAPT-CLIP v4
# ============================================================================
# Replaces naive concat+Linear with a 2-stage hierarchical cross-attention:
#
#   Stage 1 — Local Interaction (Face ↔ Body):
#     Face queries Body: "When face looks tired, does body slouch?"
#     Body queries Face: "When body leans forward, is face engaged?"
#     → face_refined, body_refined
#
#   Stage 2 — Global Grounding (Face+Body → Context):
#     Fused(face+body) queries Context: "Does the classroom scene match?"
#     → final_fused (512-dim)
#
# References:
#   - CAER-Net (Lee et al., ICCV 2019): face + context dual-stream with
#     attention mechanism. arXiv:1908.05913
#   - JMT (Gnana Praveen et al., 2024): joint multimodal transformer with
#     key-based cross-attention. arXiv:2403.10488
#   - EmoVCLIP (Sun et al., 2024): modality dropout for robust fusion.
#     arXiv:2409.07078
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention block: query attends to key/value.

    Args:
        d_model (int): Feature dimension.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model: int = 512, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query:     (B, D) — the querying modality
            key_value: (B, D) — the modality being attended to
        Returns:
            out: (B, D) — refined query
        """
        # Unsqueeze to (B, 1, D) for MHA
        q = self.norm_q(query).unsqueeze(1)
        kv = self.norm_kv(key_value).unsqueeze(1)

        attended, _ = self.cross_attn(q, kv, kv)  # (B, 1, D)
        attended = attended.squeeze(1)             # (B, D)

        # Residual + FFN
        out = query + attended
        out = out + self.ffn(self.norm_ffn(out))
        return out


class HierarchicalCrossAttentionFusion(nn.Module):
    """
    2-stage hierarchical cross-attention fusion for 3 streams.

    Stage 1 — Local Face↔Body interaction:
        face_refined  = CrossAttn(Q=face,  KV=body)
        body_refined  = CrossAttn(Q=body,  KV=face)

    Stage 2 — Global grounding with context:
        fb_fused = LayerNorm(face_refined + body_refined)
        final    = CrossAttn(Q=fb_fused, KV=context)
        output   = project(final)  →  (B, 512)

    Modality Dropout (training only):
        With probability p_drop, randomly zero out one of the 3 streams.
        Forces the model to be robust when a stream is missing (e.g., face
        occluded, body out of frame). Inspired by EmoVCLIP (arXiv:2409.07078).

    Args:
        d_model (int): Feature dimension (512 for CLIP ViT-B/16).
        n_heads (int): Attention heads per block.
        dropout (float): Dropout inside attention/FFN.
        modality_dropout_p (float): Probability of dropping one stream during
            training. 0.0 disables modality dropout.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 4,
        dropout: float = 0.1,
        modality_dropout_p: float = 0.15,
    ):
        super().__init__()
        self.d_model = d_model
        self.modality_dropout_p = modality_dropout_p

        # Stage 1: Face ↔ Body cross-attention
        self.face_queries_body = CrossAttentionBlock(d_model, n_heads, dropout)
        self.body_queries_face = CrossAttentionBlock(d_model, n_heads, dropout)

        # Stage 1 fusion: combine refined face + body
        self.stage1_norm = nn.LayerNorm(d_model)
        self.stage1_proj = nn.Linear(d_model * 2, d_model)

        # Stage 2: (Face+Body) queries Context
        self.fb_queries_context = CrossAttentionBlock(d_model, n_heads, dropout)

        # Final projection + norm
        self.final_norm = nn.LayerNorm(d_model)
        self.final_proj = nn.Linear(d_model, d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.stage1_proj.weight)
        nn.init.constant_(self.stage1_proj.bias, 0)
        nn.init.xavier_uniform_(self.final_proj.weight)
        nn.init.constant_(self.final_proj.bias, 0)

    def _modality_dropout(
        self,
        face: torch.Tensor,
        body: torch.Tensor,
        context: torch.Tensor,
    ):
        """
        Randomly zero out one stream per sample during training.
        Each sample independently decides which stream (if any) to drop.
        """
        if not self.training or self.modality_dropout_p <= 0:
            return face, body, context

        B = face.shape[0]
        # For each sample, draw a random number:
        #   [0, p/3)       → drop face
        #   [p/3, 2p/3)    → drop body
        #   [2p/3, p)      → drop context
        #   [p, 1)         → keep all
        r = torch.rand(B, device=face.device)
        p = self.modality_dropout_p
        p3 = p / 3.0

        face_mask    = (r >= p3).float().unsqueeze(1)       # 1 = keep
        body_mask    = ((r < p3) | (r >= 2 * p3)).float().unsqueeze(1)
        context_mask = (r < 2 * p3).float().unsqueeze(1)

        return face * face_mask, body * body_mask, context * context_mask

    def forward(
        self,
        face: torch.Tensor,
        body: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            face:    (B, 512) — face stream features
            body:    (B, 512) — body stream features
            context: (B, 512) — full-frame context features
        Returns:
            fused: (B, 512) — hierarchically fused representation
        """
        # ── Modality Dropout ──────────────────────────────────────────
        face, body, context = self._modality_dropout(face, body, context)

        # ── Stage 1: Face ↔ Body local interaction ────────────────────
        face_refined = self.face_queries_body(face, body)    # face learns from body
        body_refined = self.body_queries_face(body, face)    # body learns from face

        # Combine: concat → project → (B, 512)
        fb_cat = torch.cat([face_refined, body_refined], dim=-1)  # (B, 1024)
        fb_fused = self.stage1_proj(fb_cat)                        # (B, 512)
        fb_fused = self.stage1_norm(fb_fused)

        # ── Stage 2: (Face+Body) queries Context ──────────────────────
        final = self.fb_queries_context(fb_fused, context)   # (B, 512)

        # ── Final projection ──────────────────────────────────────────
        out = self.final_proj(self.final_norm(final))
        return out
