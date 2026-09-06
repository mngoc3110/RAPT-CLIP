import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttentionFusion(nn.Module):
    """
    Bidirectional Cross-Modal Attention Fusion (CMAF).

    Enables face and body modalities to attend to each other before fusion,
    allowing the model to learn which aspects of one modality are most
    relevant given the other.

    Architecture:
        Face(Q) × Body(K,V) → Cross-Attention → face_out  (face informed by body)
        Body(Q) × Face(K,V) → Cross-Attention → body_out  (body informed by face)
        Output = concat(face_out, body_out) → 1024-d

    Uses residual connections and LayerNorm for stable training.
    
    When context_gating=True (for EMOTIC), adds learnable modality importance
    weights that the model can learn to prioritize context-heavy information.

    Args:
        dim (int): Feature dimension of each modality. Default: 512.
        num_heads (int): Number of attention heads. Default: 4.
        dropout (float): Dropout rate for attention weights. Default: 0.1.
        use_context (bool): Enable 3-stream (face, body, context) fusion.
        context_gating (bool): Enable learnable modality importance gating.
    """

    def __init__(self, dim=512, num_heads=4, dropout=0.1, use_context=False, context_gating=False):
        super().__init__()
        self.use_context = use_context
        self.context_gating = context_gating

        if not self.use_context:
            # Cross-attention: Face queries, Body keys/values
            self.cross_attn_f2b = nn.MultiheadAttention(
                embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.norm_f = nn.LayerNorm(dim)

            # Cross-attention: Body queries, Face keys/values
            self.cross_attn_b2f = nn.MultiheadAttention(
                embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.norm_b = nn.LayerNorm(dim)
        else:
            # Tridirectional Cross-attention for 3 streams (Face, Body, Context)
            self.cross_attn_f2bc = nn.MultiheadAttention(
                embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.norm_f = nn.LayerNorm(dim)

            self.cross_attn_b2fc = nn.MultiheadAttention(
                embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.norm_b = nn.LayerNorm(dim)

            self.cross_attn_c2fb = nn.MultiheadAttention(
                embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.norm_c = nn.LayerNorm(dim)
        
        # ========== Context-Priority Gating ==========
        if self.context_gating and self.use_context:
            # Learnable modality importance: [face, body, context] → softmax → weights
            # EMOTIC-specific: face often invisible (distant/back shots) → low face prior
            # softmax([-0.5, 0.5, 1.0]) → [face≈12%, body≈33%, context≈55%]
            # (was [0.2, 0.3, 0.5] → [face=29%, body=32%, context=39%] — too much face!)
            self.modality_importance = nn.Parameter(torch.tensor([-0.5, 0.5, 1.0]))
            
            # Input-dependent gating network (adaptive per sample)
            # Takes concatenated features (3*dim) → 3 weights
            self.adaptive_gate = nn.Sequential(
                nn.Linear(dim * 3, dim),
                nn.GELU(),
                nn.Linear(dim, 3)
            )
            
            # Projection to match expected output dim (1536 for 3-stream concat)
            self.gate_proj = nn.Linear(dim, dim * 3)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for cross-attention projections."""
        modules = []
        if not self.use_context:
            modules = [self.cross_attn_f2b, self.cross_attn_b2f]
        else:
            modules = [self.cross_attn_f2bc, self.cross_attn_b2fc, self.cross_attn_c2fb]
            
        for module in modules:
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.constant_(module.in_proj_bias, 0.0)
            nn.init.constant_(module.out_proj.weight, 0.0) # Zero init to preserve CLIP features initially
            nn.init.constant_(module.out_proj.bias, 0.0)

    def get_modality_weights(self):
        """Get current learned modality importance weights (for logging/monitoring)."""
        if self.context_gating and self.use_context:
            return F.softmax(self.modality_importance, dim=0).detach().cpu().tolist()
        return None

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, face_feat, body_feat, context_feat=None):
        """
        Forward pass for CMAF.
        Args:
            face_feat: (B, dim) face features
            body_feat: (B, dim) body features
            context_feat: (B, dim) optional context features
        Returns:
            fused: concatenated cross-attended features
                   - 2-stream: (B, 2*dim) = (B, 1024)
                   - 3-stream: (B, 3*dim) = (B, 1536)
                   - 3-stream + context_gating: (B, 3*dim) = (B, 1536)
        """
        face_feat = face_feat.float()
        body_feat = body_feat.float()
        if context_feat is not None:
            context_feat = context_feat.float()
            
        if not self.use_context:
            # Reshape to sequence format: (B, 1, dim)
            face_q = face_feat.unsqueeze(1)
            body_kv = body_feat.unsqueeze(1)

            # Face attends to Body
            face_cross, _ = self.cross_attn_f2b(face_q, body_kv, body_kv)
            face_out = self.norm_f(face_cross.squeeze(1) + face_feat)  # residual

            # Body attends to Face
            body_q = body_feat.unsqueeze(1)
            face_kv = face_feat.unsqueeze(1)
            body_cross, _ = self.cross_attn_b2f(body_q, face_kv, face_kv)
            body_out = self.norm_b(body_cross.squeeze(1) + body_feat)  # residual

            return torch.cat((face_out, body_out), dim=-1)
        else:
            assert context_feat is not None, "context_feat must be provided when use_context=True"

            # Face queries (Body + Context)
            face_q = face_feat.unsqueeze(1)
            body_context_kv = torch.cat((body_feat.unsqueeze(1), context_feat.unsqueeze(1)), dim=1)
            face_cross, _ = self.cross_attn_f2bc(face_q, body_context_kv, body_context_kv)
            face_out = self.norm_f(face_cross.squeeze(1) + face_feat)

            # Body queries (Face + Context)
            body_q = body_feat.unsqueeze(1)
            face_context_kv = torch.cat((face_feat.unsqueeze(1), context_feat.unsqueeze(1)), dim=1)
            body_cross, _ = self.cross_attn_b2fc(body_q, face_context_kv, face_context_kv)
            body_out = self.norm_b(body_cross.squeeze(1) + body_feat)

            # Context queries (Face + Body)
            context_q = context_feat.unsqueeze(1)
            face_body_kv = torch.cat((face_feat.unsqueeze(1), body_feat.unsqueeze(1)), dim=1)
            context_cross, _ = self.cross_attn_c2fb(context_q, face_body_kv, face_body_kv)
            context_out = self.norm_c(context_cross.squeeze(1) + context_feat)

            if self.context_gating:
                # ===== Numerically-Stable Context-Priority Gating =====
                static_weights = F.softmax(self.modality_importance, dim=0)  # (3,)
                
                # Adaptive gate: depends on the actual features in this batch
                concat_for_gate = torch.cat([face_out, body_out, context_out], dim=-1)  # (B, 3*dim)
                adaptive_weights = F.softmax(self.adaptive_gate(concat_for_gate), dim=-1)  # (B, 3)
                
                # Mix static prior (0.7) + adaptive (0.3)
                combined_weights = 0.7 * static_weights.unsqueeze(0) + 0.3 * adaptive_weights  # (B, 3)
                
                # Weighted average of modalities → (B, dim)
                # Then project to (B, 3*dim) via gate_proj to preserve output dimension
                # This keeps feature magnitude stable (sum of weights=1) unlike scale*3.0
                w_f = combined_weights[:, 0:1]  # (B, 1)
                w_b = combined_weights[:, 1:2]
                w_c = combined_weights[:, 2:3]
                weighted_avg = w_f * face_out + w_b * body_out + w_c * context_out  # (B, dim)
                projected = self.gate_proj(weighted_avg)  # (B, 3*dim)
                
                # Residual: preserve per-stream features + apply gated correction
                concat_plain = torch.cat((face_out, body_out, context_out), dim=-1)  # (B, 1536)
                return concat_plain + 0.1 * projected  # small residual gate
            else:
                return torch.cat((face_out, body_out, context_out), dim=-1)

