import torch
import torch.nn as nn


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

    Args:
        dim (int): Feature dimension of each modality. Default: 512.
        num_heads (int): Number of attention heads. Default: 4.
        dropout (float): Dropout rate for attention weights. Default: 0.1.
    """

    def __init__(self, dim=512, num_heads=4, dropout=0.1):
        super().__init__()

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

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for cross-attention projections."""
        for module in [self.cross_attn_f2b, self.cross_attn_b2f]:
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.constant_(module.in_proj_bias, 0.0)
            nn.init.xavier_uniform_(module.out_proj.weight)
            nn.init.constant_(module.out_proj.bias, 0.0)

    def forward(self, face_feat, body_feat):
        """
        Args:
            face_feat: (B, dim) face features from temporal transformer
            body_feat: (B, dim) body features from temporal transformer

        Returns:
            fused: (B, 2*dim) concatenated cross-attended features
        """
        # Reshape to sequence format: (B, 1, dim)
        face_q = face_feat.unsqueeze(1)
        body_kv = body_feat.unsqueeze(1)

        # Face attends to Body (what body cues inform this face expression?)
        face_cross, _ = self.cross_attn_f2b(face_q, body_kv, body_kv)
        face_out = self.norm_f(face_cross.squeeze(1) + face_feat)  # residual

        # Body attends to Face (what face cues inform this body posture?)
        body_q = body_feat.unsqueeze(1)
        face_kv = face_feat.unsqueeze(1)
        body_cross, _ = self.cross_attn_b2f(body_q, face_kv, face_kv)
        body_out = self.norm_b(body_cross.squeeze(1) + body_feat)  # residual

        # Concatenate → 1024-d (project_fc maps this to 512)
        return torch.cat((face_out, body_out), dim=-1)
