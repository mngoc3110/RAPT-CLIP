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

    def __init__(self, dim=512, num_heads=4, dropout=0.1, use_context=False):
        super().__init__()
        self.use_context = use_context

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
            nn.init.xavier_uniform_(module.out_proj.weight)
            nn.init.constant_(module.out_proj.bias, 0.0)

    def forward(self, face_feat, body_feat, context_feat=None):
        """
        Args:
            face_feat: (B, dim) face features
            body_feat: (B, dim) body features
            context_feat: (B, dim) optional context features
        Returns:
            fused: concatenated cross-attended features
        """
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

            return torch.cat((face_out, body_out, context_out), dim=-1)
