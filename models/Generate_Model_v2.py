import torch
from torch import nn
from models.Temporal_Model import *
from models.Prompt_Learner import *
from models.Adapter import Adapter
from models.clip import clip
import copy
import itertools
import random

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, x_q, x_kv):
        """
        x_q: [B, N_q, D] - Query (e.g. Face)
        x_kv: [B, N_kv, D] - Key/Value (e.g. Body or Context)
        """
        b, n_q, _, h = *x_q.shape, self.heads
        _, n_kv, _ = x_kv.shape
        
        x_q = self.norm_q(x_q)
        x_kv = self.norm_kv(x_kv)
        
        q = self.to_q(x_q)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        
        kv = self.to_kv(x_kv).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), kv)
        
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Residual connection
        return self.to_out(out) + x_q


class GenerateModel_v2(nn.Module):
    def __init__(self, input_text, clip_model, args):
        super().__init__()
        self.args = args
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual
        
        self.is_ensemble = any(isinstance(i, list) for i in input_text)
        if self.is_ensemble:
            self.num_classes = len(input_text)
            self.num_prompts_per_class = len(input_text[0])
            self.input_text = list(itertools.chain.from_iterable(input_text))
            print(f"=> Using Prompt Ensembling with {self.num_prompts_per_class} prompts per class.")
        else:
            self.input_text = input_text
            self.num_classes = len(input_text)

        # Text Part
        self.prompt_learner = PromptLearner(self.input_text, clip_model, args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)

        if hasattr(args, 'freeze_image_encoder') and args.freeze_image_encoder:
            print("=> Freezing Image Encoder")
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        # Visual Part: Triple Stream Adapters & Temporal Nets
        self.face_adapter = Adapter(c_in=512, reduction=4)
        
        # We use AttnPool for all streams to get (B, 512) vectors
        self.temporal_net_face = Temporal_Transformer_AttnPool(num_patches=args.num_segments, input_dim=512, depth=args.temporal_layers, heads=8, mlp_dim=1024, dim_head=64)
        self.temporal_net_body = Temporal_Transformer_AttnPool(num_patches=args.num_segments, input_dim=512, depth=args.temporal_layers, heads=8, mlp_dim=1024, dim_head=64)
        self.temporal_net_context = Temporal_Transformer_AttnPool(num_patches=args.num_segments, input_dim=512, depth=args.temporal_layers, heads=8, mlp_dim=1024, dim_head=64)
        
        # Hierarchical Cross-Attention Modules
        # Level 1: Face queries Body -> Outputs Face-Body fused features
        self.cross_attn_fb = CrossAttention(dim=512, heads=8, dim_head=64)
        # Level 2: Face-Body queries Context -> Outputs final features
        self.cross_attn_fbc = CrossAttention(dim=512, heads=8, dim_head=64)

        # Project fused features to CLIP embedding space
        self.project_fc = nn.Linear(512, 512)
        
        self.modality_dropout_p = getattr(args, 'modality_dropout', 0.3)
        print(f"=> Building RAPT-CLIP v2 (Triple-Stream Cross-Attn + Modality Dropout p={self.modality_dropout_p})")

    def forward(self, image_face, image_body, image_context):
        # 1. Image Encoder
        n, t, c, h, w = image_face.shape
        
        # --- Face Stream ---
        # Apply Modality Dropout
        drop_face = self.training and random.random() < self.modality_dropout_p
        if drop_face:
            # Zero out face features if dropped
            video_face_features = torch.zeros(n, 512, device=image_face.device, dtype=self.dtype)
        else:
            face_reshaped = image_face.contiguous().view(-1, c, h, w)
            face_feat = self.image_encoder(face_reshaped.type(self.dtype))
            face_feat = self.face_adapter(face_feat)
            face_feat = face_feat.contiguous().view(n, t, -1)
            video_face_features = self.temporal_net_face(face_feat)
            
        # --- Body Stream ---
        drop_body = self.training and random.random() < self.modality_dropout_p
        if drop_body:
            video_body_features = torch.zeros(n, 512, device=image_body.device, dtype=self.dtype)
        else:
            body_reshaped = image_body.contiguous().view(-1, c, h, w)
            body_feat = self.image_encoder(body_reshaped.type(self.dtype))
            body_feat = body_feat.contiguous().view(n, t, -1)
            video_body_features = self.temporal_net_body(body_feat)
            
        # --- Context Stream --- (Never dropped, serves as anchor)
        context_reshaped = image_context.contiguous().view(-1, c, h, w)
        context_feat = self.image_encoder(context_reshaped.type(self.dtype))
        context_feat = context_feat.contiguous().view(n, t, -1)
        video_context_features = self.temporal_net_context(context_feat)
        
        # 2. Hierarchical Cross-Attention
        # Need shape (B, N, D) for attention. We have (B, D), so unsqueeze to (B, 1, D)
        vf = video_face_features.unsqueeze(1)
        vb = video_body_features.unsqueeze(1)
        vc = video_context_features.unsqueeze(1)
        
        # If face is dropped, use body as query. If both dropped, use context.
        if drop_face and not drop_body:
            fused_1 = vb
        elif not drop_face and drop_body:
            fused_1 = vf
        elif drop_face and drop_body:
            fused_1 = vc # Extreme case, just pass context
        else:
            # Face queries Body
            fused_1 = self.cross_attn_fb(vf, vb)
            
        # Level 2: Fused_1 queries Context
        fused_final = self.cross_attn_fbc(fused_1, vc)
        
        video_features = fused_final.squeeze(1) # (B, 512)
        video_features = self.project_fc(video_features)
        
        # Robust normalization
        video_features = video_features / (video_features.norm(dim=-1, keepdim=True) + 1e-6)

        # 3. Text Part
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        
        with torch.amp.autocast('cuda', enabled=False):
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features = text_features.float()
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)

        # 4. Classification
        if self.is_ensemble:
            text_features = text_features.view(self.num_classes, self.num_prompts_per_class, -1)
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
            logits = torch.einsum('bd,cpd->bcp', video_features, text_features)
            output = torch.mean(logits, dim=2) / self.args.temperature
        else:
            output = video_features @ text_features.t() / self.args.temperature

        # V2 doesn't use MoCo for simplicity in this baseline to isolate the gain from the new architecture
        moco_logits = None
        hand_crafted_text_features = None

        return output, text_features, hand_crafted_text_features, moco_logits
