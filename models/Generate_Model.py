from torch import nn
from models.Temporal_Model import *
from models.Prompt_Learner import *
from models.Text import class_descriptor_5_only_face
from models.Adapter import Adapter
from models.CrossModalAttentionFusion import CrossModalAttentionFusion
from models.clip import clip
import copy
import itertools
import numpy as np
import torch
import torch.nn.functional as F


class Q2LabelHead(nn.Module):
    """
    Query2Label (Q2L) Multi-Label Classification Head.
    
    Uses learnable label queries that attend to visual features via cross-attention.
    Label queries are initialized from CLIP text features to leverage pretrained 
    semantic knowledge about each emotion class.
    
    Reference: Q2L (CVPR 2021) — adapted for CLIP-based dual-stream architecture.
    """
    
    def __init__(self, num_classes, dim=512, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        
        # Learnable label queries — will be initialized from CLIP text features
        self.label_query = nn.Parameter(torch.randn(num_classes, dim))
        
        # Cross-attention layers: label queries attend to visual features
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.label_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final classification: each label query → 1 logit
        self.fc_out = nn.Linear(dim, 1)
        
        # Layer norm for visual features before attention
        self.norm_vis = nn.LayerNorm(dim)
        
    def init_from_text_features(self, text_features):
        """Initialize label queries from CLIP text features (26, 512)."""
        with torch.no_grad():
            self.label_query.data.copy_(text_features.float())
        print(f"=> Q2L: Initialized {self.num_classes} label queries from CLIP text features")
    
    def forward(self, visual_features):
        """
        Args:
            visual_features: (B, D) pooled visual features from temporal transformer
        Returns:
            logits: (B, num_classes) raw logits for sigmoid
        """
        B = visual_features.shape[0]
        
        # Prepare visual features as memory: (B, 1, D)
        memory = self.norm_vis(visual_features).unsqueeze(1)  # (B, 1, D)
        
        # Expand label queries for batch: (B, num_classes, D)
        queries = self.label_query.unsqueeze(0).expand(B, -1, -1)
        
        # Cross-attention: label queries attend to visual features
        # tgt=queries, memory=visual_features
        label_features = self.label_decoder(queries, memory)  # (B, num_classes, D)
        
        # Project each label feature to a single logit
        logits = self.fc_out(label_features).squeeze(-1)  # (B, num_classes)
        
        return logits


class GenerateModel(nn.Module):
    def __init__(self, input_text, clip_model, args):
        super().__init__()
        self.args = args
        self.is_multilabel = (args.dataset == "EMOTIC")
        
        self.is_ensemble = any(isinstance(i, list) for i in input_text)
        
        if self.is_ensemble:
            self.num_classes = len(input_text)
            self.num_prompts_per_class = len(input_text[0])
            self.input_text = list(itertools.chain.from_iterable(input_text))
            print(f"=> Using Prompt Ensembling with {self.num_prompts_per_class} prompts per class.")
        else:
            self.input_text = input_text

        self.prompt_learner = PromptLearner(self.input_text, clip_model, args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual

        # Freeze Image Encoder if requested
        if hasattr(args, 'freeze_image_encoder') and args.freeze_image_encoder:
            print("=> Freezing Image Encoder")
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        # For EAA
        self.face_adapter = Adapter(c_in=512, reduction=4)

        # For MI Loss
        if args.dataset == "RAER":
            hand_crafted_prompts = class_descriptor_5_only_face
        elif args.dataset == "CK+":
            from models.Text import class_descriptor_ckplus
            hand_crafted_prompts = class_descriptor_ckplus
        elif args.dataset == "DAiSEE":
            from models.Text import class_descriptor_daisee
            hand_crafted_prompts = class_descriptor_daisee
        elif args.dataset == "EMOTIC":
            # Use the first prompt of each class as the hand_crafted prompt for MI Loss
            hand_crafted_prompts = [prompts[0] if isinstance(prompts, list) else prompts for prompts in input_text]
        else:
            # Fallback to some generic or 7-class descriptors if available
            from models.Text import class_descriptor_7_only_face
            hand_crafted_prompts = class_descriptor_7_only_face
            
        self.tokenized_hand_crafted_prompts = torch.cat([clip.tokenize(p) for p in hand_crafted_prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(self.tokenized_hand_crafted_prompts.to(clip_model.token_embedding.weight.device)).type(self.dtype)
        self.register_buffer("hand_crafted_prompt_embeddings", embedding)


        # Context Stream Configuration
        self.use_context = getattr(args, 'use_context', False)
        print(f"=> Using Context Stream: {self.use_context}")

        in_dim = 1536 if self.use_context else 1024

        self.unified_temporal_net = Temporal_Transformer_AttnPool(num_patches=16,
                                                     input_dim=in_dim,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)

        # Store clip_model_ as a plain Python attribute (NOT an nn.Module submodule).
        self.project_fc = nn.Linear(in_dim, 512)
        # Initialize project_fc as an average projection of input streams to preserve CLIP visual features from step 0
        with torch.no_grad():
            num_streams = 3 if self.use_context else 2
            eye_blocks = [torch.eye(512) / float(num_streams) for _ in range(num_streams)]
            self.project_fc.weight.copy_(torch.cat(eye_blocks, dim=1))
            if self.project_fc.bias is not None:
                self.project_fc.bias.zero_()

        # Fusion Selection: gfi (Gated Feature Integration) or cmaf (Cross-Modal Attention Fusion)
        self.fusion_type = getattr(args, 'fusion_type', 'cmaf')
        print(f"=> Using Fusion Type: {self.fusion_type}")
        
        if self.fusion_type == 'gfi':
            self.gate_fc = nn.Sequential(
                nn.Linear(in_dim, in_dim // 4),
                nn.ReLU(),
                nn.Linear(in_dim // 4, in_dim),
                nn.Sigmoid()
            )
        else:
            self.cmaf = CrossModalAttentionFusion(dim=512, num_heads=4, dropout=0.1, use_context=self.use_context, context_gating=self.is_multilabel)

        # ==================== Multi-Label Class Bias (EMOTIC) ====================
        # FIXED (non-learnable) log-odds prior bias per class, computed from dataset statistics.
        # Making this learnable collapses training to predict class frequency priors.
        # Formula: log(p / (1-p)) where p = class frequency. Initialized from EMOTIC class frequencies.
        # This can be overridden via model.set_class_prior(freq_vector) after model construction.
        if self.is_multilabel:
            num_cls = self.num_classes if hasattr(self, 'num_classes') else len(input_text)
            # Default: uniform prior matching typical multi-label sparsity (p=0.1 → logit=-2.2)
            prior_logit = torch.full((num_cls,), -2.2)
            # Make it a learnable parameter again. ASL loss creates a bias offset that is DIFFERENT
            # from the dataset prior. If class_bias is fixed, the text embeddings are forced
            # to learn this bias offset, destroying their semantic meaning (catastrophic forgetting).
            self.class_bias = nn.Parameter(prior_logit)
            print(f"=> Initialized FIXED class bias (log-odds prior, p=0.1) for {num_cls} multi-label classes")

        # ==================== Q2L Multi-Label Head (EMOTIC) ====================
        if self.is_multilabel:
            print("=> EMOTIC Mode: Initializing Q2L Multi-Label Classification Head")
            self.q2l_head = Q2LabelHead(
                num_classes=self.num_classes if hasattr(self, 'num_classes') else len(input_text),
                dim=512,
                num_heads=8,
                num_layers=2,
                dropout=0.1
            )
            # Initialize label queries from CLIP text features (done after model is on device)
            self._q2l_initialized = False
        
        # ==================== Co-occurrence Matrix (EMOTIC) ====================
        if self.is_multilabel:
            # Will be set externally from training data statistics
            nc = self.num_classes if hasattr(self, 'num_classes') else len(input_text)
            self.register_buffer('label_cooccurrence', torch.zeros(nc, nc))

        # ==================== MoCo Initialization ====================
        if hasattr(args, 'use_moco') and args.use_moco:
            print("=> Initializing MoCoRank...")
            self._init_moco(args)

    def set_class_prior(self, freq_vector: torch.Tensor):
        """Update the fixed class_bias buffer from actual training data class frequencies.
        
        Call this once after model construction, before training.
        freq_vector: (C,) float tensor with values in [0, 1] = class frequency p_c in training data.
        bias = log(p / (1-p)), clipped to avoid ±inf.
        """
        if hasattr(self, 'class_bias'):
            freq = freq_vector.clamp(0.005, 0.995)
            log_odds = torch.log(freq / (1.0 - freq))
            self.class_bias.data.copy_(log_odds)
            print(f"=> Updated class_bias (learnable log-odds prior) from training data: min={log_odds.min():.2f}, max={log_odds.max():.2f}")

    def _init_moco(self, args):
        """Initialize MoCo momentum encoders. Called from __init__ when use_moco=True."""
        self.moco_dim = 512
        self.moco_k = args.moco_k
        self.moco_m = args.moco_m
        self.moco_t = args.moco_t

        # Create momentum encoders
        self.image_encoder_m = copy.deepcopy(self.image_encoder)
        self.face_adapter_m = copy.deepcopy(self.face_adapter)
        self.unified_temporal_net_m = copy.deepcopy(self.unified_temporal_net)
        self.project_fc_m = copy.deepcopy(self.project_fc)
        
        if self.fusion_type == 'gfi':
            self.gate_fc_m = copy.deepcopy(self.gate_fc)
        else:
            self.cmaf_m = copy.deepcopy(self.cmaf)

        # Freeze momentum encoders
        for param in self.image_encoder_m.parameters(): param.requires_grad = False
        for param in self.face_adapter_m.parameters(): param.requires_grad = False
        for param in self.unified_temporal_net_m.parameters(): param.requires_grad = False
        for param in self.project_fc_m.parameters(): param.requires_grad = False
        
        if self.fusion_type == 'gfi':
            for param in self.gate_fc_m.parameters(): param.requires_grad = False
        else:
            for param in self.cmaf_m.parameters(): param.requires_grad = False

        # Create queue
        self.register_buffer("queue", torch.randn(self.moco_dim, self.moco_k))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    def init_q2l_from_text(self):
        """Initialize Q2L label queries from hand-crafted CLIP text features.
        Called once after model is on device and text encoder is ready."""
        if self.is_multilabel and not self._q2l_initialized:
            with torch.no_grad():
                hand_crafted_prompts = self.hand_crafted_prompt_embeddings
                tokenized = self.tokenized_hand_crafted_prompts.to(hand_crafted_prompts.device)
                text_feats = self.text_encoder(hand_crafted_prompts, tokenized)
                text_feats = text_feats.float()
                text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-6)
            self.q2l_head.init_from_text_features(text_feats)
            self._q2l_initialized = True

    def set_label_cooccurrence(self, cooccurrence_matrix):
        """Set the label co-occurrence matrix from training data.
        Args:
            cooccurrence_matrix: (num_classes, num_classes) numpy or tensor
        """
        if isinstance(cooccurrence_matrix, np.ndarray):
            cooccurrence_matrix = torch.from_numpy(cooccurrence_matrix).float()
        self.label_cooccurrence.copy_(cooccurrence_matrix)
        print(f"=> Set label co-occurrence matrix: {cooccurrence_matrix.shape}")

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.image_encoder.parameters(), self.image_encoder_m.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)
        for param_q, param_k in zip(self.face_adapter.parameters(), self.face_adapter_m.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)
        for param_q, param_k in zip(self.unified_temporal_net.parameters(), self.unified_temporal_net_m.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)
        for param_q, param_k in zip(self.project_fc.parameters(), self.project_fc_m.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)
        
        if self.fusion_type == 'gfi':
            for param_q, param_k in zip(self.gate_fc.parameters(), self.gate_fc_m.parameters()):
                param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)
        else:
            for param_q, param_k in zip(self.cmaf.parameters(), self.cmaf_m.parameters()):
                param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys) # Removed distributed gather for single GPU simplicity

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size > self.moco_k: # Handle wrap-around if batch size > remaining space
             batch_size = self.moco_k - ptr # truncate to fit
             keys = keys[:batch_size]
        
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.moco_k  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def forward_momentum(self, image_face, image_body, image_context=None):
        n, t, c, h, w = image_face.shape

        # Face Part
        image_face = image_face.contiguous().view(-1, c, h, w)
        image_face_features = self.image_encoder_m(image_face.type(self.dtype))
        image_face_features = self.face_adapter_m(image_face_features)
        
        # Body Part
        image_body = image_body.contiguous().view(-1, c, h, w)
        image_body_features = self.image_encoder_m(image_body.type(self.dtype))

        # Context Part
        if self.use_context:
            assert image_context is not None
            image_context = image_context.contiguous().view(-1, c, h, w)
            image_context_features = self.image_encoder_m(image_context.type(self.dtype))

        # Frame-Level Fusion (momentum)
        if self.fusion_type == 'gfi':
            features_to_concat = [image_face_features, image_body_features]
            if self.use_context:
                features_to_concat.append(image_context_features)
            fused_frame_features = torch.cat(features_to_concat, dim=-1)
            gate = self.gate_fc_m(fused_frame_features)
            fused_frame_features = fused_frame_features * gate
        else:
            if self.use_context:
                fused_frame_features = self.cmaf_m(image_face_features, image_body_features, image_context_features)
            else:
                fused_frame_features = self.cmaf_m(image_face_features, image_body_features)
            
        # Unified Temporal Transformer (momentum)
        fused_frame_features = fused_frame_features.contiguous().view(n, t, -1)
        video_features = self.unified_temporal_net_m(fused_frame_features)
        
        video_features = self.project_fc_m(video_features)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        return video_features
        
    def forward(self, image_face, image_body, image_context=None):
        ################# Visual Part #################
        n, t, c, h, w = image_face.shape

        # Face Part
        image_face_reshaped = image_face.contiguous().view(-1, c, h, w)
        if torch.isnan(image_face_reshaped).any(): print("[DEBUG] NaN detected in RAW image_face INPUT!")
        # Extract feature from modalities
        face_feat = self.image_encoder(image_face_reshaped.type(self.dtype))
        if torch.isnan(face_feat).any(): print("[DEBUG] NaN detected in face_feat from image_encoder!")
            
        image_face_features = self.face_adapter(face_feat) # Apply EAA
        
        # Body Part
        image_body_reshaped = image_body.contiguous().view(-1, c, h, w)
        if torch.isnan(image_body_reshaped).any(): print("[DEBUG] NaN detected in RAW image_body INPUT!")
        body_feat = self.image_encoder(image_body_reshaped.type(self.dtype))
        if torch.isnan(body_feat).any(): print("[DEBUG] NaN detected in body_feat from image_encoder!")
            
        image_body_features = body_feat

        # Context Part
        if self.use_context:
            assert image_context is not None, "image_context must be provided when use_context=True"
            image_context_reshaped = image_context.contiguous().view(-1, c, h, w)
            context_feat = self.image_encoder(image_context_reshaped.type(self.dtype))
            if torch.isnan(context_feat).any(): print("[DEBUG] NaN detected in context_feat from image_encoder!")
            image_context_features = context_feat

        # Frame-Level Fusion (CMAF or GFI)
        if self.fusion_type == 'gfi':
            features_to_concat = [image_face_features, image_body_features]
            if self.use_context:
                features_to_concat.append(image_context_features)
            fused_frame_features = torch.cat(features_to_concat, dim=-1)
            gate = self.gate_fc(fused_frame_features)
            fused_frame_features = fused_frame_features * gate
        else:
            if self.use_context:
                fused_frame_features = self.cmaf(image_face_features, image_body_features, image_context_features)
            else:
                fused_frame_features = self.cmaf(image_face_features, image_body_features)
            
        # Unified Temporal Transformer
        fused_frame_features = fused_frame_features.contiguous().view(n, t, -1)
        video_features = self.unified_temporal_net(fused_frame_features)

        if torch.isnan(video_features).any(): print("[DEBUG] NaN detected after CMAF!")
            
        video_features = self.project_fc(video_features)
        if torch.isnan(video_features).any(): print("[DEBUG] NaN detected after project_fc!")
        
        # Robust normalization to avoid NaN on MPS
        video_features = video_features / (video_features.norm(dim=-1, keepdim=True) + 1e-6)
        if torch.isnan(video_features).any(): print("[DEBUG] NaN detected after video_features normalization!")
        
        # Save video features for feature-level knowledge distillation
        self.last_video_features = video_features.detach()

        ################# Text Part ###################
        # Learnable prompts (always computed for both training and validation)
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        
        # FORCE FP32 for Text Encoder to avoid NaN on MPS
        with torch.cuda.amp.autocast(enabled=False):
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features = text_features.float()  # Ensure float32
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
            
        if torch.isnan(text_features).any(): print("[DEBUG] NaN detected after text_encoder!")

        # Hand-crafted prompts (for MI Loss, CAD, and Text Distillation)
        hand_crafted_text_features = None
        need_hand_crafted = (
            (self.training and getattr(self.args, 'lambda_mi', 0) > 0) or
            getattr(self.args, 'lambda_cad', 0) > 0 or
            getattr(self.args, 'lambda_text', 0) > 0
        )
        if need_hand_crafted:
            hand_crafted_prompts = self.hand_crafted_prompt_embeddings
            tokenized_hand_crafted_prompts = self.tokenized_hand_crafted_prompts.to(hand_crafted_prompts.device)
            
            with torch.no_grad():
                hand_crafted_text_features = self.text_encoder(hand_crafted_prompts, tokenized_hand_crafted_prompts)
                hand_crafted_text_features = hand_crafted_text_features.float()
                hand_crafted_text_features = hand_crafted_text_features / (hand_crafted_text_features.norm(dim=-1, keepdim=True) + 1e-6)

        ################# MoCo Updates ###################
        moco_logits = None
        if self.training and hasattr(self.args, 'use_moco') and self.args.use_moco:
            with torch.no_grad():
                self._momentum_update_key_encoder()
                k_video_features = self.forward_momentum(image_face, image_body, image_context)
            
            # Compute MoCo Logits
            # Positive logits: similarity between query and key
            l_pos = torch.einsum('nc,nc->n', [video_features, k_video_features]).unsqueeze(-1)
            # Negative logits: similarity between query and queue
            l_neg = torch.einsum('nc,ck->nk', [video_features, self.queue.clone().detach()])

            # logits: Nx(1+K)
            moco_logits = torch.cat([l_pos, l_neg], dim=1)
            moco_logits /= self.moco_t

            self._dequeue_and_enqueue(k_video_features)

        ################# Spatial Patches for CAD (PromptCAD) ###################
        patch_features = None
        if getattr(self.args, 'lambda_cad', 0) > 0:
            # Extract 196 spatial patch tokens from face stream for spatial attention distillation
            _, face_patches = self.image_encoder(image_face_reshaped.type(self.dtype), return_patches=True)
            patch_features = face_patches.float()
            patch_features = patch_features / (patch_features.norm(dim=-1, keepdim=True) + 1e-6)

        ################# Classification ###################
        if getattr(self.args, 'use_q2l', False) and self.is_multilabel:
            # ===== Q2L Multi-Label Path (Optional) =====
            if not self._q2l_initialized:
                self.init_q2l_from_text()
            output = self.q2l_head(video_features)
        else:
            # ===== Standard CLIP Prompt Alignment (Single-Label & Multi-Label) =====
            if self.is_ensemble:
                # Reshape text features for ensembling: (C*P, D) -> (C, P, D)
                text_features_ens = text_features.view(self.num_classes, self.num_prompts_per_class, -1)
                text_features_ens = text_features_ens / (text_features_ens.norm(dim=-1, keepdim=True) + 1e-6)
                
                # Compute logits per prompt: (B, D) @ (D, P, C) -> (B, P, C)
                logits = torch.einsum('bd,cpd->bcp', video_features, text_features_ens)
                
                # Average the logits across the prompts for each class
                output = torch.mean(logits, dim=2) / self.args.temperature
            else:
                output = (video_features @ text_features.t()) / self.args.temperature

            if hasattr(self, 'class_bias') and self.is_multilabel:
                if torch.isnan(self.class_bias).any(): print("[DEBUG] class_bias contains NaN!")
                output = output + self.class_bias
                
        if torch.isnan(output).any(): print("[DEBUG] NaN detected in final output!")

        if getattr(self.args, 'lambda_cad', 0) > 0:
            return output, text_features, hand_crafted_text_features, moco_logits, patch_features
        return output, text_features, hand_crafted_text_features, moco_logits