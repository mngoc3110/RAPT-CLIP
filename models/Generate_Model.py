from models.Temporal_Model import *
from models.Prompt_Learner import *
from models.Text import class_descriptor_5_only_face
from models.Adapter import Adapter
from models.CrossModalAttentionFusion import CrossModalAttentionFusion
from models.cgla_head import CrossModalGlobalLocalAlignment
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

        # Face & Body Adapters (EAA)
        # With frozen image encoder, adapters carry ALL domain adaptation.
        # ratio=0.5 gives more adaptation budget (was 0.2 → too conservative for frozen backbone)
        self.face_adapter = Adapter(c_in=512, reduction=4, ratio=0.5)
        # Body stream has ~50% modality weight but previously had NO adapter. Fix:
        self.body_adapter = Adapter(c_in=512, reduction=4, ratio=0.5)

        # For MI Loss
        # Concept Generation & Refinement (CGR) - PromptCAD
        # Precompute the Center Concept Tokens for each class using all available LLM descriptors
        # This creates a highly accurate semantic anchor for Text Distillation and CAD Loss.
        print("=> Generating Center Concept Tokens (CGR) from all class descriptions...")
        all_concept_tokens = []
        with torch.no_grad():
            for i, class_prompts in enumerate(input_text):
                # Ensure it's a list of prompts
                if not isinstance(class_prompts, list):
                    class_prompts = [class_prompts]
                
                # Tokenize all prompts for this class
                tokenized = torch.cat([clip.tokenize(p) for p in class_prompts]).to(clip_model.token_embedding.weight.device)
                
                # Get token embeddings
                token_embeddings = clip_model.token_embedding(tokenized).type(self.dtype)
                
                # Pass through text encoder
                # Note: GenerateModel.text_encoder is already created above, but it requires prompts and tokenized.
                # Actually, clip_model.encode_text takes tokenized directly!
                # Wait, our Custom TextEncoder takes (prompts, tokenized) to allow soft prompts.
                # For hard prompts, we just pass the raw token_embeddings.
                class_text_features = self.text_encoder(token_embeddings, tokenized)
                class_text_features = class_text_features.float()
                class_text_features = class_text_features / (class_text_features.norm(dim=-1, keepdim=True) + 1e-6)
                
                # Average them to get the Center Concept Token (K-means with K=1)
                center_token = class_text_features.mean(dim=0, keepdim=True)
                center_token = center_token / (center_token.norm(dim=-1, keepdim=True) + 1e-6)
                
                all_concept_tokens.append(center_token)
                
        # Shape: (num_classes, 512)
        center_concept_tokens = torch.cat(all_concept_tokens, dim=0)
        self.register_buffer("hand_crafted_text_features", center_concept_tokens)
        print(f"=> CGR: Generated Center Concept Tokens with shape: {self.hand_crafted_text_features.shape}")


        # Context Stream Configuration
        self.use_context = getattr(args, 'use_context', False)
        print(f"=> Using Context Stream: {self.use_context}")

        in_dim = 1536 if self.use_context else 1024

        self.use_temporal = (getattr(args, 'temporal_layers', 1) > 0) and (getattr(args, 'num_segments', 1) > 1)
        if self.use_temporal:
            print(f"=> Temporal AttnPool ENABLED (layers={args.temporal_layers}, segments={args.num_segments})")
            self.unified_temporal_net = Temporal_Transformer_AttnPool(num_patches=16,
                                                         input_dim=in_dim,
                                                         depth=args.temporal_layers,
                                                         heads=8,
                                                         mlp_dim=1024,
                                                         dim_head=64)
        else:
            print(f"=> Temporal AttnPool DISABLED (Bypassed for image dataset or temporal_layers=0)")
            self.unified_temporal_net = nn.Identity()

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
        # Learnable per-class bias initialized from empirical log-odds prior: log(p / (1-p)).
        # Keeps full Zero-shot semantic alignment of CLIP text embeddings (mAP ~30% at Epoch 0)
        # while independently shifting decision thresholds per class to solve extreme imbalance.
        if self.is_multilabel:
            num_cls = self.num_classes if hasattr(self, 'num_classes') else len(input_text)
            prior_logit = torch.full((num_cls,), -2.2)
            self.class_bias = nn.Parameter(prior_logit)
            print(f"=> EMOTIC: Initialized learnable class_bias for {num_cls} classes (log-odds prior)")

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
            # Freeze Q2L when not actively used to save memory and gradients
            if not getattr(args, 'use_q2l', False):
                for param in self.q2l_head.parameters():
                    param.requires_grad = False
                print("=> Q2L head FROZEN (use_q2l=False). Enable with --use-q2l to unfreeze.")
        
        # ==================== CGLA (MPA-FER) ====================
        self.use_cgla = getattr(args, 'use_cgla', False)
        if self.use_cgla:
            top_k = getattr(args, 'cgla_topk', 16)
            alpha_local = getattr(args, 'cgla_alpha', 1.0)
            use_disc = not getattr(args, 'cgla_no_discriminative', False)
            print(f"=> CGLA (MPA-FER): Initializing Global-Local Alignment Head (top_k={top_k}, alpha={alpha_local}, discriminative={use_disc})")
            self.cgla_head = CrossModalGlobalLocalAlignment(top_k=top_k, temperature=args.temperature, alpha_local=alpha_local, use_class_discriminative=use_disc)

        # ==================== Co-occurrence Matrix (EMOTIC) ====================
        if self.is_multilabel:
            # Will be set externally from training data statistics
            nc = self.num_classes if hasattr(self, 'num_classes') else len(input_text)
            self.register_buffer('label_cooccurrence', torch.zeros(nc, nc))

    def set_class_prior(self, freq_vector: torch.Tensor):
        """Update the class_bias parameter from actual training data class frequencies.
        
        freq_vector: (C,) float tensor with values in [0, 1] = class frequency p_c.
        bias = log(p / (1-p)), clamped to avoid ±inf.
        """
        if hasattr(self, 'class_bias'):
            freq = freq_vector.clamp(0.01, 0.95)  # wide range: handles p=0.008 to p=0.52+
            log_odds = torch.log(freq / (1.0 - freq))
            with torch.no_grad():
                self.class_bias.copy_(log_odds)
            print(f"=> EMOTIC: Initialized class_bias (log-odds prior) from training data:")
            print(f"   min={log_odds.min():.2f} (Rare class), max={log_odds.max():.2f} (Common class)")



    def init_q2l_from_text(self):
        """Initialize Q2L label queries from hand-crafted CLIP text features.
        Called once after model is on device and text encoder is ready."""
        if self.is_multilabel and not self._q2l_initialized:
            with torch.no_grad():
                text_feats = self.hand_crafted_text_features
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
        
    def forward(self, image_face, image_body, image_context=None):
        ################# Visual Part #################
        n, t, c, h, w = image_face.shape

        # Face Part
        image_face_reshaped = image_face.contiguous().view(-1, c, h, w)
        if torch.isnan(image_face_reshaped).any(): print("[DEBUG] NaN detected in RAW image_face INPUT!")
        
        # Extract feature from modalities in true float32 to bypass specific image overflow in ViT
        with torch.cuda.amp.autocast(enabled=False):
            face_feat = self.image_encoder(image_face_reshaped.float())
            if torch.isnan(face_feat).any(): print("[DEBUG] NaN detected in face_feat from image_encoder!")
            
        image_face_features = self.face_adapter(face_feat) # Apply EAA
        
        # Body Part
        image_body_reshaped = image_body.contiguous().view(-1, c, h, w)
        if torch.isnan(image_body_reshaped).any(): print("[DEBUG] NaN detected in RAW image_body INPUT!")
        
        with torch.cuda.amp.autocast(enabled=False):
            body_feat = self.image_encoder(image_body_reshaped.float())
            if torch.isnan(body_feat).any(): print("[DEBUG] NaN detected in body_feat from image_encoder!")
            
        image_body_features = self.body_adapter(body_feat)  # Apply EAA (symmetric with face)

        if self.use_context:
            assert image_context is not None, "image_context must be provided when use_context=True"
            image_context_reshaped = image_context.contiguous().view(-1, c, h, w)
            with torch.cuda.amp.autocast(enabled=False):
                context_feat = self.image_encoder(image_context_reshaped.float())
                if torch.isnan(context_feat).any(): print("[DEBUG] NaN detected in context_feat from image_encoder!")
            image_context_features = context_feat

        # Extract body patches for CAD loss (body patches are more semantically aligned
        # with emotion text prompts than context/scene patches)
        patch_features = None
        need_patches = getattr(self.args, 'lambda_cad', 0) > 0 or getattr(self.args, 'use_cgla', False)
        if need_patches:
            with torch.cuda.amp.autocast(enabled=False):
                _, body_spatial_patches = self.image_encoder(image_body_reshaped.float(), return_patches=True)
            patch_features = body_spatial_patches.float()
            patch_features = patch_features / (patch_features.norm(dim=-1, keepdim=True) + 1e-6)
            # For video (t>1): image_body_reshaped is (B*T, C, H, W) → patches are (B*T, N, D).
            # CAD loss expects batch dim = B (to match target). Reshape to (B, T*N, D):
            # all T frames' patches are treated as the spatial token pool for each video clip.
            if t > 1:
                _, N_patches, D_feat = patch_features.shape
                patch_features = patch_features.view(n, t * N_patches, D_feat)  # (B, T*N, D)

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
            
        # Unified Temporal Transformer (only needed for video t > 1, bypass for image t=1)
        fused_frame_features = fused_frame_features.contiguous().view(n, t, -1)
        if t > 1:
            video_features = self.unified_temporal_net(fused_frame_features)
        else:
            # Single-frame image (e.g. EMOTIC t=1): Direct passthrough to preserve CLIP feature fidelity
            video_features = fused_frame_features.squeeze(1)

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
            hand_crafted_text_features = self.hand_crafted_text_features

        ################# Spatial Patches for CAD & CGLA (MPA-FER) ###################
        if need_patches and patch_features is None:
            # Fallback when context stream is disabled (e.g. body-only)
            patch_img = image_body_reshaped
            with torch.cuda.amp.autocast(enabled=False):
                _, spatial_patches = self.image_encoder(patch_img.float(), return_patches=True)
            patch_features = spatial_patches.float()
            patch_features = patch_features / (patch_features.norm(dim=-1, keepdim=True) + 1e-6)

        ################# Classification ###################
        if getattr(self.args, 'use_cgla', False) and patch_features is not None:
            # ===== CGLA: Global-Local Alignment (MPA-FER) =====
            if t > 1:
                cgla_patches = patch_features.view(n, t, -1, patch_features.shape[-1])
            else:
                cgla_patches = patch_features.view(n, -1, patch_features.shape[-1])
            if self.is_ensemble:
                text_input = text_features.view(self.num_classes, self.num_prompts_per_class, -1)
            else:
                text_input = text_features
            output = self.cgla_head(video_features, cgla_patches, text_input)

        elif getattr(self.args, 'use_q2l', False) and self.is_multilabel:
            # ===== Q2L Multi-Label Path (Optional, kept for ablation) =====
            if not self._q2l_initialized:
                self.init_q2l_from_text()
            output = self.q2l_head(video_features)

        else:
            # ===== Standard CLIP Cosine Similarity (Leverages Zero-Shot Semantic Prior) =====
            if self.is_ensemble:
                text_features_ens = text_features.view(self.num_classes, self.num_prompts_per_class, -1)
                text_features_ens = text_features_ens / (text_features_ens.norm(dim=-1, keepdim=True) + 1e-6)
                logits = torch.einsum('bd,cpd->bcp', video_features, text_features_ens)
                output = torch.mean(logits, dim=2) / self.args.temperature
            else:
                output = (video_features @ text_features.t()) / self.args.temperature

        # Apply learnable per-class bias for multi-label tasks (EMOTIC)
        # Keeps zero-shot CLIP semantics while dynamically balancing class decision thresholds
        if hasattr(self, 'class_bias') and self.is_multilabel:
            if torch.isnan(self.class_bias).any():
                print("[DEBUG] class_bias contains NaN!")
            output = output + self.class_bias

        if torch.isnan(output).any(): print("[DEBUG] NaN detected in final output!")

        if getattr(self.args, 'lambda_cad', 0) > 0:
            return output, text_features, hand_crafted_text_features, None, patch_features
        return output, text_features, hand_crafted_text_features, None