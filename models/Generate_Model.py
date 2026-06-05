from torch import nn
from models.Temporal_Model import *
from models.Prompt_Learner import *
from models.Text import class_descriptor_5_only_face
from models.Adapter import Adapter
from models.CrossModalAttentionFusion import CrossModalAttentionFusion
from models.clip import clip
import copy
import itertools

class LightweightMotionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1)) # pools H, W
        )
        self.fc = nn.Linear(128, 512)
        
    def forward(self, x):
        # x: (B, T, C, H, W) -> Conv3d expects (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        out = self.net(x) # (B, 128, T, 1, 1)
        out = out.squeeze(-1).squeeze(-1).permute(0, 2, 1) # (B, T, 128)
        out = self.fc(out) # (B, T, 512)
        return out

class GenerateModel(nn.Module):
    def __init__(self, input_text, clip_model, args):
        super().__init__()
        self.args = args
        
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

        # Motion Stream Configuration
        self.use_motion = getattr(args, 'use_motion', False)
        print(f"=> Using Motion Stream (RGB-Diff): {self.use_motion}")

        self.temporal_net = Temporal_Transformer_AttnPool(num_patches=16,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)
        
        self.temporal_net_body = Temporal_Transformer_AttnPool(num_patches=16,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)

        if self.use_context:
            self.temporal_net_context = Temporal_Transformer_AttnPool(num_patches=16,
                                                         input_dim=512,
                                                         depth=args.temporal_layers,
                                                         heads=8,
                                                         mlp_dim=1024,
                                                         dim_head=64)

        if self.use_motion:
            self.motion_encoder = LightweightMotionEncoder()
            self.temporal_net_motion = Temporal_Transformer_AttnPool(num_patches=16,
                                                         input_dim=512,
                                                         depth=args.temporal_layers,
                                                         heads=8,
                                                         mlp_dim=1024,
                                                         dim_head=64)

        # Store clip_model_ as a plain Python attribute (NOT an nn.Module submodule).
        object.__setattr__(self, 'clip_model_', clip_model)
        
        in_dim = 1024
        if self.use_context: in_dim += 512
        if self.use_motion: in_dim += 512
        self.project_fc = nn.Linear(in_dim, 512)

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
            self.cmaf = CrossModalAttentionFusion(dim=512, num_heads=4, dropout=0.1, use_context=self.use_context)

        # MoCo Initialization
        if hasattr(args, 'use_moco') and args.use_moco:
            print("=> Initializing MoCoRank...")
            self.moco_dim = 512
            self.moco_k = args.moco_k
            self.moco_m = args.moco_m
            self.moco_t = args.moco_t

            # Create momentum encoders
            self.image_encoder_m = copy.deepcopy(self.image_encoder)
            self.face_adapter_m = copy.deepcopy(self.face_adapter)
            self.temporal_net_m = copy.deepcopy(self.temporal_net)
            self.temporal_net_body_m = copy.deepcopy(self.temporal_net_body)
            if self.use_context:
                self.temporal_net_context_m = copy.deepcopy(self.temporal_net_context)
            if self.use_motion:
                self.motion_encoder_m = copy.deepcopy(self.motion_encoder)
                self.temporal_net_motion_m = copy.deepcopy(self.temporal_net_motion)
            self.project_fc_m = copy.deepcopy(self.project_fc)
            
            if self.fusion_type == 'gfi':
                self.gate_fc_m = copy.deepcopy(self.gate_fc)
            else:
                self.cmaf_m = copy.deepcopy(self.cmaf)

            # Freeze momentum encoders
            for param in self.image_encoder_m.parameters(): param.requires_grad = False
            for param in self.face_adapter_m.parameters(): param.requires_grad = False
            for param in self.temporal_net_m.parameters(): param.requires_grad = False
            for param in self.temporal_net_body_m.parameters(): param.requires_grad = False
            if self.use_context:
                for param in self.temporal_net_context_m.parameters(): param.requires_grad = False
            if self.use_motion:
                for param in self.motion_encoder_m.parameters(): param.requires_grad = False
                for param in self.temporal_net_motion_m.parameters(): param.requires_grad = False
            for param in self.project_fc_m.parameters(): param.requires_grad = False
            
            if self.fusion_type == 'gfi':
                for param in self.gate_fc_m.parameters(): param.requires_grad = False
            else:
                for param in self.cmaf_m.parameters(): param.requires_grad = False

            # Create queue
            self.register_buffer("queue", torch.randn(self.moco_dim, self.moco_k))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.image_encoder.parameters(), self.image_encoder_m.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)
        for param_q, param_k in zip(self.face_adapter.parameters(), self.face_adapter_m.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)
        for param_q, param_k in zip(self.temporal_net.parameters(), self.temporal_net_m.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)
        for param_q, param_k in zip(self.temporal_net_body.parameters(), self.temporal_net_body_m.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)
        if self.use_context:
            for param_q, param_k in zip(self.temporal_net_context.parameters(), self.temporal_net_context_m.parameters()):
                param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)
        if self.use_motion:
            for param_q, param_k in zip(self.motion_encoder.parameters(), self.motion_encoder_m.parameters()):
                param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)
            for param_q, param_k in zip(self.temporal_net_motion.parameters(), self.temporal_net_motion_m.parameters()):
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
        # Motion Part
        if self.use_motion:
            image_motion = torch.zeros_like(image_face)
            image_motion[:, :-1] = image_face[:, 1:] - image_face[:, :-1]
            image_motion[:, -1] = image_motion[:, -2]
            image_motion_features = self.motion_encoder_m(image_motion)
            video_motion_features = self.temporal_net_motion_m(image_motion_features)

        # Face Part
        n, t, c, h, w = image_face.shape
        image_face = image_face.contiguous().view(-1, c, h, w)
        image_face_features = self.image_encoder_m(image_face.type(self.dtype))
        image_face_features = self.face_adapter_m(image_face_features)
        image_face_features = image_face_features.contiguous().view(n, t, -1)
        video_face_features = self.temporal_net_m(image_face_features)
        
        # Body Part
        n, t, c, h, w = image_body.shape
        image_body = image_body.contiguous().view(-1, c, h, w)
        image_body_features = self.image_encoder_m(image_body.type(self.dtype))
        image_body_features = image_body_features.contiguous().view(n, t, -1)
        video_body_features = self.temporal_net_body_m(image_body_features)

        # Context Part
        if self.use_context:
            assert image_context is not None
            n, t, c, h, w = image_context.shape
            image_context = image_context.contiguous().view(-1, c, h, w)
            image_context_features = self.image_encoder_m(image_context.type(self.dtype))
            image_context_features = image_context_features.contiguous().view(n, t, -1)
            video_context_features = self.temporal_net_context_m(image_context_features)

        # Fusion (momentum)
        if self.fusion_type == 'gfi':
            features_to_concat = [video_face_features, video_body_features]
            if self.use_context:
                features_to_concat.append(video_context_features)
            video_features = torch.cat(features_to_concat, dim=-1)
            gate = self.gate_fc_m(video_features)
            video_features = video_features * gate
        else:
            if self.use_context:
                video_features = self.cmaf_m(video_face_features, video_body_features, video_context_features)
            else:
                video_features = self.cmaf_m(video_face_features, video_body_features)
            
        if self.use_motion:
            video_features = torch.cat((video_features, video_motion_features), dim=-1)

        video_features = self.project_fc_m(video_features)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        return video_features
        
    def forward(self, image_face, image_body, image_context=None):
        ################# Visual Part #################
        # Motion Part
        if self.use_motion:
            image_motion = torch.zeros_like(image_face)
            image_motion[:, :-1] = image_face[:, 1:] - image_face[:, :-1]
            image_motion[:, -1] = image_motion[:, -2]
            image_motion_features = self.motion_encoder(image_motion)
            video_motion_features = self.temporal_net_motion(image_motion_features)

        # Face Part
        n, t, c, h, w = image_face.shape
        image_face_reshaped = image_face.contiguous().view(-1, c, h, w)
        image_face_features = self.image_encoder(image_face_reshaped.type(self.dtype))
        image_face_features = self.face_adapter(image_face_features) # Apply EAA
        image_face_features = image_face_features.contiguous().view(n, t, -1)
        video_face_features = self.temporal_net(image_face_features)  # (4*512)
        
        # Body Part
        n, t, c, h, w = image_body.shape
        image_body_reshaped = image_body.contiguous().view(-1, c, h, w)
        image_body_features = self.image_encoder(image_body_reshaped.type(self.dtype))
        image_body_features = image_body_features.contiguous().view(n, t, -1)
        video_body_features = self.temporal_net_body(image_body_features)

        # Context Part
        if self.use_context:
            assert image_context is not None, "image_context must be provided when use_context=True"
            n, t, c, h, w = image_context.shape
            image_context_reshaped = image_context.contiguous().view(-1, c, h, w)
            image_context_features = self.image_encoder(image_context_reshaped.type(self.dtype))
            image_context_features = image_context_features.contiguous().view(n, t, -1)
            video_context_features = self.temporal_net_context(image_context_features)

        # Fusion (CMAF or GFI)
        if self.fusion_type == 'gfi':
            features_to_concat = [video_face_features, video_body_features]
            if self.use_context:
                features_to_concat.append(video_context_features)
            video_features = torch.cat(features_to_concat, dim=-1)
            gate = self.gate_fc(video_features)
            video_features = video_features * gate
        else:
            if self.use_context:
                video_features = self.cmaf(video_face_features, video_body_features, video_context_features)
            else:
                video_features = self.cmaf(video_face_features, video_body_features)
            
        if self.use_motion:
            video_features = torch.cat((video_features, video_motion_features), dim=-1)

        video_features = self.project_fc(video_features)
        # Robust normalization to avoid NaN on MPS
        video_features = video_features / (video_features.norm(dim=-1, keepdim=True) + 1e-6)
        
        # Save video features for feature-level knowledge distillation
        self.last_video_features = video_features

        ################# Text Part ###################
        # Learnable prompts
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        
        # FORCE FP32 for Text Encoder to avoid NaN on MPS
        with torch.cuda.amp.autocast(enabled=False):
            # Text Encoder might contain layers incompatible with AMP on MPS or just unstable
            text_features = self.text_encoder(prompts, tokenized_prompts)
            # Robust normalization
            text_features = text_features.float() # Ensure float32
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)

        # Hand-crafted prompts (for MI Loss, not used for classification)
        hand_crafted_prompts = self.hand_crafted_prompt_embeddings
        tokenized_hand_crafted_prompts = self.tokenized_hand_crafted_prompts.to(hand_crafted_prompts.device)
        
        with torch.cuda.amp.autocast(enabled=False):
            hand_crafted_text_features = self.text_encoder(hand_crafted_prompts, tokenized_hand_crafted_prompts)
            hand_crafted_text_features = hand_crafted_text_features.float()
            # Robust normalization
            hand_crafted_text_features = hand_crafted_text_features / (hand_crafted_text_features.norm(dim=-1, keepdim=True) + 1e-6)

        ################# MoCo Updates ###################
        moco_logits = None
        if self.training and hasattr(self.args, 'use_moco') and self.args.use_moco:
            with torch.no_grad():
                self._momentum_update_key_encoder()
                k_video_features = self.forward_momentum(image_face, image_body)
            
            # Compute MoCo Logits
            # Positive logits: similarity between query and key
            l_pos = torch.einsum('nc,nc->n', [video_features, k_video_features]).unsqueeze(-1)
            # Negative logits: similarity between query and queue
            l_neg = torch.einsum('nc,ck->nk', [video_features, self.queue.clone().detach()])

            # logits: Nx(1+K)
            moco_logits = torch.cat([l_pos, l_neg], dim=1)
            moco_logits /= self.moco_t

            self._dequeue_and_enqueue(k_video_features)

        ################# Classification ###################
        # Calculate logits
        if self.is_ensemble:
            # Reshape text features for ensembling: (C*P, D) -> (C, P, D)
            text_features = text_features.view(self.num_classes, self.num_prompts_per_class, -1)
            # Normalize again just in case (optional but safe) - Robust version
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
            
            # Compute logits per prompt: (B, D) @ (D, P, C) -> (B, P, C)
            # Note: We use einsum for clarity with batch and ensemble dimensions
            logits = torch.einsum('bd,cpd->bcp', video_features, text_features)
            
            # Average the logits across the prompts for each class
            output = torch.mean(logits, dim=2) / self.args.temperature

        else:
            output = video_features @ text_features.t() / self.args.temperature

        return output, text_features, hand_crafted_text_features, moco_logits