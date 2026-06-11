from torch import nn
from models.Temporal_Model import *
from models.Prompt_Learner import *
from models.Text import class_descriptor_5_only_face
from models.Adapter import Adapter
from models.CrossModalAttentionFusion import CrossModalAttentionFusion
from models.clip import clip
import copy
import itertools

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

        in_dim = 1536 if self.use_context else 1024

        self.unified_temporal_net = Temporal_Transformer_AttnPool(num_patches=16,
                                                     input_dim=in_dim,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)

        # Store clip_model_ as a plain Python attribute (NOT an nn.Module submodule).
        object.__setattr__(self, 'clip_model_', clip_model)
        
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
        image_face_features = self.image_encoder(image_face_reshaped.type(self.dtype))
        image_face_features = self.face_adapter(image_face_features) # Apply EAA
        
        # Body Part
        image_body_reshaped = image_body.contiguous().view(-1, c, h, w)
        image_body_features = self.image_encoder(image_body_reshaped.type(self.dtype))

        # Context Part
        if self.use_context:
            assert image_context is not None, "image_context must be provided when use_context=True"
            image_context_reshaped = image_context.contiguous().view(-1, c, h, w)
            image_context_features = self.image_encoder(image_context_reshaped.type(self.dtype))

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