from torch import nn
from models.Temporal_Model import *
from models.Prompt_Learner import *
from models.Text import class_descriptor_5_only_face
from models.Adapter import Adapter
from clip import clip
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

        # [LUỒNG 3.2.1: TEXT BRANCH]
        # Khởi tạo Prompt Learner (CoOp)
        self.prompt_learner = PromptLearner(self.input_text, clip_model, args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual

        # [LUỒNG 3.2.2: VISUAL ADAPTER]
        # Khởi tạo Face Adapter (Efficient Fine-tuning)
        self.face_adapter = Adapter(c_in=512, reduction=4)

        # For MI Loss
        hand_crafted_prompts = class_descriptor_5_only_face
        self.tokenized_hand_crafted_prompts = torch.cat([clip.tokenize(p) for p in hand_crafted_prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(self.tokenized_hand_crafted_prompts).type(self.dtype)
        self.register_buffer("hand_crafted_prompt_embeddings", embedding)

        # [LUỒNG 3.2.3: TEMPORAL MODULE]
        # Khởi tạo Temporal Attention Pooling
        if hasattr(args, 'temporal_type') and args.temporal_type == 'cls':
            print("=> Using Temporal_Transformer_Cls (Baseline style)")
            TemporalClass = Temporal_Transformer_Cls
        else:
            print("=> Using Temporal_Transformer_AttnPool (Proposed style)")
            TemporalClass = Temporal_Transformer_AttnPool

        self.temporal_net = TemporalClass(num_patches=16,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)
        
        self.temporal_net_body = TemporalClass(num_patches=16,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)
        self.clip_model_ = clip_model
        self.project_fc = nn.Linear(1024, 512)

    def forward(self, image_face,image_body):
        # [LUỒNG 7: FORWARD PASS]
        # Đây là luồng xử lý chính của model cho mỗi batch
        
        ################# Visual Part #################
        # 1. Face Part
        n, t, c, h, w = image_face.shape
        image_face_reshaped = image_face.contiguous().view(-1, c, h, w)
        image_face_features = self.image_encoder(image_face_reshaped.type(self.dtype))
        
        # Apply Face Adapter Only if Enabled
        if not hasattr(self.args, 'use_adapter') or self.args.use_adapter == 'True':
            image_face_features = self.face_adapter(image_face_features) # Apply EAA
            
        image_face_features = image_face_features.contiguous().view(n, t, -1)
        video_face_features = self.temporal_net(image_face_features)  # (4*512)
        
        # 2. Body Part
        n, t, c, h, w = image_body.shape
        image_body_reshaped = image_body.contiguous().view(-1, c, h, w)
        image_body_features = self.image_encoder(image_body_reshaped.type(self.dtype))
        image_body_features = image_body_features.contiguous().view(n, t, -1)
        video_body_features = self.temporal_net_body(image_body_features)

        # 3. Concatenate and Project (Fusion)
        video_features = torch.cat((video_face_features, video_body_features), dim=-1)
        video_features = self.project_fc(video_features)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        ################# Text Part ###################
        # 4. Learnable prompts (Context Vectors + Class Names)
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 5. Hand-crafted prompts (Only for MI Loss, not for prediction)
        hand_crafted_prompts = self.hand_crafted_prompt_embeddings
        tokenized_hand_crafted_prompts = self.tokenized_hand_crafted_prompts.to(hand_crafted_prompts.device)
        hand_crafted_text_features = self.text_encoder(hand_crafted_prompts, tokenized_hand_crafted_prompts)
        hand_crafted_text_features = hand_crafted_text_features / hand_crafted_text_features.norm(dim=-1, keepdim=True)

        ################# Classification ###################
        # 6. Calculate Similarity (Video <-> Text)
        if self.is_ensemble:
            # Reshape text features for ensembling: (C*P, D) -> (C, P, D)
            text_features = text_features.view(self.num_classes, self.num_prompts_per_class, -1)
            # Normalize again just in case (optional but safe)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute logits per prompt: (B, D) @ (D, P, C) -> (B, P, C)
            logits = torch.einsum('bd,cpd->bcp', video_features, text_features)
            
            # Average the logits across the prompts for each class
            output = torch.mean(logits, dim=2) / self.args.temperature

        else:
            output = video_features @ text_features.t() / self.args.temperature

        return output, text_features, hand_crafted_text_features, video_features