from torch import nn
from models.Temporal_Model import *
from models.Adapter import Adapter
import copy
import torch

class GenerateModel_NoText(nn.Module):
    def __init__(self, clip_model, args):
        super().__init__()
        self.args = args
        
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual

        # Freeze Image Encoder if requested
        if hasattr(args, 'freeze_image_encoder') and args.freeze_image_encoder:
            print("=> Freezing Image Encoder")
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        # For EAA
        self.face_adapter = Adapter(c_in=512, reduction=4)

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
        self.clip_model_ = clip_model
        
        # Project Face and Body
        self.project_fc = nn.Linear(1024, 512)

        # Standard Linear Classifier for Ablation
        self.num_classes = args.num_classes
        self.classifier = nn.Linear(512, self.num_classes)

        # MoCo Initialization
        if hasattr(args, 'use_moco') and args.use_moco:
            print("=> Initializing MoCoRank...")
            self.moco_dim = 512
            self.moco_k = args.moco_k
            self.moco_m = args.moco_m
            self.moco_t = args.moco_t

            self.image_encoder_m = copy.deepcopy(self.image_encoder)
            self.face_adapter_m = copy.deepcopy(self.face_adapter)
            self.temporal_net_m = copy.deepcopy(self.temporal_net)
            self.temporal_net_body_m = copy.deepcopy(self.temporal_net_body)
            self.project_fc_m = copy.deepcopy(self.project_fc)

            for param in self.image_encoder_m.parameters(): param.requires_grad = False
            for param in self.face_adapter_m.parameters(): param.requires_grad = False
            for param in self.temporal_net_m.parameters(): param.requires_grad = False
            for param in self.temporal_net_body_m.parameters(): param.requires_grad = False
            for param in self.project_fc_m.parameters(): param.requires_grad = False

            self.register_buffer("queue", torch.randn(self.moco_dim, self.moco_k))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.image_encoder.parameters(), self.image_encoder_m.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)
        for param_q, param_k in zip(self.face_adapter.parameters(), self.face_adapter_m.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)
        for param_q, param_k in zip(self.temporal_net.parameters(), self.temporal_net_m.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)
        for param_q, param_k in zip(self.temporal_net_body.parameters(), self.temporal_net_body_m.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)
        for param_q, param_k in zip(self.project_fc.parameters(), self.project_fc_m.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.moco_k:
             batch_size = self.moco_k - ptr
             keys = keys[:batch_size]
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.moco_k
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def forward_momentum(self, image_face, image_body, image_context=None):
        n, t, c, h, w = image_face.shape
        image_face = image_face.contiguous().view(-1, c, h, w)
        image_face_features = self.image_encoder_m(image_face.type(self.dtype))
        image_face_features = self.face_adapter_m(image_face_features)
        image_face_features = image_face_features.contiguous().view(n, t, -1)
        video_face_features = self.temporal_net_m(image_face_features)
        
        n, t, c, h, w = image_body.shape
        image_body = image_body.contiguous().view(-1, c, h, w)
        image_body_features = self.image_encoder_m(image_body.type(self.dtype))
        image_body_features = image_body_features.contiguous().view(n, t, -1)
        video_body_features = self.temporal_net_body_m(image_body_features)

        video_features = torch.cat((video_face_features, video_body_features), dim=-1)
        video_features = self.project_fc_m(video_features)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        return video_features
        
    def forward(self, image_face, image_body, image_context=None):
        ################# Visual Part #################
        n, t, c, h, w = image_face.shape
        image_face_reshaped = image_face.contiguous().view(-1, c, h, w)
        image_face_features = self.image_encoder(image_face_reshaped.type(self.dtype))
        image_face_features = self.face_adapter(image_face_features)
        image_face_features = image_face_features.contiguous().view(n, t, -1)
        video_face_features = self.temporal_net(image_face_features)
        
        n, t, c, h, w = image_body.shape
        image_body_reshaped = image_body.contiguous().view(-1, c, h, w)
        image_body_features = self.image_encoder(image_body_reshaped.type(self.dtype))
        image_body_features = image_body_features.contiguous().view(n, t, -1)
        video_body_features = self.temporal_net_body(image_body_features)

        video_features = torch.cat((video_face_features, video_body_features), dim=-1)
        video_features = self.project_fc(video_features)
        video_features_norm = video_features / (video_features.norm(dim=-1, keepdim=True) + 1e-6)

        ################# MoCo Updates ###################
        moco_logits = None
        if self.training and hasattr(self.args, 'use_moco') and self.args.use_moco:
            with torch.no_grad():
                self._momentum_update_key_encoder()
                k_video_features = self.forward_momentum(image_face, image_body)
            
            l_pos = torch.einsum('nc,nc->n', [video_features_norm, k_video_features]).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', [video_features_norm, self.queue.clone().detach()])

            moco_logits = torch.cat([l_pos, l_neg], dim=1)
            moco_logits /= self.moco_t

            self._dequeue_and_enqueue(k_video_features)

        ################# Classification ###################
        # Generate Logits using traditional Linear Layer on normalized video features
        output = self.classifier(video_features_norm) / self.args.temperature
        
        return output, None, None, moco_logits
