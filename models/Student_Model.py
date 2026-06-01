# models/Student_Model.py
import torch
import torch.nn as nn
import torchvision.models as models
from models.Temporal_Model import Temporal_Transformer_AttnPool

class StudentImageEncoder(nn.Module):
    """
    A lightweight, CPU-efficient image encoder using pretrained MobileNetV3-Large.
    Projects the final 960-dimensional features to 512 dimensions to match CLIP features.
    """
    def __init__(self):
        super().__init__()
        try:
            # Try loading with torchvision >= 0.13 weights argument
            from torchvision.models import mobilenet_v3_large, MobileNetV3_Large_Weights
            self.backbone = mobilenet_v3_large(weights=MobileNetV3_Large_Weights.DEFAULT)
        except ImportError:
            # Fallback for older torchvision
            self.backbone = models.mobilenet_v3_large(pretrained=True)
            
        # Extract features and average pooling from MobileNetV3
        self.features = self.backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Projection layer to match the 512-dim embedding space of CLIP
        self.proj = nn.Sequential(
            nn.Linear(960, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.features(x)      # [B, 960, h, w]
        x = self.pool(x)          # [B, 960, 1, 1]
        x = torch.flatten(x, 1)   # [B, 960]
        x = self.proj(x)          # [B, 512]
        return x

class StudentModel(nn.Module):
    """
    Lightweight Dual-Stream Student Model (~20MB) designed for real-time
    CPU inference on cheap VPS or local compute.
    
    Structure:
      - Shared StudentImageEncoder (MobileNetV3-Large) for Face and Body.
      - Lightweight Face Adapter (residual adapter mapping).
      - Two lightweight Temporal Attention Pooling Transformers.
      - Gated Feature Fusion (mimics the optimized Teacher).
      - FC Projection & Classification Head.
    """
    def __init__(self, num_classes=5, num_segments=16):
        super().__init__()
        self.num_segments = num_segments
        
        # Shared visual encoder
        self.shared_encoder = StudentImageEncoder()
        
        # Residual Face Adapter (EAA equivalent)
        self.face_adapter = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 512)
        )
        
        # Lightweight temporal transformers (highly parameter-efficient)
        self.temporal_net = Temporal_Transformer_AttnPool(
            num_patches=num_segments,
            input_dim=512,
            depth=1,
            heads=4,
            mlp_dim=512,
            dim_head=32
        )
        
        self.temporal_net_body = Temporal_Transformer_AttnPool(
            num_patches=num_segments,
            input_dim=512,
            depth=1,
            heads=4,
            mlp_dim=512,
            dim_head=32
        )

        # Gated Feature Fusion (element-wise gate)
        self.gate_fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.Sigmoid()
        )
        
        self.project_fc = nn.Linear(1024, 512)
        
        # Classification head
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, image_face, image_body):
        # 1. Face Stream
        n, t, c, h, w = image_face.shape
        image_face_reshaped = image_face.contiguous().view(-1, c, h, w)
        # Pass through shared MobileNetV3 encoder
        image_face_features = self.shared_encoder(image_face_reshaped) # [N*T, 512]
        # Apply EAA face adapter
        image_face_features = image_face_features + self.face_adapter(image_face_features)
        image_face_features = image_face_features.contiguous().view(n, t, -1) # [N, T, 512]
        # Aggregate temporal features
        video_face_features = self.temporal_net(image_face_features)  # [N, 512]
        
        # 2. Body Stream
        n, t, c, h, w = image_body.shape
        image_body_reshaped = image_body.contiguous().view(-1, c, h, w)
        # Pass through shared MobileNetV3 encoder
        image_body_features = self.shared_encoder(image_body_reshaped) # [N*T, 512]
        image_body_features = image_body_features.contiguous().view(n, t, -1) # [N, T, 512]
        # Aggregate temporal features
        video_body_features = self.temporal_net_body(image_body_features) # [N, 512]

        # 3. Gated Feature Fusion
        video_features = torch.cat((video_face_features, video_body_features), dim=-1) # [N, 1024]
        gate = self.gate_fc(video_features)
        video_features = video_features * gate
        video_features = self.project_fc(video_features) # [N, 512]
        
        # 4. Feature Normalization
        video_features = video_features / (video_features.norm(dim=-1, keepdim=True) + 1e-6)
        
        # 5. Classifier Logits
        logits = self.classifier(video_features)
        
        return logits, video_features
