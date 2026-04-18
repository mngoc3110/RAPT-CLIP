"""
inference_utils.py — Preprocess một webcam frame (base64 JPEG) 
thành tensor phù hợp với input của RAPT-CLIP model.

Model expect:  (N, T, C, H, W) với T=16 segments.
Với webcam realtime, ta chỉ có 1 frame → duplicate thành 16 frames.
Face stream và Body stream đều dùng cùng frame gốc (single-stream mode).
"""

import base64
import io
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

# Constants từ training config
IMAGE_SIZE = 224
NUM_SEGMENTS = 16

# Transform như trong test_data_loader (không augment, chỉ resize + normalize)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


def base64_to_pil(b64_string: str) -> Image.Image:
    """Decode base64 JPEG/PNG string sang PIL Image (RGB)."""
    # Xử lý data URL prefix nếu có: "data:image/jpeg;base64,..."
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    raw = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img


def preprocess_frame(b64_string: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Nhận 1 frame webcam (base64) → trả về (face_tensor, body_tensor).
    [TỐI ƯU 4GB RAM & REALTIME]
    Thay vì nhân bản lên 16 frame ở ngay bước ảnh (khiến tensor phình to và ngốn RAM/CPU),
    ta chỉ giữ 1 frame duy nhất: Shape (1, 1, 3, 224, 224).
    Mô hình bọc (Wrapper) trên Server sẽ hiểu và tự động copy ở tầng Features (rất nhẹ).
    """
    img = base64_to_pil(b64_string)
    frame_tensor = inference_transform(img)  # (C, H, W)
    
    # Chỉ bọc batch và temporal dim=1: (1, 1, C, H, W)
    video_tensor = frame_tensor.unsqueeze(0).unsqueeze(0).to(device)
    
    return video_tensor, video_tensor  # face=body (single-stream mode)
