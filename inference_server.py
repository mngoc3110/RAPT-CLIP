"""
inference_server.py — FastAPI microservice cho RAPT-CLIP realtime inference.
Port: 8001

Usage:
    python inference_server.py

Endpoints:
    POST /analyze   — nhận base64 frame, trả về emotion label + probabilities
    GET  /health    — kiểm tra server đang sống
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import uvicorn
import anyio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, os.path.dirname(__file__))
from inference_utils import preprocess_frame
from utils.builders import build_model, get_class_info

# ─── Constants ─────────────────────────────────────────────
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__),
    "outputs",
    "RAER-ramp-down",
    "model_best_slim.pth"
)

CLASS_LABELS = ["Neutral", "Enjoyment", "Confusion", "Fatigue", "Distraction"]
CLASS_EMOJIS = {
    "Neutral":     "😐",
    "Enjoyment":   "😊",
    "Confusion":   "😕",
    "Fatigue":     "😴",
    "Distraction": "📵",
}

# ─── Build minimal args namespace (match training config) ──
def get_inference_args():
    args = argparse.Namespace(
        dataset="RAER",
        clip_path="ViT-B/16",
        temporal_layers=3,
        contexts_number=8,
        class_token_position="end",
        class_specific_contexts="True",
        load_and_tune_prompt_learner="True",
        num_segments=16,
        duration=1,
        image_size=224,
        crop_body=False,
        temperature=0.07,
        text_type="prompt_ensemble",
        use_moco=False,
        moco_k=2048,
        moco_m=0.999,
        moco_t=0.07,
        lr_image_encoder=0.0,
        freeze_image_encoder=True,
        drop_path_rate=0.0,
        use_weighted_sampler=False,
        # Dummy annotation paths (not used in inference)
        train_annotation="",
        val_annotation="",
        test_annotation="",
        root_dir="",
        bounding_box_face="",
        bounding_box_body="",
    )
    return args


# ─── Global model state ────────────────────────────────────
model_state = {}

import torch.nn as nn

class RAPTCLIPRealtimeWrapper(nn.Module):
    """
    [TỐI ƯU SIÊU NHẸ]
    Nhận đầu vào tensor của ĐÚNG 1 frame (T=1), pass qua ViT để lấy đặc trưng 512,
    Sau đó mới copy đặc trưng 512 đó lên 16 lần để đút vào Temporal Model.
    -> Tiết kiệm 15 lần (15x) tính toán ViT và giảm 15 lần RAM cần cho Activation.
    """
    def __init__(self, base_model, num_segments=16):
        super().__init__()
        self.base_model = base_model
        self.num_segments = num_segments
        
    def forward(self, image_face, image_body):
        base = self.base_model
        
        # --- Visual Part ---
        n, t, c, h, w = image_face.shape
        # Ép t=1 vì ta chỉ giải nén 1 ảnh
        img_f = image_face[:, 0, :, :, :].view(-1, c, h, w)
        img_b = image_body[:, 0, :, :, :].view(-1, c, h, w)
        
        # Trích xuất đặc trưng ViT đúng 1 LẦN
        face_feat = base.image_encoder(img_f.type(base.dtype))
        face_feat = base.face_adapter(face_feat) # EAA
        body_feat = base.image_encoder(img_b.type(base.dtype))
        
        # Nhân bản đặc trưng vector (rất nhẹ) lên 16 lần (T=16)
        face_feat = face_feat.unsqueeze(1).repeat(1, self.num_segments, 1)
        body_feat = body_feat.unsqueeze(1).repeat(1, self.num_segments, 1)
        
        # Temporal Net
        vid_face = base.temporal_net(face_feat)
        vid_body = base.temporal_net_body(body_feat)
        
        vid_feat = torch.cat((vid_face, vid_body), dim=-1)
        vid_feat = base.project_fc(vid_feat)
        vid_feat = vid_feat / (vid_feat.norm(dim=-1, keepdim=True) + 1e-6)
        
        # --- Text Part ---
        prompts = base.prompt_learner()
        token = base.tokenized_prompts
        
        # Text Encoder
        with torch.amp.autocast(device_type="cpu" if not torch.cuda.is_available() else "cuda", enabled=False):
            txt_feat = base.text_encoder(prompts, token)
            txt_feat = txt_feat.float()
            txt_feat = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-6)
            
        # Classify
        if base.is_ensemble:
            txt_feat = txt_feat.view(base.num_classes, base.num_prompts_per_class, -1)
            txt_feat = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-6)
            logits = torch.einsum('bd,cpd->bcp', vid_feat, txt_feat)
            output = torch.mean(logits, dim=2) / base.args.temperature
        else:
            output = vid_feat @ txt_feat.t() / base.args.temperature
            
        return output, None, None, None



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once on startup."""
    print("=" * 50)
    # ─── Tối ưu hóa Threading CPU (Chống Thrashing) ───
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = 1
        print("⚡ Cấu hình Threading: Giới hạn xuống 1 Luồng để tránh CPU Thrashing!")
    except Exception as e:
        print(f"⚠️ Không thể giới hạn Thread: {e}")
    print("🧠 Loading RAPT-CLIP model...")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")

    # Device: MPS on Apple Silicon, CUDA if available, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"   Device: {device}")

    args = get_inference_args()
    args.device = device

    _, input_text = get_class_info(args)
    model = build_model(args, input_text)
    model = model.to(device)
    model.eval()

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        raise RuntimeError("Checkpoint not found!")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    if device.type == "cpu":
        print("⚡ [Kỹ Thuật Ép INT8] Đang thực hiện Dynamic Quantization (Linear Layers)...")
        # torch.ao.quantization được hỗ trợ trong PyTorch >= 1.13
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear}, # Hàm này quét toàn bộ mạng Nơ-ron và ép kiểu các lớp FNN
            dtype=torch.qint8
        )
        print("⚡ [Kỹ Thuật Ép INT8] Hoàn Tất Thu Nhỏ Mô Hình RAPT-CLIP xuống Hệ số INT8 (1-byte)!")

    # Bọc mô hình lại bằng Wrapper Tối ưu Siêu Nhẹ RAM
    model = RAPTCLIPRealtimeWrapper(model, num_segments=16)
    
    # Kích hoạt Torch Compile nếu PyTorch >= 2.0 (tăng % tốc độ CPU/GPU)
    try:
        model = torch.compile(model)
        print("🚀 Đã kích hoạt PyTorch 2.0 torch.compile!")
    except Exception as e:
        print(f"⚠️ Không dùng được torch.compile (có thể do phiên bản cũ): {e}")

    print("✅ Model loaded successfully!")
    print("=" * 50)

    model_state["model"] = model
    model_state["device"] = device

    yield  # Server is running

    # Shutdown
    model_state.clear()
    print("🛑 RAPT-CLIP server stopped.")


# ─── FastAPI App ───────────────────────────────────────────
app = FastAPI(
    title="RAPT-CLIP Emotion Inference API",
    description="Realtime emotion detection for MINDA Live Classroom",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response schemas ────────────────────────────
class AnalyzeRequest(BaseModel):
    frame_b64: str  # base64 encoded JPEG/PNG image


class AnalyzeResponse(BaseModel):
    label: str
    emoji: str
    confidence: float
    probabilities: dict  # {label: probability}


# ─── Endpoints ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in model_state}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if "model" not in model_state:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    model = model_state["model"]
    device = model_state["device"]

    try:
        face_tensor, body_tensor = preprocess_frame(req.frame_b64, device)

        # Sử dụng Autocast để giảm một nửa lượng RAM ở tầng tính toán (BFloat16/Float16)
        amp_device = "cpu" if device.type == "cpu" else "cuda"
        amp_dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
        
        with torch.no_grad():
            with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
                output, _, _, _ = model(face_tensor, body_tensor)
            
            probs = F.softmax(output, dim=1)[0]  # (5,)

        probs_np = probs.cpu().float().numpy()
        pred_idx = int(probs_np.argmax())
        pred_label = CLASS_LABELS[pred_idx]
        confidence = float(probs_np[pred_idx])

        prob_dict = {CLASS_LABELS[i]: round(float(probs_np[i]), 4) for i in range(len(CLASS_LABELS))}

        return AnalyzeResponse(
            label=pred_label,
            emoji=CLASS_EMOJIS.get(pred_label, "❓"),
            confidence=round(confidence, 4),
            probabilities=prob_dict,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
