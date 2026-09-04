# models/cgla_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalGlobalLocalAlignment(nn.Module):
    """
    Cross-modal Global-Local Alignment (CGLA) Module từ MPA-FER (arXiv:2506.21017).
    
    Giải quyết trực tiếp vấn đề:
    "Body bounding box là hình chữ nhật nên chứa nhiều nền (background) trùng lặp với Context".
    
    Cơ chế:
    1. Chia ảnh thành N=196 spatial patch tokens (14x14) qua Frozen ViT.
    2. Tính độ tương quan ngữ nghĩa giữa từng patch token với Text Prompt của từng lớp cảm xúc.
    3. Lọc Top-k patches (mặc định k=16) có điểm tương quan cao nhất (tự động chọn vùng nét mặt,
       bàn tay, tư thế cơ thể và LOẠI BỎ hoàn toàn phần nền lọt vào trong bounding box).
    4. Kết hợp Global similarity S_g và Top-k Local similarity S_local để sinh logits chuẩn xác.
    
    Hỗ trợ cả:
    - 2D Image (EMOTIC): (B, N, D)
    - 3D Video (RAER): (B, T, N, D) với T khung hình liên tục
    - Prompt Ensembling: text_feats có thể là (C, D) hoặc (C, P, D)
    """
    def __init__(self, top_k: int = 16, temperature: float = 0.07, alpha_local: float = 1.0, use_class_discriminative: bool = True):
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature
        self.alpha_local = alpha_local
        self.use_class_discriminative = use_class_discriminative

    def forward(self, global_feat: torch.Tensor, patch_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_feat: (B, D) vector đặc trưng toàn cục sau khi qua CMAF và Temporal Transformer
            patch_feats: (B, N, D) hoặc (B, T, N, D) spatial patch tokens từ ViT-B/16 (N=196, D=512)
            text_feats:  (C, D) hoặc (C, P, D) text embeddings của các lớp cảm xúc
        Returns:
            logits: (B, C) điểm tương quan kết hợp phục vụ phân loại
        """
        B, D = global_feat.shape
        
        # Chuẩn hóa L2 cho global feature
        global_norm = F.normalize(global_feat.float(), p=2, dim=-1)  # (B, D)

        # Xử lý Text Features (hỗ trợ cả prompt ensemble (C, P, D) và standard (C, D))
        is_ensemble = (text_feats.dim() == 3)
        if is_ensemble:
            C, P, _ = text_feats.shape
            # Trung bình qua P prompts cho mỗi lớp
            text_mean = text_feats.mean(dim=1)  # (C, D)
            text_norm = F.normalize(text_mean.float(), p=2, dim=-1)
        else:
            C = text_feats.shape[0]
            text_norm = F.normalize(text_feats.float(), p=2, dim=-1)  # (C, D)

        # 1. Global Cosine Similarity: S_g = (B, D) @ (D, C) -> (B, C)
        s_global = torch.matmul(global_norm, text_norm.T)

        # 2. Xử lý Patch Features & Class-Discriminative Filtering
        # Background patches (giá sách, tường, cây cỏ) có tương quan xấp xỉ nhau với TẤT CẢ các lớp text.
        # Mean-Class Subtraction (Centering) trừ đi độ tương quan trung bình trên toàn bộ C lớp:
        # -> Điểm số bối cảnh bị triệt tiêu về 0
        # -> Chỉ các patch mang tính phân biệt cảm xúc độc thù (lông mày, khóe miệng, cánh tay, dáng người)
        #    mới có disc_sim > 0 và được chọn vào Top-k!
        if patch_feats.dim() == 4:
            # RAER Video: B, T, N, D
            B_v, T_v, N_v, D_v = patch_feats.shape
            patch_norm = F.normalize(patch_feats.float(), p=2, dim=-1)  # (B, T, N, D)
            
            # Tính tương quan từng patch với text prompt: (B, T, N, D) x (C, D) -> (B, T, N, C)
            patch_sim = torch.einsum('btnd,cd->btnc', patch_norm, text_norm)
            patch_sim_flat = patch_sim.view(B_v, T_v * N_v, C)  # (B, T*N, C)
            
            if self.use_class_discriminative and C > 1:
                mean_class_sim = patch_sim_flat.mean(dim=-1, keepdim=True)  # (B, T*N, 1)
                disc_sim = F.relu(patch_sim_flat - mean_class_sim)           # (B, T*N, C)
                k_eff = min(self.top_k, disc_sim.size(1))
                _, topk_indices = torch.topk(disc_sim, k=k_eff, dim=1)      # (B, k, C)
                # Lấy giá trị cosine similarity thực tế tại các vị trí patch giải phẫu đã lọc sạch nền
                topk_sim = torch.gather(patch_sim_flat, dim=1, index=topk_indices) # (B, k, C)
            else:
                k_eff = min(self.top_k, patch_sim_flat.size(1))
                topk_sim, _ = torch.topk(patch_sim_flat, k=k_eff, dim=1)  # (B, k, C)
                
            s_local = torch.mean(topk_sim, dim=1)  # (B, C)
        else:
            # EMOTIC Image: (B, N, D)
            patch_norm = F.normalize(patch_feats.float(), p=2, dim=-1)  # (B, N, D)
            
            # Tính tương quan từng patch với text prompt: (B, N, D) x (C, D) -> (B, N, C)
            patch_sim = torch.einsum('bnd,cd->bnc', patch_norm, text_norm)  # (B, N, C)
            
            if self.use_class_discriminative and C > 1:
                # Triệt tiêu nền bằng cách trừ mean across classes
                mean_class_sim = patch_sim.mean(dim=-1, keepdim=True)  # (B, N, 1)
                disc_sim = F.relu(patch_sim - mean_class_sim)           # (B, N, C)
                k_eff = min(self.top_k, disc_sim.size(1))
                _, topk_indices = torch.topk(disc_sim, k=k_eff, dim=1) # (B, k, C)
                # Lấy giá trị cosine similarity thực tế tại các vị trí patch giải phẫu đã lọc sạch nền
                topk_sim = torch.gather(patch_sim, dim=1, index=topk_indices) # (B, k, C)
            else:
                k_eff = min(self.top_k, patch_sim.size(1))
                topk_sim, _ = torch.topk(patch_sim, k=k_eff, dim=1)  # (B, k, C)
                
            s_local = torch.mean(topk_sim, dim=1)  # (B, C)

        # 3. Kết hợp Global và Sparse Local theo công thức MPA-FER
        logits = (s_global + self.alpha_local * s_local) / self.temperature
        return logits
