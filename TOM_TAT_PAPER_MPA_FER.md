# 📘 BÁO CÁO PHÂN TÍCH CHUYÊN SÂU BÀI BÁO KHOA HỌC
## Multimodal Prompt Alignment for Facial Expression Recognition (MPA-FER)

* **Tác giả:** Fuyan Ma, Yiran He, Bin Sun, Shutao Li
* **Định danh arXiv:** [arXiv:2506.21017](https://arxiv.org/abs/2506.21017) (v1, Tháng 06/2025)
* **Lĩnh vực:** Computer Vision (cs.CV), Artificial Intelligence (cs.AI), Vision-Language Models (VLMs), Facial Expression Recognition (FER)

---

## 1. 📌 TỔNG QUAN & BÀI TOÁN NGHIÊN CỨU (CORE PROBLEM)

### 1.1. Thách thức trong bài toán FER thực tế (In-the-wild FER)
Nhận dạng biểu cảm khuôn mặt trong môi trường không ràng buộc (in-the-wild) là bài toán then chốt trong tương tác người-máy (HRI), y tế thông minh và tâm lý học. Tuy nhiên, dữ liệu thực tế gặp phải các thách thức lớn:
* **Độ biến thiên biểu cảm tinh tế (Subtle nuances):** Sự khác biệt giữa các sắc thái cảm xúc (ví dụ: tức giận nhẹ vs. khinh miệt, ngạc nhiên vs. sợ hãi) thường chỉ thể hiện qua vài cơ mặt vi mô (khóe môi, nếp nhăn mày, mí mắt).
* **Nhiễu môi trường:** Góc quay nghiêng, vật cản (khẩu trang, kính, tay), điều kiện ánh sáng thay đổi và vùng nền không liên quan làm phân tán sự tập trung của mô hình.

### 1.2. Hạn chế của các phương pháp VLM / CLIP hiện có
Khi ứng dụng các mô hình nền tảng Thị giác - Ngôn ngữ (như CLIP) vào FER qua kỹ thuật học prompt (Prompt Learning):
1. **Prompt văn bản quá thô sơ (Coarse Prompts):** Các template thủ công như `"a photo of [class]"` chỉ đại diện cho một danh mục trừu tượng, hoàn toàn thiếu mô tả về mặt thị giác chi tiết để phân biệt các chuyển động cơ mặt đặc thù.
2. **Nguy cơ Overfit khi Fine-tune:** Hầu hết các công trình VLM-FER trước đây đều fine-tune toàn bộ hoặc một phần Image Encoder. Cách tiếp cận này làm mô hình dễ bị overfit trên tập dữ liệu FER có kích thước vừa phải, đồng thời phá hủy không gian biểu diễn khái quát hóa (generalization) mạnh mẽ của CLIP gốc và tốn kém tài nguyên tính toán.

---

## 2. 🚀 ĐÓNG GÓP KỸ THUẬT CỐT LÕI (KEY CONTRIBUTIONS)

1. **Khái niệm Frozen CLIP Backbone cho FER:** Đề xuất một mô hình FER giữ **đóng băng 100% trọng số** của cả Image Encoder và Text Encoder, chỉ tối ưu một lượng tham số prompt cực nhỏ (**0.218 MB - 0.443 MB**).
2. **Cơ chế Soft-Hard Prompt Alignment (SPA):** Sử dụng LLM (ChatGPT-3.5) sinh tri thức mô tả đặc trưng cơ mặt chi tiết (Hard Prompts đa tầng), sau đó căn chỉnh để "bơm" tri thức này vào các Soft Prompts học được ở cả cấp độ token và cấp độ câu (prompt-level).
3. **Cơ chế Prototype-guided Visual Alignment (PVA):** Tính toán các vector Class Prototypes từ không gian đặc trưng CLIP đóng băng để làm "mỏ neo", ràng buộc các visual prompts không bị trôi dạt và bảo toàn khả năng khái quát hóa.
4. **Mô-đun Cross-modal Global-Local Alignment (CGLA):** Tách biệt và tập trung vào top-$k$ vùng đặc trưng cục bộ liên quan mật thiết đến biểu cảm (mắt, mũi, miệng), loại bỏ triệt để nhiễu nền.
5. **Thiết lập đỉnh cao mới (New SOTA):** Vượt qua tất cả các phương pháp CNN, ViT và VLM fine-tuning trên cả 3 tập dữ liệu benchmark phổ biến: **RAF-DB, FERPlus, AffectNet-7 và AffectNet-8**.

---

## 3. 🔬 BÓC TÁCH PHƯƠNG PHÁP LUẬN (METHODOLOGY)

### 3.1. Nhánh Văn bản: Sinh và Căn chỉnh Prompt (SPA)
* **Tạo Hard Prompt từ LLM:** Query ChatGPT bằng câu lệnh:  
  *“What are the most useful visual features to distinguish the facial expression of [class]?”*  
  Kết hợp template chung + nhãn lớp + mô tả từ LLM thành prompt đa tầng $\bm{t}_c^*$.
* **Căn chỉnh cấp Token (Token-level Alignment):** Huấn luyện soft prompt $\bm{t}_c$ sao cho được phân loại chính xác trong không gian embedding theo trọng số của hard prompt:
  $$\mathcal{L}_{ta} = - \sum_{d=1}^C y_d \log \frac{\exp(\text{sim}(\bm{t}_c, \bm{t}_d^*) / \tau)}{\sum_{j=1}^C \exp(\text{sim}(\bm{t}_c, \bm{t}_j^*) / \tau)}$$
* **Căn chỉnh cấp Prompt (Prompt-level Alignment):** Đo độ lệch embedding sau khi qua Text Encoder $\theta$:
  $$\mathcal{L}_{pa} = \|\theta(\bm{t}_k) - \theta(\bm{t}_k^*)\|_1$$
* **Tổng mất mát nhánh Text:**
  $$\mathcal{L}_t = \mathcal{L}_{ta} + \mathcal{L}_{pa}$$

### 3.2. Nhánh Thị giác: Visual Prompts & Prototype Anchoring (PVA)
* **Deep Visual Prompting:** Tại mỗi layer $l \in [1, K]$ của Image Encoder $\phi$, chèn $N_p$ token visual prompt học được:
  $$[\bm{z}^l, \_] = \phi^l([\bm{z}^{l-1}, \{p_i^l\}_{i=1}^{N_p}])$$
* **Class Prototypes Anchoring:** Định nghĩa prototype $\bm{p}_c$ của lớp $c$ bằng giá trị trung bình (mean) feature toàn cục của tập con ảnh từ frozen CLIP encoder:
  $$\bm{p}_c = \frac{1}{N_{subset}^c} \sum_{i \in \mathcal{D}_{subset}^c} \bm{z}_{i, \text{frozen}}^g$$
* **Mất mát neo Prototype:**
  $$\mathcal{L}_v = \mathcal{M}(\bm{z}^g, \bm{p}_c) \quad (\text{dùng khoảng cách } L_1 \text{ hoặc Cosine Similarity})$$

### 3.3. Căn chỉnh Đa phương thức Toàn cục - Cục bộ (CGLA)
Tính độ tương đồng kết hợp giữa Text Feature $\theta(\bm{t}_d)$ với:
* **Global visual feature:** $S_g = \text{sim}(\bm{z}^g, \theta(\bm{t}_d))$
* **Top-$k$ Sparse Local features:** Chỉ lấy $k$ patches có điểm tương đồng cao nhất (vùng mắt, mày, miệng), loại bỏ nền:
  $$S_{local} = \frac{1}{k} \sum_{i=1}^{N_l} \mathbb{I}_{\text{top-}k}(i) \cdot \text{sim}(\bm{z}_i^l, \theta(\bm{t}_d))$$
* **Logits & Cross-Entropy Loss:**
  $$\text{Logits} = S_g + S_{local} \quad \longrightarrow \quad \mathcal{L}_{v\_t} = \text{CE}(\text{Logits}, y)$$

### 3.4. Hàm tối ưu tổng thể (Total Loss)
$$\mathcal{L}_{total} = \mathcal{L}_{v\_t} + \beta \mathcal{L}_t + \gamma \mathcal{L}_v \quad (\text{trong bài báo đặt } \beta = \gamma = 1)$$

---

## 4. 📊 SƠ ĐỒ KIẾN TRÚC MÔ HÌNH (SCIENTIFIC SCHEMATICS)

```mermaid
flowchart TD
    subgraph TEXT_BRANCH ["Nhánh Văn Bản (Text Branch)"]
        LLM["LLM (ChatGPT-3.5)<br>Sinh tri thức mô tả nét mặt"] --> HP["Multi-Granularity Hard Prompts (t*)"]
        SP["Trainable Soft Prompts (t)"]
        
        HP -->|Token Embeddings| TA["Token-level Alignment (L_ta)"]
        SP -->|Token Embeddings| TA
        
        HP --> TE["Frozen CLIP Text Encoder θ"]
        SP --> TE
        
        TE --> PA["Prompt-level Alignment (L_pa)"]
        TA & PA --> LT["Loss Text: L_t = L_ta + L_pa"]
    end

    subgraph VISION_BRANCH ["Nhánh Thị Giác (Visual Branch)"]
        IMG["Ảnh khuôn mặt đầu vào"] --> IE["Frozen CLIP Image Encoder φ<br>(ViT-B/16 hoặc ViT-L/14)"]
        VP["Trainable Visual Prompts {p_i^l}"] -->|Inject vào từng Layer| IE
        
        IE --> ZG["Global Token (z^g)"]
        IE --> ZL["Local Patch Tokens {z_i^l}"]
        
        PROTO["Class Prototypes (p_c)<br>(Mean Frozen CLIP Features)"] --> PVA["Prototype Alignment Loss (L_v)"]
        ZG --> PVA
    end

    subgraph CGLA_MODULE ["Căn Chỉnh Đa Phương Thức Cục Bộ - Toàn Cục (CGLA)"]
        TE -->|Text Features θ(t_d)| MATCH["Cross-Modal Fusion"]
        ZG -->|Global Sim S_g| MATCH
        ZL -->|Lọc Top-k Patches liên quan| SPARSE["Sparse Local Sim S_local"]
        SPARSE --> MATCH
        
        MATCH --> LOGITS["Logits = S_g + S_local"]
        LOGITS --> LVT["Cross-Entropy Loss (L_v_t)"]
    end

    subgraph OBJECTIVE ["Hàm Mất Mát Toàn Diện"]
        LT & PVA & LVT --> TOTAL_LOSS["L_total = L_v_t + β·L_t + γ·L_v"]
    end

    classDef frozen fill:#e2e8f0,stroke:#64748b,stroke-width:2px;
    classDef learnable fill:#fef3c7,stroke:#d97706,stroke-width:2px;
    classDef loss fill:#fee2e2,stroke:#ef4444,stroke-width:2px;
    
    class TE,IE,PROTO frozen;
    class SP,VP learnable;
    class LT,PVA,LVT,TOTAL_LOSS loss;
```

---

## 5. 📈 KẾT QUẢ THỰC NGHIỆM & ĐÁNH GIÁ SOTA

### 5.1. Thí nghiệm bóc tách thành phần (Ablation on Components)
Đánh giá mức độ đóng góp của từng module trên tập **RAF-DB** và **AffectNet-7**:
* **Baseline (Frozen CLIP + Contrastive):** Độ chính xác cơ bản.
* **+ Visual Prompts (VP):** Tăng **+2.32%** (RAF-DB), **+1.63%** (AffectNet-7) $\rightarrow$ Thích nghi nhanh với phân phối ảnh khuôn mặt.
* **+ Prototype Alignment (PVA):** Tăng thêm **+0.95%** (RAF-DB), **+1.12%** (AffectNet-7) $\rightarrow$ Đóng vai trò mỏ neo, ngăn chặn overfit.
* **+ Soft-Hard Alignment (SPA):** Tăng thêm **+1.76%** (RAF-DB), **+1.56%** (AffectNet-7) $\rightarrow$ Bổ sung tri thức mô tả chi tiết từ LLM.
* **+ Global-Local Alignment (CGLA):** Tăng thêm **+1.33%** (RAF-DB), **+1.27%** (AffectNet-7) $\rightarrow$ Tập trung vào các cơ mặt trọng yếu, loại bỏ nhiễu nền.

### 5.2. Hiệu quả của các loại Prompt văn bản
* **Cấu hình 1:** `"a photo of [class]"` (Chuẩn CoOp thông thường).
* **Cấu hình 2:** `"a photo of a person making a facial expression of [class]"`.
* **Cấu hình 3:** Cấu hình 2 kèm *mô tả chi tiết từ LLM*.
* **Kết luận:** Cấu hình 3 khi kết hợp cơ chế SPA đạt độ chính xác cao nhất (**92.51%** trên RAF-DB và **91.15%** trên FERPlus).

### 5.3. So sánh với các phương pháp State-of-the-Art (SOTA)

| Mô hình | Phương pháp | RAF-DB | FERPlus | AffectNet-7 | AffectNet-8 | Tham số học thêm |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **RAN** *(Wang et al.)* | CNN | 86.90% | 88.55% | 56.40% | 52.97% | Toàn bộ model |
| **TransFER** *(Xue et al.)* | ViT | 90.91% | 90.83% | 66.23% | 61.24% | Toàn bộ model |
| **APViT** *(Ma et al.)* | ViT | 91.98% | 90.86% | 66.91% | 61.87% | Toàn bộ model |
| **CLIPER** *(Li et al.)* | CLIP Fine-tuned | 92.47% | 90.72% | 67.31% | 62.30% | Fine-tune nặng |
| **CEPrompt** *(Zhou et al.)* | CLIP Prompting | 92.44% | 90.89% | 67.49% | 62.62% | Fine-tune một phần |
| **MPA-FER (ViT-B/16)** | **Frozen CLIP** | **92.51%** | **91.15%** | **67.85%** | **62.80%** | **0.218 MB** |
| **MPA-FER (ViT-L/14)** | **Frozen CLIP** | **93.74%** | **91.81%** | **68.89%** | **63.74%** | **0.443 MB** |

---

## 6. ⚖️ ĐÁNH GIÁ PHẢN BIỆN (CRITICAL REVIEW & LIMITATIONS)

### 🌟 Ưu điểm nổi bật
* **Cực kỳ nhẹ về mặt tính toán:** Huấn luyện hoàn tất trên 1 card GPU NVIDIA V100 (32GB), số lượng tham số cần cập nhật dưới 0.5 MB.
* **Bảo toàn tính khái quát của CLIP:** Tránh được bẫy overfit kinh điển trong bài toán FER nhờ class prototypes và frozen backbone.
* **Tính trực quan và giải thích được (Explainability):** Attention map trực quan chứng minh mô hình chỉ tập trung vào các điểm mấu chốt (mắt, miệng, mày) thay vì bị xao nhãng bởi tóc hay nền ảnh.

### ⚠️ Điểm hạn chế (Limitations)
* **Phụ thuộc vào chất lượng mô tả của LLM:** Hard prompt được tạo tĩnh một lần bởi ChatGPT-3.5; nếu LLM đưa ra mô tả sai lệch hoặc trùng lặp giữa các biểu cảm tương đồng, soft prompt sẽ bị định hướng sai.
* **Cơ chế Top-$k$ Patch tĩnh:** Tham số $k=16$ được cố định cho mọi ảnh, trong khi các khuôn mặt có tỉ lệ diện tích khác nhau trong ảnh (ảnh cận cảnh vs. ảnh toàn thân) có thể cần số lượng patch biểu cảm khác nhau.
* **Chưa khai thác chiều thời gian (Video FER):** Mô hình hiện tại chỉ thiết kế cho ảnh tĩnh (static images), chưa mở rộng cho các chuỗi khung hình video liên tục.
