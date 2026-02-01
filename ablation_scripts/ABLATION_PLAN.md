# Kế hoạch Thực Nghiệm Ablation Study

Dưới đây là danh sách 16 kịch bản thí nghiệm (Experiments) nhằm đánh giá hiệu quả của từng thành phần trong mô hình đề xuất.

| STT | Tên Script | Mục Đích (Objective) | Thay Đổi Chính (vs Full) | Ý Nghĩa Kỳ Vọng |
| :--- | :--- | :--- | :--- | :--- |
| **1** | `exp1_full_method.sh` | **Proposed Method (SOTA)** | Full Options (Chuẩn) | Đạt kết quả cao nhất, làm mốc so sánh. |
| **2** | `exp2_loss_no_ldl.sh` | Đánh giá **Semantic LDL Loss** | Tắt `--use-ldl` | Chứng minh học phân bố nhãn tốt hơn nhãn cứng. |
| **3** | `exp2_loss_no_aux.sh` | Đánh giá **Auxiliary Losses** | `--lambda_mi 0.0, --lambda_dc 0.0`| Chứng minh tối ưu không gian Text giúp tăng độ chính xác. |
| **4** | `exp3_arch_no_adapter.sh` | Đánh giá **Face Adapter** | `--use-adapter False` | Chứng minh Adapter giúp tinh chỉnh đặc trưng khuôn mặt. |
| **5** | `exp3_arch_cls_token.sh` | Đánh giá **Attention Pooling** | `--temporal-type cls` | Chứng minh cơ chế Attn-Pool tốt hơn CLS Token (Baseline). |
| **6** | `exp4_data_no_sampler.sh` | Đánh giá **Imbalance Handling** | Tắt `--use-weighted-sampler` | Chứng minh giải quyết mất cân bằng dữ liệu đầu vào. |
| **7** | `exp4_data_no_mixup.sh` | Đánh giá **Data Augmentation** | `--mixup-alpha 0.0` | Chứng minh Mixup giúp chống Overfitting. |
| **8** | `exp5_prompt_no_ensemble.sh` | Đánh giá **Prompt Ensemble** | `--text-type class_descriptor` | Chứng minh Ensemble tốt hơn Prompt đơn lẻ. |
| **9** | `exp6_no_moco.sh` | Đánh giá **Contrastive Learning** | Tắt `--use-moco` | Chứng minh MoCo giúp học đặc trưng bền vững hơn. |
| **10** | `exp7_no_slerp.sh` | Đánh giá **SLERP (IEC)** | `--slerp-weight 0.0` | Chứng minh IEC giúp cá thể hoá mô tả video. |
| **11** | `exp8_backbone_vit_b32.sh` | Đánh giá **Backbone Size** | `--clip-path ViT-B/32` | So sánh ViT-B/16 vs ViT-B/32. |
| **12** | `exp9_temporal_segments_8.sh` | Đánh giá **Temporal Context (8)** | `--num-segments 8` | Kiểm tra ảnh hưởng của độ dài chuỗi frame (Ngắn). |
| **13** | `exp10_temporal_segments_32.sh`| Đánh giá **Temporal Context (32)**| `--num-segments 32` | Kiểm tra ảnh hưởng của độ dài chuỗi frame (Dài). |
| **14** | `exp11_prompt_class_names.sh` | Đánh giá **Prompt Engineering** | `--text-type class_names` | So sánh với Prompt thô (chỉ tên lớp). |
| **15** | `exp12_context_length_4.sh` | Đánh giá **Context Length (4)** | `--contexts-number 4` | Kiểm tra ảnh hưởng của số lượng vector ngữ cảnh (Ít). |
| **16** | `exp13_context_length_16.sh` | Đánh giá **Context Length (16)** | `--contexts-number 16` | Kiểm tra ảnh hưởng của số lượng vector ngữ cảnh (Nhiều). |

## Hướng dẫn chạy

1.  Di chuyển vào thư mục script:
    ```bash
    cd ablation_scripts
    ```

2.  Chạy lần lượt từng file (ví dụ):
    ```bash
    bash exp1_full_method.sh
    ```
