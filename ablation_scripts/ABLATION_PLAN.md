# Kế hoạch Thực Nghiệm Ablation Study (Phiên bản Ổn định)

Dưới đây là danh sách 16 kịch bản thí nghiệm. Lưu ý: **Kiến trúc MoCo (Momentum Encoder)** được giữ lại trong bản Full để ổn định đặc trưng, nhưng **MoCo Loss** đã bị loại bỏ do gây mất ổn định trong quá trình huấn luyện.

| STT | Tên Script | Mục Đích (Objective) | Thay Đổi Chính (vs Full) | Ý Nghĩa Kỳ Vọng |
| :--- | :--- | :--- | :--- | :--- |
| **1** | `exp1_full_method.sh` | **Proposed Method (SOTA)** | Cấu hình đạt UAR cao nhất | Làm mốc so sánh chuẩn. |
| **2** | `exp2_loss_no_ldl.sh` | Đánh giá **Semantic LDL Loss** | Tắt `--use-ldl` | So sánh LDL vs CrossEntropy. |
| **3** | `exp2_loss_no_aux.sh` | Đánh giá **Auxiliary Losses** | `--lambda_mi 0, --lambda_dc 0` | Hiệu quả của MI & DC Loss. |
| **4** | `exp3_arch_no_adapter.sh` | Đánh giá **Face Adapter** | `--use-adapter False` | Vai trò của Adapter tinh chỉnh mặt. |
| **5** | `exp3_arch_cls_token.sh` | Đánh giá **Attention Pooling** | `--temporal-type cls` | Attn-Pool vs CLS Token. |
| **6** | `exp4_data_no_sampler.sh` | Đánh giá **Imbalance Handling** | Tắt `--use-weighted-sampler` | Tầm quan trọng của cân bằng dữ liệu. |
| **7** | `exp4_data_no_mixup.sh` | Đánh giá **Data Augmentation** | `--mixup-alpha 0.0` | Vai trò của Mixup chống Overfitting. |
| **8** | `exp5_prompt_no_ensemble.sh` | Đánh giá **Prompt Ensemble** | `--text-type class_descriptor` | Hiệu quả của việc dùng nhiều câu mô tả. |
| **9** | `exp6_no_moco` | Đánh giá **Momentum Architecture**| Tắt `--use-moco` | Chứng minh kiến trúc Momentum ổn định feature. |
| **10** | `exp7_no_slerp.sh` | Đánh giá **SLERP (IEC)** | `--slerp-weight 0.0` | Tầm quan trọng của nội suy đặc trưng. |
| **11** | `exp8_backbone_vit_b32.sh` | Đánh giá **Backbone Size** | `--clip-path ViT-B/32` | So sánh ViT-B/16 vs ViT-B/32. |
| **12** | `exp9_temporal_segments_8.sh` | Đánh giá **Temporal Context (8)** | `--num-segments 8` | Ảnh hưởng của chuỗi frame ngắn. |
| **13** | `exp10_temporal_segments_32.sh`| Đánh giá **Temporal Context (32)**| `--num-segments 32` | Ảnh hưởng của chuỗi frame dài. |
| **14** | `exp11_prompt_class_names.sh` | Đánh giá **Prompt Engineering** | `--text-type class_names` | Sức mạnh của thiết kế Prompt. |
| **15** | `exp12_context_length_4.sh` | Đánh giá **Context Length (4)** | `--contexts-number 4` | Hiệu quả của độ dài chuỗi ngữ cảnh. |
| **16** | `exp13_context_length_16.sh` | Đánh giá **Context Length (16)** | `--contexts-number 16` | Hiệu quả của độ dài chuỗi ngữ cảnh. |

## Quy trình chạy
1. Chạy `exp1` lấy chuẩn.
2. Chạy lần lượt các bản bỏ bớt tính năng (exp2 - exp13).
3. Tổng hợp kết quả WAR/UAR vào bảng so sánh.