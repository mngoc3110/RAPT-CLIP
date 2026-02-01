
## Phân Tích và Gỡ Lỗi Hiện Tượng Sập Mô Hình (Model Collapse)

Trong quá trình thử nghiệm, mô hình gặp phải hiện tượng "sập" (model collapse), tức là chỉ có khả năng dự đoán một hoặc một vài lớp thay vì tất cả các lớp, dẫn đến chỉ số UAR (Unweighted Average Recall) rất thấp.

### Phân tích

Sau khi thực hiện một loạt các bước gỡ lỗi, bao gồm in thông tin dữ liệu đầu vào, kiểm tra ảnh mẫu và thử nghiệm với các learning rate khác nhau, chúng tôi đã so sánh script huấn luyện gặp sự cố (`train_final_fix.sh`) với một script baseline ổn định hơn (`train.sh`). Kết quả phân tích cho thấy sự khác biệt mấu chốt không nằm ở hàm loss `CrossEntropy` cơ bản, mà nằm ở **sự thiếu vắng các kỹ thuật điều chuẩn (regularization) và xử lý mất cân bằng dữ liệu** trong script gặp sự cố.

Script `train.sh` (baseline) có kết quả ổn định hơn vì nó áp dụng đồng thời các kỹ thuật sau:

1.  **`--use-weighted-sampler`**: Xử lý mất cân bằng lớp ở cấp độ dữ liệu. Thay vì thay đổi hàm loss, phương pháp này thay đổi cách lấy mẫu, đảm bảo mỗi batch dữ liệu mà mô hình thấy có sự phân bổ cân bằng hơn giữa các lớp. Cách tiếp cận này thường mang lại sự ổn định cao hơn so với việc chỉ thay đổi trọng số của hàm loss.

2.  **`--lambda_mi 0.2` và `--lambda_dc 0.3`**: Sử dụng hai hàm loss phụ (`MILoss` và `DCLoss`) để điều chuẩn không gian feature. Các loss này khuyến khích các feature mà mô hình học được phải vừa đa dạng, vừa giàu thông tin, giúp ngăn mô hình tìm đến các "đường tắt" (shortcut) và bị sập.

3.  **`--label-smoothing 0.02`**: Áp dụng kỹ thuật làm mịn nhãn, giúp mô hình không trở nên quá "tự tin" vào dự đoán của mình. Điều này cũng là một hình thức điều chuẩn hiệu quả để tránh overfitting và model collapse.

### Kết luận

Việc script `train_final_fix.sh` chỉ sử dụng `--class-balanced-loss` (thay đổi trọng số loss) mà không có các kỹ thuật bổ trợ trên đã khiến quá trình huấn luyện trở nên không ổn định và dẫn đến sập mô hình.

### Hướng giải quyết

Để khắc phục, chúng ta cần cấu hình lại script huấn luyện để bao gồm các kỹ thuật giúp ổn định quá trình học, tương tự như script baseline. Các bước đề xuất bao gồm:
-   Sử dụng `--use-weighted-sampler` thay cho `--class-balanced-loss`.
-   Bật lại các hàm loss phụ `lambda_mi` và `lambda_dc`.
-   Áp dụng `label-smoothing`.
-   Sử dụng một mức learning rate phù hợp và ổn định (ví dụ: `1e-5`).
