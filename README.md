# Asian Translator (Hệ Thống Dịch Máy AI)

## Giới thiệu
**Asian Translator** là một ứng dụng dịch máy toàn diện (full-stack Machine Translation web application) sử dụng mô hình ngôn ngữ lớn **NLLB-200** của Meta (thông qua Hugging Face Transformers). 
Dự án cung cấp giao diện người dùng thân thiện, kết nối với backend linh hoạt hỗ trợ chạy mô hình trực tiếp trên máy cục bộ hoặc thông qua API Server chuyên dụng trên GPU. Ngoài ra, hệ thống còn tích hợp tính năng Pivot Translation (Dịch qua ngôn ngữ trung gian) và thu thập dữ liệu RLHF (Reinforcement Learning from Human Feedback) phục vụ tinh chỉnh mô hình.

## Các tính năng chính
1. **Kiến trúc linh hoạt (Local & API Server)**:
   - **Chạy cục bộ**: Tải và chạy trực tiếp mô hình ngay trong ứng dụng web (`model_local.py`).
   - **Chạy qua API**: Tách biệt backend xử lý GPU (`gpu_server.py`) và ứng dụng web, ứng dụng web sẽ giao tiếp với backend thông qua API (`model_api.py`), giúp dễ dàng triển khai ở các môi trường phân tán hoặc máy chủ giới hạn tài nguyên.

2. **Dịch thuật Nâng cao**:
   - Tùy chỉnh các tham số sinh văn bản (Generation parameters): Cung cấp các thuật toán sinh văn bản đa dạng như Greedy Search, Beam Search, Sampling (cấu hình được Top-K, Top-P, Temperature).
   - **Pivot Translation**: Cho phép sử dụng một ngôn ngữ làm cầu nối trung gian (ví dụ Tiếng Anh) để cải thiện chất lượng dịch giữa các ngôn ngữ hiếm.
   - **Sliding Window Chunking**: Thuật toán chia nhỏ câu thông minh tự động (trong API Server) bảo toàn cấu trúc dấu câu và độ dài token tối đa, giúp hệ thống không bị tràn bộ nhớ khi dịch các đoạn văn bản dài.

3. **Luồng thu thập phản hồi người dùng (RLHF Workflow)**:
   - Hệ thống được thiết kế với một xác suất ngẫu nhiên sẽ tạo ra 2 bản dịch bằng các tham số mô hình khác nhau.
   - Người dùng có thể đánh giá và chọn bản dịch tốt hơn.
   - Dữ liệu phản hồi sẽ được tự động lưu vào tệp nhật ký `rlhf_data.jsonl` để làm bộ dữ liệu huấn luyện RLHF trong tương lai.

4. **Nghiên cứu & Tinh chỉnh mô hình (Core ML)**:
   - Bao gồm toàn bộ quy trình tinh chỉnh mô hình NLLB bằng phương pháp **Supervised Fine-Tuning (SFT)** với QDoRA (`Core/SFT_QDoRA_NLLB200.py`).
   - Thuật toán **Reinforcement Learning** cho dịch máy (`Core/RL_NLLB200.py`).
   - Các kịch bản và notebook đánh giá hiệu năng dịch thuật (`Core/Eval.ipynb`, `Core/Fine_Tranlation_Process.ipynb`).

## Cấu trúc thư mục
```text
HE THONG DỊCH/
├── app.py                   # Ứng dụng Web chính (Flask) xử lý giao diện và định tuyến
├── gpu_server.py            # API Server chạy mô hình NLLB trên GPU (Sử dụng Peft, Transformers)
├── model_api.py             # Lớp Client gọi API dịch từ ứng dụng web sang gpu_server
├── model_local.py           # Lớp tải và chạy mô hình trực tiếp trên ứng dụng
├── requirements.txt         # Danh sách các thư viện phụ thuộc của Python
├── rlhf_data.jsonl          # Dữ liệu thu thập từ phản hồi người dùng (RLHF)
├── test_client.py           # Kịch bản kiểm thử API nhanh
├── Core/                    # Mã nguồn cho quá trình Huấn luyện và Đánh giá Mô hình
│   ├── SFT_QDoRA_NLLB200.py # Code Fine-tuning mô hình với QDoRA/LoRA
│   ├── RL_NLLB200.py        # Code học tăng cường (RL) tối ưu hoá mô hình
│   ├── Eval.ipynb           # Notebook đánh giá chất lượng mô hình
│   └── Fine_Tranlation_Process.ipynb
├── static/                  # Các file tĩnh (style.css, script.js)
└── templates/               # Các trang HTML (Giao diện người dùng)
```

## Hướng dẫn Cài đặt

1. **Clone repository**:
   ```bash
   git clone https://github.com/NguyenPhuongDng/AI-Tranlator.git
   cd AI-Tranlator
   ```

2. **Cài đặt thư viện môi trường**:
   Khuyến khích bạn sử dụng môi trường ảo (virtual environment).
   ```bash
   pip install -r requirements.txt
   ```

## Hướng dẫn Sử dụng

Dự án cung cấp hai phương pháp tiếp cận để chạy mô hình dịch thuật:

### Cách 1: Chạy mô hình qua GPU API Server (Khuyến nghị cho Production)
Nếu máy chủ của bạn có GPU, bạn có thể khởi động riêng rẽ server dịch vụ:

1. Chạy API Server:
   ```bash
   python gpu_server.py
   ```
   *Mặc định server sẽ chạy tại cổng `50001`.*

2. Chạy ứng dụng Web (Client):
   Bật cấu hình chỉ định sử dụng mô hình API thay vì mô hình Local trong terminal:
   ```bash
   export USE_LOCAL_MODEL=False
   export TRANSLATION_API_URL=http://127.0.0.1:50001/api/translate
   python app.py
   ```

### Cách 2: Chạy trực tiếp Local
Mô hình sẽ được tải thẳng vào cùng process của ứng dụng Flask. Cách này tiện lợi khi chạy thử nghiệm nghiệm thu nghiệm thu trên máy cá nhân nhanh gọn:
```bash
export USE_LOCAL_MODEL=True
python app.py
```

Sau khi chạy xong `app.py`, hãy mở trình duyệt và truy cập `http://127.0.0.1:5000` để sử dụng dịch vụ.

## Công nghệ sử dụng
- **Backend**: Python, Flask
- **Học máy & Xử lý ngôn ngữ (ML/NLP)**: PyTorch, Hugging Face Transformers, Peft (LoRA/QDoRA)
- **Frontend**: HTML5, Vanilla CSS, JavaScript
- **Mô hình**: NLLB-200 (Meta)
