# CO5085 — Học sâu và Ứng dụng trong Thị giác máy tính

_Deep Learning and Its applications in Computer Vision_

Trang GitHub Pages: [https://tieutrikiet.github.io/co5085_tieutrikiet_dl_cv/](https://tieutrikiet.github.io/co5085_tieutrikiet_dl_cv/)

## Tổng quan

| Thư mục | Nội dung                                               |
| ------- | ------------------------------------------------------ |
| `E1/`   | Phân loại ảnh: Softmax, MLP, CNN                       |
| `E2/`   | Multi-Head Self-Attention                              |
| `E3/`   | Mô hình chuỗi: LSTM, GRU                               |
| `A1/`   | So sánh CNN và Transformer trên Food-101               |
| `A2/`   | Phát hiện khuôn mặt với YOLO5Face và phân tích bài báo |

## Cài đặt

Phần dưới đây chỉ thiết lập môi trường cơ bản để chạy notebook, không bao gồm môi trường huấn luyện đầy đủ cho YOLO5Face.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision numpy matplotlib pandas scikit-learn pillow jupyter
jupyter lab
```

## Dữ liệu

Các bộ dữ liệu không được commit hay đi kèm trong repository.
