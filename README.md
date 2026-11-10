# 🎓 HƯỚNG DẪN DỰ ÁN FACE RECOGNITION (DÀNH CHO NHÓM 4 NGƯỜI)

Dự án này hiện thực hóa toàn bộ các kiến thức học thuật từ bài giảng Face Recognition:
1. **One-shot learning:** Nhận diện người mới từ 1 ảnh duy nhất mà không cần train lại mạng.
2. **Siamese Network:** Mạng nơ-ron song sinh với chung trọng số (shared weights).
3. **Cách 1 - Binary Classification (Slide trang 23-28):** Đưa hiệu số $|f(x_1) - f(x_2)|$ qua Logistic Regression $\hat{y} = \sigma(W \cdot \text{diff} + b)$.
4. **Cách 2 - Triplet Loss & Hard Mining (Slide trang 13-22):** Huấn luyện với bộ ba $(A, P, N)$ và Margin $\alpha$.
5. **Pre-computed Encodings (Slide trang 26):** Lưu trước vector đặc trưng 128 chiều để nhận diện thời gian thực.

---

## 📁 CẤU TRÚC THƯ MỤC DỰ ÁN

```text
facedetect/
├── notebook_colab_training.ipynb  # File Notebook huấn luyện toàn bộ trên Google Colab
├── models.py                      # Định nghĩa mạng Siamese (Backbone + 128-D Embedding + Binary Head)
├── loss.py                        # Triplet Loss & Hard Triplet Mining
├── dataset.py                     # Bộ tạo cặp ảnh (Pairs) & bộ ba (Triplets)
├── train_binary.py                # Script train Cách 1: Binary Classification
├── train_triplet.py               # Script train Cách 2: Triplet Loss
├── enroll.py                      # Module Đăng ký 1 ảnh (One-shot) -> Lưu face_db.pkl
├── recognize.py                   # Nhận diện thời gian thực qua Webcam hoặc File ảnh
├── known_faces/                   # Thư mục chứa ảnh các thành viên để đăng ký vào hệ thống
├── checkpoints/                   # Thư mục lưu trọng số sau khi train (.pth)
└── requirements.txt               # Các thư viện cần thiết
```

---

## 👥 PHÂN CHIA TRÁCH NHIỆM CHO 4 THÀNH VIÊN KHI BÁO CÁO

| Thành viên | File phụ trách | Nội dung lý thuyết trong slide |
| :--- | :--- | :--- |
| **Thành viên 1** | `dataset.py` | - Khái niệm Face Verification ($1:1$) vs Face Recognition ($1:K$).<br>- Bài toán One-shot learning & tại sao Softmax thất bại.<br>- Cách tạo cặp ảnh Cùng/Khác người và bộ ba $(A, P, N)$. |
| **Thành viên 2** | `models.py` & `train_binary.py` | - Kiến trúc mạng Siamese Network.<br>- Phương pháp Binary Classification: $\hat{y} = \sigma(\sum w_k \|f(x_1)_k - f(x_2)_k\| + b)$.<br>- Quá trình huấn luyện với Binary Cross Entropy Loss. |
| **Thành viên 3** | `loss.py` & `train_triplet.py` | - Công thức Triplet Loss: $\mathcal{L} = \max(\|f(A)-f(P)\|^2 - \|f(A)-f(N)\|^2 + \alpha, 0)$.<br>- Ý nghĩa của Margin $\alpha = 0.2$.<br>- Hard Triplet Mining: Tại sao chọn ngẫu nhiên thì mô hình học chậm. |
| **Thành viên 4** | `enroll.py` & `recognize.py` | - Khái niệm Pre-computed Encodings (Slide 26).<br>- Cách tính khoảng cách Euclidean/Cosine & chọn ngưỡng $\tau$.<br>- Chạy Demo Webcam trực tiếp nhận diện các thành viên nhóm. |

---

## ⚡ BƯỚC 1: HUẤN LUYỆN TRÊN GOOGLE COLAB (RẤT ĐƠN GIẢN)

1. Mở trình duyệt, truy cập: [https://colab.research.google.com/](https://colab.research.google.com/)
2. Chọn **Upload** $\to$ Chọn file [`notebook_colab_training.ipynb`](./notebook_colab_training.ipynb).
3. Trên thanh menu Colab: Chọn **Runtime** $\to$ **Change runtime type** $\to$ Chọn **T4 GPU** $\to$ **Save**.
4. Chạy tuần tự từng cell (hoặc bấm **Runtime $\to$ Run all**):
   - Notebook sẽ tự động tải dữ liệu khuôn mặt LFW.
   - Huấn luyện mô hình Binary Classification và Triplet Loss.
   - Vẽ biểu đồ Loss, khoảng cách $d(A, P)$, $d(A, N)$.
   - Vẽ không gian đặc trưng gom cụm bằng biểu đồ **t-SNE**.
   - Demo One-shot recognition trực quan ngay trong notebook.
   - Tự động xuất và tải về 2 file:
     + `siamese_triplet.pth` (trọng số mô hình đã train).
     + `face_db.pkl` (cơ sở dữ liệu vector mẫu).

---

## 💻 BƯỚC 2: CHẠY DEMO WEBCAM TRÊN MÁY TÍNH CÁ NHÂN

Sau khi tải file `siamese_triplet.pth` từ Colab về máy:

### 1. Cài đặt thư viện trên máy cá nhân:
```bash
pip install -r requirements.txt
```

### 2. Đặt trọng số đã train vào thư mục `checkpoints/`:
Chuyển file `siamese_triplet.pth` vào thư mục `checkpoints/siamese_triplet.pth`.

### 3. Đăng ký khuôn mặt của 4 thành viên trong nhóm (One-shot Learning):
Chụp mỗi thành viên **1 tấm ảnh rõ mặt**, đặt vào thư mục `known_faces/`:
```text
known_faces/
├── ThanhVien1.jpg
├── ThanhVien2.jpg
├── ThanhVien3.jpg
└── ThanhVien4.jpg
```

Chạy lệnh trích xuất vector (Pre-computed Encodings):
```bash
python enroll.py --known_dir known_faces --weights checkpoints/siamese_triplet.pth --output face_db.pkl
```

### 4. Bật Webcam nhận diện trực tiếp:
```bash
python recognize.py --db face_db.pkl --weights checkpoints/siamese_triplet.pth --threshold 0.65 --webcam
```
- Khi webcam bật lên:
  - Nếu là thành viên nhóm: Hiện khung màu **xanh lá cây**, hiển thị Tên và khoảng cách $d$.
  - Nếu là người lạ: Hiện khung màu **đỏ**, hiển thị **Unknown**.
  - Nhấn phím `q` trên cửa sổ camera để thoát.

---

## ❓ CÂU HỎI VẤN ĐÁP GIẢNG VIÊN THƯỜNG HỎI & CÁCH TRẢ LỜI

1. **Q:** *Tại sao trong bài toán nhận diện khuôn mặt người ta không dùng hàm Softmax phân loại $K$ người?*  
   **A:** Vì bài toán gặp vấn đề **One-shot learning**. Trong thực tế, dữ liệu mỗi người thường chỉ có 1 hoặc rất ít ảnh. Nếu dùng Softmax, mỗi khi thêm nhân viên/học sinh mới, ta phải thay đổi cấu trúc mạng và huấn luyện lại từ đầu. Thay vào đó, mạng Siamese học một hàm khoảng cách $d(x_1, x_2)$ để so sánh sự giống nhau.

2. **Q:** *Ý nghĩa của Margin $\alpha$ trong hàm Triplet Loss là gì?*  
   **A:** Nếu không có $\alpha$, mạng có thể đưa ra nghiệm tầm thường (trivial solution) là triệt tiêu tất cả $f(x) = 0$. Margin $\alpha$ là khoảng cách an toàn tối thiểu bắt buộc khoảng cách giữa người khác $d(A, N)$ phải lớn hơn khoảng cách cùng người $d(A, P)$ ít nhất là $\alpha$.

3. **Q:** *Thế nào là "Hard Triplets" và tại sao cần Hard Triplet Mining?*  
   **A:** Khi chọn bộ ba ngẫu nhiên, phần lớn các mẫu âm đều rất khác nhau nên điều kiện $d(A, P) + \alpha \le d(A, N)$ dễ dàng thỏa mãn, khiến hàm loss bằng 0 và gradient không truyền về để cập nhật mạng. Hard Triplet Mining chọn các mẫu âm khó (người khác nhưng có nét tương đồng) để buộc mạng nơ-ron phải học các đặc trưng tinh vi nhất.

4. **Q:** *Pre-computed Encodings mang lại lợi ích gì trong hệ thống thực tế?*  
   **A:** Khi có người đứng trước camera, hệ thống chỉ cần chạy mạng CNN đúng 1 lần cho khuôn mặt đó để lấy vector 128 chiều, sau đó so khớp bằng phép tính đại số đơn giản với các vector đã tính sẵn trong database. Điều này giúp tốc độ nhận diện đạt hàng nghìn người trong vài mili-giây mà không bị trễ FPS.
