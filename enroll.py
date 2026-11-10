"""
enroll.py
Module Đăng ký người mới (One-shot Enrollment) & Lưu trữ Pre-computed Encodings:
Bám sát Slide trang 7 và 26:
- Chỉ cần 1 ảnh duy nhất của người mới (One-shot Learning).
- Trích xuất sẵn vector đặc trưng 128 chiều f(x) và lưu vào file database (Pre-computed Encodings).
- Không cần phải train lại mạng nơ-ron mỗi khi có thành viên mới!
"""

import os
import sys
import pickle
import argparse
import cv2
import numpy as np
import torch
from PIL import Image

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi UnicodeEncodeError
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from models import get_embedding_model
from dataset import get_transforms


# Khởi tạo bộ phát hiện khuôn mặt mặc định của OpenCV (Haar Cascade)
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)


def detect_and_crop_face(image_bgr_or_rgb, is_rgb=False):
    """
    Phát hiện và cắt khuôn mặt lớn nhất trong ảnh.
    Nếu không tìm thấy khuôn mặt rõ ràng, giữ nguyên ảnh gốc.
    """
    if is_rgb:
        img_rgb = image_bgr_or_rgb
        img_gray = cv2.cvtColor(image_bgr_or_rgb, cv2.COLOR_RGB2GRAY)
    else:
        img_rgb = cv2.cvtColor(image_bgr_or_rgb, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(image_bgr_or_rgb, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

    if len(faces) == 0:
        # Không tìm thấy mặt thì trả về ảnh toàn cảnh đã convert sang RGB
        return Image.fromarray(img_rgb), None

    # Lấy khuôn mặt có diện tích (w * h) lớn nhất
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    # Thêm margin biên nhỏ xung quanh mặt
    margin_w = int(w * 0.1)
    margin_h = int(h * 0.1)
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(img_rgb.shape[1], x + w + margin_w)
    y2 = min(img_rgb.shape[0], y + h + margin_h)

    face_crop = img_rgb[y1:y2, x1:x2]
    return Image.fromarray(face_crop), (x, y, w, h)


def load_model(weights_path=None, device='cpu'):
    """Tải mạng trích xuất embedding 128 chiều"""
    model = get_embedding_model(backbone_name='resnet18', embedding_dim=128, pretrained=True)
    if weights_path and os.path.exists(weights_path):
        print(f"[*] Đang tải trọng số từ: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=True)
    else:
        print("[!] Không có weights_path hoặc file chưa tồn tại, sử dụng mô hình ResNet18 pretrained ban đầu.")
    model.to(device)
    model.eval()
    return model


def compute_embedding(model, pil_image, transform, device='cpu'):
    """Trích xuất vector 128-d từ 1 ảnh PIL"""
    tensor_img = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor_img).cpu().numpy().flatten()
    return embedding


def build_database(known_faces_dir='known_faces', weights_path=None, db_output='face_db.pkl'):
    """
    Quét thư mục known_faces/:
    Mỗi file ảnh hoặc thư mục mang tên người đó (VD: 'NguyenVanA.jpg' hoặc 'NguyenVanA/1.jpg').
    Tính toán sẵn vector 128 chiều và lưu vào file face_db.pkl.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(weights_path=weights_path, device=device)
    transform = get_transforms(is_train=False)

    database = {}
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir, exist_ok=True)
        print(f"[!] Thư mục {known_faces_dir} chưa có ảnh. Hãy đặt ảnh vào thư mục này để đăng ký.")
        return database

    print(f"[*] Bắt đầu trích xuất Pre-computed Encodings từ thư mục: {known_faces_dir}")
    entries = sorted(os.listdir(known_faces_dir))

    for entry in entries:
        full_path = os.path.join(known_faces_dir, entry)

        if os.path.isfile(full_path) and entry.lower().endswith(('.jpg', '.png', '.jpeg')):
            name = os.path.splitext(entry)[0]
            img_bgr = cv2.imread(full_path)
            if img_bgr is None:
                continue
            face_pil, _ = detect_and_crop_face(img_bgr)
            emb = compute_embedding(model, face_pil, transform, device=device)
            database[name] = emb
            print(f"   [+] Đã đăng ký thành công: {name} (1 vector 128 chiều)")

        elif os.path.isdir(full_path):
            name = entry
            embeddings = []
            for img_file in os.listdir(full_path):
                if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_bgr = cv2.imread(os.path.join(full_path, img_file))
                    if img_bgr is not None:
                        face_pil, _ = detect_and_crop_face(img_bgr)
                        emb = compute_embedding(model, face_pil, transform, device=device)
                        embeddings.append(emb)
            if len(embeddings) > 0:
                # Nếu có nhiều ảnh, lấy vector trung bình đại diện cho người đó
                mean_emb = np.mean(embeddings, axis=0)
                mean_emb = mean_emb / np.linalg.norm(mean_emb)  # chuẩn hóa lại L2
                database[name] = mean_emb
                print(f"   [+] Đã đăng ký thành công: {name} ({len(embeddings)} ảnh -> vector trung bình)")

    with open(db_output, 'wb') as f:
        pickle.dump(database, f)

    print(f"[✓] Đã lưu cơ sở dữ liệu khuôn mặt với {len(database)} người vào: {db_output}")
    return database


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Đăng ký khuôn mặt và lưu Pre-computed Encodings")
    parser.add_argument('--known_dir', type=str, default='known_faces', help="Thư mục chứa ảnh thành viên")
    parser.add_argument('--weights', type=str, default='checkpoints/siamese_triplet.pth', help="File weights mô hình")
    parser.add_argument('--output', type=str, default='face_db.pkl', help="File kết quả lưu database")
    args = parser.parse_args()

    build_database(known_faces_dir=args.known_dir, weights_path=args.weights, db_output=args.output)
