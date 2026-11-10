"""
recognize.py
Chương trình Nhận diện khuôn mặt thời gian thực (Webcam hoặc File ảnh):
Bám sát Slide trang 4, 7, 8:
1. Phát hiện tất cả khuôn mặt trong khung hình.
2. Trích xuất vector 128 chiều f(x).
3. So khớp với K người trong cơ sở dữ liệu đã tính sẵn (Pre-computed Encodings):
   d(img1, img2) = || f(img1) - f(img2) ||^2
4. Nếu d <= tau (ngưỡng) -> Trả về Tên người đó.
   Nếu d > tau -> Trả về "Unknown".
"""

import os
import sys
import time
import pickle
import argparse
import cv2
import numpy as np
import torch
from PIL import Image

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from models import get_embedding_model
from dataset import get_transforms
from enroll import detect_and_crop_face, load_model, compute_embedding, face_cascade


def recognize_face(embedding, database, threshold=0.6, metric='euclidean'):
    """
    So sánh vector đặc trưng hiện tại với tất cả vector trong database.
    database: dict {name: embedding_128d}
    Trả về: (best_name, min_distance, is_match)
    """
    if len(database) == 0:
        return "Unknown", 999.0, False

    best_name = "Unknown"
    min_dist = float('inf')

    for name, db_emb in database.items():
        if metric == 'euclidean':
            # Bình phương khoảng cách L2 Euclidean
            dist = np.sum((embedding - db_emb) ** 2)
        elif metric == 'cosine':
            # Khoảng cách Cosine: 1 - cosine_similarity
            dot_prod = np.dot(embedding, db_emb)
            dist = 1.0 - dot_prod
        else:
            dist = np.linalg.norm(embedding - db_emb)

        if dist < min_dist:
            min_dist = dist
            best_name = name

    # Kiểm tra điều kiện ngưỡng tau
    if min_dist <= threshold:
        return best_name, min_dist, True
    else:
        return "Unknown", min_dist, False


def crop_face_box(img_bgr, x, y, w, h):
    """Cắt khuôn mặt kèm viền an toàn đồng nhất với enroll.py"""
    margin_w = int(w * 0.1)
    margin_h = int(h * 0.1)
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(img_bgr.shape[1], x + w + margin_w)
    y2 = min(img_bgr.shape[0], y + h + margin_h)
    face_crop = img_bgr[y1:y2, x1:x2]
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face_rgb)


def run_recognition_on_image(image_path, model, database, transform, threshold=0.6, device='cpu', output_path='result.jpg'):
    """Chạy nhận diện trên 1 file ảnh tĩnh"""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[!] Không tìm thấy ảnh: {image_path}")
        return

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

    if len(faces) == 0:
        print("[!] Không tìm thấy khuôn mặt nào trong ảnh.")
        return

    print(f"[*] Tìm thấy {len(faces)} khuôn mặt trong ảnh.")
    for (x, y, w, h) in faces:
        face_pil = crop_face_box(img_bgr, x, y, w, h)
        emb = compute_embedding(model, face_pil, transform, device=device)
        name, dist, is_match = recognize_face(emb, database, threshold=threshold)

        # Vẽ bounding box và nhãn
        color = (0, 255, 0) if is_match else (0, 0, 255)
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), color, 2)
        label = f"{name} (d={dist:.2f})"
        cv2.putText(img_bgr, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        print(f"   --> Khuôn mặt tại ({x}, {y}): {label}")

    cv2.imwrite(output_path, img_bgr)
    print(f"[✓] Đã lưu ảnh kết quả vào: {output_path}")


def run_webcam(model, database, transform, threshold=0.6, device='cpu'):
    """Bật webcam trực tiếp để nhận diện khuôn mặt theo thời gian thực"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Không thể mở Webcam. Hãy kiểm tra lại kết nối camera.")
        return

    print("[*] Đang mở Webcam... Nhấn phím 'q' trên cửa sổ video để thoát.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Tính FPS
        cur_time = time.time()
        fps = 1.0 / (cur_time - prev_time + 1e-6)
        prev_time = cur_time

        # Phát hiện các khuôn mặt trong khung hình
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(70, 70))

        for (x, y, w, h) in faces:
            face_pil = crop_face_box(frame, x, y, w, h)
            emb = compute_embedding(model, face_pil, transform, device=device)
            name, dist, is_match = recognize_face(emb, database, threshold=threshold)

            color = (0, 255, 0) if is_match else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            text = f"{name} | d={dist:.2f}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"FPS: {int(fps)} | Threshold tau: {threshold}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Database: {len(database)} persons enrolled", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Face Recognition Demo - Press Q to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chương trình nhận diện khuôn mặt")
    parser.add_argument('--db', type=str, default='face_db.pkl', help="Đường dẫn file database khuôn mặt")
    parser.add_argument('--weights', type=str, default='checkpoints/siamese_triplet.pth', help="Đường dẫn file trọng số mô hình")
    parser.add_argument('--threshold', type=float, default=0.6, help="Ngưỡng tau (tau càng nhỏ thì càng khắt khe)")
    parser.add_argument('--image', type=str, default='', help="Đường dẫn ảnh kiểm tra (để trống nếu muốn chạy webcam)")
    parser.add_argument('--webcam', action='store_true', help="Chạy webcam trực tiếp")
    args = parser.parse_args()

    # Tải database
    if not os.path.exists(args.db):
        print(f"[!] File database '{args.db}' chưa tồn tại. Hãy chạy enroll.py trước để đăng ký thành viên.")
        database = {}
    else:
        with open(args.db, 'rb') as f:
            database = pickle.load(f)
        print(f"[*] Đã nạp cơ sở dữ liệu với {len(database)} người: {list(database.keys())}")

    # Tải mô hình
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(weights_path=args.weights, device=device)
    transform = get_transforms(is_train=False)

    if args.image:
        run_recognition_on_image(args.image, model, database, transform, threshold=args.threshold, device=device)
    else:
        run_webcam(model, database, transform, threshold=args.threshold, device=device)
