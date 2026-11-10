"""
train_binary.py
Huấn luyện Siamese Network theo cách 1 (Binary Classification):
Lập trình bám sát Slide trang 23-28.
Input: 2 ảnh (x1, x2) -> Backbone -> Embeddings (128-d) -> |f(x1) - f(x2)| -> FC -> Sigmoid -> y_hat
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from models import get_embedding_model, SiameseBinaryNet
from dataset import SiamesePairDataset, get_transforms


def train_binary(data_dir, epochs=10, batch_size=32, lr=1e-4, device=None, save_path='checkpoints/siamese_binary.pth'):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Sử dụng thiết bị: {device}")

    # 1. Khởi tạo Dataset & DataLoader
    train_transform = get_transforms(is_train=True)
    dataset = SiamesePairDataset(root_dir=data_dir, num_pairs=1000, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"[*] Tổng số cặp ảnh huấn luyện: {len(dataset)}")

    # 2. Khởi tạo Mô hình
    embedding_net = get_embedding_model(backbone_name='resnet18', embedding_dim=128, pretrained=True)
    model = SiameseBinaryNet(embedding_net).to(device)

    # 3. Hàm Loss & Optimizer
    # Sử dụng BCEWithLogitsLoss (đã tích hợp Sigmoid bên trong để ổn định số học hơn)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. Vòng lặp huấn luyện
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch [{epoch}/{epochs}]")
        for img1, img2, labels in pbar:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, _, _ = model(img1, img2)
            logits = logits.squeeze(1)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Tính độ chính xác nhị phân (ngưỡng 0.0 của logit tương ứng 0.5 của sigmoid)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            running_loss += loss.item() * labels.size(0)

            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{correct_predictions / total_samples:.2%}"})

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        print(f"--> Kết thúc Epoch {epoch}: Loss trung bình = {epoch_loss:.4f} | Accuracy = {epoch_acc:.2%}")

    # 5. Lưu trọng số mô hình
    torch.save(model.state_dict(), save_path)
    print(f"[✓] Đã lưu mô hình Binary thành công tại: {save_path}")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Huấn luyện Siamese Binary Classification")
    parser.add_argument('--data_dir', type=str, default='data/train', help="Đường dẫn thư mục dữ liệu")
    parser.add_argument('--epochs', type=int, default=5, help="Số epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Kích thước batch")
    parser.add_argument('--lr', type=float, default=1e-4, help="Tốc độ học")
    parser.add_argument('--save_path', type=str, default='checkpoints/siamese_binary.pth')
    args = parser.parse_args()

    train_binary(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, save_path=args.save_path)
