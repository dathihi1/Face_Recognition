"""
train_triplet.py
Huấn luyện Siamese Network theo cách 2 (Triplet Loss & Hard Triplet Mining):
Lập trình bám sát Slide trang 13-22 (FaceNet style).
Loss = max( ||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + alpha, 0 )
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from models import get_embedding_model
from dataset import SiameseTripletDataset, get_transforms
from loss import TripletLoss, BatchHardTripletLoss


def train_triplet(data_dir, epochs=10, batch_size=32, lr=1e-4, alpha=0.2, device=None, save_path='checkpoints/siamese_triplet.pth'):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Sử dụng thiết bị: {device}")

    # 1. Khởi tạo Dataset bộ ba (Anchor, Positive, Negative)
    train_transform = get_transforms(is_train=True)
    dataset = SiameseTripletDataset(root_dir=data_dir, num_triplets=1200, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"[*] Tổng số bộ ba (triplets) huấn luyện: {len(dataset)}")

    # 2. Khởi tạo Mạng trích xuất đặc trưng (Backbone ResNet18 -> 128 chiều)
    model = get_embedding_model(backbone_name='resnet18', embedding_dim=128, pretrained=True).to(device)

    # 3. Hàm Triplet Loss với margin alpha
    criterion = TripletLoss(alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. Vòng lặp huấn luyện
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        total_samples = 0
        sum_pos_dist = 0.0
        sum_neg_dist = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch [{epoch}/{epochs}]")
        for anchor, positive, negative in pbar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            # Chạy qua cùng 1 mạng backbone trích xuất 3 vector f(A), f(P), f(N)
            f_a = model(anchor)
            f_p = model(positive)
            f_n = model(negative)

            loss = criterion(f_a, f_p, f_n)
            loss.backward()
            optimizer.step()

            # Thống kê khoảng cách d(A, P) và d(A, N) để quan sát sự hội tụ
            with torch.no_grad():
                d_pos = torch.norm(f_a - f_p, p=2, dim=1).pow(2).mean().item()
                d_neg = torch.norm(f_a - f_n, p=2, dim=1).pow(2).mean().item()

            batch_size_cur = anchor.size(0)
            running_loss += loss.item() * batch_size_cur
            total_samples += batch_size_cur
            sum_pos_dist += d_pos * batch_size_cur
            sum_neg_dist += d_neg * batch_size_cur

            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'd(A,P)': f"{d_pos:.3f}",
                'd(A,N)': f"{d_neg:.3f}"
            })

        epoch_loss = running_loss / total_samples
        avg_pos = sum_pos_dist / total_samples
        avg_neg = sum_neg_dist / total_samples
        print(f"--> Kết thúc Epoch {epoch}: Loss = {epoch_loss:.4f} | Trung bình d(A,P)={avg_pos:.3f} | Trung bình d(A,N)={avg_neg:.3f}")

    # 5. Lưu trọng số mạng Embedding
    torch.save(model.state_dict(), save_path)
    print(f"[✓] Đã lưu mô hình Triplet Embedding thành công tại: {save_path}")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Huấn luyện Siamese với Triplet Loss")
    parser.add_argument('--data_dir', type=str, default='data/train', help="Đường dẫn thư mục dữ liệu")
    parser.add_argument('--epochs', type=int, default=5, help="Số epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Kích thước batch")
    parser.add_argument('--lr', type=float, default=1e-4, help="Tốc độ học")
    parser.add_argument('--alpha', type=float, default=0.2, help="Margin alpha trong Triplet Loss")
    parser.add_argument('--save_path', type=str, default='checkpoints/siamese_triplet.pth')
    args = parser.parse_args()

    train_triplet(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, alpha=args.alpha, save_path=args.save_path)
