"""
loss.py
Cài đặt Triplet Loss & Hard Triplet Mining theo đúng công thức trong slide:
L(A, P, N) = max( ||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + alpha, 0 )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet Loss cơ bản nhận vào 3 vector embedding:
    - anchor: f(A)
    - positive: f(P) (cùng người với A)
    - negative: f(N) (khác người với A)
    - alpha: margin (khoảng đệm), mặc định 0.2
    """
    def __init__(self, alpha=0.2):
        super(TripletLoss, self).__init__()
        self.alpha = alpha

    def forward(self, anchor, positive, negative):
        # Tính bình phương khoảng cách Euclidean: d(A, P) = ||f(A) - f(P)||^2
        distance_positive = F.pairwise_distance(anchor, positive, p=2).pow(2)
        # Tính bình phương khoảng cách Euclidean: d(A, N) = ||f(A) - f(N)||^2
        distance_negative = F.pairwise_distance(anchor, negative, p=2).pow(2)

        # L = max( d(A, P) - d(A, N) + alpha, 0 )
        losses = F.relu(distance_positive - distance_negative + self.alpha)
        return losses.mean()


def pairwise_distances(embeddings):
    """
    Tính ma trận khoảng cách bình phương Euclidean giữa tất cả các cặp trong batch:
    ||x - y||^2 = ||x||^2 - 2<x, y> + ||y||^2
    Vì embeddings đã được L2 normalized nên ||x||^2 = 1, ||y||^2 = 1:
    ||x - y||^2 = 2 - 2<x, y>
    """
    dot_product = torch.matmul(embeddings, embeddings.t())
    # Kẹp giá trị tránh số âm nhỏ do sai số float
    sq_distances = 2.0 - 2.0 * dot_product
    sq_distances = torch.clamp(sq_distances, min=0.0)
    return sq_distances


class BatchHardTripletLoss(nn.Module):
    """
    Hard Triplet Mining (Slide trang 16-18):
    Trong 1 batch gồm nhiều ảnh của các cá nhân khác nhau:
    Với mỗi mẫu Anchor (A):
    - Hardest Positive: Mẫu cùng nhãn nhưng có khoảng cách xa nhất với A: max d(A, P)
    - Hardest Negative: Mẫu khác nhãn nhưng có khoảng cách gần nhất với A: min d(A, N)
    Từ đó tính loss trên các bộ ba khó này giúp mạng học nhanh và hội tụ tốt nhất.
    """
    def __init__(self, alpha=0.2):
        super(BatchHardTripletLoss, self).__init__()
        self.alpha = alpha

    def forward(self, embeddings, labels):
        """
        embeddings: Tensor kích thước [BatchSize, EmbeddingDim] (đã L2 normalized)
        labels: Tensor kích thước [BatchSize] chứa ID nhãn của từng người
        """
        pdist_matrix = pairwise_distances(embeddings)

        # Tạo mask nhãn: mask_same_label[i, j] = True nếu cùng người
        labels = labels.unsqueeze(1)
        mask_same_label = torch.eq(labels, labels.t())
        mask_diff_label = ~mask_same_label

        # 1. Hardest Positive: Trong số những người cùng nhãn, tìm khoảng cách lớn nhất
        # Với các cặp khác nhãn, gán 0 để không ảnh hưởng đến phép tìm max
        pdist_pos = pdist_matrix * mask_same_label.float()
        hardest_positive_dist, _ = torch.max(pdist_pos, dim=1)

        # 2. Hardest Negative: Trong số những người khác nhãn, tìm khoảng cách nhỏ nhất
        # Với các cặp cùng nhãn, gán giá trị cực đại để không ảnh hưởng đến phép tìm min
        max_dist = torch.max(pdist_matrix)
        pdist_neg = pdist_matrix + max_dist * mask_same_label.float()
        hardest_negative_dist, _ = torch.min(pdist_neg, dim=1)

        # 3. Tính Triplet Loss trên các cặp khó: max(d(A, P_hard) - d(A, N_hard) + alpha, 0)
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.alpha)

        # Chỉ lấy trung bình trên những triplet có loss > 0 (active triplets)
        valid_triplets = triplet_loss[triplet_loss > 1e-16]
        if len(valid_triplets) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        return valid_triplets.mean()
