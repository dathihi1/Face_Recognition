"""
models.py
Kiến trúc mạng Siamese Network phục vụ 2 phương pháp trong slide:
1. FaceEmbeddingNet: Mạng xương sống (Backbone) trích xuất vector đặc trưng 128 chiều, chuẩn hóa L2.
2. SiameseBinaryNet: Nhánh phân loại nhị phân tính hiệu số |f(x1) - f(x2)| -> Linear -> Sigmoid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FaceEmbeddingNet(nn.Module):
    """
    Mạng trích xuất đặc trưng khuôn mặt f(x) -> 128 chiều.
    Hỗ trợ backbone ResNet-18 hoặc MobileNetV2.
    Đầu ra luôn được chuẩn hóa L2 norm: ||f(x)||_2 = 1.
    """
    def __init__(self, backbone_name='resnet18', embedding_dim=128, pretrained=True):
        super(FaceEmbeddingNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.backbone_name = backbone_name

        if backbone_name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
            in_features = base_model.fc.in_features
            # Loại bỏ lớp phân loại gốc 1000 class của ImageNet
            base_model.fc = nn.Identity()
            self.backbone = base_model
        elif backbone_name == 'mobilenet_v2':
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            base_model = models.mobilenet_v2(weights=weights)
            in_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()
            self.backbone = base_model
        else:
            raise ValueError(f"Không hỗ trợ backbone: {backbone_name}")

        # Lớp chiếu xuống vector 128 chiều (theo đúng slide bài giảng)
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.head(features)
        # Chuẩn hóa L2 để các vector nằm trên mặt cầu đơn vị
        # ||f(x)||_2 = 1 giúp việc đo khoảng cách Euclidean và Cosine tương đương nhau
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings


class SiameseBinaryNet(nn.Module):
    """
    Phương pháp 1 trong slide (Trang 23-28):
    Face verification as binary classification:
    y_hat = sigma( sum_{k=1}^{128} w_k * |f(x1)_k - f(x2)_k| + b )
    """
    def __init__(self, embedding_net):
        super(SiameseBinaryNet, self).__init__()
        self.embedding_net = embedding_net
        embedding_dim = embedding_net.embedding_dim

        # Lớp logistic regression trên vector sai khác |f(x1) - f(x2)|
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward_one(self, x):
        return self.embedding_net(x)

    def forward(self, x1, x2):
        # Trích xuất 2 vector đặc trưng f(x1) và f(x2) bằng cùng một mạng (shared weights)
        f1 = self.embedding_net(x1)
        f2 = self.embedding_net(x2)

        # Tính độ lệch tuyệt đối giữa 2 embedding: |f(x1) - f(x2)|
        diff = torch.abs(f1 - f2)

        # Đưa qua lớp phân loại nhị phân
        logits = self.classifier(diff)
        return logits, f1, f2


def get_embedding_model(backbone_name='resnet18', embedding_dim=128, pretrained=True):
    return FaceEmbeddingNet(backbone_name=backbone_name, embedding_dim=embedding_dim, pretrained=pretrained)
