"""
dataset.py
Quản lý nạp và sinh dữ liệu cho 2 phương pháp:
1. SiamesePairDataset: Tạo các cặp ảnh (img1, img2, label in {0, 1}) cho Binary Classification.
2. SiameseTripletDataset: Tạo các bộ ba (anchor, positive, negative) cho Triplet Loss.
3. FaceFolderDataset: Dùng cho Batch Hard Triplet Loss (mỗi mẫu đi kèm person_id).
"""

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_transforms(is_train=True, img_size=(160, 160)):
    """
    Tiền xử lý ảnh theo chuẩn các mạng Deep Learning:
    - Resize về 160x160
    - Chuẩn hóa theo ImageNet mean & std
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def load_images_from_folder(root_dir):
    """
    Quét thư mục theo cấu trúc:
    root_dir/
      person_A/
         img1.jpg, img2.jpg
      person_B/
         img1.jpg, img2.jpg
    Trả về dict: {person_name: [list_of_image_paths]}
    """
    data = {}
    if not os.path.exists(root_dir):
        return data

    for person_name in sorted(os.listdir(root_dir)):
        person_path = os.path.join(root_dir, person_name)
        if os.path.isdir(person_path):
            images = [
                os.path.join(person_path, f) for f in os.listdir(person_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if len(images) > 0:
                data[person_name] = images
    return data


class SiamesePairDataset(Dataset):
    """
    Dataset phục vụ Phương pháp 1 (Binary Classification):
    Mỗi lần lấy mẫu trả về (img1, img2, label):
    - label = 1.0 nếu cùng 1 người (Positive pair)
    - label = 0.0 nếu khác người (Negative pair)
    Tỉ lệ 50% cùng người, 50% khác người.
    """
    def __init__(self, root_dir, num_pairs=2000, transform=None):
        self.person_to_images = load_images_from_folder(root_dir)
        self.person_names = [p for p, imgs in self.person_to_images.items() if len(imgs) >= 2]
        self.all_persons = list(self.person_to_images.keys())
        self.num_pairs = num_pairs
        self.transform = transform if transform else get_transforms(is_train=True)

        if len(self.person_names) < 1:
            raise ValueError(f"Thư mục {root_dir} cần ít nhất 1 người có >= 2 ảnh!")

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        # 50% sinh cặp cùng người (Positive), 50% sinh cặp khác người (Negative)
        is_same = (idx % 2 == 0)

        if is_same:
            person = random.choice(self.person_names)
            img_path1, img_path2 = random.sample(self.person_to_images[person], 2)
            label = 1.0
        else:
            p1, p2 = random.sample(self.all_persons, 2)
            img_path1 = random.choice(self.person_to_images[p1])
            img_path2 = random.choice(self.person_to_images[p2])
            label = 0.0

        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


class SiameseTripletDataset(Dataset):
    """
    Dataset phục vụ Phương pháp 2 (Triplet Loss):
    Mỗi lần lấy mẫu trả về bộ ba: (Anchor, Positive, Negative)
    - Anchor & Positive: 2 ảnh khác nhau của CÙNG 1 người
    - Negative: 1 ảnh của người KHÁC
    """
    def __init__(self, root_dir, num_triplets=2000, transform=None):
        self.person_to_images = load_images_from_folder(root_dir)
        self.eligible_persons = [p for p, imgs in self.person_to_images.items() if len(imgs) >= 2]
        self.all_persons = list(self.person_to_images.keys())
        self.num_triplets = num_triplets
        self.transform = transform if transform else get_transforms(is_train=True)

        if len(self.eligible_persons) < 1 or len(self.all_persons) < 2:
            raise ValueError(f"Cần ít nhất 2 người và người được chọn cần có >= 2 ảnh!")

    def __len__(self):
        return self.num_triplets

    def __getitem__(self, idx):
        # 1. Chọn người cho Anchor và Positive
        person_pos = random.choice(self.eligible_persons)
        anchor_path, pos_path = random.sample(self.person_to_images[person_pos], 2)

        # 2. Chọn người khác cho Negative
        person_neg = random.choice([p for p in self.all_persons if p != person_pos])
        neg_path = random.choice(self.person_to_images[person_neg])

        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(pos_path).convert('RGB')
        negative = Image.open(neg_path).convert('RGB')

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative


class FaceFolderDataset(Dataset):
    """
    Dataset chuẩn hóa cho Batch Hard Triplet Mining:
    Trả về (image, person_id) để đưa vào DataLoader bình thường.
    """
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.person_to_id = {}
        self.transform = transform if transform else get_transforms(is_train=True)

        person_to_images = load_images_from_folder(root_dir)
        for idx, (person, img_list) in enumerate(person_to_images.items()):
            self.person_to_id[person] = idx
            for img_path in img_list:
                self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, person_id = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, person_id
