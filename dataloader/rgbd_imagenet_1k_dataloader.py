import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as F
import torchvision.transforms as T
import os
import torch
import numpy as np


class ImageNetRGBDDataset(Dataset):
    def __init__(self, rgb_root, depth_root, transform=None):
        self.rgb_root = rgb_root
        self.depth_root = depth_root
        self.transform = transform

        self.classes = sorted(
            entry.name for entry in os.scandir(rgb_root) if entry.is_dir()
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for subdir, _, files in os.walk(rgb_root):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    rel_path = os.path.relpath(os.path.join(subdir, file), rgb_root)
                    self.samples.append(rel_path)
        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. 경로 설정
        rel_path = self.samples[idx]
        rgb_path = os.path.join(self.rgb_root, rel_path)
        depth_rel_path = os.path.splitext(rel_path)[0] + ".npy"
        depth_path = os.path.join(self.depth_root, depth_rel_path)

        # [안전장치 1] Depth 파일 존재 여부 확인 (없으면 다른 인덱스 탐색)
        # 루프를 사용하여 재귀 깊이 문제를 방지합니다.
        max_attempts = 10
        current_idx = idx
        for _ in range(max_attempts):
            if os.path.exists(depth_path):
                break
            print(f"Warning: Depth not found: {depth_path}. Trying another...")
            current_idx = random.randint(0, len(self.samples) - 1)
            rel_path = self.samples[current_idx]
            rgb_path = os.path.join(self.rgb_root, rel_path)
            depth_rel_path = os.path.splitext(rel_path)[0] + ".npy"
            depth_path = os.path.join(self.depth_root, depth_rel_path)
        else:
            raise FileNotFoundError(
                f"Could not find any valid depth map after {max_attempts} attempts."
            )

        # 2. 데이터 로드
        rgb_img = Image.open(rgb_path).convert("RGB")
        depth_map = np.load(depth_path).astype(np.float32)

        # Depth 값 범위 정규화 (0~1)
        d_max = depth_map.max()
        if d_max > 0:  # 0으로 나누기 방지
            depth_map = depth_map / d_max

        depth_img = Image.fromarray(depth_map)

        # 3. Augmentation & Preprocessing
        if self.transform:
            # Random Resized Crop
            # get_params를 사용하여 동일한 파라미터를 RGB와 Depth에 적용
            i, j, h, w = T.RandomResizedCrop.get_params(
                rgb_img, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)
            )
            rgb_img = F.resized_crop(rgb_img, i, j, h, w, (224, 224))
            depth_img = F.resized_crop(depth_img, i, j, h, w, (224, 224))

            # Random Horizontal Flip
            if random.random() > 0.5:
                rgb_img = F.hflip(rgb_img)
                depth_img = F.hflip(depth_img)

            # To Tensor
            rgb_tensor = F.to_tensor(rgb_img)
            depth_tensor = F.to_tensor(depth_img)

            # Normalize
            rgb_tensor = F.normalize(
                rgb_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            depth_tensor = F.normalize(depth_tensor, mean=[0.5], std=[0.5])

        else:
            # Validation: Simple Resize & Normalize
            rgb_img = F.resize(rgb_img, [224, 224])
            depth_img = F.resize(depth_img, [224, 224])

            rgb_tensor = F.to_tensor(rgb_img)
            depth_tensor = F.to_tensor(depth_img)

            # Validation set normalization (Good practice to include)
            rgb_tensor = F.normalize(
                rgb_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            depth_tensor = F.normalize(depth_tensor, mean=[0.5], std=[0.5])

        # 5. 최종 결합 및 라벨 반환
        # 여기서 더 이상 Size mismatch가 발생하지 않습니다.
        rgbd_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)
        label = self._get_label_from_path(rel_path)

        return rgbd_tensor, label

    def _get_label_from_path(self, path: str):
        class_name = path.split(os.sep)[0]
        return self.class_to_idx[class_name]


def get_rgbd_imagenet_loaders(
    rgb_root,
    depth_root,
    batch_size=16,
    num_workers=None,
):
    if num_workers is None:
        num_workers = os.cpu_count() or 4

    # 1. 학습용 데이터셋 (Augmentation 포함)
    train_dataset = ImageNetRGBDDataset(
        rgb_root=os.path.join(rgb_root, "train"),
        depth_root=os.path.join(depth_root, "train"),
        transform=True,  # 내부에서 RandomFlip 등이 작동하도록 설정
    )

    # 2. 검증용 데이터셋 (단순 Resize만)
    val_dataset = ImageNetRGBDDataset(
        rgb_root=os.path.join(rgb_root, "val"),
        depth_root=os.path.join(depth_root, "val"),
        transform=False,
    )

    # 3. DataLoader 생성 (Windows 최적화)
    # persistent_workers=True: 에폭마다 프로세스를 재생성하지 않음 (Windows 필수)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, val_loader
