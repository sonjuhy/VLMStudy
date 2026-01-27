import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as F
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

        # 3. 리사이즈 및 텐서 변환 (에러 방지 핵심 구간)
        # transform 여부와 상관없이 모델 입력 크기(224)는 맞춰야 합니다.

        # [안전장치 2] RGB와 Depth의 리사이즈를 항상 수행하여 크기 불일치 원천 차단
        rgb_img = F.resize(rgb_img, [224, 224])
        depth_img = F.resize(depth_img, [224, 224])

        rgb_tensor = F.to_tensor(rgb_img)
        depth_tensor = F.to_tensor(depth_img)

        # 4. Augmentation 및 정규화
        if self.transform:
            # 추가적인 정규화 적용
            rgb_tensor = F.normalize(
                rgb_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            depth_tensor = F.normalize(depth_tensor, mean=[0.5], std=[0.5])
        else:
            # transform이 없더라도 최소한의 정규화는 수행하는 것이 좋습니다.
            # (이미 위에서 F.to_tensor를 통해 0~1 스케일링은 완료됨)
            pass

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

    # 3. DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # GPU 메모리 복사 속도 향상
        prefetch_factor=2,  # CPU가 미리 데이터를 준비
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
