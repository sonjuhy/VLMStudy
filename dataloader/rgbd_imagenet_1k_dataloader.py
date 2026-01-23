from PIL import Image
from torch.utils.data import Dataset

import torchvision.transforms.functional as F
import random
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

        # 파일 목록 수집 (RGB 기준으로 수집)
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
        rel_path = self.samples[idx]

        # 1. RGB 이미지 로드
        rgb_path = os.path.join(self.rgb_root, rel_path)
        rgb_img = Image.open(rgb_path).convert("RGB")

        # 2. Depth (.npy) 로드
        depth_rel_path = os.path.splitext(rel_path)[0] + ".npy"
        depth_path = os.path.join(self.depth_root, depth_rel_path)
        depth_map = np.load(depth_path)  # [224, 224]
        depth_img = Image.fromarray(depth_map)  # PIL로 변환해야 transform 적용이 쉬움

        # 3. 동일한 증강(Augmentation) 적용
        # RandomResizedCrop과 Flip을 RGB와 Depth에 똑같이 적용하는 로직
        if self.transform:
            # i, j, h, w 등 파라미터를 고정해서 양쪽에 동일하게 적용
            # (학습용 복잡한 Augmentation이 필요할 때 아래처럼 functional 사용)

            # 예시: 랜덤 수평 뒤집기
            if random.random() > 0.5:
                rgb_img = F.hflip(rgb_img)
                depth_img = F.hflip(depth_img)

            # 예시: Resize 및 텐서 변환
            rgb_img = F.resize(rgb_img, (224, 224))
            depth_img = F.resize(depth_img, (224, 224))

            rgb_tensor = F.to_tensor(rgb_img)
            depth_tensor = F.to_tensor(depth_img)  # [1, 224, 224]
        else:
            rgb_tensor = F.to_tensor(rgb_img)
            depth_tensor = F.to_tensor(depth_img)

        # 4. RGB(3) + Depth(1) 합치기 -> [4, 224, 224]
        rgbd_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)

        # 라벨(Label) 추출 (ImageNet 폴더 구조인 경우 폴더명이 클래스)
        label = self._get_label_from_path(rel_path)

        return rgbd_tensor, label

    def _get_label_from_path(self, path: str):
        class_name = path.split(os.sep)[0]
        return self.class_to_idx[class_name]
