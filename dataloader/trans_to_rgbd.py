from tqdm import tqdm
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
from depth_anything_3.api import DepthAnything3

import os
import cv2
import torch
import numpy as np


class ImageNetDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_paths = [
            os.path.join(dp, f)
            for dp, dn, filenames in os.walk(root_dir)
            for f in filenames
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        # ViT 학습 표준 해상도인 224x224로 통일
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        from PIL import Image

        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), os.path.relpath(img_path, self.root_dir)


# 2. GPU별 워커 함수
def worker(gpu_id, world_size, root_dir, save_dir):
    # GPU 환경 설정
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # 모델 로드 (가장 강력한 DA3 Giant 모델 사용)
    print(f"[GPU {gpu_id}] Loading DA3-Giant-Large-1.1...")
    model = DepthAnything3.from_pretrained("depth-anything/da3nested-giant-large-1.1")
    model = model.to(device=device).eval()

    # 데이터셋 준비 및 해당 GPU 담당 구역(Subset) 설정
    full_dataset = ImageNetDataset(root_dir)
    indices = list(range(gpu_id, len(full_dataset), world_size))
    subset_dataset = Subset(full_dataset, indices)

    # DataLoader
    loader = DataLoader(
        subset_dataset, batch_size=1, num_workers=8, pin_memory=True, drop_last=False
    )

    print(f"[GPU {gpu_id}] Total batches: {len(loader)}")

    with torch.no_grad():
        for _, rel_paths in tqdm(loader, desc=f"GPU {gpu_id}"):
            try:
                full_paths = [os.path.join(root_dir, p) for p in rel_paths]
                prediction = model.inference(full_paths)

                depth_maps = prediction.depth
                for j, rel_path in enumerate(rel_paths):
                    save_path = os.path.join(
                        save_dir, os.path.splitext(rel_path)[0] + ".npy"
                    )
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    depth_map = depth_maps[j]
                    if depth_map.shape != (224, 224):
                        depth_map = cv2.resize(
                            depth_map, (224, 224), interpolation=cv2.INTER_LINEAR
                        )
                    np.save(save_path, depth_map.astype(np.float32))

            except Exception as e:
                import traceback

                traceback.print_exc()
                continue
