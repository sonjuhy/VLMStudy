from dataloader.rgbd_imagenet_1k_dataloader import get_rgbd_imagenet_loaders

import os
import torch

from vision.vit_model import ViTDepthEncoder


def depth_valid_test(pth_file_path: str):
    if os.path.exists(pth_file_path):
        checkpoint = torch.load(pth_file_path, map_location="cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 모델 초기화 (기존 설정과 동일하게)
        model = ViTDepthEncoder(
            img_size=224,
            patch_size=16,
            embedding_size=768,
            num_class=1000,
            num_heads=12,
            in_channels=4,
        ).to(device)

        # 2. 체크포인트 로드
        checkpoint = torch.load("checkpoints/vit_imagenet_1k_checkpoint_epoch_40.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

        ROOT_PATH = os.path.join("C:", os.sep, "WorkSpace", "DataSets")
        RGB_DATA_DIR = os.path.join(
            "imagenet_1k_origin", "imagenet_1k", "raw_data", "ILSVRC", "Data", "CLS-LOC"
        )
        DEPTH_DATA_DIR = os.path.join(
            "imagenet_1k_depth", "imagenet_1k_depth"
        )  # npy가 저장된 경로

        # 3. 데이터 로더 (Validation만)
        _, val_loader = get_rgbd_imagenet_loaders(
            rgb_root=os.path.join(ROOT_PATH, RGB_DATA_DIR),
            depth_root=os.path.join(ROOT_PATH, DEPTH_DATA_DIR),
            batch_size=32,  # 검증은 배치를 조금 더 키워도 됩니다
            num_workers=8,
        )

        # 4. 검증 수행
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if i % 100 == 0:
                    print(f"Progress: {i}/{len(val_loader)}")

        print(f"\n[Result] Epoch 40 Accuracy: {100 * correct / total:.2f}%")
    else:
        print(f"Can't find file. [{pth_file_path}]")
