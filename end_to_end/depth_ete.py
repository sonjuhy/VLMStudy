from train.depth.depth_vit_train import depth_vit_encoder_train
from torch import GradScaler
from dataloader.rgbd_imagenet_1k_dataloader import get_rgbd_imagenet_loaders

import os
import torch
import torch.nn as nn
import torch.optim as optim

from vision.vit_model import ViTDepthEncoder


def depth_vlm_end_to_end(
    train_continue: bool = False,
    start_epoch: int = 0,
    end_epoch: int = 100,
    best_acc: float = 0.0,
):
    # 1. 모델, 데이터로더, 옵티마이저 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 224  # ImageNet 표준 해상도
    patch_size = 16  # 224/16 = 14x14 총 196개의 패치 생성
    embedding_size = 768  # ViT-Base 표준 임베딩 차원 (반드시 num_heads의 배수여야 함)
    num_class = 1000  # ImageNet-1K의 클래스 개수
    num_heads = 12  # 768 / 12 = head당 64차원 (표준 설정)
    print("==" * 50)
    print("Train Default Info")
    print("==" * 50)
    print("device:", device)
    print("img_size:", img_size)
    print("patch_size:", patch_size)
    print("embedding_size:", embedding_size)
    print("num_class:", num_class)
    print("num_heads:", num_heads)
    print("train_continue:", train_continue)
    print("start_epoch:", start_epoch)
    print("end_epoch:", end_epoch)
    print()

    ROOT_PATH = os.path.join("C:", os.sep, "WorkSpace", "DataSets")
    RGB_DATA_DIR = os.path.join(
        "imagenet_1k_origin", "imagenet_1k", "raw_data", "ILSVRC", "Data", "CLS-LOC"
    )
    DEPTH_DATA_DIR = os.path.join(
        "imagenet_1k_depth", "imagenet_1k_depth"
    )  # npy가 저장된 경로
    if not os.path.exists(
        os.path.join(ROOT_PATH, DEPTH_DATA_DIR)
    ) or not os.path.exists(os.path.join(ROOT_PATH, RGB_DATA_DIR)):
        print("Path is not exist")
        return False

    print("Model Loading...")
    model = ViTDepthEncoder(
        img_size=img_size,
        patch_size=patch_size,
        embedding_size=embedding_size,
        num_class=num_class,
        num_heads=num_heads,
        in_channels=4,  # RGB-D
    ).to(device)
    train_loader, val_loader = get_rgbd_imagenet_loaders(
        rgb_root=os.path.join(ROOT_PATH, RGB_DATA_DIR),
        depth_root=os.path.join(ROOT_PATH, DEPTH_DATA_DIR),
        batch_size=32,  # 저사양 GPU (RTX 3080 Laptop (8GB/16GB)) 기준 상향 조정
        num_workers=8,  # i7-12700H  오버헤드 방지
    )

    print("Train Tools(Optimizer, Scheduler, Loss) Loading...")
    # 2. 혼합 정밀도를 위한 GradScaler 초기화
    epochs = end_epoch - start_epoch + 1
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scaler = GradScaler()
    if train_continue:
        print("==" * 50)
        print("Train Continue")
        print("==" * 50)

        # Resume 로직
        checkpoint_path = "checkpoints/best_vit_depth.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

            finetune_lr = 1e-5
            for param_group in optimizer.param_groups:
                param_group["lr"] = finetune_lr

            start_epoch = checkpoint["epoch"] + 1

            epochs_left = end_epoch - start_epoch + 1
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs_left, eta_min=1e-7
            )

            print(f"Resuming training from epoch {start_epoch}...")
            print(f"Current Best Acc: {best_acc:.2f}%")
        else:
            print("No checkpoint found. Starting from scratch.")
        # 함수 호출
        depth_vit_encoder_train(
            epochs=end_epoch,
            start_epoch=start_epoch,
            device=device,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            scheduler=scheduler,
            val_loader=val_loader,
            accumulation_steps=16,
            best_acc=best_acc,
        )
    else:
        print("==" * 50)
        print("Train Start")
        print("==" * 50)
        depth_vit_encoder_train(
            epochs=epochs,
            start_epoch=0,
            device=device,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            scheduler=scheduler,
            val_loader=val_loader,
            accumulation_steps=16,
            best_acc=0.0,
        )
