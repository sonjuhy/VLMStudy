from train.depth.depth_vit_train import depth_vit_encoder_train
from torch import GradScaler
from dataloader.rgbd_imagenet_1k_dataloader import get_rgbd_imagenet_loaders
from datetime import date, datetime

import os
import torch
import torch.nn as nn
import torch.optim as optim

from vision.vit_model import ViTDepthEncoder


def depth_vlm_end_to_end(
    train_continue: bool = False,
    start_epoch: int = 0,
    end_epoch: int = 300,
    warmup_epochs: int = 20,
    patience: int = 60,
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

    RGB_DATA_DIR = os.path.join("datasets", "imagenet_1k")
    DEPTH_DATA_DIR = os.path.join("datasets", "imagenet_1k_depth")
    if not os.path.exists(DEPTH_DATA_DIR) or not os.path.exists(RGB_DATA_DIR):
        print("Path is not exist")
        return False

    # --- 학습 시작 날짜로 로그 폴더 생성 (순수 ViT 학습과 동일한 방식) ---
    start_date = date.today().isoformat()
    log_dir = os.path.join("logs", "depth_vit", start_date)
    log_path = os.path.join(log_dir, "train_log.txt")
    os.makedirs(log_dir, exist_ok=True)

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
        rgb_root=RGB_DATA_DIR,
        depth_root=DEPTH_DATA_DIR,
        batch_size=32,  # 저사양 GPU (RTX 3080 Laptop (8GB/16GB)) 기준 상향 조정
        num_workers=8,  # i7-12700H  오버헤드 방지
    )

    print("Train Tools(Optimizer, Scheduler, Loss) Loading...")
    # 2. 혼합 정밀도를 위한 GradScaler 초기화
    epochs = end_epoch - start_epoch + 1
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    # LinearWarmup -> CosineAnnealing (순수 ViT 학습과 동일한 레시피)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-7
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scaler = GradScaler()

    ckpt = None
    if train_continue:
        print("==" * 50)
        print("Train Continue")
        print("==" * 50)

        # Resume 로직
        checkpoint_path = "checkpoints/depth_final_model/best_vit_depth.pth"
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scaler.load_state_dict(ckpt["scaler_state_dict"])

            finetune_lr = 1e-5
            for param_group in optimizer.param_groups:
                param_group["lr"] = finetune_lr

            start_epoch = ckpt["epoch"] + 1
            best_acc = ckpt.get("best_acc", best_acc)

            epochs_left = end_epoch - start_epoch + 1
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs_left, eta_min=1e-7
            )

            print(f"Resuming training from epoch {start_epoch}...")
            print(f"Current Best Acc: {best_acc:.2f}%")
        else:
            print("No checkpoint found. Starting from scratch.")

    resume_note = (
        f"Resumed from: checkpoints/depth_final_model/best_vit_depth.pth "
        f"(epoch {start_epoch - 1}, best_acc={best_acc:.2f}%)\n"
        if ckpt is not None
        else ""
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"Training Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if resume_note:
            f.write(resume_note)
        f.write("ViT Version: ViT-Base/16 (official spec, in_channels=4 RGB-D)\n")
        f.write(
            f"Model Config: embedding_size={embedding_size}, num_heads={num_heads}, "
            f"patch_size={patch_size}, img_size={img_size}, num_class={num_class}\n"
        )
        f.write("Dataset: ImageNet-1K RGB-D (RGB + estimated Depth, 1000 classes)\n")
        f.write(
            "Augmentations: RandomResizedCrop(224), RandomHorizontalFlip, "
            "AutoAugment(ImageNet policy, RGB only), Mixup(alpha=0.8, p=0.5), "
            "Label Smoothing(0.1)\n"
        )
        f.write(
            f"Hyperparameters: epochs={end_epoch}, warmup_epochs={warmup_epochs}, "
            f"batch_size=32 x accumulation=16 (effective 512), "
            f"optimizer=AdamW(lr=1e-3, weight_decay=0.05), "
            f"AMP=bfloat16, scheduler=LinearWarmup+CosineAnnealing, "
            f"early_stopping_patience={patience}\n"
        )
        f.write("=" * 70 + "\n")

    best_acc = depth_vit_encoder_train(
        epochs=end_epoch if train_continue else epochs,
        start_epoch=start_epoch if train_continue else 0,
        device=device,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        scaler=scaler,
        scheduler=scheduler,
        log_path=log_path,
        val_loader=val_loader,
        accumulation_steps=16,
        best_acc=best_acc if train_continue else 0.0,
        patience=patience,
    )

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"Training End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best Val Acc: {best_acc:.2f}%\n")
        f.write("=" * 70 + "\n\n")
