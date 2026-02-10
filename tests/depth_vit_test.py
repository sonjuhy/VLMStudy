from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from torch import autocast, GradScaler
from dataloader.rgbd_imagenet_1k_dataloader import get_rgbd_imagenet_loaders

import os
import torch
import torch.nn as nn
import torch.optim as optim

from vision.vit_model import ViTDepthEncoder


def depth_vit_encoder_train(
    epochs: int,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    scheduler: LRScheduler,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader = None,
    accumulation_steps: int = 16,  # 16(Batch) * 16 = 실제 배치 256 효과
):
    best_acc: float = 0.0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()  # 루프 밖에서 초기화

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            # 1. autocast 적용
            with autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # 누적을 위해 loss를 steps로 나눔
                loss = loss / accumulation_steps

            # 2. 스케일링된 역전파
            scaler.scale(loss).backward()

            # 3. 가중치 업데이트 (accumulation_steps 마다)
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}")

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            current_loss = loss.item() * accumulation_steps
            save_checkpoint(epoch + 1, model, optimizer, scaler, current_loss)
        # Validation
        val_loss, val_acc = depth_vit_encoder_validate(
            model, val_loader, criterion, device
        )
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # 3. Best 모델 저장 (노트북 용량 관리)
        if val_acc > best_acc:
            print(f"New Best Accuracy! ({val_acc:.2f}%) Saving model...")
            best_acc = val_acc
            save_checkpoint(
                epoch + 1,
                model,
                optimizer,
                scaler,
                val_loss,
                path="checkpoints",
                filename="best_vit_depth.pth",
            )
        scheduler.step()

        # if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
        #     save_checkpoint(epoch + 1, model, optimizer, scaler, loss.item())


def save_checkpoint(
    epoch, model, optimizer, scaler, loss, path="checkpoints", filename=None
):
    if not os.path.exists(path):
        os.makedirs(path)

    if filename is None:
        checkpoint_path = os.path.join(
            path, f"vit_imagenet_1k_checkpoint_epoch_{epoch}.pth"
        )
    else:
        checkpoint_path = os.path.join(path, filename)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loss": loss,
        },
        checkpoint_path,
    )
    print(f"--- Checkpoint saved at: {checkpoint_path} ---")


@torch.no_grad()  # 검증 시 메모리 절약 필수
def depth_vit_encoder_validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(val_loader, desc="Validating")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        # 학습과 동일하게 autocast 적용 (속도 및 메모리 효율)
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        val_loss += loss.item() * inputs.size(0)

        # 정확도 계산
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix(loss=f"{val_loss/total:.4f}", acc=f"{100.*correct/total:.2f}%")

    avg_loss = val_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def depth_end_to_end_test():
    # 1. 모델, 데이터로더, 옵티마이저 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 224  # ImageNet 표준 해상도
    patch_size = 16  # 224/16 = 14x14 총 196개의 패치 생성
    embedding_size = 768  # ViT-Base 표준 임베딩 차원 (반드시 num_heads의 배수여야 함)
    num_class = 1000  # ImageNet-1K의 클래스 개수
    num_heads = 12  # 768 / 12 = head당 64차원 (표준 설정)

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
        batch_size=16,
        num_workers=12,
    )

    # 2. 혼합 정밀도를 위한 GradScaler 초기화
    epochs = 50
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scaler = GradScaler()
    depth_vit_encoder_train(
        epochs=epochs,
        device=device,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        scaler=scaler,
        scheduler=scheduler,
        val_loader=val_loader,
        accumulation_steps=16,
    )


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
