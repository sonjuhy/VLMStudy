from torch.amp import autocast, GradScaler
from dataloader.imagenet_1k_dataloader import (
    get_imagenet_loaders,
    get_imagenet_loaders_fsdp,
)
from vision.vit_model import ViTEncoder
from tqdm import tqdm

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)

import os
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import glob


def save_checkpoint(epoch, model, optimizer, scaler, loss, path="checkpoints"):
    if not os.path.exists(path):
        os.makedirs(path)

    checkpoint_path = os.path.join(
        path, f"vit_imagenet_1k_checkpoint_epoch_{epoch}.pth"
    )

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


def load_checkpoint(path, model, optimizer, scaler):
    if os.path.isfile(path):
        print(f"--- Loading checkpoint: {path} ---")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"--- Resuming from epoch {start_epoch} ---")
        return start_epoch
    else:
        print("--- No checkpoint found, starting from scratch ---")
        return 0


def train(
    epochs: int,
    device: torch.device = torch.device("cpu"),
    model: nn.Module = None,
    train_loader: torch.utils.data.DataLoader = None,
    optimizer: optim.Optimizer = None,
    criterion: nn.Module = None,
    scaler: GradScaler = None,
    scheduler: optim.lr_scheduler = None,
):
    for epoch in range(epochs):
        model.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in pbar:
            # for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # --- 혼합 정밀도 핵심 구간 ---
            # 3. autocast를 사용하여 순전파(Forward) 연산 수행
            with autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # 4. 스케일링된 Loss로 역전파(Backward)
            scaler.scale(loss).backward()

            # 5. 가중치 업데이트 (내부적으로 스케일 조정 및 Gradient Clipping 가능)
            scaler.step(optimizer)
            scaler.update()
            # scheduler.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        # if (epoch + 1) % 10 == 0:
        save_checkpoint(epoch + 1, model, optimizer, scaler, loss.item())


def evaluate_all_checkpoints(
    checkpoint_dir,
    device=torch.device("cuda"),
    save_name="evaluation_results.csv",
):
    _, val_loader = get_imagenet_loaders(batch_size=256)

    img_size = 224  # ImageNet 표준 해상도
    patch_size = 16  # 224/16 = 14x14 총 196개의 패치 생성
    embedding_size = 768  # ViT-Base 표준 임베딩 차원 (반드시 num_heads의 배수여야 함)
    num_class = 1000  # ImageNet-1K의 클래스 개수
    num_heads = 12  # 768 / 12 = head당 64차원 (표준 설정)

    model = ViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        embedding_size=embedding_size,
        num_class=num_class,
        num_heads=num_heads,
        in_channels=3,
    ).to(device)
    model.eval()

    # 1. pth 파일 목록 가져오기 및 정렬 (에포크 순서대로)
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    # 파일명에서 숫자를 추출하여 정렬 (예: epoch_10.pth -> 10)
    checkpoint_files.sort(
        key=lambda x: int("".join(filter(str.isdigit, os.path.basename(x))))
    )

    results = []

    print(f"Found {len(checkpoint_files)} checkpoints. Starting evaluation...")

    for cp_path in checkpoint_files:
        epoch_num = "".join(filter(str.isdigit, os.path.basename(cp_path)))
        print(f"\n[Epoch {epoch_num}] Loading {os.path.basename(cp_path)}...")

        # 모델 로드
        checkpoint = torch.load(cp_path, map_location=device)
        # 만약 체크포인트가 dict 형태(model_state_dict 등)라면 아래와 같이 수정 필요
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        top1_correct = 0
        top5_correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(
                val_loader, desc=f"Evaluating Epoch {epoch_num}", leave=False
            ):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                # Top-1 및 Top-5 정확도 계산
                _, pred = outputs.topk(5, 1, True, True)
                pred = pred.t()
                correct = pred.eq(targets.view(1, -1).expand_as(pred))

                top1_correct += (
                    correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
                )
                top5_correct += (
                    correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
                )
                total += targets.size(0)

        top1_acc = 100.0 * top1_correct / total
        top5_acc = 100.0 * top5_correct / total

        print(f"Done. Top-1: {top1_acc:.2f}%, Top-5: {top5_acc:.2f}%")

        results.append(
            {
                "epoch": int(epoch_num),
                "top1_acc": top1_acc,
                "top5_acc": top5_acc,
                "path": cp_path,
            }
        )

        # 중간 저장 (혹시 모를 중단 대비)
        df = pd.DataFrame(results)
        df.to_csv(save_name, index=False)

    print(f"\n✨ All evaluations finished! Results saved to {save_name}")

    # 최적의 모델 찾기
    best_row = df.loc[df["top1_acc"].idxmax()]
    print(
        f"🏆 Best Model: Epoch {best_row['epoch']} with {best_row['top1_acc']:.2f}% Top-1 Acc"
    )
    return df


def imagenet_1k_end_to_end_test():
    # 1. 모델, 데이터로더, 옵티마이저 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 224  # ImageNet 표준 해상도
    patch_size = 16  # 224/16 = 14x14 총 196개의 패치 생성
    embedding_size = 768  # ViT-Base 표준 임베딩 차원 (반드시 num_heads의 배수여야 함)
    num_class = 1000  # ImageNet-1K의 클래스 개수
    num_heads = 12  # 768 / 12 = head당 64차원 (표준 설정)

    model = ViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        embedding_size=embedding_size,
        num_class=num_class,
        num_heads=num_heads,
        in_channels=3,
    ).to(device)
    train_loader, _ = get_imagenet_loaders(batch_size=256)

    # 2. 혼합 정밀도를 위한 GradScaler 초기화
    epochs = 100
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scaler = GradScaler()
    train(
        epochs=epochs,
        device=device,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        scaler=scaler,
        scheduler=scheduler,
    )


def save_fsdp_model(
    model: nn.Module,
    optimizer: torch.optim.Adam,
    epoch: int,
    path: str,
):
    # 1. 모든 프로세스가 가중치를 모으도록 설정
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()

    # 2. Rank 0(마스터 GPU)에서만 파일로 기록
    if dist.get_rank() == 0:
        print(f"--> Saving checkpoint to {path}...")
        checkpoint = {
            "model_state": cpu_state,
            "optimizer_state": optimizer.state_dict(),  # 옵티마이저는 추가 처리가 복잡할 수 있음
            "epoch": epoch,
        }
        torch.save(checkpoint, path)
        print("--> Checkpoint saved.")


def imagenet_1k_multi_gpu_train_test():
    if not torch.cuda.is_available():
        print("Must CUDA Avalialbe")
        return

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()

    img_size = 224  # ImageNet 표준 해상도
    patch_size = 16  # 224/16 = 14x14 총 196개의 패치 생성
    embedding_size = 768  # ViT-Base 표준 임베딩 차원 (반드시 num_heads의 배수여야 함)
    num_class = 1000  # ImageNet-1K의 클래스 개수
    num_heads = 12  # 768 / 12 = head당 64차원 (표준 설정)
    epochs = 100

    model = ViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        embedding_size=embedding_size,
        num_class=num_class,
        num_heads=num_heads,
        in_channels=3,
    ).cuda()
    fsdp_model = FSDP(model)

    # DataSet
    train_loader, _, train_sampler = get_imagenet_loaders_fsdp(batch_size=256)
    optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scaler = GradScaler()
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch=epoch)
        fsdp_model.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = fsdp_model(inputs)

            optimizer.zero_grad()
            with torch.autocast():
                loss = criterion(outputs, targets)
                # 4. 스케일링된 Loss로 역전파(Backward)
                scaler.scale(loss).backward()

            # 5. 가중치 업데이트 (내부적으로 스케일 조정 및 Gradient Clipping 가능)
            scaler.step(optimizer)
            scaler.update()

            if rank == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        # [추가] 매 에폭 혹은 특정 주기에 저장
        if (epoch + 1) % 10 == 0:
            save_fsdp_model(
                fsdp_model, optimizer, epoch, f"vit_fsdp_epoch_{epoch+1}.pth"
            )
    dist.destroy_process_group()

    # # GPU 2개를 모두 사용하도록 설정
    # torchrun --nproc_per_node=2 test.py
