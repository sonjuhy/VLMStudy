from vision.vit_model import ViTEncoder
from dataloader.imagenet_1k_dataloader import get_imagenet_loaders_fsdp
from train.imagenet.imagenet_vit_train import imagenet_vit_encoder_train
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import date, datetime

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist


def imagenet_vit_end_to_end(
    epochs: int = 300,
    warmup_epochs: int = 20,
    start_epoch: int = 1,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.05,
    patience: int = 60,
    resume_checkpoint: str = "",
):
    """
    실행 방법 (2-GPU DDP, 레포 루트에서):
        torchrun --nproc_per_node=2 -m end_to_end.imagenet_ete

    재개 방법:
        imagenet_vit_end_to_end(resume_checkpoint="checkpoints/final_model/best_vit_imagenet_1k.pth")
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA가 필요합니다.")

    # 이 서버는 NCCL의 CUMEM 기반 GPU P2P 전송 경로에서 핸드셰이크는 성공하지만
    # 실제 데이터 전송 시 영구 hang이 발생함이 실측으로 확인됨 (드라이버/NCCL 조합 이슈로 추정).
    # P2P만 비활성화하면 SHM 경로로 정상 동작하며, 실측상 단일 GPU 대비 약 1.96x로
    # 거의 선형에 가까운 DDP 스케일링이 나오므로 성능 손실은 미미함.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")

    # --- DDP 초기화 (torchrun이 RANK/WORLD_SIZE/LOCAL_RANK 환경변수를 설정해줌) ---
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    is_main_process = dist.get_rank() == 0

    # --- 체크포인트 로드 (DDP 래핑 전, bare model에 state dict 적용) ---
    ckpt = None
    best_acc = 0.0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        ckpt = torch.load(resume_checkpoint, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        if is_main_process:
            print(f"[Resume] Checkpoint loaded: epoch={ckpt['epoch']}, best_acc={best_acc:.2f}%")

    # --- 학습 시작 날짜로 로그 폴더 생성 ---
    start_date = date.today().isoformat()
    log_dir = os.path.join("logs", "imagenet_vit", start_date)
    log_path = os.path.join(log_dir, "train_log.txt")
    if is_main_process:
        os.makedirs(log_dir, exist_ok=True)

    model_config = {
        "vit_version": "ViT-Base/16 (official spec)",
        "embedding_size": 768,
        "img_size": 224,
        "patch_size": 16,
        "num_heads": 12,
        "n_layers": 12,
        "num_class": 1000,
    }

    model = ViTEncoder(
        embedding_size=model_config["embedding_size"],
        img_size=model_config["img_size"],
        patch_size=model_config["patch_size"],
        num_class=model_config["num_class"],
        num_heads=model_config["num_heads"],
        in_channels=3,
        n_layers=model_config["n_layers"],
    ).to(device)

    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])

    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # Linear Warmup -> Cosine Annealing (DeiT 학습 레시피)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # 재개 시 스케줄러를 resume 지점까지 fast-forward (LR 위치 복원)
    if ckpt is not None:
        for _ in range(start_epoch - 1):
            scheduler.step()
        if is_main_process:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"[Resume] Scheduler fast-forwarded to epoch {start_epoch}, LR={current_lr:.6f}")

    scaler = torch.GradScaler()

    train_loader, val_loader, train_sampler = get_imagenet_loaders_fsdp(
        data_dir="datasets/imagenet_1k", batch_size=batch_size
    )

    if is_main_process:
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        resume_note = f"Resumed from: {resume_checkpoint} (epoch {start_epoch - 1}, best_acc={best_acc:.2f}%)\n" if ckpt else ""
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(f"Training Start: {start_time}\n")
            if resume_note:
                f.write(resume_note)
            f.write(f"ViT Version: {model_config['vit_version']}\n")
            f.write(
                f"Model Config: embedding_size={model_config['embedding_size']}, "
                f"num_heads={model_config['num_heads']}, "
                f"n_layers={model_config['n_layers']}, "
                f"patch_size={model_config['patch_size']}, "
                f"img_size={model_config['img_size']}, "
                f"num_class={model_config['num_class']}\n"
            )
            f.write("Dataset: ImageNet-1K (ILSVRC2012, 1000 classes)\n")
            f.write(
                "Augmentations: RandomResizedCrop(224), RandomHorizontalFlip, "
                "AutoAugment(ImageNet policy), Mixup(alpha=0.8, p=0.5), "
                "Label Smoothing(0.1)\n"
            )
            f.write(
                f"Hyperparameters: epochs={epochs}, warmup_epochs={warmup_epochs}, "
                f"batch_size={batch_size}(per GPU) x world_size={world_size} GPUs, "
                f"optimizer=AdamW(lr={learning_rate}, weight_decay={weight_decay}), "
                f"AMP=bfloat16, scheduler=LinearWarmup+CosineAnnealing, "
                f"early_stopping_patience={patience}\n"
            )
            f.write("=" * 70 + "\n")

    best_acc = imagenet_vit_encoder_train(
        epochs=epochs,
        start_epoch=start_epoch,
        model=model,
        train_loader=train_loader,
        train_sampler=train_sampler,
        optimizer=optimizer,
        criterion=criterion,
        scaler=scaler,
        scheduler=scheduler,
        device=device,
        log_path=log_path,
        val_loader=val_loader,
        patience=patience,
        best_acc=best_acc,
    )

    if is_main_process:
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Training End: {end_time}\n")
            f.write(f"Best Val Acc: {best_acc:.2f}%\n")
            f.write("=" * 70 + "\n\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    imagenet_vit_end_to_end(
        resume_checkpoint="checkpoints/final_model/best_vit_imagenet_1k.pth",
    )
