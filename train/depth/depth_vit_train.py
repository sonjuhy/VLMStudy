from valid.depth.depth_vit_valid import depth_vit_encoder_validate
from utils.utils import save_checkpoint
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from torch import autocast, GradScaler

import os
import torch
import torch.nn as nn
import torch.optim as optim


def depth_vit_encoder_train(
    epochs: int,
    start_epoch: int,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    scheduler: LRScheduler,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader = None,
    accumulation_steps: int = 16,  # 16(Batch) * 16 = 실제 배치 256 효과
    best_acc: float = 0.0,
):
    print(
        f"Starting training from Epoch {start_epoch} with current Best Acc: {best_acc:.2f}%"
    )

    for epoch in range(start_epoch - 1, epochs):
        model.train()
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}")

        # Validation
        val_loss, val_acc = depth_vit_encoder_validate(
            model, val_loader, criterion, device
        )
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # --- [추가] 로그 유실 방지: CSV 파일에 기록 ---
        with open("train_log.csv", "a", encoding="utf-8") as f:
            # 파일이 비어있다면 헤더 작성
            if os.path.getsize("train_log.csv") == 0:
                f.write("epoch,val_loss,val_acc\n")
            f.write(f"{epoch+1},{val_loss:.4f},{val_acc:.2f}\n")
        # -------------------------------------------

        # 매 5에폭마다 혹은 마지막에 체크포인트 저장
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            current_loss = loss.item() * accumulation_steps
            save_checkpoint(epoch + 1, model, optimizer, scaler, current_loss)

        # Best 모델 저장
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

        # 스케줄러 업데이트
        scheduler.step()
