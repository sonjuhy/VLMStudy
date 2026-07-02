from utils.train_utils import mixup_criterion, mixup_data
from valid.imagenet.imagenet_vit_valid import imagenet_vit_encoder_validate
from utils.utils import save_checkpoint
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from torch import autocast, GradScaler
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist


def _log(log_path: str, message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def imagenet_vit_encoder_train(
    epochs: int,
    start_epoch: int,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    scheduler: LRScheduler,
    device: torch.device,
    log_path: str,
    train_sampler=None,
    val_loader: torch.utils.data.DataLoader = None,
    accumulation_steps: int = 1,
    best_acc: float = 0.0,
    patience: int = 20,
):
    # DDP로 실행된 경우(torchrun) rank0만 로그/체크포인트를 기록하도록 구분
    is_distributed = dist.is_available() and dist.is_initialized()
    is_main_process = (not is_distributed) or dist.get_rank() == 0

    if is_main_process:
        _log(
            log_path,
            f"Starting training from Epoch {start_epoch} with current Best Acc: {best_acc:.2f}%",
        )

    epochs_without_improvement = 0

    for epoch in range(start_epoch - 1, epochs):
        # DistributedSampler는 매 epoch마다 시드를 갱신해야 shuffle이 제대로 동작함
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad()

        running_train_loss = 0.0
        train_correct = 0.0
        train_total = 0

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not is_main_process
        )
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            use_mixup = np.random.rand() < 0.5
            if use_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(
                    inputs, targets, alpha=0.8
                )
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(inputs)
                if use_mixup:
                    loss = mixup_criterion(
                        criterion, outputs, targets_a, targets_b, lam
                    )
                else:
                    loss = criterion(outputs, targets)

                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            with torch.no_grad():
                _, predicted = outputs.max(1)
                batch_size = targets.size(0)
                train_total += batch_size

                if use_mixup:
                    correct_a = (predicted == targets_a).sum().item()
                    correct_b = (predicted == targets_b).sum().item()
                    train_correct += (lam * correct_a) + ((1 - lam) * correct_b)
                else:
                    train_correct += (predicted == targets).sum().item()

                running_train_loss += loss.item() * accumulation_steps * batch_size

            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if is_main_process:
                pbar.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}")

        # DDP 환경에서는 각 rank가 데이터 샤드만 처리하므로 전체 지표를 합산
        if is_distributed:
            metrics = torch.tensor(
                [running_train_loss, train_correct, train_total],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            running_train_loss, train_correct, train_total = metrics.tolist()

        epoch_train_loss = running_train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total

        val_loss, val_top1, val_top5 = imagenet_vit_encoder_validate(
            model, val_loader, criterion, device
        )

        if is_main_process:
            current_lr = optimizer.param_groups[0]["lr"]
            _log(
                log_path,
                f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | "
                f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}%",
            )

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                model_to_save = model.module if hasattr(model, "module") else model
                save_checkpoint(
                    epoch + 1,
                    model_to_save,
                    optimizer,
                    scaler,
                    epoch_train_loss,
                    val_top1,
                    path="checkpoints/final_model",
                    filename=f"vit_imagenet_1k_checkpoint_epoch_{epoch+1}.pth",
                )

        if val_top1 > best_acc:
            best_acc = val_top1
            epochs_without_improvement = 0
            if is_main_process:
                _log(log_path, f"New Best Accuracy! Top-1: {val_top1:.2f}% | Top-5: {val_top5:.2f}% — Saving model...")
                model_to_save = model.module if hasattr(model, "module") else model
                save_checkpoint(
                    epoch + 1,
                    model_to_save,
                    optimizer,
                    scaler,
                    val_loss,
                    best_acc,
                    path="checkpoints/final_model",
                    filename="best_vit_imagenet_1k.pth",
                )
        else:
            epochs_without_improvement += 1

        scheduler.step()

        if epochs_without_improvement >= patience:
            if is_main_process:
                _log(
                    log_path,
                    f"Early stopping triggered at epoch {epoch+1} "
                    f"(no improvement for {patience} epochs, best val acc: {best_acc:.2f}%)",
                )
            break

    return best_acc
