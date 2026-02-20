from tqdm import tqdm
from vision.vit_model import ViTDepthEncoder
from dataloader.rgbd_imagenet_1k_dataloader import get_rgbd_imagenet_loaders

import os
import torch
import torch.nn as nn


@torch.no_grad()  # 검증 시 메모리 절약 필수
def depth_vit_encoder_validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
):
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
