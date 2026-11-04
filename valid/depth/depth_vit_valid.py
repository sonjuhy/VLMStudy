from tqdm import tqdm
from typing import Tuple

import torch
import torch.nn as nn


@torch.no_grad()
def depth_vit_encoder_validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float, float]:
    """
    Returns:
        (val_loss, top1_acc, top5_acc)
    """
    model.eval()
    val_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0

    pbar = tqdm(val_loader, desc="Validating")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        val_loss += loss.item() * inputs.size(0)
        total += targets.size(0)

        # Top-1 / Top-5 동시 계산
        _, pred = outputs.topk(5, dim=1, largest=True, sorted=True)
        pred = pred.t()  # [5, Batch]
        correct = pred.eq(targets.view(1, -1).expand_as(pred))  # [5, Batch]
        top1_correct += correct[:1].reshape(-1).float().sum().item()
        top5_correct += correct[:5].reshape(-1).float().sum().item()

        pbar.set_postfix(
            loss=f"{val_loss/total:.4f}",
            top1=f"{100.*top1_correct/total:.2f}%",
            top5=f"{100.*top5_correct/total:.2f}%",
        )

    return val_loss / total, 100.0 * top1_correct / total, 100.0 * top5_correct / total
