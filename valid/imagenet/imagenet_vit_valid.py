from tqdm import tqdm
from typing import Tuple

import torch
import torch.nn as nn
import torch.distributed as dist


@torch.no_grad()
def imagenet_vit_encoder_validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float, float]:
    """
    Returns:
        (val_loss, top1_acc, top5_acc)

    Top-1 / Top-5 는 ImageNet 분류 벤치마크의 표준 지표입니다.
    - Top-1: 모델의 최고 예측 클래스가 정답인 비율
    - Top-5: 상위 5개 예측 중 정답이 포함된 비율 (클래스 1000개 기준 표준 지표)
    """
    model.eval()
    is_distributed = dist.is_available() and dist.is_initialized()
    is_main_process = (not is_distributed) or dist.get_rank() == 0

    val_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0

    pbar = tqdm(val_loader, desc="Validating", disable=not is_main_process)
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

        if is_main_process:
            pbar.set_postfix(
                loss=f"{val_loss/total:.4f}",
                top1=f"{100.*top1_correct/total:.2f}%",
                top5=f"{100.*top5_correct/total:.2f}%",
            )

    if is_distributed:
        metrics = torch.tensor(
            [val_loss, top1_correct, top5_correct, total],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        val_loss, top1_correct, top5_correct, total = metrics.tolist()

    return val_loss / total, 100.0 * top1_correct / total, 100.0 * top5_correct / total
