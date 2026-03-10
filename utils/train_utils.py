import torch
import torch.nn as nn

import numpy as np


class DropPath(nn.Module):
    """
    Stochastic Depth (DropPath) 모듈
    참고: 기존 가중치를 불러올 때 이 모듈은 학습되는 파라미터가 없으므로
    에러 없이 100% 완벽하게 호환됩니다.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        # 배치 차원(차원 0)을 제외한 나머지 차원을 1로 설정하여 브로드캐스팅
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        # 확률에 따라 0 또는 1을 생성
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)

        return x * random_tensor


def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """배치 데이터에 Mixup을 적용합니다."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    # 배치의 순서를 무작위로 섞음
    index = torch.randperm(batch_size).to(x.device)

    # 이미지 섞기
    mixed_x = lam * x + (1 - lam) * x[index, :]

    # 기존 라벨, 섞인 라벨, 비율을 반환 (Loss 계산용)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Mixup이 적용된 라벨에 대해 Loss를 계산합니다."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
