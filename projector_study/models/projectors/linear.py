import torch
import torch.nn as nn

from projector_study.models.projectors.base import BaseProjector


class LinearProjector(BaseProjector):
    """
    P1 — Linear Projection (baseline)

    파라미터 수: v_dim * l_dim  (768→4096: ~3.1M)
    출력 토큰 수: N (변화 없음)
    """

    def __init__(self, v_dim: int = 768, l_dim: int = 4096) -> None:
        super().__init__()
        self.proj = nn.Linear(v_dim, l_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x   : (B, N, v_dim)
        # out : (B, N, l_dim)
        return self.proj(x)

    @property
    def name(self) -> str:
        return "linear"
