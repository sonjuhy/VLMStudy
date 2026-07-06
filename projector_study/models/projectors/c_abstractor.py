import math

import torch
import torch.nn as nn

from projector_study.models.projectors.base import BaseProjector


class CAbstractorProjector(BaseProjector):
    """
    P5 — C-Abstractor (HoneyBee)

    ViT 출력을 2D 공간으로 reshape → Depthwise Separable Conv → Linear.
    지역적 공간 구조를 유지하면서 경량으로 작동.
    파라미터 수: ~10M
    출력 토큰 수: N (변화 없음)
    """

    def __init__(
        self,
        v_dim: int = 768,
        l_dim: int = 4096,
        num_patches: int = 196,  # 14×14
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.h = self.w = int(math.isqrt(num_patches))

        # Depthwise conv: 각 채널 독립 처리로 공간 구조 집약
        self.dw_conv = nn.Conv2d(
            v_dim, v_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=v_dim,
            bias=False,
        )
        # Pointwise conv: 채널 간 정보 혼합
        self.pw_conv = nn.Conv2d(v_dim, v_dim, kernel_size=1, bias=False)

        self.norm = nn.LayerNorm(v_dim)
        self.proj = nn.Linear(v_dim, l_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x   : (B, N, v_dim)  N = H*W
        B, N, C = x.shape

        # (B, N, C) → (B, C, H, W)
        feat = x.permute(0, 2, 1).reshape(B, C, self.h, self.w)

        # Depthwise Separable Conv
        feat = self.dw_conv(feat)                                 # (B, C, H, W)
        feat = self.pw_conv(feat)                                 # (B, C, H, W)

        # (B, C, H, W) → (B, N, C)
        feat = feat.reshape(B, C, N).permute(0, 2, 1)

        feat = self.norm(x + feat)                               # residual + norm
        # out : (B, N, l_dim)
        return self.proj(feat)

    @property
    def name(self) -> str:
        return "c_abstractor"
