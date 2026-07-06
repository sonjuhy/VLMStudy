import math

import torch
import torch.nn as nn

from projector_study.models.projectors.base import BaseProjector


class PixelShuffleProjector(BaseProjector):
    """
    P6 — Pixel Shuffle (InternVL2)

    인접 2x2 패치를 채널 방향으로 합산(unshuffle)해 토큰 수를 1/4로 줄인 뒤 Linear 투영.
    공간 구조를 유지하면서 LLM 입력 시퀀스를 단축 → 추론 속도 개선.

    파라미터 수: ~12M  (4*v_dim → l_dim: 4*768*4096 ≈ 12.6M)
    출력 토큰 수: N/4 (196 → 49)
    """

    def __init__(
        self,
        v_dim: int = 768,
        l_dim: int = 4096,
        num_patches: int = 196,  # 14×14
        scale_factor: int = 2,
    ) -> None:
        super().__init__()
        self.h = self.w = int(math.isqrt(num_patches))
        self.scale_factor = scale_factor

        # scale_factor² 배로 늘어난 채널을 l_dim으로 압축
        merged_dim = v_dim * (scale_factor ** 2)
        self.proj = nn.Sequential(
            nn.LayerNorm(merged_dim),
            nn.Linear(merged_dim, l_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x   : (B, N, v_dim)   N = H * W
        B, N, C = x.shape
        s = self.scale_factor

        # (B, N, C) → (B, C, H, W)
        feat = x.permute(0, 2, 1).reshape(B, C, self.h, self.w)

        # PixelUnshuffle: (B, C, H, W) → (B, C*s², H/s, W/s)
        feat = nn.functional.pixel_unshuffle(feat, s)             # (B, C*s², H/s, W/s)

        H2, W2 = self.h // s, self.w // s
        # (B, C*s², H/s, W/s) → (B, H/s * W/s, C*s²)
        feat = feat.reshape(B, C * s * s, H2 * W2).permute(0, 2, 1)

        # out : (B, N/s², l_dim)
        return self.proj(feat)

    @property
    def name(self) -> str:
        return "pixel_shuffle"
