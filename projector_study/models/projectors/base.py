from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseProjector(nn.Module, ABC):
    """
    모든 Projector의 공통 인터페이스.

    Input : (B, N,  v_dim)  — ViT patch 특징
    Output: (B, N', l_dim)  — LLM 입력 임베딩
    N'은 projector에 따라 N과 다를 수 있음 (Q-Former, Resampler, PixelShuffle).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
