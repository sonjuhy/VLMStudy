import torch
import torch.nn as nn

from projector_study.models.projectors.base import BaseProjector


class ResamplerLayer(nn.Module):
    """
    Perceiver-style 단일 레이어 (Flamingo).

    Q-Former와의 차이: self-attn 없음.
    latent가 (latent + visual) 전체를 KV로 참조 → 자기 자신도 갱신 가능.
    """

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)

    def forward(self, latents: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        # latents: (B, L, hidden_dim)
        # visual : (B, N, hidden_dim)

        # KV = latent와 visual 합산 → latent가 visual + 자신 모두 참조
        kv = self.norm_kv(torch.cat([latents, visual], dim=1))    # (B, L+N, hidden_dim)
        q  = self.norm1(latents)
        latents = latents + self.cross_attn(q, kv, kv)[0]         # (B, L, hidden_dim)
        latents = latents + self.ffn(self.norm2(latents))          # (B, L, hidden_dim)
        return latents


class ResamplerProjector(BaseProjector):
    """
    P4 — Perceiver Resampler (Flamingo / IDEFICS)

    고정 크기 Latent 토큰이 Cross-Attention으로 시각 특징 집약.
    파라미터 수: ~50M (num_latents=64, num_layers=6 기준)
    출력 토큰 수: num_latents (N → L, 시퀀스 압축)
    """

    def __init__(
        self,
        v_dim: int = 768,
        l_dim: int = 4096,
        hidden_dim: int = 1024,
        num_latents: int = 64,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 4096,
    ) -> None:
        super().__init__()
        self.num_latents = num_latents

        self.latents    = nn.Parameter(torch.randn(1, num_latents, hidden_dim))
        self.input_proj = nn.Linear(v_dim, hidden_dim) if v_dim != hidden_dim else nn.Identity()

        self.layers = nn.ModuleList([
            ResamplerLayer(hidden_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.norm     = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, l_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x   : (B, N, v_dim)
        B = x.size(0)

        visual  = self.input_proj(x)                              # (B, N, hidden_dim)
        latents = self.latents.expand(B, -1, -1)                  # (B, L, hidden_dim)

        for layer in self.layers:
            latents = layer(latents, visual)                      # (B, L, hidden_dim)

        latents = self.norm(latents)                              # (B, L, hidden_dim)
        # out : (B, L, l_dim)
        return self.out_proj(latents)

    @property
    def name(self) -> str:
        return "resampler"
