import torch
import torch.nn as nn

from projector_study.models.projectors.base import BaseProjector


class QFormerLayer(nn.Module):
    """Self-Attn(query) + Cross-Attn(query→visual) + FFN — BLIP-2 스타일 단일 레이어."""

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, queries: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        # queries: (B, Q, hidden_dim)
        # visual : (B, N, hidden_dim)

        # 1. Self-attention: query끼리 상호작용
        q = self.norm1(queries)
        queries = queries + self.self_attn(q, q, q)[0]           # (B, Q, hidden_dim)

        # 2. Cross-attention: query가 visual 특징 참조
        q = self.norm2(queries)
        queries = queries + self.cross_attn(q, visual, visual)[0] # (B, Q, hidden_dim)

        # 3. FFN
        queries = queries + self.ffn(self.norm3(queries))         # (B, Q, hidden_dim)
        return queries


class QFormerProjector(BaseProjector):
    """
    P3 — Q-Former (BLIP-2)

    학습 가능한 Query 토큰이 Cross-Attention으로 ViT 특징 압축.
    파라미터 수: ~100M+ (num_queries=32, num_layers=6 기준)
    출력 토큰 수: num_queries (N → Q, 시퀀스 압축)
    """

    def __init__(
        self,
        v_dim: int = 768,
        l_dim: int = 4096,
        hidden_dim: int = 768,
        num_queries: int = 32,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 3072,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries

        # 학습 가능한 query 토큰
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim))

        # visual 특징을 hidden_dim으로 맞추는 입력 projection
        self.input_proj = nn.Linear(v_dim, hidden_dim) if v_dim != hidden_dim else nn.Identity()

        self.layers = nn.ModuleList([
            QFormerLayer(hidden_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # 최종 l_dim으로 출력
        self.out_proj = nn.Linear(hidden_dim, l_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x   : (B, N, v_dim)
        B = x.size(0)

        visual = self.input_proj(x)                               # (B, N, hidden_dim)
        queries = self.query_tokens.expand(B, -1, -1)             # (B, Q, hidden_dim)

        for layer in self.layers:
            queries = layer(queries, visual)                      # (B, Q, hidden_dim)

        queries = self.norm(queries)                              # (B, Q, hidden_dim)
        # out : (B, Q, l_dim)
        return self.out_proj(queries)

    @property
    def name(self) -> str:
        return "qformer"
