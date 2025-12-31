import torch
import torch.nn as nn


class ImageEmbeddingLayer(nn.Module):
    def __init__(self, in_channels: int, img_size: int, patch_size: int):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.embedding_size = in_channels * patch_size**2

        self.div_img = nn.Conv2d(
            in_channels, self.embedding_size, kernel_size=patch_size, stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_size))
        self.position_div_imgs = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.embedding_size)
        )

    def forward(self, x):
        x = self.div_img(x)  # [B, E, H_p, W_p]
        x = x.flatten(2).transpose(1, 2)  # [B, Num_patches, E]

        b = x.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.position_div_imgs
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size: int, num_heads: int) -> None:
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.2,
        )
        self.query = nn.Linear(embedding_size, embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size)
        self.value = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_output, attention = self.multihead_attention(query, key, value)
        return attention_output, attention


class FeedForwardLayer(nn.Sequential):
    def __init__(
        self, embedding_size: int, expansion: int = 4, drop_out: float = 0.2
    ) -> None:
        super().__init__(
            nn.Linear(embedding_size, expansion * embedding_size),
            nn.GELU(),
            nn.Dropout(p=drop_out),
            nn.Linear(expansion * embedding_size, embedding_size),
        )


class ViTModule(nn.Module):
    def __init__(self, embedding_size: int, num_heads: int = 8):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(
            embedding_size, num_heads=num_heads
        )
        self.FFL = FeedForwardLayer(embedding_size)
        self.norm = nn.LayerNorm(embedding_size)

    def forward(self, x):
        # Attention + Skip Connection
        norm_x = self.norm(x)
        attn_out, attension = self.multihead_attention(norm_x)
        x = x + attn_out

        # FFL + Skip Connection
        x = x + self.FFL(self.norm(x))
        return x, attension


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_size: int, n_layer: int = 5) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ViTModule(
                    embedding_size=embedding_size,
                )
                for _ in range(n_layer)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)

        return x


class ViTEncoder(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        img_size: int,
        patch_size: int,
        num_class: int,
        num_heads: int = 8,
        in_channels: int = 1,
    ):
        super().__init__()
        # 임베딩은 한 번만 실행되도록 여기에 배치
        self.embedding = ImageEmbeddingLayer(in_channels, img_size, patch_size)

        self.layers = nn.ModuleList(
            [ViTModule(embedding_size, num_heads=num_heads) for _ in range(5)]
        )

        self.classifer = nn.Linear(embedding_size, num_class)

    def forward(self, x):
        # 이미지 -> 패치 임베딩 (한 번만)
        x = self.embedding(x)

        # 트랜스포머 블록 반복
        for layer in self.layers:
            x = layer(x)

        # CLS 토큰 추출 및 분류
        cls_token = x[:, 0]
        return self.classifer(cls_token)
