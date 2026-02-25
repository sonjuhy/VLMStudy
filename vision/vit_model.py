import torch
import torch.nn as nn


class ImageEmbeddingLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: int,
        patch_size: int,
        embedding_size: int = 768,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.embedding_size = embedding_size

        self.div_img = nn.Conv2d(
            in_channels, self.embedding_size, kernel_size=patch_size, stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_size))
        self.position_div_imgs = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.embedding_size)
        )

    def forward(self, x):
        x = self.div_img(x)
        x = x.flatten(2).transpose(1, 2)
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
        if isinstance(x, tuple):
            x = x[0]
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


class Projector(nn.Module):
    def __init__(self, input_size: int, projection_size: int) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.GELU(),
            nn.Linear(input_size, projection_size),
            nn.LayerNorm(projection_size),
        )

    def forward(self, x):
        return self.projector(x)


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
        self.embedding = ImageEmbeddingLayer(
            in_channels, img_size, patch_size, embedding_size
        )

        self.layers = nn.ModuleList(
            [ViTModule(embedding_size, num_heads=num_heads) for _ in range(5)]
        )

        self.classifer = nn.Linear(embedding_size, num_class)

    def extract_features(self, x):
        # 1. 임베딩 레이어 통과
        x = self.embedding(x)

        # 2. Transformer 블록 통과
        for layer in self.layers:
            x = layer(x)

            # [중요] 만약 블록의 출력이 튜플이라면 텐서만 추출하여 업데이트
            if isinstance(x, tuple):
                x = x[0]

        # 3. 최종 결과물 x가 텐서인지 다시 한번 확인 (방어적 코딩)
        if isinstance(x, tuple):
            x = x[0]

        # 4. CLS 토큰 제외하고 196개의 패치 특징만 반환
        # 이제 x는 확실히 [Batch, 197, 768] 모양의 텐서입니다.
        return x[:, 1:, :]

    def forward(self, x):
        # 이미지 -> 패치 임베딩 (한 번만)
        x = self.embedding(x)

        # 트랜스포머 블록 반복
        for layer in self.layers:
            x, _ = layer(x)

        # CLS 토큰 추출 및 분류
        cls_token = x[:, 0]
        return self.classifer(cls_token)


class ViTDepthEncoder(nn.Module):
    def __init__(
        self,
        embedding_size: int = 768,
        img_size: int = 224,
        patch_size: int = 16,
        num_class: int = 1000,
        num_heads: int = 12,  # ViT-Base 표준은 12개입니다.
        in_channels: int = 4,  # [수정] RGB(3) + Depth(1) = 4
        n_layers: int = 12,  # ViT-Base 표준은 12레이어입니다.
    ):
        super().__init__()
        # 1. 패치 임베딩 (입력 채널 4개 수용)
        self.embedding = ImageEmbeddingLayer(
            in_channels, img_size, patch_size, embedding_size
        )

        # 2. Transformer 블록들 (기존 ModuleList를 TransformerEncoder 클래스로 통합 관리 권장)
        # 메모리 절약을 위해 n_layers를 조절 가능하게 변경
        self.transformer = TransformerEncoder(embedding_size, n_layer=n_layers)

        # 3. 분류 헤드
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_size), nn.Linear(embedding_size, num_class)
        )

    def extract_features(self, x):
        """VLM의 시각 인코더로 사용할 때 호출 (CLS 제외 패치 토큰만 반환)"""
        x = self.embedding(x)
        x = self.transformer(x)
        # x shape: [Batch, 197, 768] -> [Batch, 196, 768] (CLS 제외)
        return x[:, 1:, :]

    def forward(self, x):
        """ImageNet 1K 학습 시 호출 (CLS 토큰 활용)"""
        x = self.embedding(x)
        x = self.transformer(x)

        # CLS 토큰만 추출 [Batch, 768]
        cls_token = x[:, 0]
        return self.classifier(cls_token)
