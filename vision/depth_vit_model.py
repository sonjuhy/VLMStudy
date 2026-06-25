from vision.vit_model import Projector, ViTDepthEncoder
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from torch import autocast, GradScaler

import os
import torch
import torch.nn as nn
import torch.optim as optim


class MultiModalProjector(nn.Module):
    """
    시각 특징(Vision Features)을 언어 모델의 임베딩 공간으로 변환하는 투사 계층입니다.
    LLaVA 아키텍처의 표준인 2-Layer MLP 구조를 사용합니다.
    """

    def __init__(self, vision_dim: int, language_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(vision_dim, language_dim)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(language_dim, language_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.gelu(self.linear_1(x)))


class DepthVLM(nn.Module):
    def __init__(
        self,
        vit: ViTDepthEncoder,
        llm_model: PreTrainedModel,
        vision_dim: int = 768,
        llm_hidden_size: int = 4096,
    ):
        super().__init__()
        self.vit_encoder: ViTDepthEncoder = vit
        self.llm: PreTrainedModel = llm_model
        self.llm_hidden_size: int = llm_hidden_size
        self.projector = MultiModalProjector(vision_dim, self.llm_hidden_size)

        # LLM은 학습에서 제외 (Frozen)
        for param in self.vit_encoder.parameters():
            param.requires_grad = False
        for param in self.llm.parameters():
            param.requires_grad = False

        # Projector만 학습 가능하도록 설정
        for param in self.projector.parameters():
            param.requires_grad = True

    def forward(self, images, input_ids, labels=None):
        # 1. ViT에서 이미지 특징 추출 (Batch, 196, 768)
        dtype = self.llm.dtype
        images = images.to(device=images.device, dtype=dtype)
        with torch.no_grad():
            image_features = self.vit_encoder.extract_features(images)
            image_features = image_features.to(dtype)

        # 2. Projector 차원 변환 (Batch, 196, 4096)
        image_embeddings = self.projector(image_features)

        # 3. 텍스트 임베딩 (Batch, Seq_Len, 4096)
        text_embeddings = self.llm.get_input_embeddings()(input_ids)

        # 4. 결합 [Image; Text] (Batch, 196 + Seq_Len, 4096)
        inputs_embeds = torch.cat([image_embeddings, text_embeddings], dim=1)

        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device
        num_image_tokens = 196

        # 5. Labels 길이 맞추기
        # 이미지 토큰 위치(196개)에는 정답이 없으므로 -100(Ignore Index)으로 채웁니다.
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            image_mask = torch.ones(
                (batch_size, num_image_tokens),
                dtype=attention_mask.dtype,
                device=device,
            )
            full_attention_mask = torch.cat([image_mask, attention_mask], dim=1)
        else:
            full_attention_mask = None

        # 6. Labels 길이 맞추기
        if labels is not None:
            labels = labels.to(device)
            # -100은 해당 토큰을 무시하라는 의미
            ignore_labels = torch.full(
                (batch_size, num_image_tokens), -100, dtype=labels.dtype, device=device
            )
            full_labels = torch.cat([ignore_labels, labels], dim=1)
        else:
            full_labels = None

        # 7. LLM 통과
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )
        return outputs
