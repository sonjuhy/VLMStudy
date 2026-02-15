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


class DepthVLM(nn.Module):
    def __init__(
        self,
        vit: ViTDepthEncoder,
        llm_model: PreTrainedModel,
        llm_hidden_size: int = 4096,
    ):
        super().__init__()
        self.vit_encoder: ViTDepthEncoder = vit
        self.llm: PreTrainedModel = llm_model
        self.llm_hidden_size: int = llm_hidden_size
        self.projector: Projector = Projector(
            input_size=768, projection_size=self.llm_hidden_size
        )

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

        # 5. Labels 길이 맞추기
        # 이미지 토큰 위치(196개)에는 정답이 없으므로 -100(Ignore Index)으로 채웁니다.
        if labels is not None:
            # labels: [Batch, Seq_Len]
            labels = labels.to(device=images.device)
            device = labels.device
            batch_size = labels.shape[0]

            # 이미지 토큰 개수만큼 -100 채우기
            ignore_labels = torch.full((batch_size, 196), -100, device=device)
            # 최종 결합: [Batch, 196 + Seq_Len]
            full_labels = torch.cat([ignore_labels, labels], dim=1)
        else:
            full_labels = None

        # 6. LLM 통과
        outputs = self.llm(inputs_embeds=inputs_embeds, labels=full_labels)
        return outputs
