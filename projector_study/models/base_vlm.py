import torch
import torch.nn as nn
from transformers import PreTrainedModel

from projector_study.models.projectors.base import BaseProjector
from vision.vit_model import ViTEncoder


class ProjectorVLM(nn.Module):
    """
    ViT (Frozen) + Projector (학습 대상) + LLM (Frozen) 공통 래퍼.

    Projector만 교체하여 P1~P6 비교 실험을 진행한다.
    LLM 입력 시퀀스 구조: [image_tokens ; text_tokens]

    image_tokens 수는 projector에 따라 다름:
      - Linear / MLP / C-Abstractor : 196 tokens
      - Q-Former                    :  32 tokens
      - Resampler                   :  64 tokens
      - Pixel Shuffle               :  49 tokens
    """

    def __init__(
        self,
        vit: ViTEncoder,
        projector: BaseProjector,
        llm: PreTrainedModel,
    ) -> None:
        super().__init__()
        self.vit       = vit
        self.projector = projector
        self.llm       = llm

        # ViT와 LLM은 Freeze
        for p in self.vit.parameters():
            p.requires_grad = False
        for p in self.llm.parameters():
            p.requires_grad = False

        # Projector만 학습
        for p in self.projector.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        # images : (B, 3, 224, 224)
        dtype = self.llm.dtype
        images = images.to(dtype=dtype)
        with torch.no_grad():
            feats = self.vit.extract_features(images)  # (B, 196, 768)
        feats = feats.to(dtype=dtype)
        return self.projector(feats)                   # (B, N', l_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> "CausalLMOutputWithPast":
        # 1. 이미지 인코딩 + Projector
        img_emb = self._encode_image(images)           # (B, N', l_dim)
        N_img   = img_emb.size(1)

        # 2. 텍스트 임베딩
        txt_emb = self.llm.get_input_embeddings()(input_ids)  # (B, T, l_dim)

        # 3. [Image ; Text] 연결
        inputs_embeds = torch.cat([img_emb, txt_emb], dim=1)  # (B, N'+T, l_dim)

        # 4. Attention Mask 확장
        if attention_mask is not None:
            B = images.size(0)
            img_mask = torch.ones(
                (B, N_img), device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([img_mask, attention_mask], dim=1)  # (B, N'+T)

        # 5. Labels 확장 — 이미지 토큰 위치는 -100 (loss 무시)
        if labels is not None:
            B = images.size(0)
            ignore = torch.full(
                (B, N_img), -100, device=labels.device, dtype=labels.dtype
            )
            labels = torch.cat([ignore, labels], dim=1)                    # (B, N'+T)

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        img_emb = self._encode_image(images)                               # (B, N', l_dim)
        N_img   = img_emb.size(1)
        txt_emb = self.llm.get_input_embeddings()(input_ids)              # (B, T, l_dim)
        inputs_embeds = torch.cat([img_emb, txt_emb], dim=1)              # (B, N'+T, l_dim)

        if attention_mask is not None:
            B = images.size(0)
            img_mask = torch.ones(
                (B, N_img), device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([img_mask, attention_mask], dim=1)

        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    # ------------------------------------------------------------------
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
