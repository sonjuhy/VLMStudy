from vision.vit_model import ImageEmbeddingLayer, ViTModule
from transformers import PreTrainedModel

import torch
import torch.nn as nn


class MNISTViTEncoder(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        img_size: int,
        patch_size: int,
        return_token_type: bool,  # True: Origin Data Token, False: CLS Token
        num_heads: int = 8,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.in_channels = in_channels

        self.return_token_type = return_token_type
        self.embedding = ImageEmbeddingLayer(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            embedding_size=embedding_size,
        )

        self.layers = nn.ModuleList(
            [
                ViTModule(embedding_size=embedding_size, num_heads=num_heads)
                for _ in range(5)
            ]
        )

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x, _ = layer(x)

        if self.return_token_type:
            return x  # Origin Data Token
        else:
            return x[:, 0]  # CLS Token


class MNISTViTHyperClovaX(nn.Module):
    def __init__(
        self,
        vit_encoder: MNISTViTEncoder,
        llm_model: PreTrainedModel,
        llm_hidden_size: int = 1024,
    ) -> None:
        super().__init__()
        self.vit = vit_encoder
        self.llm = llm_model
        self.projector = nn.Linear(self.vit.embedding_size, llm_hidden_size)

        # LLM은 학습에서 제외 (Frozen)
        for param in self.llm.parameters():
            param.requires_grad = False

        # ViT와 Projector만 학습 가능하도록 설정
        self.vit.train()
        self.projector.train()

    def forward(self, images, input_ids, attention_mask, labels=None):
        # ViT 특징 추출 (Trainable)
        visual_tokens = self.vit(images)
        # Projector 통과 (Trainable)
        image_embeds = self.projector(visual_tokens)

        # 텍스트 임베딩 및 결합
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        # Mask 및 Labels 설정
        visual_mask = torch.ones(
            (images.size(0), image_embeds.size(1)),
            device=images.device,
            dtype=attention_mask.dtype,
        )
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)

        if labels is not None:
            visual_labels = torch.full(
                (images.size(0), image_embeds.size(1)),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            combined_labels = torch.cat([visual_labels, labels], dim=1)
            return self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=combined_mask,
                labels=combined_labels,
            )

        return self.llm(inputs_embeds=inputs_embeds, attention_mask=combined_mask)
