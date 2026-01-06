import os
import torch
import torch.nn as nn

from torch.optim import AdamW
from dataloader.mnist_dataloader import (
    MNISTVLMDataset,
    vlm_collate_fn,
)
from vision.vit_model import ImageEmbeddingLayer, ViTModule
from transformers import PreTrainedModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


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


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # HyperCLOVA X의 경우 패딩 토큰 설정 필수
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 모델 로드
    llm = AutoModelForCausalLM.from_pretrained(model_name).to(device)  # type: ignore

    vit = MNISTViTEncoder(
        embedding_size=768,
        img_size=28,
        patch_size=7,
        return_token_type=True,
        num_heads=12,
    )
    vlm_model = MNISTViTHyperClovaX(vit, llm).to(device)

    # 파라미터 최적화 대상: ViT + Projector
    optimizer = AdamW(
        [
            {"params": vlm_model.vit.parameters(), "lr": 1e-5},
            {"params": vlm_model.projector.parameters(), "lr": 1e-4},
        ]
    )

    # MNIST 데이터셋 준비
    # 1. 표준 MNIST 로드 (원본 데이터)
    data_path = os.path.join("datasets", "mnist")
    raw_train_dataset = datasets.MNIST(root=data_path, train=True, download=True)

    # 2. 작성하신 MNISTVLMDataset으로 감싸기
    # 여기서 텍스트 템플릿과 토크나이징이 적용됩니다.
    train_vlm_dataset = MNISTVLMDataset(raw_train_dataset, tokenizer)

    # 3. DataLoader에 vlm_collate_fn 적용
    # collate_fn을 넣어야 리스트 형태의 출력을 텐서 배치로 묶어줍니다.
    train_loader = DataLoader(
        train_vlm_dataset, batch_size=64, shuffle=True, collate_fn=vlm_collate_fn
    )

    # 학습 루프
    vlm_model.train()
    for epoch in range(5):
        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            # 루나 레이크 AMX 가속을 위한 bfloat16 사용
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                outputs = vlm_model(
                    images=batch["images"].to(device),
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["input_ids"].to(device),
                )
                loss = outputs.loss

            loss.backward()
            optimizer.step()

    # 학습 완료 후 저장
    torch.save(vlm_model.vit.state_dict(), "vit_768_mnist.pth")
    torch.save(vlm_model.projector.state_dict(), "projector_768_to_1024.pth")
