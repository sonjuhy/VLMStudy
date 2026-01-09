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
from torchvision import datasets, transforms
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


def train(epochs: int = 10):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # HyperCLOVA X의 경우 패딩 토큰 설정 필수
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 모델 로드
    llm = AutoModelForCausalLM.from_pretrained(model_name).to(device)

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

    # 2. 커스텀 MNISTVLMDataset으로 감싸기
    # 여기서 텍스트 템플릿과 토크나이징이 적용.
    train_vlm_dataset = MNISTVLMDataset(raw_train_dataset, tokenizer)

    # 3. DataLoader에 vlm_collate_fn 적용
    # collate_fn을 넣어야 리스트 형태의 출력을 텐서 배치로 결합.
    train_loader = DataLoader(
        train_vlm_dataset, batch_size=64, shuffle=True, collate_fn=vlm_collate_fn
    )

    # 학습 루프
    vlm_model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            optimizer.zero_grad()

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

            # 통계 업데이트
            current_loss = loss.item()
            epoch_loss += current_loss

            # tqdm 상태바에 현재 배치의 Loss 표시
            pbar.set_postfix(loss=f"{current_loss:.4f}")

        # 한 에포크가 끝나면 평균 Loss 출력
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(
            f"\n>>> Epoch [{epoch+1}/5] Completed. Average Loss: {avg_epoch_loss:.4f}"
        )
        print("-" * 50)

    # 학습 완료 후 저장
    torch.save(vlm_model.vit.state_dict(), "vit_768_mnist.pth")
    torch.save(vlm_model.projector.state_dict(), "projector_768_to_1024.pth")


def valid():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"

    ##########DataSet###########
    # 1. 이미지 전처리 정의 (학습 시와 동일해야 함)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # 2. 데이터셋에서 샘플 하나 추출
    test_ds = datasets.MNIST(root="./datasets/mnist", train=False, download=True)
    sample_img, sample_label = test_ds[0]  # 첫 번째 데이터 (숫자 7)

    # 3. 텐서 변환 및 배치 차원 추가 [1, 1, 28, 28]
    input_tensor = transform(sample_img).unsqueeze(0)

    ##############Inference###################
    # 1. 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # 학습 시와 동일한 구조의 ViT 선언
    vit = MNISTViTEncoder(
        embedding_size=768,
        img_size=28,
        patch_size=7,
        return_token_type=True,
        num_heads=12,
    ).to(device)

    # VLM 클래스
    vlm_model = MNISTViTHyperClovaX(vit, llm).to(device)

    # 2. 학습된 가중치 로드
    vlm_model.vit.load_state_dict(torch.load("vit_768_mnist.pth", map_location=device))
    vlm_model.projector.load_state_dict(
        torch.load("projector_768_to_1024.pth", map_location=device)
    )
    vlm_model.eval()

    # 3. 추론용 프롬프트 구성 (답변 직전까지만 입력)
    prompt = "질문: 이 이미지에 있는 숫자는 무엇인가요?\n답변: 이 숫자는"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 4. 생성 (Inference)
    with torch.no_grad():
        target_device_type = "cpu" if device == "cpu" else "cuda"
        with torch.autocast(device_type=target_device_type, dtype=torch.bfloat16):
            # 이미지 특징 추출 및 프로젝션
            visual_tokens = vlm_model.vit(input_tensor.to(device))
            image_embeds = vlm_model.projector(visual_tokens)

            # 텍스트 임베딩 추출
            text_embeds = vlm_model.llm.get_input_embeddings()(inputs["input_ids"])

            # [이미지 임베딩 + 텍스트 임베딩] 결합
            combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)

            output_ids = vlm_model.llm.generate(
                inputs_embeds=combined_embeds,
                max_new_tokens=10,  # " 5입니다." 정도만 생성하면 되므로 짧게 설정
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    # 5. 결과 디코딩
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"--- 추론 결과 ---")
    if sample_label is not None:
        print(f"실제 정답: {sample_label}")
    print(f"모델 답변: {result}")


# Epoch 1/5: 100%|█████████████████████████████████████████████████████████████████████| 938/938 [10:07<00:00,  1.54it/s, loss=0.0060]

# >>> Epoch [1/5] Completed. Average Loss: 0.0827
# --------------------------------------------------
# Epoch 2/5: 100%|█████████████████████████████████████████████████████████████████████| 938/938 [10:07<00:00,  1.54it/s, loss=0.0006]

# >>> Epoch [2/5] Completed. Average Loss: 0.0035
# --------------------------------------------------
# Epoch 3/5: 100%|█████████████████████████████████████████████████████████████████████| 938/938 [10:07<00:00,  1.55it/s, loss=0.0023]

# >>> Epoch [3/5] Completed. Average Loss: 0.0022
# --------------------------------------------------
# Epoch 4/5: 100%|█████████████████████████████████████████████████████████████████████| 938/938 [10:07<00:00,  1.54it/s, loss=0.0003]

# >>> Epoch [4/5] Completed. Average Loss: 0.0016
# --------------------------------------------------
# Epoch 5/5: 100%|█████████████████████████████████████████████████████████████████████| 938/938 [10:07<00:00,  1.55it/s, loss=0.0003]

# >>> Epoch [5/5] Completed. Average Loss: 0.0013
# --------------------------------------------------
# --- 추론 결과 ---
# 실제 정답: 7
# 모델 답변:  7입니다.
