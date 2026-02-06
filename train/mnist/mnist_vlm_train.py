from vision.mnist_vit_model import MNISTViTEncoder, MNISTViTHyperClovaX

from torch.optim import AdamW
from dataloader.mnist_dataloader import (
    MNISTVLMDataset,
    vlm_collate_fn,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

import os
import torch


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
    torch.save(vlm_model.vit.state_dict(), "vit_768_multi_prompt_mnist.pth")
    torch.save(
        vlm_model.projector.state_dict(), "projector_multi_prompt_768_to_1024.pth"
    )


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
