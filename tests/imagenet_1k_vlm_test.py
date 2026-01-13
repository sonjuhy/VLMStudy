from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedModel
from torch.optim import AdamW
from torchvision import transforms
from dataloader.llava_dataloader import get_blip_laion_cc_558k_dataloader
from vision.vit_model import Projector, ViTEncoder

import os
import torch
import torch.nn as nn


class ImageNet1KVLM(nn.Module):
    def __init__(
        self,
        vit: ViTEncoder,
        llm_model: PreTrainedModel,
        llm_hidden_size: int = 4096,
    ):
        super().__init__()
        self.vit_encoder: ViTEncoder = vit
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
        with torch.no_grad():
            image_features = self.vit_encoder.extract_features(images)

        # 2. Projector 차원 변환 (Batch, 196, 4096)
        image_embeddings = self.projector(image_features)

        # 3. 텍스트 임베딩 (Batch, Seq_Len, 4096)
        text_embeddings = self.llm.get_input_embeddings()(input_ids)

        # 4. 결합 [Image; Text] (Batch, 196 + Seq_Len, 4096)
        inputs_embeds = torch.cat([image_embeddings, text_embeddings], dim=1)

        # 5. [중요!] Labels 길이 맞추기
        # 이미지 토큰 위치(196개)에는 정답이 없으므로 -100(Ignore Index)으로 채웁니다.
        if labels is not None:
            # labels: [Batch, Seq_Len]
            device = labels.device
            batch_size = labels.shape[0]

            # 이미지 토큰 개수만큼 -100 채우기
            ignore_labels = torch.full((batch_size, 196), -100, device=device)
            # 최종 정답지 결합: [Batch, 196 + Seq_Len]
            full_labels = torch.cat([ignore_labels, labels], dim=1)
        else:
            full_labels = None

        # 6. LLM 통과
        outputs = self.llm(inputs_embeds=inputs_embeds, labels=full_labels)
        return outputs


def projector_train(
    model: nn.Module,
    train_path: str,
    valid_path: str,
    json_path: str,
    img_root: str,
    epochs: int = 1,
):
    if os.path.exists(train_path) is False:
        raise ValueError(f"Train path {train_path} does not exist.")
    if os.path.exists(valid_path) is False:
        raise ValueError(f"Valid path {valid_path} does not exist.")

    optimizer = AdamW(model.projector.parameters(), lr=1e-3, weight_decay=0.1)

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 1 Epoch만 하므로 Warmup을 짧고 강하게 가져갑니다.
    train_loader = get_blip_laion_cc_558k_dataloader(
        model_name="upstage/SOLAR-10.7B-Instruct-v1.0",
        vis_processor=val_transform,
        json_path=json_path,
        img_root=img_root,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader)
    )
    scaler = torch.GradScaler()

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            # 데이터 로드 (device 이동)
            images = batch["image"].to("cuda", dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to("cuda")
            labels = batch["labels"].to("cuda")

            optimizer.zero_grad()

            # Mixed Precision 학습 (Bfloat16 사용 권장)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(images, input_ids, labels)
                loss = outputs.loss

            # 역전파
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 스케줄러 업데이트 (Step 단위)
            scheduler.step()

            # 로그 기록
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} 완료. 평균 Loss: {avg_loss:.4f}")

        # 체크포인트 저장 (Projector 가중치만 저장하여 용량 아끼기)
        save_path = f"solar_projector_epoch_{epoch+1}.pth"
        torch.save(model.projector.state_dict(), save_path)
        print(f"Projector saved to {save_path}")

    print("Stage 2 Alignment Finished!")
    return model.projector


def projector_train_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 모델 경로 및 설정
    llm_id = "upstage/SOLAR-10.7B-Instruct-v1.0"
    vit_checkpoint = "./checkpoints/final_model/vit_imagenet_1k_checkpoint_epoch_99.pth"  # 최고 성능 에포크
    train_json = "./DataSets/VLMDatasets/LlavaJson/blip_laion_cc_sbu_558k.json"
    valid_json = "./DataSets/VLMDatasets/LlavaJson/llava_instruct_150k.json"  # 검증용으로 활용 가능
    img_root = "./DataSets/VLMDatasets/images/558_images"  # 이미지가 모인 상위 폴더

    print("--- 1. Loading Vision Encoder (ViT-Base) ---")
    # 기존에 정의하신 ViTEncoder 클래스 인스턴스 생성
    vit = ViTEncoder(
        img_size=224,
        patch_size=16,
        embedding_size=768,
        num_class=1000,
        num_heads=12,
        in_channels=3,
    )
    checkpoint = torch.load(vit_checkpoint, map_location="cpu")

    # 체크포인트 로드 (dict 형태인지 직접 인스턴스 형태인지 확인 필요)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        vit.load_state_dict(checkpoint["model_state_dict"])
    else:
        vit.load_state_dict(checkpoint)
    vit.cuda()

    print(f"--- 2. Loading Language Model (SOLAR-10.7B) ---")
    # A6000에서 10.7B 모델을 8-bit로 로드하여 VRAM 절약
    llm = AutoModelForCausalLM.from_pretrained(
        llm_id,
        load_in_8bit=True,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print("--- 3. Initializing ImageNet1KVLM Wrapper ---")
    model = ImageNet1KVLM(
        vit=vit, llm_model=llm, llm_hidden_size=llm.config.hidden_size
    ).to(device=device)
    model.llm.gradient_checkpointing_enable()

    print("--- 4. Starting Projector Alignment Training ---")

    final_projector = projector_train(
        model=model,
        train_path=train_json,
        valid_path=valid_json,
        json_path=train_json,
        img_root=img_root,
        epochs=1,  # Stage 2는 1 에포크면 충분
    )

    print("--- All Processes Completed Successfully! ---")


def end_to_end_test():
    pass
