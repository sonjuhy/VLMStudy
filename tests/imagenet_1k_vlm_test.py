from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
from dataloader.llava_dataloader import (
    LlavaStage3Dataset,
    get_blip_laion_cc_558k_dataloader,
)
from vision.vit_model import Projector, ViTEncoder
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
from utils.enums.e_path import JSONPathEnum, ImagePathEnum, CheckPointPathEnum

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
    train_json = "/media/edint/64d115f7-57cc-417b-acf0-7738ac091615/Ivern/DataSets/VLMDatasets/LlavaJson/blip_laion_cc_sbu_558k.json"
    valid_json = "/media/edint/64d115f7-57cc-417b-acf0-7738ac091615/Ivern/DataSets/VLMDatasets/LlavaJson/llava_instruct_150k.json"  # 검증용으로 활용 가능
    img_root = "/media/edint/64d115f7-57cc-417b-acf0-7738ac091615/Ivern/DataSets/VLMDatasets/images/558_images"  # 이미지가 모인 상위 폴더
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


def stage3_train(epochs: int = 1):
    # 1. 초기화 및 환경 설정
    accelerator = Accelerator(
        gradient_accumulation_steps=16
    )  # 메모리 효율을 위해 accumulation 사용
    device = accelerator.device

    model_id = "upstage/SOLAR-10.7B-Instruct-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token  # Solar 패딩 토큰 설정

    # 2. 모델 로드 및 LoRA 설정
    print("Solar LLM 로드 중...")
    llm = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
        dtype=torch.bfloat16,
        device_map={"": device},
    )
    llm = prepare_model_for_kbit_training(llm)  # 양자화 위한 옵션

    # LoRA 설정: Solar의 핵심 레이어들에 어댑터 추가
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm = get_peft_model(llm, lora_config)

    # 3. VLM 구조 결합 및 Stage 2 가중치 이식
    vit = ViTEncoder(
        img_size=224,
        patch_size=16,
        embedding_size=768,
        num_class=1000,
        num_heads=12,
        in_channels=3,
    )
    vit_encoder = prepare_model_for_kbit_training(vit)
    vlm_model = (
        ImageNet1KVLM(
            llm_model=llm,
            llm_hidden_size=llm.config.hidden_size,
            vit=vit_encoder,
        )
        .to(device)
        .to(dtype=torch.bfloat16)
    )
    vlm_model.llm.gradient_checkpointing_enable()

    for name, param in vlm_model.named_parameters():
        if "lora_" in name or "projector" in name:
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in vlm_model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in vlm_model.parameters())

    if accelerator.is_main_process:
        print(f"--- 학습 파라미터 체크 ---")
        print(f"학습 가능 파라미터: {trainable_params:,} 개")
        print(f"전체 파라미터: {all_params:,} 개")
        print(f"비중: {100 * trainable_params / all_params:.2f}%")
        print(f"------------------------")

    print("Stage 2 Projector 가중치 이식 중...")
    projector_path = CheckPointPathEnum.SOLAR_PROJECTOR_STAGE_2.value
    if os.path.exists(projector_path):
        vlm_model.projector.load_state_dict(
            torch.load(projector_path, map_location="cpu")
        )

    # 4. 데이터셋 준비
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = LlavaStage3Dataset(
        json_path=JSONPathEnum.LLAVA_1_5_MIX665K_CLEAN.value,
        img_root=ImagePathEnum.LLAVA_ALL_IMAGES.value,
        tokenizer=tokenizer,
        vis_processor=val_transform,  # Stage 2와 동일한 프로세서
    )

    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)

    # 5. 옵티마이저 및 스케줄러
    optimizer = torch.optim.AdamW(vlm_model.parameters(), lr=2e-5)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=len(train_loader)
    )

    # 6. Accelerate 준비 (모델, 옵티마이저 등 배분)
    vlm_model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        vlm_model, optimizer, train_loader, lr_scheduler
    )

    # 7. 학습 루프
    vlm_model.train()
    for epoch in range(epochs):
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            disable=not accelerator.is_local_main_process,
        )
        for step, batch in pbar:
            if step == 0 and accelerator.is_main_process:
                # DDP wrapper를 벗기고 실제 파라미터의 타입을 확인
                unwrapped_model = accelerator.unwrap_model(vlm_model)
                # LLM 임베딩 레이어의 타입을 확인하는 것이 가장 확실합니다.
                current_dtype = unwrapped_model.llm.dtype

                print(f"--- 디버깅 정보 ---")
                print(
                    f"이미지 텐서 모양: {batch['image'].shape}"
                )  # [batch, 3, 224, 224]
                print(f"이미지 텐서 dtype: {batch['image'].dtype}")
                print(f"인풋 아이디 모양: {batch['input_ids'].shape}")
                print(f"모델(LLM) dtype: {current_dtype}")
            with accelerator.accumulate(vlm_model):
                outputs = vlm_model(
                    images=batch["image"],
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")

            # 8. 중간 저장 (5000 스텝마다)
            if step % 5000 == 0 and step > 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_dir = f"checkpoints/vlm/stage3/step_{step}"
                    os.makedirs(save_dir, exist_ok=True)

                    # LoRA 가중치 저장
                    unwrapped_model = accelerator.unwrap_model(vlm_model)
                    unwrapped_model.llm.save_pretrained(save_dir)

                    # Projector 가중치 별도 저장
                    torch.save(
                        unwrapped_model.projector.state_dict(),
                        os.path.join(save_dir, "projector.bin"),
                    )
                    print(f"💾 Step {step} 저장 완료")

    # 최종 저장
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_dir = "checkpoints/vlm/stage3/final_model"
        os.makedirs(save_dir, exist_ok=True)

        # 1. 분산 학습 환경에서 모델 꺼내기
        unwrapped_model = accelerator.unwrap_model(vlm_model)

        # 2. LLM 부분(LoRA) 저장 (폴더 경로만 지정)
        unwrapped_model.llm.save_pretrained(save_dir)

        # 3. Projector 부분 저장
        torch.save(
            unwrapped_model.projector.state_dict(),
            os.path.join(save_dir, "projector.bin"),
        )

        # 4. 토크나이저도 함께 저장
        tokenizer.save_pretrained(save_dir)

        print(f"모든 모델 구성 요소가 {save_dir}에 저장되었습니다.")


def end_to_end_test():
    pass
