from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoTokenizer
from tqdm import tqdm

import json
import torch
import os


class BlipLaionCC558KDataset(Dataset):
    def __init__(self, json_path, img_root, tokenizer, vis_processor, max_length=128):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.img_root = img_root
        self.tokenizer = tokenizer
        self.vis_processor = vis_processor  # ViT용 전처리 (Resize, Normalize 등)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 2. 이미지 로드
        img_path = os.path.join(self.img_root, item["image"])

        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.vis_processor(image)  # [3, 224, 224]
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self.data))

        # 3. 대화 데이터에서 캡션(GPT 답변) 추출
        # conversations[0]: Human 질문, conversations[1]: GPT 답변
        convs = item["conversations"]
        caption = convs[1]["value"]

        # 4. 토큰화 (프롬프트 구성)
        full_text = f"Describe this image: {caption}{self.tokenizer.eos_token}"

        tokenized = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

        input_ids = tokenized.input_ids.squeeze()

        # 5. Labels 생성 (패딩 토큰 Loss 제외 처리)
        labels = input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            # Solar/Llama 계열에서 pad_token이 설정되지 않았을 경우 eos_token 사용 가능성 대비
            labels[labels == self.tokenizer.eos_token_id] = -100

        return {"image": image_tensor, "input_ids": input_ids, "labels": labels}


def get_blip_laion_cc_558k_dataloader(
    model_name: str,
    vis_processor,
    json_path: str,
    img_root: str,
):
    # 1. 토크나이저 준비 (Llama-3 기준)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 데이터셋 인스턴스 생성
    # img_root는 이미지들이 들어있는 최상위 폴더 경로입니다.
    dataset = BlipLaionCC558KDataset(
        json_path=json_path,
        img_root=img_root,
        tokenizer=tokenizer,
        vis_processor=vis_processor,  # 이전에 사용한 ViT 전처리 함수
    )

    # 3. 로더 생성
    train_loader = DataLoader(
        dataset,
        batch_size=8,  # 8-bit 양자화 사용 시 더 키울 수 있음
        shuffle=True,
        num_workers=8,  # A6000 서버라면 CPU 코어에 맞춰 8~16 권장
        pin_memory=True,
    )

    return train_loader


class LlavaStage3Dataset(Dataset):
    def __init__(self, json_path, img_root, tokenizer, vis_processor, max_length=1024):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.vis_processor = vis_processor
        self.max_length = max_length

        # 하위 폴더 이미지 경로 미리 맵핑 (학습 시 속도 저하 방지)
        print("이미지 경로 인덱싱 중...")
        self.image_map = {}
        for root, _, files in os.walk(img_root):
            for file in files:
                self.image_map[file] = os.path.join(root, file)
        print(f"인덱싱 완료: {len(self.image_map)}개의 이미지 탐지")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. 이미지 처리
        image_tensor = None
        if "image" in item:
            file_name = os.path.basename(item["image"])
            actual_path = self.image_map.get(file_name)
            if actual_path:
                image = Image.open(actual_path).convert("RGB")
                image_tensor = self.vis_processor(image)

        # 이미지가 없는 데이터(텍스트 전용)인 경우 처리
        if image_tensor is None:
            image_tensor = torch.zeros(3, 224, 224)

        # 2. Solar 대화 포맷 구성
        # 포맷: ### User: <image>\n질문\n\n### Assistant: 답변</s>
        convs = item["conversations"]
        full_text = ""
        for i, conv in enumerate(convs):
            role = "### User" if conv["from"] == "human" else "### Assistant"
            value = conv["value"]

            if i == 0 and "<image>" in value:  # 이미지 토큰 처리
                value = value.replace("<image>", "").strip()
                full_text += f"{role}: <image>\n{value}\n\n"
            else:
                full_text += f"{role}: {value}\n\n"

        full_text = full_text.strip() + self.tokenizer.eos_token

        # 3. 토큰화 및 레이블 생성
        encodings = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = encodings.input_ids.squeeze()
        labels = input_ids.clone()

        return {"image": image_tensor, "input_ids": input_ids, "labels": labels}


def verify_and_clean_dataset_recursive(json_path, img_root, output_json_path):
    # 1. JSON 로드
    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"총 데이터 개수: {len(data)}")

    # 2. 모든 하위 파일 경로 미리 맵핑 (속도 향상을 위해)
    # 파일명 -> 실제 전체 경로 로 딕셔너리를 만듭니다.
    print("디렉토리 구조 스캔 중... (시간이 조금 걸릴 수 있습니다)")
    image_map = {}
    for root, dirs, files in os.walk(img_root):
        for file in files:
            # 파일명을 키로, 전체 경로를 값으로 저장
            image_map[file] = os.path.join(root, file)

    print(f"스캔 완료! 발견된 총 이미지 파일 수: {len(image_map)}")

    clean_data = []
    missing_count = 0

    # 3. 데이터 검증
    print("데이터셋 필터링 중...")
    for item in tqdm(data):
        if "image" not in item:
            clean_data.append(item)
            continue

        file_name = os.path.basename(item["image"])

        # 미리 만들어둔 맵에서 파일이 있는지 확인
        if file_name in image_map:
            # 나중에 학습할 때 편하도록 실제 경로로 업데이트해주면 더 좋습니다.
            # item['image'] = image_map[file_name]
            clean_data.append(item)
        else:
            missing_count += 1

    # 4. 결과 보고
    print("\n" + "=" * 30)
    print(f"✅ 최종 사용 가능 데이터: {len(clean_data)}")
    print(f"❌ 실제로 누락된 데이터: {missing_count}")
    print("=" * 30)

    # 5. 저장
    with open(output_json_path, "w") as f:
        json.dump(clean_data, f, indent=4)
    print(f"💾 필터링된 JSON 저장 완료: {output_json_path}")
