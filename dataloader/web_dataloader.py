import torch
import webdataset as wds
from torchvision import transforms
from torch.utils.data import DataLoader

def get_wds_loader(urls, batch_size, world_size, rank):
    # 1. 전처리 정의
    preprocess = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. 데이터셋 정의 (URL은 .tar 파일들의 경로 패턴)
    dataset = (
        wds.WebDataset(urls, resampled=True) # 대규모 데이터셋을 위한 리샘플링
        .shuffle(1000)
        .decode("pil")                       # 이미지를 PIL 타입으로 디코딩
        .to_tuple("jpg", "txt")              # 이미지와 캡션 추출
        .map_tuple(preprocess, lambda x: x)  # 이미지 전처리, 텍스트는 그대로
    )

    # 3. 로더 생성 (DDP 환경 고려)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,                       # CPU 코어 수에 비례하여 설정
        pin_memory=True
    )
    return loader