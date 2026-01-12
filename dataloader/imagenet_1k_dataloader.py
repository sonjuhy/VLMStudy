from torchvision import datasets, transforms
from kaggle.api.kaggle_api_extended import KaggleApi

import os
import torch
import shutil
import pandas as pd  # CSV 처리를 위해 추가


def download_and_extract_from_kaggle():
    print(f'Kaggle API Token : {os.environ["KAGGLE_API_TOKEN"]}')

    api = KaggleApi()
    api.authenticate()

    target_path = os.path.join("datasets", "imagenet_1k")
    zip_file_path = os.path.join(
        target_path, "imagenet-object-localization-challenge.zip"
    )

    # 1. 다운로드
    if not os.path.exists(zip_file_path):
        api.competition_download_files(
            "imagenet-object-localization-challenge", path=target_path, quiet=False
        )

    # 2. 압축 해제 (시스템 unzip 사용 권장)
    if not os.path.exists(os.path.join(target_path, "ILSVRC")):
        print("Extracting... (Using system unzip is recommended for 160GB)")
        os.system(f"unzip -q {zip_file_path} -d {target_path}")

    # 3. 폴더 이동
    base_data_path = os.path.join(target_path, "ILSVRC", "Data", "CLS-LOC")
    train_dest = os.path.join(target_path, "train")
    val_dest = os.path.join(target_path, "val")

    if os.path.exists(os.path.join(base_data_path, "train")) and not os.path.exists(
        train_dest
    ):
        shutil.move(os.path.join(base_data_path, "train"), train_dest)

    # 4. 검증 데이터셋 분류 (중요!)
    if os.path.exists(os.path.join(base_data_path, "val")) and not os.path.exists(
        val_dest
    ):
        print("Reorganizing Val images by class...")
        temp_val_path = os.path.join(base_data_path, "val")
        # Kaggle 정답지 위치 확인
        solution_csv = os.path.join(target_path, "LOC_val_solution.csv")
        df = pd.read_csv(solution_csv)

        for _, row in df.iterrows():
            img_id = row["ImageId"]
            # PredictionString에서 첫 번째 단어가 클래스 ID(n0xxxx)입니다.
            label = row["PredictionString"].split(" ")[0]

            label_dir = os.path.join(val_dest, label)
            os.makedirs(label_dir, exist_ok=True)

            src_file = os.path.join(temp_val_path, f"{img_id}.JPEG")
            if os.path.exists(src_file):
                shutil.move(src_file, os.path.join(label_dir, f"{img_id}.JPEG"))


def get_imagenet_loaders(
    data_dir=os.path.join(
        "datasets", "imagenet_1k", "raw_data", "ILSVRC", "Data", "CLS-LOC"
    ),
    batch_size=256,
):
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print("데이터가 없거나 폴더가 비어 있습니다. 다운로드를 시작합니다...")
        download_and_extract_from_kaggle()
    else:
        print(f"데이터가 이미 존재합니다: {data_dir}")
    # ImageNet은 Train/Val 폴더 구조가 표준화되어 있음.
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    print(f"Train directory: {train_dir}, Val directory: {val_dir}")
    print("Train folder exists:", os.path.exists(train_dir))
    print("Val folder exists:", os.path.exists(val_dir))

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(
                transforms.AutoAugmentPolicy.IMAGENET
            ),  # ImageNet 최적화 증강
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    # num_workers == CPU 코어 수
    cpu_core_count = os.cpu_count() if os.cpu_count() is not None else 8
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cpu_core_count,
        pin_memory=True,
        prefetch_factor=2,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    return train_loader, val_loader
