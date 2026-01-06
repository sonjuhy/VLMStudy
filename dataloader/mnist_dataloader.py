from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

import os
import torch


def vlm_collate_fn(batch):
    images = torch.stack([item["images"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    return {"images": images, "input_ids": input_ids, "attention_mask": attention_mask}


class MNISTVLMDataset(Dataset):
    def __init__(self, mnist_dataset, tokenizer, max_length=128):
        self.mnist = mnist_dataset  # torchvision.datasets.MNIST 객체
        self.tokenizer = tokenizer
        self.max_length = max_length

        # MNIST는 28x28이므로 ViT 입력에 맞게 resize하거나 정규화합니다.
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        image = self.transform(image)

        # 1. 대화 형식 구성 (HyperCLOVA X 템플릿 적용 가능)
        # <image> 토큰은 나중에 ViT 특징값이 들어갈 자리임을 표시하는 특수 문자열입니다.
        prompt = (
            f"질문: 이 이미지에 있는 숫자는 무엇인가요?\n답변: 이 숫자는 {label}입니다."
        )

        # 2. 토크나이징
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "images": image,  # [1, 28, 28]
            "input_ids": inputs["input_ids"].squeeze(),  # [L]
            "attention_mask": inputs["attention_mask"].squeeze(),  # [L]
        }


def mnist_dataloader():

    batch_size: int = 64

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    data_path = os.path.join("datasets", "mnist")
    train_dataset = datasets.MNIST(
        root=data_path, train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(root=data_path, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
