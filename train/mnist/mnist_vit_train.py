import torch
import torch.nn as nn
import torch.optim as optim

from dataloader.mnist_dataloader import mnist_dataloader
from vision.vit_model import ViTEncoder
from torch.utils.data import DataLoader


def train(
    model: nn.Module,
    optimizer,
    criterion,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
):
    print(f"Using device: {device}")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 200 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}"
                )
        PATH = "vit_mnist_model.pth"
        torch.save(model.state_dict(), PATH)
        print(f"모델 가중치가 {PATH}에 저장되었습니다.")


def valid(model: nn.Module, device: torch.device, test_loader: DataLoader):
    # --- 4. 테스트 (Evaluation) ---
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f"\nTest Accuracy: {100. * correct / len(test_loader.dataset):.2f}%")  # type: ignore


# (venv) C:\WorkSpace\Dev\Python\VLMStudy>python test.py
# Using device: cpu
# Epoch 1/10 | Batch 0/938 | Loss: 2.7868
# Epoch 1/10 | Batch 200/938 | Loss: 0.3929
# Epoch 1/10 | Batch 400/938 | Loss: 0.4312
# Epoch 1/10 | Batch 600/938 | Loss: 0.2581
# Epoch 1/10 | Batch 800/938 | Loss: 0.2996
# 모델 가중치가 vit_mnist_model.pth에 저장되었습니다.
# Epoch 2/10 | Batch 0/938 | Loss: 0.1079
# Epoch 2/10 | Batch 200/938 | Loss: 0.1575
# Epoch 2/10 | Batch 400/938 | Loss: 0.2386
# Epoch 2/10 | Batch 600/938 | Loss: 0.1935
# Epoch 2/10 | Batch 800/938 | Loss: 0.1959
# 모델 가중치가 vit_mnist_model.pth에 저장되었습니다.
# Epoch 3/10 | Batch 0/938 | Loss: 0.2314
# Epoch 3/10 | Batch 200/938 | Loss: 0.1610
# Epoch 3/10 | Batch 400/938 | Loss: 0.2520
# Epoch 3/10 | Batch 600/938 | Loss: 0.0510
# Epoch 3/10 | Batch 800/938 | Loss: 0.0625
# 모델 가중치가 vit_mnist_model.pth에 저장되었습니다.
# Epoch 4/10 | Batch 0/938 | Loss: 0.0454
# Epoch 4/10 | Batch 200/938 | Loss: 0.0998
# Epoch 4/10 | Batch 400/938 | Loss: 0.0820
# Epoch 4/10 | Batch 600/938 | Loss: 0.1048
# Epoch 4/10 | Batch 800/938 | Loss: 0.0821
# 모델 가중치가 vit_mnist_model.pth에 저장되었습니다.
# Epoch 5/10 | Batch 0/938 | Loss: 0.0772
# Epoch 5/10 | Batch 200/938 | Loss: 0.0613
# Epoch 5/10 | Batch 400/938 | Loss: 0.1244
# Epoch 5/10 | Batch 600/938 | Loss: 0.0449
# Epoch 5/10 | Batch 800/938 | Loss: 0.0281
# 모델 가중치가 vit_mnist_model.pth에 저장되었습니다.
# Epoch 6/10 | Batch 0/938 | Loss: 0.0752
# Epoch 6/10 | Batch 200/938 | Loss: 0.0394
# Epoch 6/10 | Batch 400/938 | Loss: 0.0201
# Epoch 6/10 | Batch 600/938 | Loss: 0.0495
# Epoch 6/10 | Batch 800/938 | Loss: 0.0758
# 모델 가중치가 vit_mnist_model.pth에 저장되었습니다.
# Epoch 7/10 | Batch 0/938 | Loss: 0.0319
# Epoch 7/10 | Batch 200/938 | Loss: 0.0359
# Epoch 7/10 | Batch 400/938 | Loss: 0.1106
# Epoch 7/10 | Batch 600/938 | Loss: 0.0170
# Epoch 7/10 | Batch 800/938 | Loss: 0.0179
# 모델 가중치가 vit_mnist_model.pth에 저장되었습니다.
# Epoch 8/10 | Batch 0/938 | Loss: 0.0466
# Epoch 8/10 | Batch 200/938 | Loss: 0.1245
# Epoch 8/10 | Batch 400/938 | Loss: 0.0278
# Epoch 8/10 | Batch 600/938 | Loss: 0.0869
# Epoch 8/10 | Batch 800/938 | Loss: 0.0528
# 모델 가중치가 vit_mnist_model.pth에 저장되었습니다.
# Epoch 9/10 | Batch 0/938 | Loss: 0.0528
# Epoch 9/10 | Batch 200/938 | Loss: 0.0353
# Epoch 9/10 | Batch 400/938 | Loss: 0.0331
# Epoch 9/10 | Batch 600/938 | Loss: 0.1514
# Epoch 9/10 | Batch 800/938 | Loss: 0.1000
# 모델 가중치가 vit_mnist_model.pth에 저장되었습니다.
# Epoch 10/10 | Batch 0/938 | Loss: 0.1011
# Epoch 10/10 | Batch 200/938 | Loss: 0.0318
# Epoch 10/10 | Batch 400/938 | Loss: 0.0452
# Epoch 10/10 | Batch 600/938 | Loss: 0.0638
# Epoch 10/10 | Batch 800/938 | Loss: 0.0364
# 모델 가중치가 vit_mnist_model.pth에 저장되었습니다.

# Test Accuracy: 98.09%
