from torch.utils.data import DataLoader

import torch
import torch.nn as nn


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
