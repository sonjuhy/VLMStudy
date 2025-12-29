from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import os


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
