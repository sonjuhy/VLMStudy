from vision.vit_model import ViTEncoder
from dataloader.mnist_dataloader import mnist_dataloader
from train.mnist.mnist_vit_train import train as vit_train
from valid.mnist.mnist_vit_valid import valid as vit_valid
from train.mnist.mnist_vlm_train import train as vlm_train
from valid.mnist.mnist_vlm_valid import valid as vlm_valid

import torch
import torch.nn as nn
import torch.optim as optim


def mnist_vit_end_to_end():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size: int = 7
    embedding_batch_size: int = patch_size * patch_size
    epochs = 10
    learning_rate = 0.001

    model = ViTEncoder(
        img_size=28,
        patch_size=patch_size,
        embedding_size=embedding_batch_size,
        num_class=10,
        num_heads=7,
    ).to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, test_loader = mnist_dataloader()

    vit_train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        device=device,
        epochs=epochs,
    )

    vit_valid(model=model, device=device, test_loader=test_loader)


def mnist_vlm_end_to_end():
    epochs = 10
    vlm_train(epochs=epochs)
    vlm_valid()
