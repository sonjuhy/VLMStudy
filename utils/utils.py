from torch.cuda.amp import GradScaler
from contextlib import contextmanager

import os
import torch
import torch.nn as nn
import torch.optim as optim
import time


@contextmanager
def timer_call():
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    loss: float,
    path: str = "checkpoints",
    filename: str = None,
):
    if not os.path.exists(path):
        os.makedirs(path)

    if filename is None:
        checkpoint_path = os.path.join(path, f"checkpoint_epoch_{epoch}.pth")
    else:
        checkpoint_path = os.path.join(path, filename)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loss": loss,
        },
        checkpoint_path,
    )
    print(f"--- Checkpoint saved at: {checkpoint_path} ---")
