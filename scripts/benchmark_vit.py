"""
ViT-Base/16 ImageNet-1K 벤치마크 스크립트
학습 완료 후 체크포인트를 대상으로 단독 실행합니다.

측정 항목 (모두 ImageNet 논문 표준 지표):
  1. Top-1 / Top-5 Accuracy  — 분류 정확도 (ViT, ResNet, EfficientNet 등 모든 분류 모델 표준)
  2. Throughput (img/s)       — 추론 속도 (MLPerf, timm 라이브러리 공통 지표)
  3. Linear Probe Top-1      — 특징 추출기 품질 (DINO, MAE, SimCLR 등 SSL 모델 표준 평가)

실행 방법:
  # 단일 GPU
  cd /home/edint/test/VLMStudy
  source venv/bin/activate
  python scripts/benchmark_vit.py --checkpoint checkpoints/final_model/best_vit_imagenet_1k.pth

  # 전체 옵션
  python scripts/benchmark_vit.py \\
      --checkpoint checkpoints/final_model/best_vit_imagenet_1k.pth \\
      --data_dir datasets/imagenet_1k \\
      --batch_size 256 \\
      --linear_probe          # 피처 추출기 품질 측정 (시간 소요)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from vision.vit_model import ViTEncoder


# ---------------------------------------------------------------------------
# 모델 로드
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> ViTEncoder:
    model = ViTEncoder(
        embedding_size=768,
        img_size=224,
        patch_size=16,
        num_class=1000,
        num_heads=12,
        in_channels=3,
        n_layers=12,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 데이터 로더
# ---------------------------------------------------------------------------

def get_val_loader(data_dir: str, batch_size: int) -> DataLoader:
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(8, os.cpu_count()),
        pin_memory=True,
    )


def get_train_loader_for_probe(data_dir: str, batch_size: int) -> DataLoader:
    """Linear Probe용 — 학습셋에서 특징 추출"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(8, os.cpu_count()),
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# 1. Top-1 / Top-5 Accuracy
#    ▸ 표준 여부: YES
#      ImageNet 벤치마크의 사실상 유일한 표준 지표.
#      ViT 원논문(Dosovitskiy 2020), DeiT, Swin, ResNet 논문 모두 이 수치로 비교.
#      timm 라이브러리 기본 평가 지표이기도 함.
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_top1_top5(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    top1_correct = top5_correct = total = 0

    for inputs, targets in tqdm(val_loader, desc="[1/3] Top-1 / Top-5"):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(inputs)

        _, pred = outputs.topk(5, dim=1)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        top1_correct += correct[:1].reshape(-1).float().sum().item()
        top5_correct += correct[:5].reshape(-1).float().sum().item()
        total += targets.size(0)

    return 100.0 * top1_correct / total, 100.0 * top5_correct / total


# ---------------------------------------------------------------------------
# 2. Throughput (images/sec)
#    ▸ 표준 여부: YES
#      MLPerf, timm, TorchBench 공통 지표.
#      학술 논문에서는 GFLOPs와 함께 efficiency 테이블에 자주 등장.
#      측정 방식: 워밍업 50 step → 본 측정 200 step, 평균
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_throughput(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 256,
    warmup: int = 50,
    measure: int = 200,
) -> float:
    dummy = torch.randn(batch_size, 3, 224, 224, device=device)

    for _ in tqdm(range(warmup), desc="[2/3] Throughput warm-up", leave=False):
        with torch.amp.autocast(device_type="cuda"):
            model(dummy)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in tqdm(range(measure), desc="[2/3] Throughput measure", leave=False):
        with torch.amp.autocast(device_type="cuda"):
            model(dummy)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return (batch_size * measure) / elapsed  # img/s


# ---------------------------------------------------------------------------
# 3. Linear Probe Top-1
#    ▸ 표준 여부: YES
#      SSL 논문(DINO, MAE, MoCo, SimCLR)에서 비전 인코더의
#      representation 품질을 측정하는 사실상 표준 방법.
#      ViT를 VLM 인코더로 쓸 때 헤드 없는 feature 품질이 중요하므로 핵심 지표.
#      방법: ViT Frozen + 선형 분류기(1-layer)만 100 epoch 학습 후 val Top-1 측정.
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    model: ViTEncoder,
    loader: DataLoader,
    device: torch.device,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CLS 토큰 기반 특징 추출 (헤드 제거, 분류기 미사용)"""
    feats, labels = [], []
    for inputs, targets in tqdm(loader, desc=desc):
        inputs = inputs.to(device)
        with torch.amp.autocast(device_type="cuda"):
            # extract_features()는 patch 토큰 반환이므로
            # CLS 토큰을 직접 뽑아서 사용 (linear probe 표준)
            x = model.embedding(inputs)
            for layer in model.layers:
                x, _ = layer(x)
            cls = x[:, 0]  # [B, 768]
        feats.append(cls.cpu().float())
        labels.append(targets)
    return torch.cat(feats), torch.cat(labels)


def linear_probe(
    model: ViTEncoder,
    data_dir: str,
    device: torch.device,
    batch_size: int,
    epochs: int = 100,
) -> float:
    print("[3/3] Linear Probe: 특징 추출 중...")
    train_loader = get_train_loader_for_probe(data_dir, batch_size)
    val_loader = get_val_loader(data_dir, batch_size)

    train_feats, train_labels = extract_features(model, train_loader, device, "  train features")
    val_feats, val_labels = extract_features(model, val_loader, device, "  val features")

    # 선형 분류기 학습 (CPU로 이동해 GPU 메모리 절약)
    print(f"[3/3] Linear Probe: 선형 분류기 학습 ({epochs} epoch)...")
    classifier = nn.Linear(768, 1000).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    train_ds = torch.utils.data.TensorDataset(train_feats, train_labels)
    probe_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=4)

    for ep in range(epochs):
        classifier.train()
        for f, l in probe_loader:
            f, l = f.to(device), l.to(device)
            optimizer.zero_grad()
            criterion(classifier(f), l).backward()
            optimizer.step()
        scheduler.step()

    # 검증
    classifier.eval()
    correct = total = 0
    val_ds = torch.utils.data.TensorDataset(val_feats, val_labels)
    val_probe_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
    with torch.no_grad():
        for f, l in val_probe_loader:
            f, l = f.to(device), l.to(device)
            correct += classifier(f).argmax(dim=1).eq(l).sum().item()
            total += l.size(0)

    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# 모델 정보
# ---------------------------------------------------------------------------

def model_stats(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ViT-Base/16 ImageNet Benchmark")
    parser.add_argument("--checkpoint", required=True, help="체크포인트 .pth 경로")
    parser.add_argument("--data_dir", default="datasets/imagenet_1k", help="ImageNet 데이터 루트")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--linear_probe", action="store_true", help="Linear Probe 평가 실행 (시간 소요)")
    parser.add_argument("--probe_epochs", type=int, default=100, help="Linear Probe 학습 에폭 수")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"  ViT-Base/16 Benchmark")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Device     : {device}")
    print(f"{'='*60}\n")

    model = load_model(args.checkpoint, device)
    stats = model_stats(model)
    print(f"  Parameters : {stats['total_params']:,} ({stats['total_params']/1e6:.1f}M)\n")

    # 1. Top-1 / Top-5
    val_loader = get_val_loader(args.data_dir, args.batch_size)
    top1, top5 = evaluate_top1_top5(model, val_loader, device)
    print(f"\n  [Result 1] Top-1 Acc : {top1:.2f}%  |  Top-5 Acc : {top5:.2f}%")
    print(f"             ViT-Base/16 paper baseline (JFT pretrain) : Top-1 77.9%")
    print(f"             DeiT-Base (ImageNet only, 300ep)           : Top-1 81.8%\n")

    # 2. Throughput
    throughput = measure_throughput(model, device, args.batch_size)
    print(f"  [Result 2] Throughput : {throughput:.1f} img/s  (batch={args.batch_size}, AMP fp16)")
    print(f"             참고: timm ViT-Base/16 A100 기준 ~2500 img/s\n")

    # 3. Linear Probe (옵션)
    probe_top1 = None
    if args.linear_probe:
        probe_top1 = linear_probe(model, args.data_dir, device, args.batch_size, args.probe_epochs)
        print(f"\n  [Result 3] Linear Probe Top-1 : {probe_top1:.2f}%")
        print(f"             참고: DINO ViT-Base/16 linear probe : 78.2%")
        print(f"             참고: MAE ViT-Base/16  linear probe : 68.0%\n")

    # 최종 요약
    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  Model       : ViT-Base/16 (86M params)")
    print(f"  Checkpoint  : {os.path.basename(args.checkpoint)}")
    print(f"  Top-1 Acc   : {top1:.2f}%")
    print(f"  Top-5 Acc   : {top5:.2f}%")
    print(f"  Throughput  : {throughput:.1f} img/s")
    if probe_top1 is not None:
        print(f"  Linear Probe: {probe_top1:.2f}%")
    print(f"{'='*60}\n")

    # 결과 파일 저장
    result_path = os.path.join(
        os.path.dirname(args.checkpoint),
        f"benchmark_{os.path.splitext(os.path.basename(args.checkpoint))[0]}.txt"
    )
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Parameters: {stats['total_params']:,}\n")
        f.write(f"Top-1 Acc: {top1:.2f}%\n")
        f.write(f"Top-5 Acc: {top5:.2f}%\n")
        f.write(f"Throughput: {throughput:.1f} img/s\n")
        if probe_top1 is not None:
            f.write(f"Linear Probe Top-1: {probe_top1:.2f}%\n")
    print(f"  결과 저장: {result_path}")


if __name__ == "__main__":
    main()
