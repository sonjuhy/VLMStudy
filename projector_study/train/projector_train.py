"""
Projector 비교 실험 학습 스크립트.

실행 예시 (2-GPU DDP):
    torchrun --nproc_per_node=2 -m projector_study.train.projector_train \\
        --projector mlp \\
        --llm_id Qwen/Qwen2.5-7B-Instruct \\
        --vit_ckpt checkpoints/final_model/best_vit_imagenet_1k.pth \\
        --json_path /data/llava_665k.json \\
        --img_root /data/llava_images \\
        --epochs 3 --batch_size 8 --lr 1e-3

단일 GPU 실행:
    python -m projector_study.train.projector_train --projector linear ...
"""

import argparse
import csv
import os
import random
import signal
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataloader.llava_dataloader import BlipLaionCC558KDataset
from projector_study.models.base_vlm import ProjectorVLM
from projector_study.models.projectors import PROJECTOR_REGISTRY
from vision.vit_model import ViTEncoder


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _log(log_path: str, msg: str) -> None:
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _csv_write(csv_path: str, row: dict, write_header: bool = False) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# 데이터 전처리
# ---------------------------------------------------------------------------

VIT_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# 1 에폭 학습
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: ProjectorVLM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    accumulation_steps: int,
    pad_token_id: int,
    is_main: bool,
) -> float:
    model.train()
    optimizer.zero_grad()

    total_loss, total_samples = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", disable=not is_main)

    for step, batch in enumerate(pbar):
        images    = batch["image"].to(device, dtype=torch.bfloat16)
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        # dataloader가 attention_mask를 반환하지 않을 경우 pad_token_id로 직접 생성
        attn_mask = batch.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        else:
            attn_mask = (input_ids != pad_token_id).long()

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels,
            )
            loss = outputs.loss / accumulation_steps

        # bf16은 underflow 없음 → GradScaler 불필요
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        B = images.size(0)
        total_loss    += loss.item() * accumulation_steps * B
        total_samples += B

        if is_main:
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}",
                             lr=f"{current_lr:.2e}")

    # DDP: 전체 프로세스 loss 합산
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([total_loss, float(total_samples)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, total_samples = t[0].item(), t[1].item()

    return total_loss / max(total_samples, 1)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: ProjectorVLM,
    loader: DataLoader,
    device: torch.device,
    is_main: bool,
    max_steps: int = 200,
) -> float:
    model.eval()
    total_loss, total_samples = 0.0, 0
    pbar = tqdm(loader, desc="Validation", disable=not is_main, total=max_steps)

    for step, batch in enumerate(pbar):
        if step >= max_steps:
            break
        images    = batch["image"].to(device, dtype=torch.bfloat16)
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        attn_mask = batch.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                images=images, input_ids=input_ids,
                attention_mask=attn_mask, labels=labels,
            )

        B = images.size(0)
        total_loss    += outputs.loss.item() * B
        total_samples += B

    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([total_loss, float(total_samples)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, total_samples = t[0].item(), t[1].item()

    model.train()
    return total_loss / max(total_samples, 1)


# ---------------------------------------------------------------------------
# 메인 학습 함수
# ---------------------------------------------------------------------------

def run_training(args: argparse.Namespace) -> None:
    # ── DDP 초기화 ──────────────────────────────────────────────────────────
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    is_distributed = "RANK" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    is_main = (not is_distributed) or dist.get_rank() == 0

    seed_everything(args.seed)

    # ── 로그 / 체크포인트 경로 ─────────────────────────────────────────────
    run_name = f"{args.projector}_{date.today().isoformat()}"
    log_dir  = Path(args.output_dir) / "results" / run_name
    ckpt_dir = Path(args.output_dir) / "checkpoints" / run_name
    if is_main:
        log_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_path = str(log_dir / "train.log")
    csv_path = str(log_dir / "metrics.csv")

    # ── 모델 로드 ──────────────────────────────────────────────────────────
    if is_main:
        _log(log_path, f"=== Projector: {args.projector} | LLM: {args.llm_id} ===")
        _log(log_path, f"GPU count: {torch.cuda.device_count()}")

    # ViT
    vit = ViTEncoder(
        img_size=224, patch_size=16, embedding_size=768,
        num_class=1000, num_heads=12, in_channels=3,
    )
    if args.vit_ckpt and os.path.exists(args.vit_ckpt):
        ckpt = torch.load(args.vit_ckpt, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        vit.load_state_dict(state)
        if is_main:
            _log(log_path, f"ViT loaded from {args.vit_ckpt}")

    # LLM — 단일 프로세스: device_map="auto"로 가용 GPU에 자동 분산
    #        DDP:          각 rank의 GPU에 전체 로드
    if is_main:
        _log(log_path, f"Loading LLM: {args.llm_id} ...")
    llm_device_map = {"": local_rank} if is_distributed else "auto"
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_id,
        torch_dtype=torch.bfloat16,
        device_map=llm_device_map,
    )
    llm.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.llm_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Projector
    ProjCls   = PROJECTOR_REGISTRY[args.projector]
    projector = ProjCls(v_dim=768, l_dim=llm.config.hidden_size)

    # VLM 래퍼 — ViT·Projector만 device로 이동 (LLM은 device_map이 처리)
    model = ProjectorVLM(vit=vit, projector=projector, llm=llm)
    model.vit.to(device)
    model.projector.to(device)

    if is_main:
        _log(log_path,
             f"Trainable params: {model.trainable_params()/1e6:.1f}M / "
             f"Total: {model.total_params()/1e6:.1f}M")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    inner = model.module if is_distributed else model

    # ── 데이터로더 ────────────────────────────────────────────────────────
    dataset = BlipLaionCC558KDataset(
        json_path=args.json_path,
        img_root=args.img_root,
        tokenizer=tokenizer,
        vis_processor=VIT_TRANSFORM,
        max_length=args.max_length,
    )

    val_size   = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    train_loader  = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    # ── 옵티마이저 / 스케줄러 ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        inner.projector.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps  = (len(train_loader) // args.accumulation_steps) * args.epochs
    warmup_steps = max(1, int(total_steps * 0.05))

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=args.lr * 0.01
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps]
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_loss = float("inf")
    patience_counter = 0

    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        resume = torch.load(args.resume_ckpt, map_location="cpu")
        inner.projector.load_state_dict(resume["state_dict"])
        start_epoch   = resume["epoch"] + 1
        best_val_loss = resume.get("val_loss", float("inf"))
        if "optimizer_state_dict" in resume:
            optimizer.load_state_dict(resume["optimizer_state_dict"])
        if "scheduler_state_dict" in resume:
            scheduler.load_state_dict(resume["scheduler_state_dict"])
        if is_main:
            _log(log_path,
                 f"Resumed from {args.resume_ckpt} "
                 f"(epoch={resume['epoch']}, val_loss={best_val_loss:.4f})")

    # ── SIGTERM 핸들러 (Spot VM 선점 30초 전 신호) ────────────────────────
    # 루프 변수 epoch이 정의되기 전에 핸들러가 호출될 수 있으므로 미리 초기화
    epoch = start_epoch - 1
    emergency_path = ckpt_dir / "emergency_projector.pth"

    def _sigterm_handler(signum, frame):
        if is_main:
            save_projector(inner.projector, emergency_path,
                           epoch=epoch, val_loss=best_val_loss,
                           optimizer=optimizer, scheduler=scheduler)
            _log(log_path, f"SIGTERM 수신 — emergency checkpoint 저장: {emergency_path}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # ── CSV 헤더 초기화 ───────────────────────────────────────────────────
    if is_main and start_epoch == 1:
        _csv_write(csv_path, {
            "epoch": "", "train_loss": "", "val_loss": "",
            "lr": "", "elapsed_sec": "",
        }, write_header=True)

    # ── 학습 루프 ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.perf_counter()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, epoch, args.epochs, args.accumulation_steps,
            pad_token_id=tokenizer.pad_token_id,
            is_main=is_main,
        )
        val_loss = validate(model, val_loader, device, is_main, max_steps=200)

        elapsed    = time.perf_counter() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        if is_main:
            _log(log_path,
                 f"Epoch {epoch}/{args.epochs} | "
                 f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                 f"LR: {current_lr:.2e} | Elapsed: {elapsed:.0f}s")

            _csv_write(csv_path, {
                "epoch": epoch, "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "lr": round(current_lr, 8), "elapsed_sec": round(elapsed, 1),
            })

            # 에폭 체크포인트 (optimizer·scheduler 포함 → resume 가능)
            save_projector(
                inner.projector, ckpt_dir / f"projector_epoch_{epoch}.pth",
                epoch, val_loss, optimizer, scheduler,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_projector(
                    inner.projector, ckpt_dir / "best_projector.pth",
                    epoch, val_loss, optimizer, scheduler,
                )
                _log(log_path, f"  -> Best Val Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1

        if args.patience > 0 and patience_counter >= args.patience:
            if is_main:
                _log(log_path, f"Early stopping at epoch {epoch}.")
            break

    if is_main:
        _log(log_path, f"Training done. Best Val Loss: {best_val_loss:.4f}")

    if is_distributed:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# 체크포인트
# ---------------------------------------------------------------------------

def save_projector(
    projector: nn.Module,
    path: Path,
    epoch: int,
    val_loss: float,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
) -> None:
    ckpt = {
        "epoch":      epoch,
        "val_loss":   val_loss,
        "state_dict": projector.state_dict(),
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(ckpt, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Projector architecture comparison training")

    # 필수
    p.add_argument("--projector",  required=True, choices=list(PROJECTOR_REGISTRY.keys()))
    p.add_argument("--llm_id",     required=True, help="HuggingFace model ID (e.g. Qwen/Qwen2.5-7B-Instruct)")
    p.add_argument("--json_path",  required=True, help="LLaVA 558K JSON 경로")
    p.add_argument("--img_root",   required=True, help="이미지 루트 폴더 경로")

    # 선택
    p.add_argument("--vit_ckpt",   default="checkpoints/final_model/best_vit_imagenet_1k.pth")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--accumulation_steps", type=int, default=2)
    p.add_argument("--max_length", type=int,   default=128)
    p.add_argument("--num_workers", type=int,  default=4)
    p.add_argument("--patience",   type=int,   default=0,  help="0=비활성화")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--resume_ckpt", default=None,
                   help="재개할 체크포인트 .pth 경로 (Spot VM 선점 복구)")
    p.add_argument("--output_dir",  default="projector_study",
                   help="결과/체크포인트 루트 디렉토리 (GCS 마운트 경로 등)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
