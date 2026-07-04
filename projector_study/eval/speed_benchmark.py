"""
Projector 추론 속도 벤치마크

측정 항목:
  1. Projector Latency  — projector forward만 (ms/sample)
  2. Prefill Latency    — 이미지+텍스트 → 첫 토큰 생성 시간 (ms/sample)
  3. Throughput         — 연속 토큰 생성 속도 (tokens/sec)
  4. Peak VRAM          — 측정 중 최대 GPU 메모리 사용량 (GB)

실행 예시:
    python -m projector_study.eval.speed_benchmark \\
        --llm_id Qwen/Qwen2.5-7B-Instruct \\
        --vit_ckpt checkpoints/final_model/best_vit_imagenet_1k.pth \\
        --proj_ckpt_dir projector_study/checkpoints \\
        --projectors linear mlp qformer resampler c_abstractor pixel_shuffle
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from projector_study.models.base_vlm import ProjectorVLM
from projector_study.models.projectors import PROJECTOR_REGISTRY
from vision.vit_model import ViTEncoder


# ---------------------------------------------------------------------------
# 모델 로드
# ---------------------------------------------------------------------------

def load_vlm(
    projector_name: str,
    proj_ckpt: str | None,
    llm_id: str,
    vit_ckpt: str,
    device: torch.device,
) -> tuple[ProjectorVLM, AutoTokenizer]:
    vit = ViTEncoder(
        img_size=224, patch_size=16, embedding_size=768,
        num_class=1000, num_heads=12, in_channels=3,
    )
    if vit_ckpt and os.path.exists(vit_ckpt):
        ckpt = torch.load(vit_ckpt, map_location="cpu")
        vit.load_state_dict(ckpt.get("model_state_dict", ckpt))

    llm = AutoModelForCausalLM.from_pretrained(
        llm_id, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ProjCls   = PROJECTOR_REGISTRY[projector_name]
    projector = ProjCls(v_dim=768, l_dim=llm.config.hidden_size)

    if proj_ckpt and os.path.exists(proj_ckpt):
        saved = torch.load(proj_ckpt, map_location="cpu")
        projector.load_state_dict(saved.get("state_dict", saved))

    model = ProjectorVLM(vit=vit, projector=projector, llm=llm).to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# 1. Projector Latency
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_projector_latency(
    projector: nn.Module,
    device: torch.device,
    batch_size: int = 8,
    n_patches: int = 196,
    v_dim: int = 768,
    warmup: int = 10,
    repeat: int = 100,
) -> float:
    """projector forward만 측정. 단위: ms/sample"""
    x = torch.randn(batch_size, n_patches, v_dim, device=device, dtype=torch.bfloat16)
    projector = projector.to(device).to(torch.bfloat16)

    # warmup
    for _ in range(warmup):
        _ = projector(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        _ = projector(x)
    torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms / (repeat * batch_size)  # ms/sample


# ---------------------------------------------------------------------------
# 2. Prefill Latency
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_prefill_latency(
    model: ProjectorVLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int = 1,
    text_len: int = 32,
    warmup: int = 5,
    repeat: int = 20,
) -> float:
    """이미지+텍스트 → 첫 토큰 생성까지 시간. 단위: ms/sample"""
    images    = torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.bfloat16)
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, text_len), device=device)
    attn_mask = torch.ones_like(input_ids)

    for _ in range(warmup):
        model.generate(images, input_ids, attn_mask,
                       max_new_tokens=1, do_sample=False, use_cache=True)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        model.generate(images, input_ids, attn_mask,
                       max_new_tokens=1, do_sample=False, use_cache=True)
    torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms / (repeat * batch_size)  # ms/sample


# ---------------------------------------------------------------------------
# 3. Throughput (tokens/sec)
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_throughput(
    model: ProjectorVLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int = 1,
    text_len: int = 32,
    gen_len: int = 64,
    warmup: int = 3,
    repeat: int = 10,
) -> float:
    """연속 토큰 생성 속도. 단위: tokens/sec"""
    images    = torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.bfloat16)
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, text_len), device=device)
    attn_mask = torch.ones_like(input_ids)

    for _ in range(warmup):
        model.generate(images, input_ids, attn_mask,
                       max_new_tokens=gen_len, do_sample=False, use_cache=True)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        model.generate(images, input_ids, attn_mask,
                       max_new_tokens=gen_len, do_sample=False, use_cache=True)
    torch.cuda.synchronize()

    total_tokens = gen_len * batch_size * repeat
    elapsed      = time.perf_counter() - t0
    return total_tokens / elapsed  # tokens/sec


# ---------------------------------------------------------------------------
# 4. Peak VRAM
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_peak_vram(
    model: ProjectorVLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int = 1,
    text_len: int = 32,
) -> float:
    """측정 중 최대 VRAM. 단위: GB"""
    torch.cuda.reset_peak_memory_stats(device)

    images    = torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.bfloat16)
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, text_len), device=device)
    attn_mask = torch.ones_like(input_ids)

    model.generate(images, input_ids, attn_mask,
                   max_new_tokens=32, do_sample=False, use_cache=True)
    torch.cuda.synchronize()

    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / (1024 ** 3)  # GB


# ---------------------------------------------------------------------------
# 전체 실행
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    results_dir = Path("projector_study/results/speed")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "speed_results.csv"

    fieldnames = [
        "projector", "proj_latency_ms", "prefill_latency_ms",
        "throughput_tok_s", "peak_vram_gb", "proj_params_m",
    ]
    rows = []

    for name in tqdm(args.projectors, desc="Projectors"):
        # 해당 projector의 best checkpoint 자동 탐색
        proj_ckpt = None
        if args.proj_ckpt_dir:
            candidates = list(Path(args.proj_ckpt_dir).glob(f"{name}_*/best_projector.pth"))
            if candidates:
                proj_ckpt = str(sorted(candidates)[-1])

        print(f"\n{'='*55}")
        print(f"  Projector: {name}  |  ckpt: {proj_ckpt or 'random init'}")
        print(f"{'='*55}")

        model, tokenizer = load_vlm(
            projector_name=name,
            proj_ckpt=proj_ckpt,
            llm_id=args.llm_id,
            vit_ckpt=args.vit_ckpt,
            device=device,
        )

        inner_proj = model.projector
        proj_params = sum(p.numel() for p in inner_proj.parameters()) / 1e6

        print("  [1/4] Projector latency ...")
        p_lat = measure_projector_latency(
            inner_proj, device,
            batch_size=args.batch_size, warmup=10, repeat=100,
        )

        print("  [2/4] Prefill latency ...")
        pre_lat = measure_prefill_latency(
            model, tokenizer, device,
            batch_size=1, text_len=32, warmup=5, repeat=20,
        )

        print("  [3/4] Throughput ...")
        tput = measure_throughput(
            model, tokenizer, device,
            batch_size=1, text_len=32, gen_len=64, warmup=3, repeat=10,
        )

        print("  [4/4] Peak VRAM ...")
        vram = measure_peak_vram(model, tokenizer, device, batch_size=1)

        row = {
            "projector":          name,
            "proj_latency_ms":    round(p_lat, 4),
            "prefill_latency_ms": round(pre_lat, 2),
            "throughput_tok_s":   round(tput, 2),
            "peak_vram_gb":       round(vram, 3),
            "proj_params_m":      round(proj_params, 2),
        }
        rows.append(row)
        print(f"\n  proj_latency={p_lat:.4f}ms | prefill={pre_lat:.2f}ms | "
              f"throughput={tput:.1f} tok/s | VRAM={vram:.3f}GB")

        # 메모리 해제
        del model
        torch.cuda.empty_cache()

    # CSV 저장
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 터미널 요약표
    print(f"\n{'='*80}")
    print(f"  {'Projector':<16} {'ProjLat(ms)':<14} {'Prefill(ms)':<14} "
          f"{'Throughput(t/s)':<18} {'VRAM(GB)':<12} {'Params(M)'}")
    print(f"  {'-'*76}")
    for r in rows:
        print(f"  {r['projector']:<16} {r['proj_latency_ms']:<14.4f} "
              f"{r['prefill_latency_ms']:<14.2f} {r['throughput_tok_s']:<18.1f} "
              f"{r['peak_vram_gb']:<12.3f} {r['proj_params_m']:.1f}")
    print(f"{'='*80}")
    print(f"\n  결과 저장: {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Projector speed benchmark")
    p.add_argument("--llm_id",       required=True)
    p.add_argument("--vit_ckpt",     default="checkpoints/final_model/best_vit_imagenet_1k.pth")
    p.add_argument("--proj_ckpt_dir", default="projector_study/checkpoints",
                   help="각 projector의 best_projector.pth가 있는 상위 디렉터리")
    p.add_argument("--projectors",   nargs="+",
                   default=list(PROJECTOR_REGISTRY.keys()),
                   choices=list(PROJECTOR_REGISTRY.keys()))
    p.add_argument("--batch_size",   type=int, default=8,
                   help="Projector latency 측정용 배치 크기")
    p.add_argument("--gpu",          type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
