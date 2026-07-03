# Projector Architecture Comparison Study

## 실험 개요

ViT (Vision Encoder)와 LLM (7B)를 고정하고, **Projector 아키텍처 변형**에 따른 성능(정확도, 추론속도)을 체계적으로 비교한다.

---

## 고정 조건 (Frozen)

| 구성요소 | 모델 | 설정 |
|---|---|---|
| Vision Encoder | ViT-Base/16 (또는 ViT-L/14) | 가중치 Freeze |
| Language Model | LLM 7B (e.g. Qwen2.5-7B / LLaMA-3.1-8B) | 가중치 Freeze |
| 데이터셋 | LLaVA-665K (Instruction Tuning) | 고정 |
| 학습률 | 1e-3 (AdamW + CosineAnnealingLR) | 고정 |
| Batch Size | 16 per GPU (DDP 2-GPU) | 고정 |
| Epochs | 3 | 고정 |

> ViT output dim: 768 (Base) / 1024 (Large)  
> LLM input dim: 4096 (7B 계열)

---

## 비교 대상 Projector 목록

### P1. Linear Projection
- **구조:** `Linear(v_dim → l_dim)`
- **파라미터 수:** 3.1M (768→4096 기준)
- **특징:** 최소 구조, 속도 기준선(baseline)
- **참고:** InstructBLIP ablation

### P2. 2-Layer MLP (ReLU)
- **구조:** `Linear → ReLU → Linear`
- **파라미터 수:** ~6.3M
- **특징:** LLaVA-1.5 표준 구조
- **참고:** LLaVA-1.5 (Liu et al., 2023)

### P3. Q-Former
- **구조:** N개의 학습 가능한 Query Token + Cross-Attention (ViT output과 interact)
- **파라미터 수:** ~100M+ (Query 수에 따라 가변, Query=32 기준)
- **특징:** 시퀀스 압축 효과, BLIP-2 방식
- **참고:** BLIP-2 (Li et al., 2023)

### P4. Resampler (Perceiver-style)
- **구조:** 고정 크기 Latent Query + Cross-Attention → 고정 길이 출력
- **파라미터 수:** ~50M (latent=64 기준)
- **특징:** 입력 패치 수에 무관한 고정 출력 길이, 속도 안정적
- **참고:** Flamingo (Alayrac et al., 2022), IDEFICS

### P5. C-Abstractor (Conv-based)
- **구조:** ViT 출력을 2D reshape → Depthwise Conv → Linear
- **파라미터 수:** ~10M
- **특징:** 지역적 공간 구조 유지, 경량
- **참고:** HoneyBee (Cha et al., 2023)

### P6. Pixel Shuffle
- **구조:** ViT 출력을 2D reshape → 인접 2×2 패치를 채널 방향으로 합산(merge) → Linear
- **파라미터 수:** ~12M (토큰 수 1/4로 압축 후 4×v_dim → l_dim)
- **특징:** 토큰 수를 줄여 LLM 입력 시퀀스 단축, 공간 구조 유지, 2024년 SOTA 모델 다수 채택
- **참고:** InternVL2 (Chen et al., 2024)

---

## 평가 지표

### 정확도 (Accuracy)
| 벤치마크 | 측정 항목 |
|---|---|
| VQAv2 | Overall Accuracy |
| GQA | Accuracy |
| MMBench | Overall Score |
| POPE | F1 Score (Hallucination) |

### 추론 속도 (Inference Speed)
| 지표 | 측정 방법 |
|---|---|
| Projector Latency | 배치 단위 forward 시간 (ms/sample) |
| End-to-End Throughput | 토큰 생성 속도 (tokens/sec) |
| GPU Memory | Peak VRAM 사용량 (GB) |

> 속도 측정: `torch.cuda.synchronize()` + `time.perf_counter()` 기반, warmup 10회 후 100회 평균

---

## 폴더 구조

```
projector_study/
├── PLAN.md                        # 이 파일
├── models/
│   ├── base_vlm.py                # ViT + Projector + LLM 공통 래퍼
│   ├── projectors/
│   │   ├── linear.py              # P1
│   │   ├── mlp.py                 # P2
│   │   ├── qformer.py             # P3
│   │   ├── resampler.py           # P4
│   │   ├── c_abstractor.py        # P5
│   │   └── pixel_shuffle.py       # P6
├── train/
│   └── projector_train.py         # 학습 루프 (projector만 학습)
├── eval/
│   ├── benchmark_runner.py        # VQA/GQA/POPE 평가
│   └── speed_benchmark.py         # 추론 속도 측정
├── scripts/
│   ├── run_train.sh               # DDP 학습 실행 스크립트
│   └── run_eval.sh                # 평가 실행 스크립트
├── configs/
│   └── projector_config.yaml      # 공통 하이퍼파라미터
└── results/
    └── summary.csv                # 실험 결과 집계
```

---

## 실험 진행 순서

1. **환경 구성** - 공통 VLM 래퍼(`base_vlm.py`) 및 Projector 인터페이스 정의
2. **Projector 구현** - P1~P6 각각 구현 및 단위 테스트 (tensor shape 검증)
3. **학습** - 각 Projector에 대해 동일 조건으로 학습 (`projector_train.py`)
4. **벤치마크 평가** - VQAv2 / GQA / POPE 정확도 측정
5. **속도 측정** - `speed_benchmark.py`로 latency / throughput / VRAM 집계
6. **결과 분석** - `results/summary.csv` 기반 비교표 작성

---

## 예상 결과 가설

| Projector | 정확도 | 속도 | 메모리 |
|---|---|---|---|
| Linear | 낮음 | 가장 빠름 | 최소 |
| MLP | 중간 | 빠름 | 낮음 |
| Q-Former | 높음 | 느림 | 높음 |
| Resampler | 높음 | 중간 | 중간 |
| C-Abstractor | 중간~높음 | 중간 | 낮음 |
| Pixel Shuffle | 높음 | 빠름 | 낮음 |

---

## 참고 문헌

- LLaVA-1.5: [Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744)
- BLIP-2: [Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2301.12597)
- Flamingo: [A Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
- HoneyBee: [A Scalable Modular Framework for Creating Multimodal LLMs](https://arxiv.org/abs/2312.06235)
- InstructBLIP: [Towards General Visual-Language Models](https://arxiv.org/abs/2305.06500)
- InternVL2: [InternVL: Scaling up Vision Foundation Models](https://arxiv.org/abs/2404.16821)
