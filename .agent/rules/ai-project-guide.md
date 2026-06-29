---
trigger: always_on
---

# AI Modeling Project Context

## 1. Environment & Framework
- **Core:** PyTorch 2.0+ / HuggingFace Transformers
- **Acceleration:** NVIDIA CUDA (DistributedDataParallel 사용)
- **Monitoring:** Weights & Biases (W&B)
- **Hardware:** Multi-GPU (RTX A6000 * 2)

## 2. Model Modeling Principles
- **Reproducibility:** 모든 실험 코드에는 `seed_everything` 함수를 포함하고, 랜덤 시드를 고정합니다.
- **Memory Efficiency:** `torch.cuda.empty_cache()` 호출 및 `Gradient Accumulation` 전략을 우선 제안하세요.
- **Validation:** 학습 루프 생성 시 항상 Validation Step과 Early Stopping 로직을 기본으로 포함합니다.
- **Documentation:** 모델 아키텍처 변경 시 파라미터 수($\text{Total Parameters}$)와 레이어 구조를 요약해 주세요.

## 3. Gemini Modeling Assistance
- **Code Generation:** Tensor Shape($\text{Batch, Seq\_Len, Hidden\_Dim}$) 주석을 코드 라인마다 명시해 주세요.
- **Optimization:** `AdamW`와 `CosineAnnealingLR` 스케줄러 조합을 기본으로 추천합니다.
- **Debugging:** `RuntimeError: CUDA out of memory` 발생 시 배치 사이즈 조정 및 `fp16/bf16` 혼합 정밀도 훈련 전환 코드를 즉시 제공하세요.

## 4. Data Pipeline
- `Dataset` 및 `DataLoader` 구현 시 `num_workers`와 `pin_memory` 설정을 최적화하여 제안하세요.