# VLMStudy

Vision Transformer(ViT)와 Vision Language Model(VLM)을 PyTorch로 처음부터 직접 구현하며 학습하는 프로젝트입니다.

---

## 목표

- **ViT 구현**: 이미지 패치 임베딩, Multi-Head Attention, Feed-Forward Layer 등 Transformer 구조를 직접 구현
- **VLM 구현**: 학습된 ViT를 시각 인코더로 활용하고, LLM과 Projector를 연결하는 멀티모달 파이프라인 구축
- **다양한 데이터셋 적용**: MNIST(소규모) → ImageNet-1K(대규모) → RGB-D(멀티모달) 순서로 확장

---

## 구현 모델

### 1. ViT-Base/16 (ImageNet-1K 분류)
- **아키텍처**: Patch Embedding + CLS Token + 12개 Transformer Block + 분류 헤드
- **스펙**: `embedding=768, num_heads=12, n_layers=12, patch_size=16, img_size=224`
- **파라미터**: ~86M
- **학습**: 2-GPU DDP, DeiT 레시피 (AdamW + LinearWarmup + CosineAnnealing + Mixup + AutoAugment)

### 2. ViT (MNIST 분류)
- **아키텍처**: 경량화된 ViT (`n_layers=5, patch_size=7, img_size=28`)
- **학습**: 단일 GPU, Adam 옵티마이저, 10 epochs

### 3. ViT Depth Encoder (RGB-D 분류)
- **아키텍처**: 4채널 입력(RGB + Depth)을 받는 ViT-Base/16 변형
- **특징**: `in_channels=4`, TransformerEncoder 모듈화, DropPath 적용

### 4. Depth VLM (멀티모달)
- **아키텍처**: Frozen ViT Depth Encoder + 2-Layer MLP Projector + Frozen LLM (LLaVA 구조)
- **학습 전략**: Projector만 학습, 시각 특징(196 patch tokens, dim=768)을 LLM 임베딩 공간(dim=4096)으로 변환

---

## 프로젝트 구조

```
VLMStudy/
├── main.py                        # 진입점 (파이프라인 선택 실행)
│
├── vision/                        # 모델 정의
│   ├── vit_model.py               # ViTEncoder, ViTDepthEncoder, Projector 등
│   ├── depth_vit_model.py         # Depth VLM 통합 모델 (DepthVLM)
│   └── mnist_vit_model.py         # MNIST 전용 경량 ViT
│
├── end_to_end/                    # 학습 + 검증 통합 파이프라인
│   ├── mnist_ete.py               # MNIST ViT / VLM 파이프라인
│   ├── imagenet_ete.py            # ImageNet-1K DDP 학습 파이프라인
│   └── depth_ete.py               # RGB-D ImageNet 학습 파이프라인
│
├── train/                         # 학습 루프
│   ├── mnist/
│   │   ├── mnist_vit_train.py
│   │   └── mnist_vlm_train.py
│   ├── imagenet/
│   │   └── imagenet_vit_train.py  # DDP, EarlyStopping, Mixup, AMP, 텍스트 로그
│   └── depth/
│       ├── depth_vit_train.py
│       └── depth_vlm_train.py
│
├── valid/                         # 검증 루프
│   ├── mnist/
│   ├── imagenet/
│   │   └── imagenet_vit_valid.py  # 분산 검증 (all_reduce)
│   └── depth/
│
├── dataloader/                    # 데이터셋 로더
│   ├── mnist_dataloader.py
│   ├── imagenet_1k_dataloader.py  # Kaggle 자동 다운로드 + val 재구성
│   ├── rgbd_imagenet_1k_dataloader.py
│   ├── llava_dataloader.py
│   └── trans_to_rgbd.py           # RGB → RGB-D 변환 유틸
│
├── utils/
│   ├── utils.py                   # save_checkpoint, timer_call
│   └── train_utils.py             # mixup_data, mixup_criterion, DropPath
│
├── scripts/
│   └── run_pipeline.sh            # 데이터 추출 + DDP 학습 자동화 스크립트
│
├── checkpoints/                   # 저장된 모델 체크포인트
└── logs/                          # 학습 로그 (날짜별 폴더)
    └── imagenet_vit/
        └── YYYY-MM-DD/
            └── train_log.txt
```

---

## 실행 방법

### MNIST ViT
```bash
python main.py  # main.py 내 ViTRunning().mnist_vit_end_to_end() 호출
```

### ImageNet-1K (2-GPU DDP)
```bash
# VLMStudy 루트에서 실행
torchrun --nproc_per_node=2 -m end_to_end.imagenet_ete
```

자동화 스크립트 (데이터 추출 → 학습 순차 실행, 백그라운드):
```bash
nohup bash scripts/run_pipeline.sh > /dev/null 2>&1 &
```

### Depth VLM
```bash
python main.py  # main.py 내 VLMRunning().depth_vlm_end_to_end() 호출
```

---

## 학습 설정

### ImageNet-1K (ViT-Base/16)

| 항목 | 설정 |
|------|------|
| 옵티마이저 | AdamW (lr=1e-3, weight_decay=0.05) |
| 스케줄러 | LinearWarmup (20 epoch) → CosineAnnealing (280 epoch) |
| 증강 | RandomResizedCrop, RandomHorizontalFlip, AutoAugment(ImageNet), Mixup(α=0.8, p=0.5) |
| 손실 함수 | CrossEntropyLoss (label_smoothing=0.1) |
| 에폭 | 300 (Early Stopping patience=20) |
| 배치 크기 | 256 per GPU × 2 GPU = 512 global |
| 혼합 정밀도 | AMP fp16 (GradScaler) |
| 분산 학습 | DDP (NCCL, 2×RTX A6000) |

### Depth VLM

| 항목 | 설정 |
|------|------|
| 옵티마이저 | AdamW (lr=1e-4, weight_decay=0.05) |
| 스케줄러 | CosineAnnealingLR |
| 입력 | RGB-D (4채널, 224×224) |
| 배치 크기 | 32 (Gradient Accumulation 16 step) |
| 학습 대상 | Projector만 학습 (ViT + LLM Frozen) |

---

## 환경

```
Python 3.x
PyTorch 2.0+
torchvision
transformers (HuggingFace)
kaggle
pandas
tqdm
```

패키지 설치:
```bash
pip install torch torchvision transformers kaggle pandas tqdm
```

ImageNet-1K 데이터셋은 [Kaggle ILSVRC](https://www.kaggle.com/c/imagenet-object-localization-challenge)에서 다운로드됩니다. `~/.kaggle/kaggle.json`에 API 토큰이 필요합니다.

---

## 하드웨어

- GPU: NVIDIA RTX A6000 × 2
- DDP 통신: NCCL (SHM 경로 사용, P2P 비활성화)
- 예상 학습 시간 (ImageNet-1K 300 epoch): 약 7~8일
