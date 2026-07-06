# GCP Connection Check & Training Guide

이 프로젝트(`VLMStudy`)를 Google Cloud Platform(GCP)에 연결하여 학습할 수 있는지 확인한 결과와 GCP 학습 연동 가이드입니다.

---

## 1. 현재 GCP 연결 상태 확인 결과

로컬 환경에 설정된 `gcloud` CLI 상태를 확인한 결과, GCP 프로젝트에 정상적으로 로그인되어 있으며 API 호출이 가능합니다.

### 🔍 로그인 및 프로젝트 정보
- **계정**: `sonjuhy@gmail.com`
- **활성화된 프로젝트**: `digital-yeti-307713`
- **기본 리전/존**: `asia-northeast3` (서울) / `asia-northeast3-a`

### 🛠️ 활성화된 주요 GCP API
현재 프로젝트에서 아래 주요 API들이 활성화되어 있어 즉시 자원을 생성하고 활용할 수 있습니다.
- `compute.googleapis.com` (Compute Engine API)
- `artifactregistry.googleapis.com` (Artifact Registry API)
- `storage.googleapis.com` (Cloud Storage API)

### 📦 기존 자원 및 GPU 쿼터(Quota) 상태
- **Compute Engine VM**: 현재 활성화된 GPU/CPU VM이 없습니다. (0 items)
- **Cloud Storage (GCS) 버킷**: 
  - `gs://gcf-sources-1035118627521-asia-northeast3/` (기본 생성된 버킷)
- **GPU 쿼터 상태 (중요 ⚠️)**:
  - **글로벌 GPU 총합 한도 (`GPUS_ALL_REGIONS`)**: **`0.0`** (현재 차단됨)
  - **서울 리전 (`asia-northeast3`) 개별 GPU 한도**: 
    - `NVIDIA_L4_GPUS`: `1.0`
    - `NVIDIA_T4_GPUS`: `1.0`
    - `NVIDIA_V100_GPUS`: `1.0`
    - `NVIDIA_A100_GPUS` & `NVIDIA_A100_80GB_GPUS`: `0.0`
  - **진단 결과**: 서울 리전 내에 L4, T4 등의 개별 쿼터가 `1.0`으로 되어 있으나, 프로젝트 전체의 글로벌 GPU 한도(`GPUS_ALL_REGIONS`)가 `0.0`으로 막혀 있기 때문에 **이 상태 그대로는 GPU 인스턴스 생성이 불가능합니다.** VM 생성을 시도하면 쿼터 초과 에러가 발생합니다.

### 💡 GPU 한도 해제(승인) 방법
이 제약을 해결하려면 Google Cloud Console에서 **`GPUS_ALL_REGIONS` 할당량 증가를 신청**해야 합니다.
1. [Google Cloud Console 할당량 페이지](https://console.cloud.google.com/iam-admin/quotas)로 이동합니다.
2. 필터에 `GPUS_ALL_REGIONS`를 검색합니다.
3. 해당 항목을 선택한 후 **할당량 수정(EDIT QUOTAS)**을 클릭합니다.
4. 새로운 한도를 `1` (또는 필요 수량)로 설정하고 사유(예: "ML Model Training for VLM Study")를 입력하여 신청을 제출합니다.
5. 승인은 보통 수분에서 수시간 내에 완료됩니다.


---

## 2. GCP에서 모델 학습을 진행하는 2가지 방법

현재 프로젝트는 MNIST(소규모), ImageNet-1K(대규모, 160GB), RGB-D 데이터셋 학습을 포함하고 있습니다. GCP로 연동하여 학습하기 위한 현실적인 2가지 방안을 제안합니다.

### 💡 방법 A: Compute Engine (GCE) GPU VM 사용 (추천 🌟)
가장 직관적이고 기존 로컬 GPU 서버(`RTX A6000 * 2`)와 유사한 환경을 구축하는 방법입니다. VM을 생성하고 터미널로 접속하여 직접 코드를 실행합니다.

- **장점**: 대화식 디버깅 가능, 코드 수정 및 학습 상태 모니터링이 용이함.
- **적합한 GPU**: 
  - **L4 GPU** (24GB VRAM): 가성비가 매우 뛰어나며, PyTorch 학습에 적합 (`g2-standard-8` 등)
  - **A100 GPU** (40GB/80GB): 대규모 분산 학습이나 대량 메모리가 필요할 때 사용 (`a2-highgpu-1g` 등)

### 💡 방법 B: Vertex AI Custom Training 사용
학습 코드와 환경을 Docker 컨테이너로 패키징하여 Vertex AI에서 학습 작업을 제출하는 방식입니다.

- **장점**: 학습이 끝나면 자동으로 VM이 반환되어 비용 낭비 방지, 분산 학습 관리가 자동화됨.
- **단점**: Docker 배포 환경 세팅 필요, 실시간 디버깅이 Compute Engine에 비해 번거로움.

---

## 3. 학습 시작을 위한 단계별 실천 가이드 (Compute Engine 기준)

ImageNet-1K 데이터셋 크기(160GB)와 학습 효율을 고려했을 때 아래 워크플로우를 권장합니다.

### 1단계: Cloud Storage (GCS) 버킷 생성 및 데이터 업로드
매번 Kaggle에서 160GB 데이터셋을 다운로드받는 것은 비효율적이므로, 데이터셋을 GCS 버킷에 보관해두고 VM 시작 시 다운로드하거나 마운트하는 것이 좋습니다.

```bash
# 1. 버킷 생성 (예: vlm-study-storage)
gcloud storage buckets create gs://vlm-study-storage --location=asia-northeast3

# 2. 로컬에 준비된 ImageNet 데이터를 GCS로 고속 복사 (멀티스레드 복사 -m 옵션)
gcloud storage cp -r ./datasets/imagenet_1k gs://vlm-study-storage/datasets/
```

### 2단계: GPU VM 인스턴스 생성
Deep Learning VM Image를 활용하면 PyTorch와 CUDA 드라이버가 사전 설치되어 있어 설정 단계를 대폭 단축할 수 있습니다.

```bash
# L4 GPU 1개가 장착된 Deep Learning VM 생성 예시
gcloud compute instances create vlm-train-vm \
    --zone=asia-northeast3-a \
    --machine-type=g2-standard-8 \
    --maintenance-policy=TERMINATE \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=300GB \
    --metadata="install-nvidia-driver=True"
```
> [!NOTE]
> GCP에서 GPU를 사용하려면 프로젝트에 GPU 할당량(Quota)이 설정되어 있어야 합니다. 필요시 GCP 콘솔의 `IAM & Admin > Quotas`에서 `NVIDIA_L4_GPUS` 또는 `NVIDIA_A100_GPUS` 한도 증가를 요청해야 합니다.

### 3단계: 코드 배포 및 학습 실행
1. VM에 SSH 접속:
   ```bash
   gcloud compute ssh vlm-train-vm --zone=asia-northeast3-a
   ```
2. VM 내부에서 코드 클론 및 데이터 다운로드:
   ```bash
   # git clone 또는 로컬에서 gcloud compute scp로 복사
   git clone <your-repo-url> VLMStudy
   cd VLMStudy

   # GCS에서 데이터셋 복사 (매우 빠름)
   mkdir -p datasets
   gcloud storage cp -r gs://vlm-study-storage/datasets/imagenet_1k ./datasets/
   ```
3. 라이브러리 설치 및 학습 시작:
   ```bash
   pip install -r requirements.txt  # pandas, tqdm, transformers 등
   # DDP 학습 실행
   torchrun --nproc_per_node=1 -m end_to_end.imagenet_ete
   ```

---

## 4. 권장 액션 플랜

다음 단계를 원하신다면 말씀해주세요. 필요한 설정을 즉시 도와드리겠습니다.
1. **GCS 버킷 생성 스크립트 작성**: 데이터셋 및 모델 체크포인트를 저장할 버킷 생성 자동화
2. **GPU VM 생성 스크립트 작성**: 원하는 GPU 종류(L4, A100 등)와 디스크 크기를 반영한 커스텀 VM 생성 스크립트 작성
3. **Dockerfile 작성 (Vertex AI 대비)**: 컨테이너 기반 훈련을 원할 경우 Docker 이미지 구성 설정
