#!/usr/bin/env bash
# GCP Spot VM 원클릭 학습 스크립트
#
# 사용법:
#   bash gcp_vm_train.sh <projector1> [projector2]
#
# 예시 (VM-A: linear + resampler):
#   bash gcp_vm_train.sh linear resampler
#
# 이 스크립트는 VM 시작 시 --metadata startup-script로 실행되거나
# SSH 접속 후 직접 실행합니다.
# 완료 후 VM이 자동 종료됩니다 (Spot 비용 절감).

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <projector1> [projector2]"
    exit 1
fi

PROJECTORS=("$@")

# ── 설정 (필요 시 수정) ──────────────────────────────────────────────────────
GCS_BUCKET="${GCS_BUCKET:-gs://vlmstudy-checkpoints}"
GCS_DATA_BUCKET="${GCS_DATA_BUCKET:-gs://vlmstudy-data}"
REPO_DIR="/home/$(whoami)/VLMStudy"
LLM_ID="${LLM_ID:-Qwen/Qwen2.5-7B-Instruct}"
HF_CACHE="${HF_CACHE:-/home/$(whoami)/.cache/huggingface}"

DATA_DIR="/data"
JSON_PATH="$DATA_DIR/llava_558k.json"
IMG_ROOT="$DATA_DIR/llava_images"
VIT_CKPT="$REPO_DIR/checkpoints/final_model/best_vit_imagenet_1k.pth"
OUTPUT_DIR="$REPO_DIR/projector_study"

EPOCHS=3
BATCH_SIZE=4
ACCUM=4

LOG_FILE="/tmp/vlm_train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================"
echo "  VLMStudy GCP VM 학습 시작: $(date)"
echo "  Projectors: ${PROJECTORS[*]}"
echo "============================================"

# ── 1. 시스템 패키지 ─────────────────────────────────────────────────────────
echo "[1/6] 시스템 패키지 확인..."
if ! command -v gcsfuse &>/dev/null; then
    export GCSFUSE_REPO="gcsfuse-$(lsb_release -cs)"
    echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" \
        | sudo tee /etc/apt/sources.list.d/gcsfuse.list
    curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt-get update -qq
    sudo apt-get install -y gcsfuse
fi

# ── 2. 저장소 클론 / 업데이트 ────────────────────────────────────────────────
echo "[2/6] 저장소 준비..."
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/sonjuhy/VLMStudy.git "$REPO_DIR"
else
    cd "$REPO_DIR" && git pull --ff-only
fi
cd "$REPO_DIR"

# venv 생성 / 의존성 설치
if [ ! -f ".venv/bin/python" ]; then
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip -q
    .venv/bin/pip install -r requirements.txt -q
fi

# ── 3. 데이터 준비 ───────────────────────────────────────────────────────────
echo "[3/6] 데이터 다운로드..."
sudo mkdir -p "$DATA_DIR"
sudo chown "$(whoami)" "$DATA_DIR"

# GCS → 로컬 복사 (이미 있으면 스킵)
if [ ! -f "$JSON_PATH" ]; then
    echo "  LLaVA 558K JSON 다운로드..."
    gsutil -m cp "$GCS_DATA_BUCKET/llava_558k.json" "$JSON_PATH"
fi

if [ ! -d "$IMG_ROOT" ]; then
    echo "  이미지 다운로드 (시간 소요)..."
    mkdir -p "$IMG_ROOT"
    gsutil -m rsync -r "$GCS_DATA_BUCKET/llava_images/" "$IMG_ROOT/"
fi

# ViT 체크포인트
mkdir -p "$(dirname "$VIT_CKPT")"
if [ ! -f "$VIT_CKPT" ]; then
    gsutil cp "$GCS_BUCKET/vit/best_vit_imagenet_1k.pth" "$VIT_CKPT"
fi

# ── 4. HuggingFace 모델 사전 캐시 ───────────────────────────────────────────
echo "[4/6] LLM 사전 캐시..."
if [ ! -d "$HF_CACHE/hub/models--$(echo "$LLM_ID" | tr '/' '--')" ]; then
    HF_HOME="$HF_CACHE" .venv/bin/python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('$LLM_ID')
AutoModelForCausalLM.from_pretrained('$LLM_ID', torch_dtype='auto')
print('LLM 캐시 완료')
"
fi

# ── 5. Resume 체크포인트 GCS → 로컬 복원 ────────────────────────────────────
echo "[5/6] 기존 체크포인트 복원..."
for PROJ in "${PROJECTORS[@]}"; do
    REMOTE_EMERG="$GCS_BUCKET/checkpoints/${PROJ}/emergency_projector.pth"
    LOCAL_EMERG="$OUTPUT_DIR/checkpoints/${PROJ}_resume/emergency_projector.pth"
    if gsutil -q stat "$REMOTE_EMERG" 2>/dev/null; then
        mkdir -p "$(dirname "$LOCAL_EMERG")"
        gsutil cp "$REMOTE_EMERG" "$LOCAL_EMERG"
        echo "  $PROJ: emergency checkpoint 복원"
    fi
done

# ── 6. 학습 ─────────────────────────────────────────────────────────────────
echo "[6/6] 학습 시작..."

# SIGTERM 수신 시 GCS 업로드 후 종료
_upload_and_exit() {
    echo "SIGTERM 수신 — checkpoint를 GCS에 업로드..."
    for PROJ in "${PROJECTORS[@]}"; do
        gsutil -m rsync -r \
            "$OUTPUT_DIR/checkpoints/" \
            "$GCS_BUCKET/checkpoints/" 2>/dev/null || true
    done
    # 로그도 업로드
    gsutil cp "$LOG_FILE" "$GCS_BUCKET/logs/$(basename "$LOG_FILE")" 2>/dev/null || true
    exit 0
}
trap _upload_and_exit SIGTERM

for PROJ in "${PROJECTORS[@]}"; do
    echo "--- Projector: $PROJ ---"

    # 로컬 resume 체크포인트 자동 탐색
    RESUME_ARG=""
    EMERG_PATH=$(find "$OUTPUT_DIR/checkpoints" -name "emergency_projector.pth" \
                 -path "*${PROJ}*" 2>/dev/null | sort | tail -1 || true)
    EPOCH_PATH=$(find "$OUTPUT_DIR/checkpoints" -name "projector_epoch_*.pth" \
                 -path "*${PROJ}*" 2>/dev/null | sort -V | tail -1 || true)

    if [ -n "$EMERG_PATH" ]; then
        RESUME_ARG="--resume_ckpt $EMERG_PATH"
        echo "  Resume from: $EMERG_PATH"
    elif [ -n "$EPOCH_PATH" ]; then
        RESUME_ARG="--resume_ckpt $EPOCH_PATH"
        echo "  Resume from: $EPOCH_PATH"
    fi

    LLM_ID="$LLM_ID" \
    HF_HOME="$HF_CACHE" \
    .venv/bin/python -m projector_study.train.projector_train \
        --projector    "$PROJ" \
        --llm_id       "$LLM_ID" \
        --vit_ckpt     "$VIT_CKPT" \
        --json_path    "$JSON_PATH" \
        --img_root     "$IMG_ROOT" \
        --output_dir   "$OUTPUT_DIR" \
        --epochs       "$EPOCHS" \
        --batch_size   "$BATCH_SIZE" \
        --accumulation_steps "$ACCUM" \
        $RESUME_ARG

    echo "  $PROJ 완료 — GCS 업로드..."
    gsutil -m rsync -r \
        "$OUTPUT_DIR/checkpoints/" \
        "$GCS_BUCKET/checkpoints/"
    gsutil -m rsync -r \
        "$OUTPUT_DIR/results/" \
        "$GCS_BUCKET/results/"
done

# 로그 업로드
gsutil cp "$LOG_FILE" "$GCS_BUCKET/logs/$(basename "$LOG_FILE")"

echo "============================================"
echo "  모든 학습 완료: $(date)"
echo "  VM 5초 후 종료..."
echo "============================================"
sleep 5
sudo poweroff
