#!/usr/bin/env bash
# GCP Spot VM 3대 생성 스크립트
#
# 사용법:
#   bash projector_study/scripts/gcp_create_vms.sh
#
# 사전 준비:
#   1. gcloud auth login && gcloud config set project <PROJECT_ID>
#   2. GPUS_ALL_REGIONS 할당량 6 이상 승인 확인
#   3. GCS 버킷 생성: gsutil mb gs://vlmstudy-checkpoints gs://vlmstudy-data
#   4. 데이터 업로드: gsutil -m cp llava_558k.json gs://vlmstudy-data/
#   5. (선택) 아래 변수 수정

set -euo pipefail

# ── 설정 ────────────────────────────────────────────────────────────────────
PROJECT="${PROJECT:-$(gcloud config get-value project)}"
ZONE="${ZONE:-asia-northeast3-a}"          # 서울 (L4 가용 리전 확인 필요)
MACHINE_TYPE="g2-standard-24"              # 2× L4 GPU, 96GB RAM
DISK_SIZE="200GB"
IMAGE_FAMILY="pytorch-2-1-cu121-ubuntu-2004-py310"
IMAGE_PROJECT="deeplearning-platform-release"
GCS_BUCKET="gs://vlmstudy-checkpoints"
REPO_URL="https://github.com/sonjuhy/VLMStudy.git"

# ── VM 할당 ──────────────────────────────────────────────────────────────────
# VM-A: linear + resampler  (no-compression baseline + large query-based)
# VM-B: mlp + c_abstractor  (LLaVA-1.5 + spatial-aware)
# VM-C: qformer + pixel_shuffle  (cross-attention + tile-based)
declare -A VM_PROJECTORS
VM_PROJECTORS["vlm-train-a"]="linear resampler"
VM_PROJECTORS["vlm-train-b"]="mlp c_abstractor"
VM_PROJECTORS["vlm-train-c"]="qformer pixel_shuffle"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STARTUP_TEMPLATE="$SCRIPT_DIR/gcp_vm_train.sh"

create_vm() {
    local VM_NAME="$1"
    local PROJ1="$2"
    local PROJ2="${3:-}"

    echo "========================================"
    echo "  VM 생성: $VM_NAME ($PROJ1 $PROJ2)"
    echo "========================================"

    # startup-script: VM 기동 후 자동 실행
    local STARTUP_CMD
    STARTUP_CMD=$(cat << STARTUP_EOF
#!/usr/bin/env bash
set -e
cd /home/\$(whoami)
git clone $REPO_URL VLMStudy 2>/dev/null || (cd VLMStudy && git pull)
cd VLMStudy
GCS_BUCKET=$GCS_BUCKET bash projector_study/scripts/gcp_vm_train.sh $PROJ1 $PROJ2 \
    >> /tmp/startup.log 2>&1
STARTUP_EOF
)

    gcloud compute instances create "$VM_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --accelerator="type=nvidia-l4,count=2" \
        --maintenance-policy=TERMINATE \
        --provisioning-model=SPOT \
        --instance-termination-action=STOP \
        --boot-disk-size="$DISK_SIZE" \
        --boot-disk-type=pd-ssd \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --scopes=storage-full,logging-write \
        --metadata="startup-script=$STARTUP_CMD,install-nvidia-driver=True" \
        --no-restart-on-failure

    echo "  $VM_NAME 생성 완료"
}

# ── VM 3대 동시 생성 ─────────────────────────────────────────────────────────
echo "GCP Spot VM 3대 생성 시작 (project: $PROJECT, zone: $ZONE)"
echo ""

create_vm "vlm-train-a" linear resampler &
create_vm "vlm-train-b" mlp c_abstractor &
create_vm "vlm-train-c" qformer pixel_shuffle &

wait

echo ""
echo "========================================"
echo "  모든 VM 생성 완료"
echo ""
echo "  상태 확인:"
echo "    gcloud compute instances list --filter='name~vlm-train'"
echo ""
echo "  로그 확인 (예: VM-A):"
echo "    gcloud compute ssh vlm-train-a --zone $ZONE -- 'tail -f /tmp/startup.log'"
echo ""
echo "  GCS 결과 확인:"
echo "    gsutil ls $GCS_BUCKET/checkpoints/"
echo "    gsutil ls $GCS_BUCKET/results/"
echo "========================================"
