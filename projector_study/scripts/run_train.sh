#!/usr/bin/env bash
# Projector 학습 스크립트
#
# 로컬(DDP):
#   torchrun --nproc_per_node=2 projector_study/scripts/run_train.sh
#
# GCP 단일-프로세스 (device_map="auto"로 2× L4 자동 분산):
#   bash projector_study/scripts/run_train.sh linear mlp
#   (인수 없으면 6개 전부 순차 실행)

set -euo pipefail

LLM_ID="${LLM_ID:-Qwen/Qwen2.5-7B-Instruct}"
VIT_CKPT="${VIT_CKPT:-checkpoints/final_model/best_vit_imagenet_1k.pth}"
JSON_PATH="${JSON_PATH:-/data/llava_558k.json}"
IMG_ROOT="${IMG_ROOT:-/data/llava_images}"
OUTPUT_DIR="${OUTPUT_DIR:-projector_study}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUM="${ACCUM:-4}"
RESUME_CKPT="${RESUME_CKPT:-}"

# 인수로 projector 목록을 받고, 없으면 전부 실행
if [ "$#" -gt 0 ]; then
    PROJECTORS=("$@")
else
    PROJECTORS=(linear mlp qformer resampler c_abstractor pixel_shuffle)
fi

PYTHON=".venv/bin/python"

for PROJ in "${PROJECTORS[@]}"; do
    echo "========================================"
    echo "  Training projector: $PROJ"
    echo "========================================"

    RESUME_ARG=""
    # --resume_ckpt 자동 탐색: emergency → 최신 epoch 체크포인트 순
    if [ -n "$RESUME_CKPT" ]; then
        RESUME_ARG="--resume_ckpt $RESUME_CKPT"
    else
        LATEST=$(ls "${OUTPUT_DIR}/checkpoints/${PROJ}_"*/projector_epoch_*.pth 2>/dev/null \
                 | sort -V | tail -1 || true)
        EMERG=$(ls "${OUTPUT_DIR}/checkpoints/${PROJ}_"*/emergency_projector.pth 2>/dev/null \
                | sort -V | tail -1 || true)
        if [ -n "$EMERG" ]; then
            RESUME_ARG="--resume_ckpt $EMERG"
        elif [ -n "$LATEST" ]; then
            RESUME_ARG="--resume_ckpt $LATEST"
        fi
    fi

    $PYTHON -m projector_study.train.projector_train \
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

    echo "  ✓ $PROJ 완료"
done

echo "All projectors trained."
