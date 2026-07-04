#!/usr/bin/env bash
# 모든 projector의 속도 + 정확도 벤치마크 실행
# 사용법: bash projector_study/scripts/run_eval.sh

set -e

LLM_ID="Qwen/Qwen2.5-7B-Instruct"
VIT_CKPT="checkpoints/final_model/best_vit_imagenet_1k.pth"
CKPT_DIR="projector_study/checkpoints"

# ── 1. 속도 벤치마크 (전체 projector 일괄) ───────────────────────────────
echo "========================================"
echo "  Speed Benchmark"
echo "========================================"
.venv/bin/python -m projector_study.eval.speed_benchmark \
    --llm_id          "$LLM_ID" \
    --vit_ckpt        "$VIT_CKPT" \
    --proj_ckpt_dir   "$CKPT_DIR"

# ── 2. 정확도 벤치마크 (projector별, 데이터셋별) ─────────────────────────
POPE_ANN="/data/pope/pope_adversarial.jsonl"
POPE_IMG="/data/coco/val2014"

GQA_ANN="/data/gqa/testdev_balanced_questions.json"
GQA_IMG="/data/gqa/images"

for PROJ in linear mlp qformer resampler c_abstractor pixel_shuffle; do
    # best checkpoint 자동 탐색
    PROJ_CKPT=$(ls "$CKPT_DIR"/${PROJ}_*/best_projector.pth 2>/dev/null | sort | tail -1)
    CKPT_ARG=""
    if [ -n "$PROJ_CKPT" ]; then
        CKPT_ARG="--proj_ckpt $PROJ_CKPT"
    fi

    echo "--- POPE [$PROJ] ---"
    .venv/bin/python -m projector_study.eval.benchmark_runner \
        --dataset pope --projector "$PROJ" \
        --llm_id "$LLM_ID" --vit_ckpt "$VIT_CKPT" \
        --ann_path "$POPE_ANN" --img_root "$POPE_IMG" \
        $CKPT_ARG

    echo "--- GQA [$PROJ] ---"
    .venv/bin/python -m projector_study.eval.benchmark_runner \
        --dataset gqa --projector "$PROJ" \
        --llm_id "$LLM_ID" --vit_ckpt "$VIT_CKPT" \
        --ann_path "$GQA_ANN" --img_root "$GQA_IMG" \
        $CKPT_ARG
done

echo "All evaluations done."
echo "Results: projector_study/results/"
