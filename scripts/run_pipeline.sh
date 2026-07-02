#!/bin/bash
set -e

cd /home/edint/test/VLMStudy
source venv/bin/activate

LOG="logs/pipeline.log"
mkdir -p logs

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] Pipeline started (PID $$)." >> "$LOG"

if [ ! -d "datasets/imagenet_1k/train" ] || [ ! -d "datasets/imagenet_1k/val" ]; then
    echo "[$(ts)] Extracting and reorganizing ImageNet-1K dataset..." >> "$LOG"
    python -c "from dataloader.imagenet_1k_dataloader import download_and_extract_from_kaggle; download_and_extract_from_kaggle()" >> "$LOG" 2>&1
else
    echo "[$(ts)] train/val already exist, skipping extraction." >> "$LOG"
fi

if [ ! -d "datasets/imagenet_1k/train" ] || [ ! -d "datasets/imagenet_1k/val" ]; then
    echo "[$(ts)] ERROR: extraction failed, train/val missing. Aborting." >> "$LOG"
    exit 1
fi
echo "[$(ts)] Dataset ready." >> "$LOG"

echo "[$(ts)] Starting DDP training (torchrun, 2 GPUs)..." >> "$LOG"
torchrun --nproc_per_node=2 -m end_to_end.imagenet_ete >> "$LOG" 2>&1

echo "[$(ts)] Training finished." >> "$LOG"
