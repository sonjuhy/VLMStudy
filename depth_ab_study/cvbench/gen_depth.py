"""
CV-Bench A/B용 이미지(학습/테스트, 박스가 그려진 상태 그대로)의 DA3 depth map을 만든다.
RGB-D 인코더는 실제 사용 시에도 DA3 depth를 입력받으므로, 학습/평가 모두 같은 방식으로 생성한다.
ImageNet depth와 같은 모델(da3mono-large)과 같은 후처리(224x224 리사이즈, float32 .npy)를 쓴다.

    images/<...>.jpg  ->  depth/<...>.npy

실행:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python depth_ab_study/cvbench/gen_depth.py
"""

import json
import os

import cv2
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3
from PIL import Image
from tqdm import tqdm

ROOT = "datasets/cvbench_ab"
BATCH_SIZE = 16
MODEL_NAME = "depth-anything/da3mono-large"


def depth_path(image_path):
    p = image_path.replace(os.sep + "images" + os.sep, os.sep + "depth" + os.sep).replace("/images/", "/depth/")
    return os.path.splitext(p)[0] + ".npy"


def infer(model, paths):
    """OOM이 나면 배치를 반으로 나눠 재시도한다."""
    try:
        with torch.no_grad():
            return list(zip(paths, model.inference(paths).depth))
    except RuntimeError as e:
        if "out of memory" not in str(e).lower() or len(paths) == 1:
            raise
        torch.cuda.empty_cache()
        mid = len(paths) // 2
        return infer(model, paths[:mid]) + infer(model, paths[mid:])


def main():
    paths = []
    for split in ("test", "train"):
        with open(os.path.join(ROOT, split, "questions.jsonl"), encoding="utf-8") as f:
            paths += [json.loads(l)["image"] for l in f]
    todo = sorted({p for p in paths if not os.path.exists(depth_path(p))})
    # 배치 안의 이미지 크기가 다르면 DA3가 가장 작은 크기로 crop하므로 크기별로 묶는다
    todo.sort(key=lambda p: Image.open(p).size)
    print(f"images: {len(set(paths))}, todo: {len(todo)}")

    model = DepthAnything3.from_pretrained(MODEL_NAME).to("cuda").eval()
    for i in tqdm(range(0, len(todo), BATCH_SIZE), desc="DA3 depth"):
        batch = todo[i:i + BATCH_SIZE]
        sizes = {Image.open(p).size for p in batch}
        groups = [[p for p in batch if Image.open(p).size == s] for s in sizes]
        for g in groups:
            for p, d in infer(model, g):
                if d.shape != (224, 224):
                    d = cv2.resize(d, (224, 224), interpolation=cv2.INTER_LINEAR)
                out = depth_path(p)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                np.save(out, d.astype(np.float32))


if __name__ == "__main__":
    main()
