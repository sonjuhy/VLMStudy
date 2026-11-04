"""
depth map에서 규칙 기반으로 만든 공간 관계 QA 데이터셋.
같은 seed면 RGB 인코더와 RGB-D 인코더가 완전히 동일한 이미지/질문/정답을 보게 된다.

질문 3종 (정답은 depth map의 영역 평균으로 결정, 작을수록 카메라에 가까움):
  lr: 왼쪽/오른쪽 중 어느 쪽이 더 가까운가  -> left / right
  tb: 위/아래 중 어느 쪽이 더 가까운가      -> top / bottom
  cb: 중앙이 가장자리보다 가까운가          -> yes / no
영역 평균 차이가 작아 모호한 샘플(MARGIN 미만)은 제외한다.
"""

import math
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MARGIN = 0.10
QUESTIONS = {
    "lr": ("Which side of the image is closer to the camera, the left or the right?", ["left", "right"]),
    "tb": ("Which part of the image is closer to the camera, the top or the bottom?", ["top", "bottom"]),
    "cb": ("Is the center of the image closer to the camera than its border?", ["yes", "no"]),
}
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _region_means(d: np.ndarray):
    h, w = d.shape
    center = np.zeros_like(d, dtype=bool)
    center[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
    return {
        "lr": (d[:, : w // 2].mean(), d[:, w // 2 :].mean()),
        "tb": (d[: h // 2].mean(), d[h // 2 :].mean()),
        "cb": (d[center].mean(), d[~center].mean()),
    }


def _answer_index(qtype: str, a: float, b: float) -> int:
    # a, b: (left|top|center) 평균, (right|bottom|border) 평균. 작은 쪽이 더 가깝고,
    # 후보 순서가 (left,right) / (top,bottom) / (yes,no)라 첫 항목이 더 가까우면 0.
    return 0 if a < b else 1


class SpatialQADataset(Dataset):
    def __init__(self, rgb_root, depth_root, split, n_images, tokenizer,
                 use_depth, seed=42, max_length=64):
        self.rgb_split = os.path.join(rgb_root, split)
        self.depth_split = os.path.join(depth_root, split)
        self.tok = tokenizer
        self.use_depth = use_depth
        self.max_length = max_length

        rng = random.Random(seed)
        classes = sorted(e.name for e in os.scandir(self.rgb_split) if e.is_dir())
        per_class = math.ceil(n_images / len(classes))
        rels = []
        for c in classes:
            files = sorted(os.listdir(os.path.join(self.rgb_split, c)))
            rng.shuffle(files)
            rels += [os.path.join(c, f) for f in files[:per_class]]
        rng.shuffle(rels)
        rels = rels[:n_images]

        self.samples = []  # (rel_path, qtype, answer_idx)
        for rel in rels:
            d = np.load(os.path.join(self.depth_split, os.path.splitext(rel)[0] + ".npy"))
            for qtype, (a, b) in _region_means(d).items():
                if abs(a - b) / max(a, b, 1e-6) < MARGIN:
                    continue
                self.samples.append((rel, qtype, _answer_index(qtype, a, b)))

        # 정답이 한쪽으로 치우치면(예: 아래쪽이 거의 항상 더 가까움) 이미지를 안 봐도 점수가 높아지므로
        # 질문 유형별로 정답 비율을 50:50으로 맞춘다.
        by_key = {}
        for s in self.samples:
            by_key.setdefault((s[1], s[2]), []).append(s)
        balanced = []
        for qtype in QUESTIONS:
            groups = [by_key.get((qtype, k), []) for k in (0, 1)]
            n = min(len(g) for g in groups)
            for g in groups:
                balanced += g[:n]
        rng.shuffle(balanced)
        self.samples = balanced

        # 평가용: 후보 답의 첫 토큰이 서로 달라야 한다
        self.cand_ids = {}
        for qtype, (_, cands) in QUESTIONS.items():
            ids = [self.tok(" " + c, add_special_tokens=False).input_ids[0] for c in cands]
            assert len(set(ids)) == len(ids), f"candidate first tokens collide: {cands}"
            self.cand_ids[qtype] = ids

    def __len__(self):
        return len(self.samples)

    def _image(self, rel):
        img = Image.open(os.path.join(self.rgb_split, rel)).convert("RGB").resize((224, 224), Image.BICUBIC)
        x = (np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0 - IMG_MEAN) / IMG_STD
        if self.use_depth:
            d = np.load(os.path.join(self.depth_split, os.path.splitext(rel)[0] + ".npy")).astype(np.float32)
            m = d.max()
            if m > 0:
                d = d / m
            x = np.concatenate([x, ((d - 0.5) / 0.5)[None]], axis=0)
        return torch.from_numpy(x)

    def __getitem__(self, i):
        rel, qtype, ans = self.samples[i]
        q, cands = QUESTIONS[qtype]
        prompt_ids = self.tok(f"Question: {q}\nAnswer:", add_special_tokens=False).input_ids
        ans_ids = self.tok(" " + cands[ans], add_special_tokens=False).input_ids + [self.tok.eos_token_id]
        ids = (prompt_ids + ans_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + ans_ids)[: self.max_length]
        return {
            "image": self._image(rel),
            "input_ids": ids,
            "labels": labels,
            "prompt_len": len(prompt_ids),
            "qtype": qtype,
            "answer": ans,
        }


class Collate:
    """Windows spawn 워커로 pickle 가능해야 하므로 중첩 함수가 아닌 클래스로 둔다."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        T = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), T), self.pad_id, dtype=torch.long)
        lab = torch.full((len(batch), T), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), T), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            lab[i, :n] = torch.tensor(b["labels"])
            mask[i, :n] = 1
        return {
            "images": torch.stack([b["image"] for b in batch]),
            "input_ids": ids,
            "labels": lab,
            "attention_mask": mask,
            "prompt_len": torch.tensor([b["prompt_len"] for b in batch]),
            "qtype": [b["qtype"] for b in batch],
            "answer": torch.tensor([b["answer"] for b in batch]),
        }
