"""
순수 ViT(RGB) vs Depth ViT(RGB-D)를 CV-Bench로 비교한다.
ViT / LLM(Qwen2.5-0.5B-Instruct)은 Frozen, Projector만 학습 (이전 A/B와 같은 조건).

- 학습: build_data.py가 만든 COCO/SUN RGB-D 문항 (CV-Bench와 같은 템플릿, 독립적인 GT)
- 평가: CV-Bench test 2,638문항, 정답 보기 문자(A~F)의 로짓 argmax로 채점
- 점수: CV-Bench 공식 식 = 1/2 * ((acc_2D_ADE20K + acc_2D_COCO)/2 + acc_3D_Omni3D)

실행 예:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python depth_ab_study/cvbench/train_cvbench.py --encoder rgb --seed 0
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from depth_ab_study.cvbench.gen_depth import depth_path
from depth_ab_study.spatial_qa import IMG_MEAN, IMG_STD, Collate
from depth_ab_study.train_ab import ENCODERS, answer_loss, build_vit, hidden_states, log
from projector_study.models.base_vlm import ProjectorVLM
from projector_study.models.projectors import PROJECTOR_REGISTRY

ROOT = "datasets/cvbench_ab"
LETTERS = "ABCDEF"
SUFFIX = "\nAnswer with the option's letter from the given choices directly."


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


class CVBenchQA(Dataset):
    def __init__(self, rows, tokenizer, use_depth, max_length=192):
        self.rows, self.tok, self.use_depth, self.max_length = rows, tokenizer, use_depth, max_length

    def __len__(self):
        return len(self.rows)

    def _image(self, path):
        img = Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)
        x = (np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0 - IMG_MEAN) / IMG_STD
        if self.use_depth:
            d = np.load(depth_path(path)).astype(np.float32)
            m = d.max()
            if m > 0:
                d = d / m
            x = np.concatenate([x, ((d - 0.5) / 0.5)[None]], axis=0)
        return torch.from_numpy(x)

    def __getitem__(self, i):
        r = self.rows[i]
        prompt_ids = self.tok(f"Question: {r['prompt']}{SUFFIX}\nAnswer:", add_special_tokens=False).input_ids
        ans_ids = self.tok(" " + LETTERS[r["answer"]], add_special_tokens=False).input_ids + [self.tok.eos_token_id]
        ids = (prompt_ids + ans_ids)[-self.max_length:]
        labels = ([-100] * len(prompt_ids) + ans_ids)[-self.max_length:]
        return {
            "image": self._image(r["image"]),
            "input_ids": ids,
            "labels": labels,
            "prompt_len": len(ids) - len(ans_ids),
            "qtype": i,  # 평가 때 원래 행을 찾는 데 쓴다
            "answer": r["answer"],
        }


@torch.no_grad()
def predict(model, ds, loader, letter_ids, device):
    model.eval()
    preds = {}
    for b in tqdm(loader, desc="Eval", leave=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h, n_img = hidden_states(model, b, device)
        for j, i in enumerate(b["qtype"]):
            pos = n_img + int(b["prompt_len"][j]) - 1
            logits = model.llm.lm_head(h[j, pos]).float()
            n = len(ds.rows[i]["choices"])
            preds[i] = int(logits[letter_ids[:n]].argmax())
    return preds


def score(rows, preds):
    hit = defaultdict(list)
    for i, r in enumerate(rows):
        ok = int(preds[i] == r["answer"])
        hit[(r["type"], r["source"])].append(ok)
        hit[("task", r["task"])].append(ok)
        if r["type"] == "3D":
            hit[("3D_source", r["source_dataset"])].append(ok)
    acc = {f"{k[0]}/{k[1]}": 100.0 * sum(v) / len(v) for k, v in hit.items()}
    a2d = (acc["2D/ADE20K"] + acc["2D/COCO"]) / 2
    a3d = acc["3D/Omni3D"]
    return {"cv_bench": (a2d + a3d) / 2, "2D": a2d, "3D": a3d, **dict(sorted(acc.items()))}


def chance(rows):
    """무작위로 찍을 때의 기대 점수 (같은 공식 적용)."""
    preds_like = defaultdict(list)
    for r in rows:
        preds_like[(r["type"], r["source"])].append(1 / len(r["choices"]))
    acc = {k: 100.0 * sum(v) / len(v) for k, v in preds_like.items()}
    return ((acc[("2D", "ADE20K")] + acc[("2D", "COCO")]) / 2 + acc[("3D", "Omni3D")]) / 2


def main(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda")
    out_dir = os.path.join(a.output_dir, a.encoder, f"seed{a.seed}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_log.txt")
    log(log_path, f"args: {vars(a)}")

    tok = AutoTokenizer.from_pretrained(a.llm_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(a.llm_id, torch_dtype=torch.bfloat16).to(device)
    letter_ids = torch.tensor([tok(" " + c, add_special_tokens=False).input_ids[0] for c in LETTERS], device=device)
    assert len(set(letter_ids.tolist())) == len(LETTERS)

    vit = build_vit(a.encoder)
    projector = PROJECTOR_REGISTRY[a.projector](v_dim=768, l_dim=llm.config.hidden_size)
    model = ProjectorVLM(vit, projector, llm).to(device)
    model.projector.float()
    log(log_path, f"trainable params={model.trainable_params():,}")

    use_depth = a.encoder == "rgbd"
    train_rows = read_jsonl(os.path.join(ROOT, "train", "questions.jsonl"))
    test_rows = read_jsonl(os.path.join(ROOT, "test", "questions.jsonl"))
    # 에포크별 모니터링은 학습셋에서 떼어낸 val로만 하고, CV-Bench test는 마지막에 한 번만 본다
    rng = random.Random(0)
    rng.shuffle(train_rows)
    n_val = int(0.03 * len(train_rows))
    val_rows, train_rows = train_rows[:n_val], train_rows[n_val:]
    if a.max_train:
        train_rows = train_rows[:a.max_train]
    train_ds, val_ds, test_ds = (CVBenchQA(r, tok, use_depth) for r in (train_rows, val_rows, test_rows))
    log(log_path, f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    collate = Collate(tok.pad_token_id)
    train_dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
                          collate_fn=collate, drop_last=True, persistent_workers=a.num_workers > 0)
    mk_eval = lambda ds: DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
                                    collate_fn=collate)
    val_dl, test_dl = mk_eval(val_ds), mk_eval(test_ds)

    opt = torch.optim.AdamW(model.projector.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    total = a.epochs * (len(train_dl) // a.accumulation_steps)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    step = 0
    for ep in range(a.epochs):
        model.train(); model.vit.eval(); model.llm.eval()
        run, cnt = 0.0, 0
        for i, b in enumerate(tqdm(train_dl, desc=f"Epoch {ep+1}/{a.epochs}")):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = answer_loss(model, b, device)
            (loss / a.accumulation_steps).backward()
            run += loss.item(); cnt += 1
            if (i + 1) % a.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.projector.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 100 == 0:
                    log(log_path, f"step {step}/{total} | lr {sched.get_last_lr()[0]:.2e} | loss {run/cnt:.4f}")
                    run, cnt = 0.0, 0
        vp = predict(model, val_ds, val_dl, letter_ids, device)
        per_task = defaultdict(list)
        for i, r in enumerate(val_rows):
            per_task[r["task"]].append(int(vp[i] == r["answer"]))
        log(log_path, f"Epoch {ep+1} val: " + json.dumps({k: round(100 * sum(v) / len(v), 2) for k, v in per_task.items()}))

    tp = predict(model, test_ds, test_dl, letter_ids, device)
    res = score(test_rows, tp)
    res["chance_cv_bench"] = chance(test_rows)
    log(log_path, "CV-Bench test: " + json.dumps({k: round(v, 2) for k, v in res.items()}))

    torch.save(model.projector.state_dict(), os.path.join(out_dir, "projector.pth"))
    with open(os.path.join(out_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for i, r in enumerate(test_rows):
            f.write(json.dumps({"idx": r["idx"], "task": r["task"], "pred": LETTERS[tp[i]],
                                "answer": LETTERS[r["answer"]]}) + "\n")
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"encoder": a.encoder, "seed": a.seed, "args": vars(a), "results": res}, f, indent=2)
    log(log_path, "done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True, choices=list(ENCODERS))
    p.add_argument("--llm_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--projector", default="mlp", choices=list(PROJECTOR_REGISTRY))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--accumulation_steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_train", type=int, default=0, help="스모크 테스트용 학습 문항 수 제한 (0=전체)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="depth_ab_study/results_cvbench")
    main(p.parse_args())
