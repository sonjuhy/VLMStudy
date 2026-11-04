"""
순수 ViT(RGB) vs Depth ViT(RGB-D)를 같은 조건(같은 LLM, Projector, 이미지, 질문, 하이퍼파라미터)으로
공간 관계 QA에 Projector 정렬 학습한 뒤 정확도를 비교한다. ViT와 LLM은 Frozen.

실행 예:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python depth_ab_study/train_ab.py --encoder rgb
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python depth_ab_study/train_ab.py --encoder rgbd
"""

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from depth_ab_study.spatial_qa import Collate, SpatialQADataset
from projector_study.models.base_vlm import ProjectorVLM
from projector_study.models.projectors import PROJECTOR_REGISTRY
from vision.vit_model import ViTDepthEncoder, ViTEncoder

ENCODERS = {
    "rgb": dict(ckpt="checkpoints/final_model/best_vit_imagenet_1k.pth", channels=3),
    "rgbd": dict(ckpt="checkpoints/depth_final_model/best_vit_depth.pth", channels=4),
}


def build_vit(kind):
    cfg = ENCODERS[kind]
    common = dict(embedding_size=768, img_size=224, patch_size=16, num_class=1000,
                  num_heads=12, in_channels=cfg["channels"])
    vit = ViTEncoder(n_layers=12, **common) if kind == "rgb" else ViTDepthEncoder(**common)
    ckpt = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
    vit.load_state_dict(ckpt["model_state_dict"])
    return vit.to(torch.bfloat16).eval()


def log(path, msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def hidden_states(model, b, device):
    """이미지+텍스트를 LLM 본체에 통과시켜 마지막 hidden state만 반환한다.
    어휘 15만 크기의 로짓을 모든 위치(이미지 토큰 196개 포함)에 계산하면 8GB에서 OOM이 나서,
    lm_head는 필요한 위치에만 적용한다."""
    img_emb = model._encode_image(b["images"].to(device))
    txt = model.llm.get_input_embeddings()(b["input_ids"].to(device))
    emb = torch.cat([img_emb, txt], dim=1)
    mask = torch.cat([torch.ones(img_emb.shape[:2], dtype=torch.long, device=device),
                      b["attention_mask"].to(device)], dim=1)
    h = model.llm.model(inputs_embeds=emb, attention_mask=mask).last_hidden_state
    return h, img_emb.size(1)


def answer_loss(model, b, device):
    h, n_img = hidden_states(model, b, device)
    labels = b["labels"].to(device)
    labels = torch.cat([torch.full((labels.size(0), n_img), -100, device=device, dtype=labels.dtype), labels], 1)
    h, y = h[:, :-1], labels[:, 1:]
    sel = y != -100
    logits = model.llm.lm_head(h[sel]).float()
    return torch.nn.functional.cross_entropy(logits, y[sel])


@torch.no_grad()
def evaluate(model, loader, cand_ids, device):
    model.eval()
    hit, tot, ans_count = defaultdict(int), defaultdict(int), defaultdict(lambda: defaultdict(int))
    for b in tqdm(loader, desc="Eval", leave=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h, n_img = hidden_states(model, b, device)
        for i, qt in enumerate(b["qtype"]):
            pos = n_img + int(b["prompt_len"][i]) - 1  # 마지막 프롬프트 토큰 위치 -> 첫 답 토큰 예측
            logits = model.llm.lm_head(h[i, pos])
            pred = int(logits[cand_ids[qt]].argmax())
            gold = int(b["answer"][i])
            hit[qt] += int(pred == gold)
            tot[qt] += 1
            ans_count[qt][gold] += 1
    res = {}
    for qt in tot:
        res[qt] = {
            "acc": 100.0 * hit[qt] / tot[qt],
            "majority": 100.0 * max(ans_count[qt].values()) / tot[qt],
            "n": tot[qt],
        }
    res["overall"] = {"acc": 100.0 * sum(hit.values()) / sum(tot.values()), "n": sum(tot.values())}
    return res


def main(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda")
    out_dir = os.path.join(a.output_dir, a.encoder)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_log.txt")
    log(log_path, f"args: {vars(a)}")

    tok = AutoTokenizer.from_pretrained(a.llm_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(a.llm_id, torch_dtype=torch.bfloat16).to(device)
    l_dim = llm.config.hidden_size

    vit = build_vit(a.encoder)
    projector = PROJECTOR_REGISTRY[a.projector](v_dim=768, l_dim=l_dim)
    model = ProjectorVLM(vit, projector, llm).to(device)
    model.projector.float()
    log(log_path, f"llm hidden={l_dim}, trainable params={model.trainable_params():,}")

    use_depth = a.encoder == "rgbd"
    mk = lambda split, n: SpatialQADataset(a.rgb_root, a.depth_root, split, n, tok, use_depth, seed=a.seed)
    train_ds, test_ds = mk("train", a.n_train), mk("val", a.n_test)
    log(log_path, f"train samples={len(train_ds)}, test samples={len(test_ds)}")
    collate = Collate(tok.pad_token_id)
    train_dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
                          collate_fn=collate, drop_last=True, persistent_workers=a.num_workers > 0)
    test_dl = DataLoader(test_ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
                         collate_fn=collate)

    opt = torch.optim.AdamW(model.projector.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    total = a.epochs * (len(train_dl) // a.accumulation_steps)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    step = 0
    for ep in range(a.epochs):
        model.train(); model.vit.eval(); model.llm.eval()
        run, cnt = 0.0, 0
        pbar = tqdm(train_dl, desc=f"Epoch {ep+1}/{a.epochs}")
        for i, b in enumerate(pbar):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = answer_loss(model, b, device)
            (loss / a.accumulation_steps).backward()
            run += loss.item(); cnt += 1
            if (i + 1) % a.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.projector.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 50 == 0:
                    log(log_path, f"step {step}/{total} | lr {sched.get_last_lr()[0]:.2e} | loss {run/cnt:.4f}")
                    run, cnt = 0.0, 0
        res = evaluate(model, test_dl, test_ds.cand_ids, device)
        log(log_path, f"Epoch {ep+1} eval: " + json.dumps(res, ensure_ascii=False))

    torch.save(model.projector.state_dict(), os.path.join(out_dir, "projector.pth"))
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"encoder": a.encoder, "args": vars(a), "results": res}, f, indent=2, ensure_ascii=False)
    log(log_path, "done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True, choices=list(ENCODERS))
    p.add_argument("--llm_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--projector", default="mlp", choices=list(PROJECTOR_REGISTRY))
    p.add_argument("--rgb_root", default="datasets/imagenet_1k")
    p.add_argument("--depth_root", default="datasets/imagenet_1k_depth")
    p.add_argument("--n_train", type=int, default=20000, help="학습에 쓸 이미지 수 (이미지당 최대 3문항)")
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--accumulation_steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="depth_ab_study/results")
    main(p.parse_args())
