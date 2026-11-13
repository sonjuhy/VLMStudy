"""
CV-Bench(Cambrian-1, NeurIPS 2024) 평가용 데이터 준비.

CV-Bench는 test split만 있으므로, 벤치마크가 만들어진 방식(COCO/Omni3D 어노테이션에서 규칙 기반 생성)을
그대로 따라 학습셋을 만든다. 정답은 전부 사람 어노테이션 / 센서 기반 3D 박스에서 나오며 DA3와 무관하다.

  2D Count    : COCO train2017 인스턴스 수        (CV-Bench는 COCO val2017 / ADE20K val 사용)
  2D Relation : COCO train2017 박스 중심 좌우/상하
  3D Depth    : SUN RGB-D 공식 train split 3D 박스 중심의 카메라 깊이 (CV-Bench는 Omni3D test 사용)
  3D Distance : SUN RGB-D 공식 train split 3D 박스 중심 간 유클리드 거리

출력 (datasets/cvbench_ab/):
  train/images/{task}/{i}.jpg, train/questions.jsonl
  test/images/{2D,3D}/{idx}.{png,jpg}, test/questions.jsonl   <- CV-Bench parquet에서 그대로 추출

실행:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python depth_ab_study/cvbench/build_data.py
"""

import argparse
import io
import itertools
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import pyarrow.parquet as pq
import scipy.io as sio
from PIL import Image, ImageDraw
from tqdm import tqdm

OUT = "datasets/cvbench_ab"
RED, BLUE, GREEN = (255, 0, 0), (0, 0, 255), (0, 255, 0)
MAX_SIDE = 640

# SUN RGB-D 클래스명을 CV-Bench(Omni3D) 표기에 맞춘다. 여기 없는 클래스는 '_'만 공백으로 바꾼다.
SUN_RENAME = {
    "garbage_bin": "bin", "recycle_bin": "bin", "tv": "television", "fridge": "refrigerator",
    "bookshelf": "bookcase", "night_stand": "night stand", "shelf": "shelves", "sofa_chair": "chair",
    "kitchen_counter": "counter", "kitchen_cabinet": "cabinet", "file_cabinet": "cabinet",
    "coffee_table": "table", "endtable": "table", "back_pack": "backpack", "dresser_mirror": "mirror",
}
SUN_EXCLUDE = {"paper", "papers", "cpu"}


def line_width(w, h):
    # CV-Bench 이미지: 1024px 폭에 약 3px 선
    return max(2, round(max(w, h) / 340))


def resized(img):
    s = MAX_SIDE / max(img.size)
    if s < 1:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
        return img, s
    return img, 1.0


def mc_prompt(question, choices, select_line=True):
    opts = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
    return f"{question} Select from the following choices.\n{opts}" if select_line else f"{question}\n{opts}"


# ---------------------------------------------------------------- COCO (2D)
def build_coco(coco_root, n_count, n_rel, rng, out_dir):
    ann = json.load(open(os.path.join(coco_root, "annotations", "instances_train2017.json"), encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    imgs = {i["id"]: i for i in ann["images"]}
    by_img = defaultdict(list)
    for a in ann["annotations"]:
        by_img[a["image_id"]].append(a)
    ids = sorted(by_img)
    rng.shuffle(ids)

    count_q, rel_q = [], []
    for iid in tqdm(ids, desc="COCO"):
        if len(count_q) >= n_count and len(rel_q) >= n_rel:
            break
        info = imgs[iid]
        W, H = info["width"], info["height"]
        per_cat = defaultdict(list)
        for a in by_img[iid]:
            per_cat[a["category_id"]].append(a)
        # 크라우드 박스나 아주 작은 인스턴스가 섞인 클래스는 개수가 모호하므로 제외
        clean = {c: v for c, v in per_cat.items()
                 if not any(a["iscrowd"] for a in v) and all(a["area"] >= 0.002 * W * H for a in v)}
        src = os.path.join(coco_root, "train2017", info["file_name"])

        # 한 이미지는 Count / Relation 중 한 과제에만 쓴다
        want_count = len(count_q) < n_count and (len(rel_q) >= n_rel or rng.random() < 0.5)
        if want_count:
            cand = [(c, len(v)) for c, v in clean.items() if 1 <= len(v) <= 10]
            if not cand:
                continue
            c, ans = rng.choice(cand)
            k = rng.choice([4, 4, 5, 6])
            lo = rng.randint(max(1, ans - (k - 2)), ans)
            choices = [str(x) for x in [0] + list(range(lo, lo + k - 1))]
            rng.shuffle(choices)
            img, _ = resized(Image.open(src).convert("RGB"))
            path = os.path.join(out_dir, "images", "count", f"{len(count_q):06d}.jpg")
            img.save(path, quality=95)
            count_q.append(dict(task="Count", source="COCO", image=path, choices=choices,
                                answer=choices.index(str(ans)),
                                prompt=mc_prompt(f"How many {cats[c]}s are in the image?", choices)))
        elif len(rel_q) < n_rel:
            single = [v[0] for c, v in clean.items() if len(v) == 1 and v[0]["area"] >= 0.01 * W * H]
            pairs = [(a, b) for a, b in itertools.permutations(single, 2)]
            rng.shuffle(pairs)
            for a, b in pairs:
                ax, ay, aw, ah = a["bbox"]
                bx, by, bw, bh = b["bbox"]
                gap_x = max(bx - (ax + aw), ax - (bx + bw))
                gap_y = max(by - (ay + ah), ay - (by + bh))
                axes = [ax_ for ax_, g in (("x", gap_x), ("y", gap_y)) if g > 0]
                if not axes:
                    continue
                axis = rng.choice(axes)
                if axis == "x":
                    choices, ans = ["left", "right"], int(ax + aw / 2 > bx + bw / 2)
                else:
                    choices, ans = ["above", "below"], int(ay + ah / 2 > by + bh / 2)
                img, s = resized(Image.open(src).convert("RGB"))
                ImageDraw.Draw(img).rectangle([ax * s, ay * s, (ax + aw) * s, (ay + ah) * s], outline=RED,
                                              width=line_width(*img.size))
                path = os.path.join(out_dir, "images", "relation", f"{len(rel_q):06d}.jpg")
                img.save(path, quality=95)
                na, nb = cats[a["category_id"]], cats[b["category_id"]]
                q = (f"Considering the relative positions of the {na} (annotated by the red box) and the {nb} "
                     f"in the image provided, where is the {na} (annotated by the red box) located with respect to the {nb}?")
                rel_q.append(dict(task="Relation", source="COCO", image=path, choices=choices, answer=ans,
                                  prompt=mc_prompt(q, choices)))
                break
    return count_q, rel_q


# ---------------------------------------------------------------- SUN RGB-D (3D)
def sun_objects(r, W, H):
    """3D 박스를 이미지에 투영해 (이름, 카메라 깊이, 3D 중심, 2D 박스) 목록을 만든다.
    SUN RGB-D toolbox의 project3dPtsTo2D와 같은 변환: upright 좌표 -> Rtilt^T -> (x, -z, y)."""
    objs = []
    for o in np.atleast_1d(r.groundtruth3DBB):
        if not hasattr(o, "classname") or o.classname in SUN_EXCLUDE:
            continue
        corners = np.array([o.centroid + sum(sg[i] * o.basis[i] * o.coeffs[i] for i in range(3))
                            for sg in itertools.product([-1, 1], repeat=3)])
        p = (r.Rtilt.T @ np.vstack([corners, o.centroid[None]]).T).T
        if (p[:, 1] <= 0.1).any():
            continue
        u = p[:-1, 0] * r.K[0, 0] / p[:-1, 1] + r.K[0, 2]
        v = -p[:-1, 2] * r.K[1, 1] / p[:-1, 1] + r.K[1, 2]
        box = [u.min(), v.min(), u.max(), v.max()]
        full = (box[2] - box[0]) * (box[3] - box[1])
        clip = [max(0, box[0]), max(0, box[1]), min(W - 1, box[2]), min(H - 1, box[3])]
        vis = max(0, clip[2] - clip[0]) * max(0, clip[3] - clip[1])
        # 화면 밖으로 많이 나가거나 너무 작은 객체는 박스가 무엇을 가리키는지 애매하다
        if full <= 0 or vis / full < 0.6 or vis < 0.01 * W * H:
            continue
        name = SUN_RENAME.get(o.classname, o.classname.replace("_", " "))
        objs.append(dict(name=name, z=float(p[-1, 1]), c=o.centroid.astype(float), box=clip))
    return objs


def build_sun(sun_root, per_image, rng, out_dir, exclude_seqs):
    M = sio.loadmat(os.path.join(sun_root, "SUNRGBDMeta3DBB_v2.mat"), squeeze_me=True,
                    struct_as_record=False)["SUNRGBDMeta"]
    split = sio.loadmat(os.path.join(sun_root, "allsplit.mat"), squeeze_me=True)
    train = {s.split("/data/")[1].rstrip("/") for s in split["alltrain"]}
    rows = [r for r in M if r.sequenceName.rstrip("/") in train]
    leaked = {r.sequenceName.rstrip("/") for r in rows} & exclude_seqs
    assert not leaked, f"CV-Bench 테스트 이미지가 학습 split에 있음: {list(leaked)[:3]}"

    name_freq = Counter()
    for r in rows:
        for o in np.atleast_1d(r.groundtruth3DBB):
            if hasattr(o, "classname"):
                name_freq[SUN_RENAME.get(o.classname, o.classname.replace("_", " "))] += 1
    vocab = {n for n, c in name_freq.items() if c >= 100}

    depth_q, dist_q = [], []
    for r in tqdm(rows, desc="SUN RGB-D"):
        src = os.path.join(sun_root, r.rgbpath.split("/data/")[1])
        base = Image.open(src).convert("RGB")
        objs = [o for o in sun_objects(r, *base.size) if o["name"] in vocab]
        img0, s = resized(base)
        lw = line_width(*img0.size)

        def save(task, boxes, idx):
            img = img0.copy()
            d = ImageDraw.Draw(img)
            for box, col in boxes:
                d.rectangle([x * s for x in box], outline=col, width=lw)
            path = os.path.join(out_dir, "images", task.lower(), f"{idx:06d}.jpg")
            img.save(path, quality=95)
            return path

        pairs = [(a, b) for a, b in itertools.combinations(objs, 2)
                 if a["name"] != b["name"] and abs(a["z"] - b["z"]) >= max(0.3, 0.15 * min(a["z"], b["z"]))]
        rng.shuffle(pairs)
        for a, b in pairs[:per_image]:
            if rng.random() < 0.5:
                a, b = b, a
            choices = [a["name"], b["name"]]
            q = (f"Which object is closer to the camera taking this photo, the {a['name']} (highlighted by a red box) "
                 f"or the {b['name']} (highlighted by a blue box)?")
            path = save("Depth", [(a["box"], RED), (b["box"], BLUE)], len(depth_q))
            depth_q.append(dict(task="Depth", source="SUNRGBD", image=path, choices=choices,
                                answer=int(b["z"] < a["z"]), prompt=mc_prompt(q, choices, select_line=False)))

        triples = []
        for a, b, c in itertools.permutations(objs, 3):
            if len({a["name"], b["name"], c["name"]}) < 3 or b["name"] > c["name"]:
                continue
            dab, dac = np.linalg.norm(a["c"] - b["c"]), np.linalg.norm(a["c"] - c["c"])
            if abs(dab - dac) >= max(0.3, 0.2 * min(dab, dac)):
                triples.append((a, b, c, dab, dac))
        rng.shuffle(triples)
        for a, b, c, dab, dac in triples[:per_image]:
            if rng.random() < 0.5:
                b, c, dab, dac = c, b, dac, dab
            choices = [b["name"], c["name"]]
            q = (f"Estimate the real-world distances between objects in this image. Which object is closer to the "
                 f"{a['name']} (highlighted by a red box), the {b['name']} (highlighted by a blue box) or the "
                 f"{c['name']} (highlighted by a green box)?")
            path = save("Distance", [(a["box"], RED), (b["box"], BLUE), (c["box"], GREEN)], len(dist_q))
            dist_q.append(dict(task="Distance", source="SUNRGBD", image=path, choices=choices,
                               answer=int(dac < dab), prompt=mc_prompt(q, choices, select_line=False)))
    return depth_q, dist_q


# ---------------------------------------------------------------- CV-Bench test
def export_cvbench(cvb_root, out_dir):
    rows = []
    for sub in ("2d", "3d"):
        t = pq.read_table(os.path.join(cvb_root, f"test_{sub}.parquet"),
                          columns=["idx", "type", "task", "source", "source_dataset", "source_filename",
                                   "image", "prompt", "choices", "answer"]).to_pylist()
        for r in tqdm(t, desc=f"CV-Bench {sub}"):
            raw = r["image"]["bytes"]
            ext = "png" if raw[:4] == b"\x89PNG" else "jpg"
            path = os.path.join(out_dir, "images", r["type"], f"{r['idx']:06d}.{ext}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(raw)
            rows.append(dict(idx=r["idx"], type=r["type"], task=r["task"], source=r["source"],
                             source_dataset=r["source_dataset"], source_filename=r["source_filename"],
                             image=path, prompt=r["prompt"], choices=list(r["choices"]),
                             answer=ord(r["answer"].strip("()")) - 65))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": i, **r}, ensure_ascii=False) + "\n")


def main(a):
    rng = random.Random(a.seed)
    tr, te = os.path.join(OUT, "train"), os.path.join(OUT, "test")
    for t in ("count", "relation", "depth", "distance"):
        os.makedirs(os.path.join(tr, "images", t), exist_ok=True)

    test = export_cvbench(a.cvbench_root, te)
    write_jsonl(os.path.join(te, "questions.jsonl"), test)
    # SUN RGB-D 테스트 이미지의 sequence 경로 (예: SUNRGBD/kv1/NYUdata/NYU0358)
    exclude = {os.path.dirname(os.path.dirname(r["source_filename"].replace("//", "/")))
               for r in test if r["source_dataset"] == "Omni3D_SUNRGBD"}

    depth_q, dist_q = build_sun(a.sun_root, a.sun_per_image, rng, tr, exclude)
    count_q, rel_q = build_coco(a.coco_root, a.n_count, a.n_relation, rng, tr)
    train = count_q + rel_q + depth_q + dist_q
    write_jsonl(os.path.join(tr, "questions.jsonl"), train)

    for name, rows in (("train", train), ("test", test)):
        c = Counter((r["task"], r["answer"]) for r in rows)
        print(name, len(rows), dict(sorted(c.items())))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--coco_root", default="C:/WorkSpace/DataSets/COCO")
    p.add_argument("--sun_root", default="datasets/sunrgbd")
    p.add_argument("--cvbench_root", default="datasets/cv_bench")
    p.add_argument("--n_count", type=int, default=15000)
    p.add_argument("--n_relation", type=int, default=15000)
    p.add_argument("--sun_per_image", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
