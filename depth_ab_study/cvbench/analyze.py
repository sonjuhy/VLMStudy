"""
CV-Bench A/B 결과(seed별 predictions.jsonl)를 모아 RGB vs RGB-D의 유의성을 검정한다.

- seed 간 Welch t-test (학습 오차)
- seed + 문항을 동시에 재표집하는 bootstrap 95% CI (학습 오차 + 테스트셋 표본 오차)

실행:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python depth_ab_study/cvbench/analyze.py
결과: depth_ab_study/results_cvbench/summary.json
"""

import glob
import json
import os

import numpy as np
from scipy import stats

RES = "depth_ab_study/results_cvbench"
TEST = "datasets/cvbench_ab/test/questions.jsonl"
B = 10000


def load_correct(enc):
    seeds = sorted(int(p.split("seed")[-1]) for p in glob.glob(os.path.join(RES, enc, "seed*"))
                   if os.path.exists(os.path.join(p, "results.json")))
    out = []
    for s in seeds:
        with open(os.path.join(RES, enc, f"seed{s}", "predictions.jsonl"), encoding="utf-8") as f:
            out.append([json.loads(l) for l in f])
    return seeds, np.array([[p["pred"] == p["answer"] for p in P] for P in out], float)


def main():
    with open(TEST, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    seeds, ca = load_correct("rgb")
    seeds_b, cb = load_correct("rgbd")
    assert seeds == seeds_b
    grp = np.array([f"{r['type']}/{r['source']}" for r in rows])
    task = np.array([r["task"] for r in rows])
    G = {g: np.where(grp == g)[0] for g in ("2D/ADE20K", "2D/COCO", "3D/Omni3D")}
    rng = np.random.default_rng(0)

    def official(c, idx=G):
        a = lambda g: c[..., idx[g]].mean(-1) * 100
        return ((a("2D/ADE20K") + a("2D/COCO")) / 2 + a("3D/Omni3D")) / 2

    def compare(name, metric, resample):
        a, b = metric(ca), metric(cb)
        n = len(seeds)
        boot = []
        for _ in range(B):
            ia, ib, idx = rng.integers(0, n, n), rng.integers(0, n, n), resample()
            boot.append(metric(cb[ib].mean(0), idx) - metric(ca[ia].mean(0), idx))
        boot = np.array(boot)
        r = {"rgb_mean": a.mean(), "rgb_std": a.std(ddof=1), "rgbd_mean": b.mean(), "rgbd_std": b.std(ddof=1),
             "delta": b.mean() - a.mean(), "welch_p": stats.ttest_ind(b, a, equal_var=False).pvalue,
             "ci95": [np.percentile(boot, 2.5), np.percentile(boot, 97.5)], "p_delta_le_0": float(np.mean(boot <= 0))}
        print(f"{name:12s} rgb {r['rgb_mean']:6.2f}±{r['rgb_std']:4.2f}  rgbd {r['rgbd_mean']:6.2f}±{r['rgbd_std']:4.2f}"
              f"  Δ {r['delta']:+5.2f}  Welch p={r['welch_p']:.1e}  CI [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]")
        return r

    def subset(ii):
        return (lambda c, j=ii: c[..., j].mean(-1) * 100), (lambda: rng.choice(ii, len(ii)))

    summary = {"seeds": seeds, "n_test": len(rows),
               "per_seed": {"rgb": official(ca).tolist(), "rgbd": official(cb).tolist()}}
    summary["cv_bench"] = compare("cv_bench", official, lambda: {g: rng.choice(v, len(v)) for g, v in G.items()})
    items = [("2D", np.where(grp != "3D/Omni3D")[0]), ("3D", G["3D/Omni3D"])]
    items += [(t, np.where(task == t)[0]) for t in ("Count", "Relation", "Depth", "Distance")]
    items += [(s, np.array([i for i, r in enumerate(rows) if r.get("source_dataset") == s]))
              for s in ("Omni3D_Hypersim", "Omni3D_SUNRGBD", "Omni3D_nuScenes")]
    for name, ii in items:
        summary[name] = compare(name, *subset(ii))

    with open(os.path.join(RES, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)


if __name__ == "__main__":
    main()
