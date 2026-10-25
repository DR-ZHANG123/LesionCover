"""逐类校准分割器输出（DDR 开发区 142 张掩膜图像，概率图坐标，视野内像素）：
对每类以对数几率为单一特征拟合 Platt 逻辑回归（正像素全取、负像素按 1/20 抽样并加权 20），
再在校准概率上取像素 F1 最大的阈值作为该类分量阈值。输出 results/04b_calibration/calibration.json
（--subdir probmaps_adabn --out 04b_calibration_adabn：按整张照片统计量适配后的概率图）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from scripts.lib.segnet import decode_logit  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--subdir", default="probmaps"); ap.add_argument("--out", default="04b_calibration")
    args = ap.parse_args()
    cfg = load_config()
    set_global_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    proc = ROOT / cfg["paths"]["processed"]
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    dev = man[(man.included.astype(int) == 1) & (man.mask_role == "seg_dev")]
    xs, ys = [[] for _ in range(4)], [[] for _ in range(4)]
    for r in dev.itertuples():
        z = np.load(proc / args.subdir / f"{r.image_key}.npz")
        lg, fov = decode_logit(z["logit"]), z["fov"] > 0
        m = np.load(proc / "canon_masks" / f"{r.seg_key}.npy").astype(np.float32)
        mm = F.max_pool2d(torch.from_numpy(m)[None], m.shape[-1] // lg.shape[-1])[0].numpy() > 0
        for c in range(4):
            y, x = mm[c][fov], lg[c][fov]
            keep = y | (rng.random(len(y)) < 0.05)
            xs[c].append(x[keep]); ys[c].append(y[keep])
    out = {}
    for c, name in enumerate(cfg["segmenter"]["classes"]):
        x, y = np.concatenate(xs[c]), np.concatenate(ys[c])
        w = np.where(y, 1.0, 20.0)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(x[:, None], y, sample_weight=w)
        a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        p = 1 / (1 + np.exp(-(a * x + b)))
        grid = np.unique(np.round(np.quantile(p[y], np.linspace(0.01, 0.99, 99)), 4))
        f1 = []
        for t in grid:
            pos = p > t
            tp = w[pos & y].sum(); fp = w[pos & ~y].sum(); fn = w[~pos & y].sum()
            f1.append(2 * tp / max(2 * tp + fp + fn, 1e-9))
        k = int(np.argmax(f1))
        out[name] = {"a": a, "b": b, "tau": float(grid[k]), "pixel_f1": float(f1[k]), "n_pos": int(y.sum())}
        print(name, out[name], flush=True)
    d = ROOT / "results" / args.out; d.mkdir(parents=True, exist_ok=True)
    (d / "calibration.json").write_text(json.dumps(out, indent=2))
    write_provenance(args.out, [ROOT / "results/02_manifest/images.tsv"], [d / "calibration.json"], seed=cfg["seed"])


if __name__ == "__main__":
    main()
