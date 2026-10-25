"""分割器训练/评价共用：读取 canon 图与 canon 掩膜，计算视野内逐类像素 AUPR。"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from .repro import ROOT
from .segnet import predict_full


def load_set(man: pd.DataFrame, cfg: dict):
    proc = ROOT / cfg["paths"]["processed"]
    imgs, masks, fms = [], [], []
    for k in man.seg_key:
        imgs.append(cv2.imread(str(proc / "canon" / f"{k}.jpg"))[:, :, ::-1].copy())
        masks.append(np.load(proc / "canon_masks" / f"{k}.npy"))
        fms.append(cv2.imread(str(proc / "fov" / f"{k}.png"), 0))
    return imgs, masks, fms


def evaluate(model, imgs, masks, fms, device, size: int) -> dict:
    """在 size×size（最大池化）上计算每类视野内像素 AUPR。"""
    model.eval()
    ys, ps = [[] for _ in range(4)], [[] for _ in range(4)]
    for img, m, fm in zip(imgs, masks, fms):
        p = predict_full(model, img, device, flip_tta=False)
        f = p.shape[-1] // size
        pp = F.max_pool2d(torch.from_numpy(p)[None], f)[0].numpy()
        mm = F.max_pool2d(torch.from_numpy(m.astype(np.float32))[None], f)[0].numpy()
        fv = cv2.resize(fm, (size, size), interpolation=cv2.INTER_NEAREST) > 0
        for c in range(4):
            ys[c].append(mm[c][fv]); ps[c].append(pp[c][fv])
    out = {}
    rng = np.random.default_rng(0)
    for c, name in enumerate(["MA", "HE", "EX", "SE"]):
        y, p = np.concatenate(ys[c]), np.concatenate(ps[c])
        # 负像素按 1/20 抽样、权重 20，正像素全保留：AP 的无偏近似，计算量降一个数量级
        keep = (y > 0) | (rng.random(len(y)) < 0.05)
        w = np.where(y[keep] > 0, 1.0, 20.0)
        out[name] = float(average_precision_score(y[keep], p[keep], sample_weight=w)) if y.any() else float("nan")
    out["mean"] = float(np.nanmean([out[n] for n in ["MA", "HE", "EX", "SE"]]))
    return out
