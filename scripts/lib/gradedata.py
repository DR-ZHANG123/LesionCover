"""分级数据：读取 05_selection 的裁块坐标，构建 DataLoader。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .clsnet import GradeSet
from .repro import ROOT


def load_crops(method: str, k: int, df: pd.DataFrame, path=None, var: str = "") -> np.ndarray | None:
    """var 含 neg：读健康照片阴性监督分割器所选的裁块（05c_neg_crops）。"""
    if method.startswith("G"):
        return None
    tag = f"{method}_k{k}"
    if path is None:
        path = ROOT / ("results/05c_neg_crops/crops.tsv.gz" if "neg" in var else "results/05_selection/crops.tsv.gz")
    sel = pd.read_csv(path, sep="\t")
    sel = sel[sel.method == tag].sort_values(["image_key", "slot"])
    arr = sel[["x0", "y0"]].to_numpy().reshape(-1, k, 2)
    keys = sel.image_key.to_numpy().reshape(-1, k)[:, 0]
    lut = dict(zip(keys, range(len(keys))))
    return arr[[lut[x] for x in df.image_key]]


def hard_negative_weights(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """0 级照片中分割器候选病灶分量数位于本数据集 0 级照片上四分位以上者（最易被裁块误读为病灶的健康照片）
    抽样权重乘以 hn_factor，其余为 1。"""
    n = df[["image_key"]].merge(pd.read_csv(ROOT / "results/05_selection/pred_components.tsv", sep="\t"),
                                how="left").n_pred_components.fillna(0).to_numpy()
    cc = cfg["classifier"]
    w = np.ones(len(df))
    for role, idx in df.groupby("role").indices.items():
        z = idx[df.grade.to_numpy()[idx] == 0]
        hard = z[n[z] >= np.quantile(n[z], cc["hn_quantile"])]
        w[hard] = cc["hn_factor"]
    return w


def make_loader(df, crops, cfg, gsize, train, seed, bs, artifact_aug=False, weights=None):
    """weights：训练时按该权重有放回抽样（每轮样本数不变），缺省为均匀打乱。"""
    ds = GradeSet(df.canon_path, df.grade, crops, cfg, gsize, train, seed, artifact_aug)
    g = torch.Generator().manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(df), generator=g) if weights is not None else None
    return ds, torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=train and sampler is None, sampler=sampler,
                                           num_workers=cfg["classifier"]["num_workers"],
                                           pin_memory=True, drop_last=train, persistent_workers=True,
                                           generator=g)
