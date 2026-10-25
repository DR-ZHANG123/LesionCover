"""分级评价与患者整群 bootstrap。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .clsnet import quadratic_kappa


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """y 真实等级，p 预测等级（整数）。"""
    y, p = np.asarray(y, int), np.asarray(p, int)
    g1 = y == 1
    ge2 = y >= 2
    out = {
        "qwk": quadratic_kappa(y, p),
        "macro_f1": float(f1_score(y, p, average="macro", labels=range(5), zero_division=0)),
        "accuracy": float((y == p).mean()),
        "mae": float(np.abs(y - p).mean()),
        "mild_to_none": float((p[g1] == 0).mean()) if g1.any() else np.nan,
        "moderate_plus_under": float((p[ge2] < 2).mean()) if ge2.any() else np.nan,
        "referable_sens": float((p[ge2] >= 2).mean()) if ge2.any() else np.nan,
        "referable_spec": float((p[~ge2] < 2).mean()) if (~ge2).any() else np.nan,
    }
    for g in range(5):
        m = y == g
        out[f"recall_{g}"] = float((p[m] == g).mean()) if m.any() else np.nan
    return out


def eye_level(pred: pd.DataFrame, man: pd.DataFrame) -> pd.DataFrame:
    """同眼照片概率平均 → 眼级预测。返回 patient_id, eye, grade, p0..p4。"""
    d = pred.merge(man[["image_key", "patient_id", "eye", "grade", "quality"]], on="image_key")
    cols = [f"p{i}" for i in range(5)]
    e = d.groupby(["patient_id", "eye"]).agg({**{c: "mean" for c in cols}, "grade": "first", "quality": "min"}).reset_index()
    return e


def cluster_bootstrap(eyes: dict, ref: str, alts: list, n_boot: int, seed: int) -> pd.DataFrame:
    """eyes: {方法: [每个种子的眼级表 (同顺序)]}。按患者重抽样，统计量 = 各种子 QWK 均值之差。"""
    base = eyes[ref][0]
    pids = base.patient_id.to_numpy()
    upid, inv = np.unique(pids, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(upid))]
    y = base.grade.to_numpy()
    preds = {m: [e[[f"p{i}" for i in range(5)]].to_numpy().argmax(1) for e in v] for m, v in eyes.items()}
    rng = np.random.default_rng(seed)
    rows = []
    stats = {m: [] for m in [ref] + alts}
    for _ in range(n_boot):
        pick = rng.integers(len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        for m in stats:
            stats[m].append(np.mean([quadratic_kappa(y[idx], p[idx]) for p in preds[m]]))
    obs = {m: np.mean([quadratic_kappa(y, p) for p in preds[m]]) for m in stats}
    for a in alts:
        d = np.asarray(stats[ref]) - np.asarray(stats[a])
        p_two = 2 * min((d <= 0).mean(), (d >= 0).mean())
        rows.append({"contrast": f"{ref}-{a}", "diff": obs[ref] - obs[a], "lo": np.percentile(d, 2.5),
                     "hi": np.percentile(d, 97.5), "p_boot": min(1.0, max(p_two, 1 / n_boot))})
    return pd.DataFrame(rows)


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj
