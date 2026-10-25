"""两层 bootstrap 的公共部件：集成（三个模型概率平均）统计量，区间同时重抽评价单位与每个训练设置内的模型。

每个设置的三个模型有放回地抽三次，只有 10 种组合（可重复组合），预先算好每种组合的集成预测等级；
评价单位的重抽样用权重（每个单位被抽中的次数）表示，加权混淆矩阵即得 κ 与比例，避免逐次拼接索引。"""
from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np

COMBOS = list(combinations_with_replacement(range(3), 3))
COMBO_INDEX = {c: i for i, c in enumerate(COMBOS)}
W5 = (np.arange(5)[:, None] - np.arange(5)[None]) ** 2 / 16.0


def combo_labels(probs: list[np.ndarray]) -> np.ndarray:
    """probs：三个模型的概率（单位 × 5）。返回 10 × 单位 的集成预测等级（组合内概率平均后取最大）。"""
    assert len(probs) == 3
    st = np.stack(probs)
    return np.stack([st[list(c)].mean(0).argmax(1) for c in COMBOS]).astype(np.int8)


def full_labels(probs: list[np.ndarray]) -> np.ndarray:
    return np.mean(probs, 0).argmax(1)


def wkappa(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    """加权的二次加权 κ（w 为每行权重；None 时与 clsnet.quadratic_kappa 相同）。"""
    O = np.bincount(y * 5 + p, weights=w, minlength=25).reshape(5, 5)
    n = O.sum()
    E = np.outer(O.sum(1), O.sum(0)) / max(n, 1e-12)
    den = (W5 * E).sum()
    return float(1 - (W5 * O).sum() / den) if den > 0 else 0.0


def wshare(mask: np.ndarray, hit: np.ndarray, w: np.ndarray | None = None) -> float:
    """在 mask 行中 hit 为真的（加权）比例。"""
    ww = np.ones(len(mask)) if w is None else w
    den = ww[mask].sum()
    return float(ww[mask & hit].sum() / den) if den > 0 else np.nan


class Resampler:
    """按单位重抽样的权重发生器。units：每行所属单位（患者或照片）。"""

    def __init__(self, units: np.ndarray, rng: np.random.Generator):
        self.uu, self.inv = np.unique(units, return_inverse=True)
        self.rng = rng

    def weights(self) -> np.ndarray:
        cnt = np.bincount(self.rng.integers(len(self.uu), size=len(self.uu)), minlength=len(self.uu))
        return cnt[self.inv].astype(float)

    def combo(self) -> int:
        return COMBO_INDEX[tuple(sorted(self.rng.integers(3, size=3)))]


def summarize(obs: float, draws: list[float]) -> tuple[float, float, float, float]:
    """（观测值, 2.5%, 97.5%, 双侧 bootstrap P）；P 为两侧尾部比例较小者的两倍，下限 1/B。"""
    d = np.asarray(draws, float)
    d = d[~np.isnan(d)]
    if not len(d):
        return obs, np.nan, np.nan, np.nan
    p = min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / len(d)))
    return obs, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), p
