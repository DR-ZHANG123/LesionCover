"""等级解码：取概率最大的等级（argmax），或以期望等级 E=Σ k·p_k 配合四个阈值分级；阈值只在 DDR 开发区上
以二次加权 κ 最大为目标优化（坐标搜索），然后原样用于所有测试队列。"""
from __future__ import annotations

import numpy as np

from .clsnet import quadratic_kappa


def expected_grade(P: np.ndarray) -> np.ndarray:
    return P @ np.arange(P.shape[1])


def apply_thresholds(e: np.ndarray, th: np.ndarray) -> np.ndarray:
    return np.digitize(e, th)


def fit_thresholds(e: np.ndarray, y: np.ndarray, n_iter: int = 4) -> np.ndarray:
    """坐标搜索：从 0.5/1.5/2.5/3.5 开始，逐个阈值在网格上取使 κ 最大的值，保持单调。"""
    th = np.array([0.5, 1.5, 2.5, 3.5])
    for _ in range(n_iter):
        for i in range(4):
            lo = th[i - 1] + 0.01 if i else 0.0
            hi = th[i + 1] - 0.01 if i < 3 else 4.0
            grid = np.linspace(lo, hi, 61)
            scores = []
            for g in grid:
                t = th.copy(); t[i] = g
                scores.append(quadratic_kappa(y, apply_thresholds(e, t)))
            th[i] = grid[int(np.argmax(scores))]
    return th
