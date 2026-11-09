"""病灶 token：把校准概率图中的每个预测病灶分量变成一条记录（类别、得分、面积、位置），供 LesionCover 网络使用。

分量数超过 token 上限时按覆盖优先选取：先在每个（类别 × 空间格）中各取得分最高的一个分量（不同位置、不同类别的病灶
各计一次，与 LesionCover 选块的覆盖目标同一原则），再按得分补满。被截断的分量仍计入逐类逐象限计数。
所有坐标在概率图（prob_map_size）坐标系。"""
from __future__ import annotations

import numpy as np

from .selection import extract_components

FIELDS = ("cls", "q", "log_area", "x", "y", "w", "h")


def coverage_first(cls: np.ndarray, q: np.ndarray, cen: np.ndarray, size: int, cell: int, m: int) -> np.ndarray:
    """返回选中分量的下标（最多 m 个）。"""
    n = len(q)
    if n <= m:
        return np.argsort(-q, kind="stable")
    order = np.argsort(-q, kind="stable")
    g = (np.clip(cen[:, 1], 0, size - 1) // cell).astype(int) * (size // cell + 1) + \
        (np.clip(cen[:, 0], 0, size - 1) // cell).astype(int)
    key = cls.astype(int) * 10 ** 6 + g
    seen, first, rest = set(), [], []
    for i in order:
        (rest if key[i] in seen else first).append(i)
        seen.add(key[i])
    return np.asarray((first + rest)[:m])


def quadrant_counts(cls: np.ndarray, cen: np.ndarray, size: int, n_cls: int) -> np.ndarray:
    """逐类逐象限（以视野方框中心划分）的分量数，形状 (n_cls*4,)。"""
    out = np.zeros((n_cls, 4), np.float32)
    if len(cls):
        qd = (cen[:, 0] >= size / 2).astype(int) + 2 * (cen[:, 1] >= size / 2).astype(int)
        np.add.at(out, (cls.astype(int), qd), 1)
    return out.ravel()


def image_tokens(prob: np.ndarray, tau: np.ndarray, min_px: int, m: int, cell: int,
                 drop: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """prob: (C,H,W) 校准概率。drop：可选的 (H,W) 布尔图，分量质心落在其中者剔除（相机斑点）。
    返回 (tokens (≤m, len(FIELDS)) float32, counts (C*4,) float32)。"""
    comps = extract_components(prob, tau, min_px)
    size = prob.shape[-1]
    cls, q, cen = comps.cls, comps.q, comps.centre
    keep = np.ones(len(q), bool)
    if drop is not None and len(q):
        keep = ~drop[np.clip(cen[:, 1].astype(int), 0, size - 1), np.clip(cen[:, 0].astype(int), 0, size - 1)]
    idx_all = np.flatnonzero(keep)
    counts = quadrant_counts(cls[idx_all], cen[idx_all], size, prob.shape[0])
    if len(idx_all) == 0:
        return np.zeros((0, len(FIELDS)), np.float32), counts
    sel = idx_all[coverage_first(cls[idx_all], q[idx_all], cen[idx_all], size, cell, m)]
    rows = []
    for i in sel:
        xs, ys = comps.px[i], comps.py[i]
        rows.append((cls[i], q[i], np.log(len(xs)), cen[i, 0], cen[i, 1],
                     xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    return np.asarray(rows, np.float32), counts
