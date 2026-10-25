"""参考病灶覆盖：在原始分辨率的参考掩膜上提取连通分量，把 canon 坐标的裁块映射回原图后计算保留比例。"""
from __future__ import annotations

import cv2
import numpy as np

cv2.setNumThreads(1)   # 每个数据加载/多进程 worker 只用单线程，避免 CPU 超订

CLASSES = ["MA", "HE", "EX", "SE"]


def read_mask(path) -> np.ndarray | None:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m.max(axis=2)
    return (m > 0).astype(np.uint8)


def reference_components(mask_paths: dict, shape: tuple) -> list:
    """返回分量列表：(类别索引, 像素 x 数组, 像素 y 数组)。mask_paths: {类别: 路径或 None}。"""
    comps = []
    for ci, name in enumerate(CLASSES):
        p = mask_paths.get(name)
        if p is None:
            continue
        m = read_mask(p)
        if m is None or not m.any():
            continue
        n, lab = cv2.connectedComponents(m, connectivity=8)
        ys, xs = np.nonzero(lab)
        ids = lab[ys, xs]
        order = np.argsort(ids, kind="stable")
        ys, xs, ids = ys[order], xs[order], ids[order]
        cuts = np.flatnonzero(np.diff(ids)) + 1
        for yy, xx in zip(np.split(ys, cuts), np.split(xs, cuts)):
            comps.append((ci, xx.astype(np.int32), yy.astype(np.int32)))
    return comps


def crops_to_orig(crops_canon: np.ndarray, box: tuple, canon: int, c: int) -> np.ndarray:
    """canon 左上角 (K,2) → 原图矩形 (K,4)：x0, y0, x1, y1。"""
    s = box[2] / canon
    x0 = box[0] + crops_canon[:, 0] * s
    y0 = box[1] + crops_canon[:, 1] * s
    return np.stack([x0, y0, x0 + c * s, y0 + c * s], 1)


def retained_fraction(comps: list, rects: np.ndarray) -> np.ndarray:
    """F[l, j]：分量 l 落在矩形 j 内的像素比例。"""
    F = np.zeros((len(comps), len(rects)), np.float32)
    for l, (_, xx, yy) in enumerate(comps):
        inside = ((xx[:, None] >= rects[None, :, 0]) & (xx[:, None] < rects[None, :, 2]) &
                  (yy[:, None] >= rects[None, :, 1]) & (yy[:, None] < rects[None, :, 3]))
        F[l] = inside.mean(0)
    return F


def crop_has_lesion(comps: list, rects: np.ndarray) -> np.ndarray:
    """每个裁块内是否含任一参考病灶像素。"""
    has = np.zeros(len(rects), bool)
    for _, xx, yy in comps:
        inside = ((xx[:, None] >= rects[None, :, 0]) & (xx[:, None] < rects[None, :, 2]) &
                  (yy[:, None] >= rects[None, :, 1]) & (yy[:, None] < rects[None, :, 3]))
        has |= inside.any(0)
    return has


def overlap_fraction(rects: np.ndarray) -> float:
    """所选裁块并集面积相对 K 个裁块面积之和的冗余比例：1 - |∪|/Σ|r|（栅格近似）。"""
    x0, y0 = rects[:, 0].min(), rects[:, 1].min()
    x1, y1 = rects[:, 2].max(), rects[:, 3].max()
    step = max((x1 - x0), (y1 - y0)) / 512
    gx = np.arange(x0, x1, step) + step / 2
    gy = np.arange(y0, y1, step) + step / 2
    X, Y = np.meshgrid(gx, gy)
    cnt = np.zeros_like(X)
    for r in rects:
        cnt += (X >= r[0]) & (X < r[2]) & (Y >= r[1]) & (Y < r[3])
    tot = (cnt > 0).sum()
    return float(1 - tot / cnt.sum()) if cnt.sum() else 0.0


def ref_components_for(row: dict, cfg: dict) -> list:
    """清单中一张掩膜图像的参考分量（读官方 lesion_segmentation 原始分辨率 .tif）。"""
    from .repro import ROOT
    paths = {c: (ROOT / row[f"mask_{c}"]) if isinstance(row.get(f"mask_{c}", ""), str) and row.get(f"mask_{c}", "") else None
             for c in CLASSES}
    return reference_components(paths, (row["orig_h"], row["orig_w"]))
