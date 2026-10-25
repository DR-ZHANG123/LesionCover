"""眼底图预处理：提取视野圆盘外接正方形，统一重采样到 canon_size，记录与原图的坐标变换。"""
from __future__ import annotations

import cv2
import numpy as np

cv2.setNumThreads(1)   # 每个数据加载/多进程 worker 只用单线程，避免 CPU 超订


def foreground(img: np.ndarray, base: int) -> np.ndarray:
    """视野前景掩膜（原图分辨率）。在 512 边长的模糊缩略图上，取与背景色（四角块中最暗者的逐通道中位数）
    最大通道差 > base 的像素，保留最大连通域并以凸包填充（视野是圆或截顶圆），再放大回原图。
    可处理黑、灰、白背景与曝光不足的照片。"""
    h, w = img.shape[:2]
    sc = 512 / max(h, w)
    small = cv2.GaussianBlur(cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), (7, 7), 0)
    corners = [small[:8, :8], small[:8, -8:], small[-8:, :8], small[-8:, -8:]]
    bg = min((np.median(c.reshape(-1, 3), axis=0) for c in corners), key=lambda v: v.sum())
    diff = np.abs(small.astype(np.int16) - bg.astype(np.int16)[None, None]).max(axis=2)
    m = (diff > base).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.ones((h, w), np.uint8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pts = cv2.findNonZero((lab == k).astype(np.uint8))
    hull = np.zeros_like(m)
    cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
    return (cv2.resize(hull, (w, h), interpolation=cv2.INTER_LINEAR) > 0).astype(np.uint8)


def fundus_box(fg: np.ndarray) -> tuple[int, int, int]:
    """前景掩膜的外接正方形 (x0, y0, side)，坐标在原图像素系；正方形可越出原图边界（之后补零）。"""
    ys, xs = np.nonzero(fg)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = int(max(x1 - x0, y1 - y0))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return int(round(cx - side / 2)), int(round(cy - side / 2)), side


def crop_square(img: np.ndarray, x0: int, y0: int, side: int) -> np.ndarray:
    """按 (x0, y0, side) 截取正方形，越界处补零。"""
    h, w = img.shape[:2]
    out = np.zeros((side, side) + img.shape[2:], dtype=img.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, w), min(y0 + side, h)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def to_canon(img: np.ndarray, box: tuple[int, int, int], size: int, nearest: bool = False) -> np.ndarray:
    sq = crop_square(img, *box)
    interp = cv2.INTER_NEAREST if nearest else (cv2.INTER_AREA if box[2] > size else cv2.INTER_CUBIC)
    return cv2.resize(sq, (size, size), interpolation=interp)


def canon_to_orig(x: float, y: float, box: tuple[int, int, int], size: int) -> tuple[float, float]:
    """canon 坐标 → 原图坐标。"""
    s = box[2] / size
    return box[0] + x * s, box[1] + y * s
