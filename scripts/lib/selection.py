"""裁块选择：共享候选库 + 七种选择规则。所有坐标在概率图（prob_map_size）坐标系，裁块边长 c。

规则：U 固定均匀、R 随机、T 热图像素得分 top-K、N 分量得分 + NMS、D 分量得分 + 质量—多样性 DPP、
C 分量覆盖（贪心最大化 F(S)=Σ_l q_l max_{j∈S} a_lj）。另有消融：C 不做分量归一化（Cmass）、
按分量得分直接 top-K（S）。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

cv2.setNumThreads(1)   # 每个数据加载/多进程 worker 只用单线程，避免 CPU 超订


@dataclass
class Components:
    q: np.ndarray            # (L,) 分量平均预测概率
    mass: np.ndarray         # (L,) 分量内概率总和
    cls: np.ndarray          # (L,) 类别索引
    centre: np.ndarray       # (L, 2) 质心 (x, y)
    px: list                 # 每个分量的像素 x 坐标
    py: list
    pw: list                 # 每个像素的概率


def extract_components(prob: np.ndarray, tau, min_px: int) -> Components:
    """prob: (C, H, W) 校准概率；tau: 每类阈值（标量则各类相同）。每类单独阈值化并取 8 连通分量。"""
    tau = np.broadcast_to(np.asarray(tau, np.float32), (prob.shape[0],))
    q, mass, cls, cen, pxs, pys, pws = [], [], [], [], [], [], []
    for k in range(prob.shape[0]):
        b = (prob[k] > tau[k]).astype(np.uint8)
        n, lab = cv2.connectedComponents(b, connectivity=8)
        if n <= 1:
            continue
        ys, xs = np.nonzero(lab)
        ids = lab[ys, xs]
        order = np.argsort(ids, kind="stable")
        ys, xs, ids = ys[order], xs[order], ids[order]
        cuts = np.flatnonzero(np.diff(ids)) + 1
        for yy, xx in zip(np.split(ys, cuts), np.split(xs, cuts)):
            if len(xx) < min_px:
                continue
            w = prob[k, yy, xx]
            q.append(float(w.mean())); mass.append(float(w.sum())); cls.append(k)
            cen.append((float(xx.mean()), float(yy.mean())))
            pxs.append(xx.astype(np.int32)); pys.append(yy.astype(np.int32)); pws.append(w.astype(np.float32))
    return Components(np.asarray(q, np.float32), np.asarray(mass, np.float32), np.asarray(cls, np.int16),
                      np.asarray(cen, np.float32).reshape(-1, 2), pxs, pys, pws)


def uniform_layout(size: int, c: int, k: int) -> np.ndarray:
    """固定均匀布局（左上角坐标）：先是围绕图像中心的 2×2 象限，再接外圈 8 个位置；取前 K 个。"""
    ctr = size / 2
    off = c / 2 + size * 0.01
    pts = [(ctr - off, ctr - off), (ctr + off, ctr - off), (ctr - off, ctr + off), (ctr + off, ctr + off)]
    ring = size * 0.3
    for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        pts.append((ctr + ring * np.cos(ang), ctr + ring * np.sin(ang)))
    tl = np.clip(np.asarray(pts) - c / 2, 0, size - c)
    return np.round(tl).astype(np.int32)


def build_candidates(fmask: np.ndarray, comps: Components, c: int, stride: int,
                     min_frac: float, max_centres: int) -> np.ndarray:
    """候选左上角坐标 (M, 2)：固定网格 + 分量中心（按 q 取前 max_centres 个），只保留视野比例足够者。"""
    size = fmask.shape[0]
    g = np.arange(0, size - c + 1, stride)
    if g[-1] != size - c:
        g = np.append(g, size - c)
    grid = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    if len(comps.q):
        top = np.argsort(-comps.q, kind="stable")[:max_centres]
        cc = np.clip(np.round(comps.centre[top] - c / 2), 0, size - c).astype(np.int64)
        cand = np.concatenate([grid, cc])
    else:
        cand = grid
    cand = np.unique(cand, axis=0)
    ii = cv2.integral(fmask.astype(np.uint8))
    x, y = cand[:, 0], cand[:, 1]
    frac = (ii[y + c, x + c] - ii[y, x + c] - ii[y + c, x] + ii[y, x]) / float(c * c)
    keep = cand[frac >= min_frac]
    return keep if len(keep) else grid[:1]


def coverage_matrix(comps: Components, cand: np.ndarray, c: int, normalise: bool = True) -> np.ndarray:
    """A[l, j]：分量 l 落入候选 j 的概率质量（normalise=True 时除以分量总质量）。"""
    A = np.zeros((len(comps.q), len(cand)), np.float32)
    X0, Y0 = cand[:, 0][None], cand[:, 1][None]
    for l, (xx, yy, ww) in enumerate(zip(comps.px, comps.py, comps.pw)):
        inside = (xx[:, None] >= X0) & (xx[:, None] < X0 + c) & (yy[:, None] >= Y0) & (yy[:, None] < Y0 + c)
        A[l] = ww @ inside
        if normalise:
            A[l] /= comps.mass[l]
    return A


def iou(a: np.ndarray, b: np.ndarray, c: int) -> np.ndarray:
    """左上角 a (2,) 与 b (M,2) 的等大正方形 IoU。"""
    ix = np.clip(c - np.abs(b[:, 0] - a[0]), 0, None)
    iy = np.clip(c - np.abs(b[:, 1] - a[1]), 0, None)
    inter = ix * iy
    return inter / (2 * c * c - inter)


def _fill(sel: list, cand_tl: np.ndarray, fallback: np.ndarray, k: int, c: int) -> np.ndarray:
    """不足 K 个时按固定均匀布局补齐（跳过与已选 IoU>0.5 者），所有规则共用。"""
    out = [tuple(cand_tl[j]) for j in sel]
    for f in fallback:
        if len(out) >= k:
            break
        if not out or iou(f, np.asarray(out), c).max() <= 0.5:
            out.append(tuple(f))
    for f in fallback:
        if len(out) >= k:
            break
        out.append(tuple(f))
    return np.asarray(out[:k], np.int32)


def select_topk(score: np.ndarray, k: int) -> list:
    pos = np.flatnonzero(score > 0)
    return list(pos[np.argsort(-score[pos], kind="stable")][:k])


def select_nms(score: np.ndarray, cand: np.ndarray, k: int, c: int, thr: float) -> list:
    sel = []
    for j in np.argsort(-score, kind="stable"):
        if score[j] <= 0 or len(sel) >= k:
            break
        if not sel or iou(cand[j], cand[sel], c).max() <= thr:
            sel.append(int(j))
    return sel


def select_dpp(score: np.ndarray, cand: np.ndarray, k: int, theta: float, sigma: float) -> list:
    """质量—多样性 DPP 贪心 MAP（Chen et al. 2018）：L = diag(r) S diag(r)，r = exp(α ŝ)，α = θ/(2(1-θ))。"""
    pos = np.flatnonzero(score > 0)
    if len(pos) == 0:
        return []
    s = score[pos] / score[pos].max()
    alpha = theta / (2 * (1 - theta))
    r = np.exp(alpha * s)
    ctr = cand[pos].astype(np.float64)
    d2 = ((ctr[:, None] - ctr[None]) ** 2).sum(-1)
    L = r[:, None] * np.exp(-d2 / (2 * sigma ** 2)) * r[None]
    n = len(pos)
    cis = np.zeros((k, n))
    di2 = np.diag(L).copy()
    sel = []
    for t in range(min(k, n)):
        j = int(np.argmax(di2))
        if di2[j] < 1e-10:
            break
        sel.append(j)
        e = (L[j] - cis[:t].T @ cis[:t, j]) / np.sqrt(di2[j])
        cis[t] = e
        di2 = di2 - e ** 2
        di2[sel] = -np.inf
    return [int(pos[j]) for j in sel]


def select_coverage(A: np.ndarray, q: np.ndarray, k: int, rho: float = 0.0) -> list:
    """贪心最大化 F(S)=Σ_l q_l max_{j∈S} g(A[l,j])，g(a)=a·1[a≥ρ]：只有窗口包含病灶质量的至少 ρ 时才计分。
    g 非负，F 仍是单调子模的设施选址函数，贪心保证 1-1/e。边际收益为 0 时停止（由 _fill 补齐）。"""
    if A.size == 0:
        return []
    A = A * (A >= rho)
    best = np.zeros(A.shape[0], np.float32)
    sel = []
    for _ in range(k):
        gain = (q[:, None] * np.clip(A - best[:, None], 0, None)).sum(0)
        if sel:
            gain[sel] = -1
        j = int(np.argmax(gain))
        if gain[j] <= 1e-6:
            break
        sel.append(j)
        best = np.maximum(best, A[:, j])
    return sel


def heat_scores(prob: np.ndarray, cand: np.ndarray, c: int) -> np.ndarray:
    h = prob.max(0).astype(np.float64)
    ii = cv2.integral(h)
    x, y = cand[:, 0], cand[:, 1]
    return ii[y + c, x + c] - ii[y, x + c] - ii[y + c, x] + ii[y, x]


def shift_threshold(tau: np.ndarray, delta: float) -> np.ndarray:
    """阈值在对数几率尺度上平移 delta（用于阈值扰动分析）。"""
    t = np.clip(np.asarray(tau, np.float64), 1e-6, 1 - 1e-6)
    return (1 / (1 + np.exp(-(np.log(t / (1 - t)) + delta)))).astype(np.float32)


def select_all(prob: np.ndarray, fmask: np.ndarray, cfg: dict, k: int, rng: np.random.Generator,
               nms_thr: float, dpp_theta: float, tau, rho: float | None = None,
               methods=("U", "R", "T", "N", "D", "C", "Cmass", "S")) -> dict:
    """对一张图返回 {方法: (k,2) 左上角坐标（prob 坐标系）}。"""
    sc = cfg["selection"]
    scale = cfg["preprocess"]["canon_size"] / prob.shape[-1]
    c = int(round(sc["crop"] / scale))
    stride = int(round(sc["grid_stride"] / scale))
    comps = extract_components(prob, tau, sc["comp_min_pixels"])
    cand = build_candidates(fmask, comps, c, stride, sc["min_fundus_frac"], sc["max_comp_centres"])
    fb = uniform_layout(prob.shape[-1], c, max(k, 4))
    A = coverage_matrix(comps, cand, c, True)
    s = (comps.q[:, None] * A).sum(0) if len(comps.q) else np.zeros(len(cand), np.float32)
    out = {}
    for m in methods:
        if m == "U":
            out[m] = fb[:k].astype(np.int32)
            continue
        if m == "R":
            sel = list(rng.choice(len(cand), size=min(k, len(cand)), replace=False))
        elif m == "T":
            sel = select_topk(heat_scores(prob, cand, c), k)
        elif m == "N":
            sel = select_nms(s, cand, k, c, nms_thr)
        elif m == "D":
            sel = select_dpp(s, cand, k, dpp_theta, sc["dpp_sigma_frac"] * prob.shape[-1])
        elif m == "C":
            sel = select_coverage(A, comps.q, k, sc.get("coverage_rho", 0.0) if rho is None else rho)
        elif m == "Cmass":
            Am = coverage_matrix(comps, cand, c, False)
            sel = select_coverage(Am, np.ones_like(comps.q), k)
        elif m == "S":
            sel = select_topk(s, k)
        out[m] = _fill(sel, cand, fb, k, c)
    out["_n_comp"] = len(comps.q)
    out["_n_cand"] = len(cand)
    return out
