"""Fig. 1 的两个图形部件：
- draw_network：分级网络结构示意（特征向量、门控注意力、注意力权重、加权求和、拼接、全连接与五级概率、训练用裁块头），
  数值取自示例照片三个病灶覆盖模型的平均输出（results/07_classify），不重新计算；
- cohort_thumbnails：每个外部队列一张代表性眼底照片（重抽样后的 canon 图，按固定规则选取）。
所有坐标为毫米（Fig. 1 的数据坐标）。"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

NAVY, TEAL, RED, SAND, PURPLE, GREY = "#2B4C7E", "#2A9D8F", "#B2182B", "#E9C46A", "#6A4C93", "#5A5A5A"
GRADE_COLORS = ["#4C9F70", "#C9B33A", "#E08A2C", "#C8553D", "#7B2D26"]


def _arrow(ax, p0, p1, lw=0.7, ls="-", color="#444444"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=5.5, lw=lw, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=3))


def _vector(ax, x, y, n, cell, w, colors, edge="white"):
    """竖直特征向量：n 个小格，自下而上。"""
    for i in range(n):
        ax.add_patch(Rectangle((x, y + i * cell), w, cell, fc=colors[i % len(colors)], ec=edge, lw=0.35, zorder=4))


def example_outputs(root, key: str) -> dict:
    """示例照片在三个病灶覆盖模型下的平均五级概率、裁块头概率与注意力权重。"""
    p, a, c = [], [], []
    for sd in (11, 22, 33):
        d = root / "results/07_classify" / f"C_k4_s{sd}_aux"
        r = pd.read_csv(d / "predictions.tsv.gz", sep="\t").set_index("image_key").loc[key]
        p.append(r[[f"p{i}" for i in range(5)]].astype(float).to_numpy())
        a.append(r[[f"attn{i}" for i in range(4)]].astype(float).to_numpy())
        rc = pd.read_csv(d / "predictions_croponly.tsv.gz", sep="\t").set_index("image_key").loc[key]
        c.append(rc[[f"p{i}" for i in range(5)]].astype(float).to_numpy())
    return {"p": np.mean(p, 0), "attn": np.mean(a, 0), "p_crop": np.mean(c, 0)}


def draw_network(ax, yb: float, out: dict, aux_w: float, x0: float = 47.0) -> None:
    """从编码器输出右侧（x0）画到五级概率（约 x0+76）。"""
    # 全局特征 g 与四个裁块特征 h_j
    gx, cell = x0 + 3.0, 1.9
    _vector(ax, gx, yb + 30, 6, cell, 2.6, [NAVY, "#4A6FA5", "#6C8EBF"])
    ax.text(gx + 1.3, yb + 42.2, r"$\mathbf{g}$", ha="center", va="bottom", fontsize=7)
    hx = [gx + k * 3.4 for k in range(4)]
    for x in hx:
        _vector(ax, x, yb + 7, 5, cell, 2.6, ["#F3C6C6", "#E79A9A", "#D96C6C"], edge=RED)
    ax.text(hx[0] + 6.4, yb + 17.2, r"$\mathbf{h}_1\ldots\mathbf{h}_4$", ha="center", va="bottom", fontsize=6.4)
    _arrow(ax, (x0 + 0.3, yb + 28.0), (gx - 0.4, yb + 33.0))
    _arrow(ax, (x0 + 0.3, yb + 22.0), (gx - 0.4, yb + 14.0))
    # 门控注意力：tanh 与 sigm 两支，逐元素相乘 ⊙，softmax 得到 ω_j
    bx = hx[-1] + 5.0
    for yy, lab in ((yb + 14.5, "tanh"), (yb + 8.0, "sigm")):
        ax.add_patch(FancyBboxPatch((bx, yy), 6.6, 3.6, boxstyle="round,pad=0,rounding_size=0.8", fc="#EEF3F8",
                                    ec=NAVY, lw=0.5, zorder=4))
        ax.text(bx + 3.3, yy + 1.8, lab, ha="center", va="center", fontsize=5.4, zorder=5)
        ax.plot([hx[-1] + 2.7, bx], [yb + 11.75, yy + 1.8], color=GREY, lw=0.45, zorder=2)
    ox, oy = bx + 10.0, yb + 13.0
    ax.add_patch(Circle((ox, oy), 1.7, fc="white", ec=NAVY, lw=0.6, zorder=4))
    ax.text(ox, oy - 0.05, r"$\odot$", ha="center", va="center", fontsize=7, zorder=5)
    for yy in (yb + 16.3, yb + 9.8):
        ax.plot([bx + 6.6, ox - 1.5], [yy, oy + (0.8 if yy > oy else -0.8)], color=GREY, lw=0.45, zorder=2)
    # 注意力权重柱（示例照片的四个裁块）
    wx, base, hmax = ox + 3.4, yb + 7.0, 11.0
    att = out["attn"] / out["attn"].max()
    _arrow(ax, (ox + 1.8, oy), (wx - 0.3, oy))
    for k, v in enumerate(att):
        ax.add_patch(Rectangle((wx + k * 1.9, base), 1.4, hmax * v * 0.9, fc=RED, ec="none", alpha=0.85, zorder=4))
    ax.plot([wx - 0.3, wx + 7.6], [base, base], color=GREY, lw=0.4, zorder=4)
    ax.text(wx + 3.6, base + hmax * 0.9 + 0.6, r"$\omega_j$", ha="center", va="bottom", fontsize=6.4)
    # 加权求和 Σ ω_j h_j = z
    sx, sy = wx + 11.8, yb + 13.0
    ax.add_patch(Circle((sx, sy), 2.0, fc="white", ec=PURPLE, lw=0.7, zorder=4))
    ax.text(sx, sy - 0.1, r"$\Sigma$", ha="center", va="center", fontsize=7, zorder=5)
    _arrow(ax, (wx + 7.8, sy), (sx - 2.1, sy))
    ax.plot([hx[1] + 1.3, hx[1] + 1.3, sx, sx], [yb + 7.0, yb + 3.6, yb + 3.6, sy - 2.0], color=GREY, lw=0.45, ls=(0, (2, 1.2)), zorder=2)
    zx = sx + 4.0
    _vector(ax, zx, yb + 8.3, 5, cell, 2.6, [PURPLE, "#8A6FB3", "#A991CC"])
    _arrow(ax, (sx + 2.1, sy), (zx - 0.3, sy))
    ax.text(zx + 1.3, yb + 18.2, r"$\mathbf{z}$", ha="center", va="bottom", fontsize=7)
    # 拼接 [g; z] 与全连接层 → 五级概率
    cx = zx + 6.6
    _vector(ax, cx, yb + 13.0, 5, cell, 2.6, [PURPLE, "#8A6FB3", "#A991CC"])
    _vector(ax, cx, yb + 13.0 + 5 * cell, 6, cell, 2.6, [NAVY, "#4A6FA5", "#6C8EBF"])
    ax.text(cx + 1.3, yb + 13.0 + 11 * cell + 0.6, r"$[\mathbf{g};\mathbf{z}]$", ha="center", va="bottom", fontsize=6.4)
    ax.plot([gx + 2.6, cx - 3.0, cx - 0.3], [yb + 36.0, yb + 36.0, yb + 29.0], color=GREY, lw=0.45, zorder=2)
    _arrow(ax, (cx - 1.5, yb + 29.6), (cx - 0.2, yb + 28.6))
    _arrow(ax, (zx + 2.8, yb + 13.0), (cx - 0.3, yb + 15.5))
    nx = cx + 7.0
    ny = [yb + 30.0 - 3.6 * k for k in range(5)]
    for i in range(11):
        for y in ny:
            ax.plot([cx + 2.6, nx - 1.3], [yb + 13.0 + (i + 0.5) * cell, y], color="#C8C8C8", lw=0.25, zorder=1)
    for k, y in enumerate(ny):
        ax.add_patch(Circle((nx, y), 1.35, fc=GRADE_COLORS[k], ec="white", lw=0.4, zorder=4))
        ax.text(nx, y - 0.05, str(k), ha="center", va="center", fontsize=4.8, color="white", fontweight="bold", zorder=5)
        ax.add_patch(Rectangle((nx + 1.9, y - 0.9), 9.0 * out["p"][k], 1.8, fc=GRADE_COLORS[k], ec="none", zorder=4))
    ax.text(nx + 4.0, yb + 33.4, r"Grade, $\mathbf{p}$", ha="center", va="bottom", fontsize=6.0)
    # 训练时的裁块头（虚线，测试时不用）
    px, py = nx - 2.0, yb - 1.5
    ax.add_patch(FancyBboxPatch((px - 1.2, py - 1.4), 15.6, 9.6, boxstyle="round,pad=0,rounding_size=1.0", fc="white",
                                ec=GREY, lw=0.6, ls=(0, (3, 2)), zorder=2))
    for k in range(5):
        ax.add_patch(Rectangle((px + 1.0 + k * 2.4, py + 1.2), 1.6, 5.0 * out["p_crop"][k] + 0.05, fc=GRADE_COLORS[k],
                               ec="none", zorder=4))
    ax.plot([px + 0.6, px + 13.0], [py + 1.2, py + 1.2], color=GREY, lw=0.4, zorder=4)
    ax.text(px + 6.6, py - 2.0, r"$\mathbf{p}^{\mathrm{crop}}$ (training only)", ha="center", va="top", fontsize=5.6, color="#444444")
    _arrow(ax, (zx + 1.3, yb + 8.1), (px - 1.4, py + 3.4), ls=(0, (2.5, 1.5)))
    ax.text(x0 + 19.0, yb - 3.0, r"$\mathcal{L}=\mathrm{CE}(\mathbf{p})+" + f"{aux_w:g}" + r"\,\mathrm{CE}(\mathbf{p}^{\mathrm{crop}})$",
            ha="center", va="center", fontsize=6.2)


def pick_thumbnail(root, man: pd.DataFrame, role: str, n_scan: int = 200) -> str:
    """每个队列：参考等级 1–2 的照片中（按 image_key 排序的前 n_scan 张），视野内平均亮度处于中间一半者里最清晰
    （拉普拉斯方差最大）的一张。"""
    sub = man[(man.role == role) & (man.grade.isin(["1", "2"]))].sort_values("image_key").head(n_scan)
    bright, sharp = [], []
    for k in sub.image_key:
        im = cv2.imread(str(root / "data/processed/canon" / f"{k}.jpg"), cv2.IMREAD_GRAYSCALE)
        small = cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)
        fov = cv2.erode((small > 10).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        bright.append(float(small[fov].mean()) if fov.any() else 0.0)
        sharp.append(float(cv2.Laplacian(small, cv2.CV_64F)[fov].var()) if fov.any() else 0.0)
    b = np.asarray(bright)
    mid = (b >= np.quantile(b, 0.25)) & (b <= np.quantile(b, 0.75))
    idx = np.flatnonzero(mid)[np.argmax(np.asarray(sharp)[mid])]
    return sub.image_key.iloc[idx]


def round_thumb(root, key: str, n: int = 240) -> np.ndarray:
    """canon 图缩小并以圆形视野为界（视野外白色），供小图使用。"""
    im = cv2.imread(str(root / "data/processed/canon" / f"{key}.jpg"))[:, :, ::-1]
    im = cv2.resize(im, (n, n), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[:n, :n]
    mask = (xx - n / 2 + 0.5) ** 2 + (yy - n / 2 + 0.5) ** 2 <= (n / 2) ** 2
    rgba = np.dstack([im, (mask * 255).astype(np.uint8)])
    return rgba
