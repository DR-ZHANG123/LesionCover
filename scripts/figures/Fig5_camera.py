"""Fig. 5：相机复现斑点与其去除。
a 参考 0 级照片中，病灶覆盖裁块内预测微动脉瘤数与只看裁块判为 DR 的比例（各评价队列；08g）；
b 预测微动脉瘤位置的复现图：DDR 测试区与 Messidor-2（05d；显示时取 6×6 格的最大值，使单格大小的热点可见）；
c 各队列预测病灶的最大复现率与阈值（05d/08o）；
d Messidor-2 每张照片的复现斑点数，按相机设置与参考等级（08o）；
e 示例：参考 0 级 Messidor-2 照片的一个病灶覆盖裁块，去除复现斑点前后（08o）；
f Messidor-2 中全局图与病灶覆盖裁块在去除复现斑点前后的无 DR 判为 DR、轻度判为无 DR（08n）。只读结果表。"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle, Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from figstyle import METHOD_COLORS, W2, apply_style, panel_label, save_fig, save_source  # noqa: E402

R = ROOT / "results"
COH = {"ddr_test": ("DDR test", "o", "#222222"), "deepdrid": ("DeepDRiD", "s", "#4C72B0"),
       "idrid": ("IDRiD", "^", "#DD8452"), "messidor2": ("Messidor-2", "D", "#55A868"),
       "eyepacs": ("EyePACS", "v", "#8172B3")}
REC = [("ddr_train", "DDR training"), ("ddr_dev", "DDR development"), ("ddr_test", "DDR test"),
       ("deepdrid", "DeepDRiD"), ("eyepacs", "EyePACS"), ("idrid", "IDRiD"), ("messidor2", "Messidor-2")]
BINS = ["0", "1", "2", "3-4", "≥5"]
GRADES = [("No DR", "#BDBDBD"), ("Mild DR", "#E0A458"), ("Referable DR", "#8C2D04")]


def panel_a(ax):
    hb = pd.read_csv(R / "08g_healthy_spots/bins.tsv", sep="\t")
    hb = hb[hb.grade == 0]
    off = (len(COH) - 1) / 2
    for j, (coh, (lab, mk, col)) in enumerate(COH.items()):
        t = hb[hb.cohort == coh].set_index("bin").loc[BINS]
        x = np.arange(len(BINS)) + (j - off) * 0.07
        ax.errorbar(x, t.called_dr_pct, yerr=[t.called_dr_pct - t.lo, t.hi - t.called_dr_pct], fmt=f"{mk}-", ms=3,
                    lw=0.9, color=col, elinewidth=0.5, capsize=1.2, label=lab)
    ax.set_xticks(range(len(BINS))); ax.set_xticklabels(BINS)
    ax.set_xlabel("Predicted microaneurysms\ninside the LesionCover crops")
    ax.set_ylabel("No-DR photographs graded\nas DR from crops (%)")
    ax.set_ylim(0, 128); ax.set_yticks(range(0, 101, 20))
    ax.legend(fontsize=5.5, frameon=False, loc="upper left", ncol=2, columnspacing=0.6, handletextpad=0.3,
              borderaxespad=0.1)
    save_source(hb, ROOT, "Fig5", "a_healthy_spots")


def panel_c(ax, thr: float):
    q = pd.read_csv(R / "08o_camera_spots/recurrence_quantiles.tsv", sep="\t").set_index("role")
    vals = [100 * q.loc[k, "max"] for k, _ in REC]
    cols = [COH[k][2] if k in ("idrid", "messidor2") else "#9E9E9E" for k, _ in REC]
    y = np.arange(len(REC))[::-1]
    ax.barh(y, vals, height=0.62, color=cols, edgecolor="none")
    ax.axvline(100 * thr, color="#B2182B", lw=0.8, ls="--")
    ax.text(100 * thr + 0.8, y[0] + 0.1, "Threshold, 5%", fontsize=6, color="#B2182B", ha="left", va="center")
    ax.set_yticks(y); ax.set_yticklabels([l for _, l in REC])
    ax.set_xlabel("Highest recurrence of a\npredicted lesion (%)"); ax.set_xlim(0, 30)
    save_source(pd.DataFrame({"cohort": [k for k, _ in REC], "max_recurrence_pct": vals}), ROOT, "Fig5", "c_recurrence")


def panel_b(fig, spec):
    z = np.load(R / "05d_camera_prior/recurrence_maps.npz")
    sub = spec.subgridspec(2, 2, height_ratios=[1, 0.05], wspace=0.06, hspace=0.12)
    norm = PowerNorm(0.5, vmin=0, vmax=30)
    axes, rows = [], []
    for i, (role, lab) in enumerate((("ddr_test", "DDR test"), ("messidor2", "Messidor-2"))):
        ax = fig.add_subplot(sub[0, i])
        m = 100 * z[f"{role}_MA"]
        blk = m.reshape(128, 6, 128, 6).max(axis=(1, 3))          # 显示用：6×6 格最大值
        im = ax.imshow(blk, cmap="magma", norm=norm, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(lab, fontsize=7, pad=2)
        axes.append(ax)
        rows.append({"cohort": role, "max_pct": float(m.max()), "cells_at_or_above_5pct": int((m >= 5).sum())})
    cax = fig.add_subplot(sub[1, :])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 2, 5, 10, 20, 30], orientation="horizontal")
    cb.ax.tick_params(labelsize=5.5, width=0.5, length=2); cb.outline.set_linewidth(0.4)
    cb.set_label("Photographs with a predicted microaneurysm at the position (%)", fontsize=6, labelpad=2)
    save_source(pd.DataFrame(rows), ROOT, "Fig5", "b_recurrence_maps")
    return axes[0]


def panel_d(ax):
    sb = pd.read_csv(R / "08o_camera_spots/spots_by_camera.tsv", sep="\t")
    settings = sorted(sb.camera_setting.unique(), key=lambda s: int(s.split()[0]))
    w = 0.26
    for j, (g, col) in enumerate(GRADES):
        t = sb[sb.grade_group == g].set_index("camera_setting").loc[settings]
        x = np.arange(len(settings)) + (j - 1) * w
        ax.bar(x, t.mean_flagged, width=w, color=col, edgecolor="none", label=g)
        ax.errorbar(x, t.mean_flagged, yerr=[t.mean_flagged - t.lo, t.hi - t.mean_flagged], fmt="none",
                    ecolor="#333333", elinewidth=0.5, capsize=1.2)
    ax.set_xticks(range(len(settings))); ax.set_xticklabels([s.replace(" × ", "×") for s in settings])
    ax.set_xlabel("Messidor-2 image size (pixels)")
    ax.set_ylabel("Recurrent spots\nper photograph"); ax.set_ylim(0, 6.6)
    ax.legend(fontsize=5.5, frameon=False, loc="upper left", ncol=3, columnspacing=0.6, handletextpad=0.3,
              handlelength=1.0, borderaxespad=0.1)
    save_source(sb, ROOT, "Fig5", "d_spots_by_camera")


def panel_e(fig, spec):
    import matplotlib.image as mpimg
    sp = pd.read_csv(R / "08o_camera_spots/example_spots.tsv", sep="\t")
    sub = spec.subgridspec(1, 2, wspace=0.06)
    axes = []
    for i, (f, lab) in enumerate((("example_before.png", "Original"), ("example_after.png", "Spots removed"))):
        ax = fig.add_subplot(sub[0, i])
        img = mpimg.imread(R / "08o_camera_spots" / f)
        ax.imshow(img, interpolation="lanczos")
        if i == 0:
            for x, y in sp[["x_in_crop", "y_in_crop"]].to_numpy():
                ax.add_patch(Circle((x, y), 17, fill=False, ec="#FFD700", lw=0.8))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(lab, fontsize=7, pad=2)
        axes.append(ax)
    save_source(sp, ROOT, "Fig5", "e_example_spots")
    return axes[0]


def panel_f(ax):
    m = pd.read_csv(R / "08n_camera_prior/metrics.tsv", sep="\t")
    m = m[m.cohort == "messidor2"].set_index("input")
    bars = [("G512", "Global view", "#D0D0D0", None), ("G512cam", "Global view, spots removed", "#707070", None),
            ("C", "LesionCover", "#F2B6BB", None), ("Ccam", "LesionCover, spots removed", METHOD_COLORS["C"], None)]
    stats = [("healthy_fp", "No DR graded\nas DR"), ("mild_missed", "Mild DR graded\nas no DR")]
    w = 0.2
    rows = []
    for j, (k, lab, col, _) in enumerate(bars):
        x = np.arange(len(stats)) + (j - 1.5) * w
        v = np.array([100 * m.loc[k, s] for s, _ in stats])
        lo = np.array([100 * m.loc[k, f"{s}_lo"] for s, _ in stats]); hi = np.array([100 * m.loc[k, f"{s}_hi"] for s, _ in stats])
        ax.bar(x, v, width=w, color=col, edgecolor="none", label=lab)
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt="none", ecolor="#333333", elinewidth=0.5, capsize=1.2)
        rows += [{"input": k, "stat": s, "pct": a, "lo": b, "hi": c} for (s, _), a, b, c in zip(stats, v, lo, hi)]
    ax.set_xticks(range(len(stats))); ax.set_xticklabels([l for _, l in stats])
    ax.set_ylabel("Messidor-2 cases (%)"); ax.set_ylim(0, 92); ax.set_yticks(range(0, 61, 20))
    ax.legend(fontsize=5.5, frameon=False, loc="upper left", ncol=2, columnspacing=0.6, handletextpad=0.3,
              handlelength=1.0, borderaxespad=0.1)
    save_source(pd.DataFrame(rows), ROOT, "Fig5", "f_messidor2_screening")


def main():
    import yaml
    thr = yaml.safe_load((ROOT / "config/run.yaml").read_text())["camera_prior"]["threshold"]
    apply_style(7)
    fig = plt.figure(figsize=(W2, 0.62 * W2))
    gs = fig.add_gridspec(2, 3, wspace=0.5, hspace=0.62, left=0.085, right=0.99, top=0.95, bottom=0.11,
                          width_ratios=[1, 1.3, 1.05])
    a = fig.add_subplot(gs[0, 0]); panel_a(a); panel_label(a, "a")
    b = panel_b(fig, gs[0, 1]); panel_label(b, "b")
    c = fig.add_subplot(gs[0, 2]); panel_c(c, thr); panel_label(c, "c")
    d = fig.add_subplot(gs[1, 0]); panel_d(d); panel_label(d, "d")
    e = panel_e(fig, gs[1, 1]); panel_label(e, "e")
    f = fig.add_subplot(gs[1, 2]); panel_f(f); panel_label(f, "f")
    fig._col_tol = 0.05
    save_fig(fig, ROOT, "Fig5")


if __name__ == "__main__":
    main()
