"""Figure 3：同一病灶负荷在不同队列被划为不同等级。只读 results/08u_grading_standard 与 results/08w_ruler_check。
a 各队列每个参考等级的病灶负荷指数（中位数与四分位距；同一分割器、同一指数）
b 各队列四个划级切点相对 DDR 训练区的偏移（95% 区间，按单位重抽样）
c 只用人工标注：DDR（病灶掩膜）与 MMRDR（逐图病灶标签）中轻度伴渗出、出血判为 ≥2 级、仅渗出判为 ≥2 级的比例
d 分割器计数与人工掩膜计数的一致性（Spearman ρ），DDR 测试区与外部 IDRiD"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from cohorts import COH_COLOR, COH_LABEL, LESIONS, LES_COLOR  # noqa: E402
from figstyle import W2, apply_style, panel_label, save_fig, save_source  # noqa: E402

R = ROOT / "results"
ORDER = ["ddr_train", "ddr_test", "messidor2", "idrid", "deepdrid", "eyepacs", "mmrdr"]


def panel_a(ax):
    bg = pd.read_csv(R / "08u_grading_standard/burden_by_grade.tsv", sep="\t")
    bg = bg[bg.cohort.isin(ORDER)]
    for i, c in enumerate(ORDER):
        d = bg[bg.cohort == c].sort_values("grade")
        x = d.grade + (i - (len(ORDER) - 1) / 2) * 0.07
        lw = 1.6 if c == "ddr_train" else 0.9
        ax.plot(x, d.burden_median, "-", color=COH_COLOR[c], lw=lw, zorder=3 if c == "ddr_train" else 2)
        ax.errorbar(x, d.burden_median, yerr=[d.burden_median - d.burden_q25, d.burden_q75 - d.burden_median],
                    fmt="o", ms=2.6, color=COH_COLOR[c], elinewidth=0.6, capsize=0, zorder=3)
    ax.set_xticks(range(5)); ax.set_xticklabels(["0\nNo DR", "1\nMild", "2\nModerate", "3\nSevere", "4\nPDR"])
    ax.set_xlabel("Reference grade"); ax.set_ylabel("Lesion burden index")
    h = [Line2D([], [], color=COH_COLOR[c], marker="o", ms=2.6, lw=0.9, label=COH_LABEL[c]) for c in ORDER]
    ax.legend(handles=h, loc="upper left", frameon=False, fontsize=5.6, handlelength=1.4, borderaxespad=0.1)
    return bg


def panel_b(ax):
    cp = pd.read_csv(R / "08u_grading_standard/cutpoints.tsv", sep="\t")
    cohs = [c for c in ORDER if c != "ddr_train"]
    cols = ["#BDBDBD", "#C0392B", "#7F7F7F", "#222222"]
    for j, b in enumerate(["grade>=1", "grade>=2", "grade>=3", "grade>=4"]):
        for i, c in enumerate(cohs):
            r = cp[(cp.cohort == c) & (cp.boundary == b)].iloc[0]
            y = i + (j - 1.5) * 0.17
            ax.errorbar(r["shift"], y, xerr=[[r["shift"] - r.shift_lo], [r.shift_hi - r["shift"]]], fmt="o", ms=3,
                        color=cols[j], elinewidth=0.7, capsize=0, zorder=3)
    ax.axvline(0, color="#999999", lw=0.6, ls="--", zorder=1)
    ax.set_yticks(range(len(cohs))); ax.set_yticklabels([COH_LABEL[c] for c in cohs])
    ax.set_ylim(len(cohs) - 0.5, -0.5)
    ax.set_xlabel("Shift of grade boundary vs DDR training\n(lesion burden index, 95% CI)")
    h = [Line2D([], [], ls="", marker="o", ms=3, color=cols[j], label=l)
         for j, l in enumerate(["No DR | DR", "Mild | moderate", "Moderate | severe", "Severe | PDR"])]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=5.6,
              handletextpad=0.2, borderaxespad=0.1, columnspacing=0.8)
    return cp[cp.cohort.isin(cohs)]


def panel_c(ax):
    h = pd.read_csv(R / "08w_ruler_check/human_grade_rule.tsv", sep="\t")
    names = {"DDR (lesion masks)": ("DDR", "#424242"), "MMRDR (lesion labels)": ("MMRDR", COH_COLOR["mmrdr"])}
    items = [("Mild DR with\nhard exudate", lambda d: d[d.grade == "1"].exudate_present.iloc[0]),
             ("Grade ≥2 when\nhaemorrhage present", lambda d: d[d.grade == "grade>=2 | haemorrhage present"]["share"].iloc[0]),
             ("Grade ≥2 when exudate\nwithout haemorrhage", lambda d: d[d.grade == "grade>=2 | exudate without haemorrhage"]["share"].iloc[0])]
    rows, w = [], 0.36
    for j, (k, (lab, col)) in enumerate(names.items()):
        d = h[h.cohort == k]
        for i, (nm, f) in enumerate(items):
            v = float(f(d)) * 100
            ax.bar(i + (j - 0.5) * w, v, width=w * 0.92, color=col, zorder=2)
            ax.text(i + (j - 0.5) * w, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=5.6)
            rows.append({"cohort": lab, "quantity": nm.replace("\n", " "), "percent": v})
    ax.set_xticks(range(len(items))); ax.set_xticklabels([n for n, _ in items], fontsize=6)
    ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("Photographs (%)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c in names.values()], [l for l, _ in names.values()],
              loc="upper left", frameon=False, fontsize=5.8, ncol=2, borderaxespad=0.1, bbox_to_anchor=(0, 1.04))
    return pd.DataFrame(rows)


def panel_d(ax):
    rc = pd.read_csv(R / "08w_ruler_check/ruler_counts.tsv", sep="\t")
    grp = [("DDR test", "#424242"), ("IDRiD", COH_COLOR["idrid"])]
    w, rows = 0.36, []
    for j, (g, col) in enumerate(grp):
        for i, (l, _) in enumerate(LESIONS):
            r = rc[(rc.group == g) & (rc.lesion == l)].iloc[0]
            ax.bar(i + (j - 0.5) * w, r.spearman, width=w * 0.92, color=col, zorder=2)
            rows.append({"group": g, "lesion": l, "spearman": r.spearman, "n": r.n,
                         "median_ratio_pred_human": r.median_ratio_pred_human})
    ax.set_xticks(range(len(LESIONS))); ax.set_xticklabels([n.replace(" ", "\n") for _, n in LESIONS], fontsize=6)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Spearman ρ, predicted vs\nannotated lesion count")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c in grp],
              [f"{g} (n = {int(rc[rc.group == g].n.iloc[0])})" for g, _ in grp], loc="upper left", frameon=False,
              fontsize=5.8, ncol=2, borderaxespad=0.1, bbox_to_anchor=(0, 1.04))
    return pd.DataFrame(rows)


def main():
    apply_style()
    fig = plt.figure(figsize=(W2, 128 / 25.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1.25, 1])
    axa, axb, axc, axd = (fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2))
    src = {"a": panel_a(axa), "b": panel_b(axb), "c": panel_c(axc), "d": panel_d(axd)}
    for ax, k in ((axa, "a"), (axb, "b"), (axc, "c"), (axd, "d")):
        panel_label(ax, k)
    for k, df in src.items():
        save_source(df, ROOT, "Fig3", k)
    print(save_fig(fig, ROOT, "Fig3"))


if __name__ == "__main__":
    main()
