"""Figure 4：外部队列上，模型按训练集的分级标准判级。只读 results/08s_lcnet_eval 与 results/08y_error_direction。
a 六个队列的 κ（95% 区间）：LesionCover、只用全局图的 DINOv2-L、重新训练的骨干网络与已发表方法
b LesionCover 与只用全局图的 DINOv2-L 之差：轻度判为无 DR、可转诊敏感度、可转诊特异度（百分点，95% 区间）
c 各队列判高与判低的比例（LesionCover）
d 参考轻度判为 ≥2 级的比例与该队列在 mild | moderate 边界上的标准偏移"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from cohorts import COH_COLOR, COH_LABEL, COHORTS, LC_COL, MODEL_COLOR, MODEL_LABEL, MODEL_MARK  # noqa: E402
from figstyle import W2, apply_style, panel_label, save_fig, save_source  # noqa: E402

R = ROOT / "results"
SHOW = ["LesionCover", "DINOv2 global tokens only", "Global 1024", "Swin-T 1024", "GDRNet", "CABNet"]
REF2 = "DINOv2 global tokens only"


def panel_a(ax, met):
    rows, n = [], len(SHOW)
    for i, (coh, _) in enumerate(COHORTS):
        d = met[met.cohort == coh].set_index("input")
        for j, key in enumerate(SHOW):
            r = d.loc[key]; y = i + (j - (n - 1) / 2) * 0.13
            ax.errorbar(r.qwk, y, xerr=[[r.qwk - r.qwk_lo], [r.qwk_hi - r.qwk]], fmt=MODEL_MARK[key], ms=3.2,
                        color=MODEL_COLOR[key], elinewidth=0.7, capsize=0, zorder=3)
            rows.append({"cohort": coh, "model": MODEL_LABEL[key], "qwk": r.qwk, "qwk_lo": r.qwk_lo, "qwk_hi": r.qwk_hi, "n": r.n})
        if i:
            ax.axhline(i - 0.5, color="#DDDDDD", lw=0.5, zorder=1)
    ax.set_yticks(range(len(COHORTS))); ax.set_yticklabels([l for _, l in COHORTS])
    ax.set_ylim(len(COHORTS) - 0.5, -0.5); ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("Quadratic weighted $\\kappa$ (95% CI)")
    h = [Line2D([], [], ls="", marker=MODEL_MARK[k], color=MODEL_COLOR[k], ms=3.4, label=MODEL_LABEL[k]) for k in SHOW]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=5.4, borderaxespad=0.1,
              handletextpad=0.2, labelspacing=0.3, columnspacing=0.6)
    return pd.DataFrame(rows)


def panel_b(ax, con):
    stats = [("mild_missed", "Mild DR graded as no DR", "#7F7F7F"), ("ref_sens", "Referable DR sensitivity", LC_COL),
             ("ref_spec", "Referable DR specificity", "#4C72B0")]
    rows = []
    for i, (coh, _) in enumerate(COHORTS):
        d = con[(con.cohort == coh) & (con.contrast == f"LesionCover vs {REF2}")].set_index("stat")
        for j, (st, lab, col) in enumerate(stats):
            r = d.loc[st]; y = i + (j - 1) * 0.22
            ax.errorbar(100 * r["diff"], y, xerr=[[100 * (r["diff"] - r.lo)], [100 * (r.hi - r["diff"])]], fmt="o", ms=3,
                        color=col, elinewidth=0.7, capsize=0, zorder=3)
            rows.append({"cohort": coh, "stat": st, "diff_pp": 100 * r["diff"], "lo_pp": 100 * r.lo, "hi_pp": 100 * r.hi})
        if i:
            ax.axhline(i - 0.5, color="#DDDDDD", lw=0.5, zorder=1)
    ax.axvline(0, color="#999999", lw=0.6, ls="--", zorder=1)
    ax.set_yticks(range(len(COHORTS))); ax.set_yticklabels([l for _, l in COHORTS])
    ax.set_ylim(len(COHORTS) - 0.5, -0.5); ax.set_xlim(-10, 6)
    ax.set_xlabel("LesionCover minus DINOv2-L global view only\n(percentage points, 95% CI)")
    h = [Line2D([], [], ls="", marker="o", ms=3, color=c, label=l) for _, l, c in stats]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=5.6, borderaxespad=0.1,
              handletextpad=0.2, ncol=2, columnspacing=0.8)
    return pd.DataFrame(rows)


def panel_c(ax, ed):
    d = ed[ed.input == "LesionCover"].set_index(["cohort", "stat"])
    rows, w = [], 0.38
    for i, (coh, _) in enumerate(COHORTS):
        for j, (st, col) in enumerate((("over_graded", LC_COL), ("under_graded", "#4C72B0"))):
            r = d.loc[(coh, st)]
            ax.bar(i + (j - 0.5) * w, 100 * r.value, width=w * 0.92, color=col, zorder=2)
            ax.errorbar(i + (j - 0.5) * w, 100 * r.value, yerr=[[100 * (r.value - r.lo)], [100 * (r.hi - r.value)]],
                        fmt="none", ecolor="#222222", elinewidth=0.6, zorder=3)
            rows.append({"cohort": coh, "stat": st, "percent": 100 * r.value, "lo": 100 * r.lo, "hi": 100 * r.hi})
    ax.set_xticks(range(len(COHORTS))); ax.set_xticklabels([l.replace(" ", "\n") for _, l in COHORTS], fontsize=6)
    ax.set_ylabel("Photographs or eyes (%)"); ax.set_ylim(0, 27)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in (LC_COL, "#4C72B0")],
              ["Graded above reference", "Graded below reference"], loc="upper left", frameon=False, fontsize=5.8,
              borderaxespad=0.1, ncol=2, bbox_to_anchor=(0, 1.04))
    return pd.DataFrame(rows)


def panel_d(ax, ed):
    rows = []
    for key, filled in (("LesionCover", True), (REF2, False)):
        d = ed[(ed.input == key) & (ed.stat == "mild_to_moderate_plus")]
        for coh, _ in COHORTS:
            r = d[d.cohort == coh].iloc[0]
            col = COH_COLOR[coh]
            ax.errorbar(r.standard_shift_grade2 + (0 if filled else 0.03), 100 * r.value,
                        yerr=[[100 * (r.value - r.lo)], [100 * (r.hi - r.value)]], fmt="o", ms=4 if filled else 3.4,
                        mfc=col if filled else "white", mec=col, color=col, elinewidth=0.7, capsize=0, zorder=3)
            rows.append({"model": MODEL_LABEL[key], "cohort": coh, "shift_mild_moderate": r.standard_shift_grade2,
                         "mild_to_moderate_plus_pct": 100 * r.value, "lo": 100 * r.lo, "hi": 100 * r.hi, "n_mild": r.n})
    lab = {c: COH_LABEL[c] for c, _ in COHORTS}
    off = {"ddr_test": (0.06, 0), "messidor2": (0.06, -3), "idrid": (-0.06, 6), "deepdrid": (0.08, 0),
           "eyepacs": (0.07, -3), "mmrdr": (-0.04, 8)}
    d = ed[(ed.input == "LesionCover") & (ed.stat == "mild_to_moderate_plus")].set_index("cohort")
    for c, (dx, dy) in off.items():
        ax.text(d.loc[c, "standard_shift_grade2"] + dx, 100 * d.loc[c, "value"] + dy, lab[c], fontsize=5.6,
                ha="left" if dx > 0 else "right", va="center", color=COH_COLOR[c])
    ax.set_xlim(-0.6, 1.5); ax.set_ylim(0, 100)
    ax.set_xlabel("Shift of the mild | moderate boundary\nvs DDR training (lesion burden index)")
    ax.set_ylabel("Mild DR graded as moderate or worse (%)")
    h = [Line2D([], [], ls="", marker="o", color="#555555", ms=3.6, label="LesionCover"),
         Line2D([], [], ls="", marker="o", mfc="white", mec="#555555", ms=3.4, label="DINOv2-L, global view only")]
    ax.legend(handles=h, loc="upper left", frameon=False, fontsize=5.6, borderaxespad=0.1, handletextpad=0.2)
    return pd.DataFrame(rows)


def main():
    apply_style()
    met = pd.read_csv(R / "08s_lcnet_eval/metrics.tsv", sep="\t")
    con = pd.read_csv(R / "08s_lcnet_eval/contrasts.tsv", sep="\t")
    ed = pd.read_csv(R / "08y_error_direction/error_direction.tsv", sep="\t")
    fig = plt.figure(figsize=(W2, 135 / 25.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1.15, 1])
    axa, axb, axc, axd = (fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2))
    src = {"a": panel_a(axa, met), "b": panel_b(axb, con), "c": panel_c(axc, ed), "d": panel_d(axd, ed)}
    for ax, k in ((axa, "a"), (axb, "b"), (axc, "c"), (axd, "d")):
        panel_label(ax, k)
    for k, df in src.items():
        save_source(df, ROOT, "Fig4", k)
    print(save_fig(fig, ROOT, "Fig4"))


if __name__ == "__main__":
    main()
