"""Supplementary Fig. S4：可转诊 DR（≥2 级）的 ROC 曲线，以三个模型集成的 P(≥2 级) 为评分；每个队列一格，
全局 512、DPP 与病灶覆盖裁块三条曲线，圆点为各输入以最可能等级判定时的工作点（results/08l_screening），
图例给出 AUROC（results/08m_referable_roc）。只读结果表。"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from figstyle import METHOD_COLORS, W2, apply_style, panel_label, save_fig, save_source  # noqa: E402

COHORTS = {"ddr_test": "DDR test", "deepdrid": "DeepDRiD (eyes)", "idrid": "IDRiD", "messidor2": "Messidor-2",
           "eyepacs": "EyePACS"}
INPUTS = {"G512": "Global view", "D": "DPP crops", "C": "Coverage crops"}


def main():
    apply_style(7)
    R = ROOT / "results/08m_referable_roc"
    cur = pd.read_csv(R / "roc_curves.tsv.gz", sep="\t")
    auc = pd.read_csv(R / "auroc.tsv", sep="\t")
    met = pd.read_csv(ROOT / "results/08l_screening/metrics.tsv", sep="\t")
    fig, axes = plt.subplots(2, 3, figsize=(W2, 0.62 * W2))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.09, wspace=0.42, hspace=0.55)
    pts = []
    for k, (co, title) in enumerate(COHORTS.items()):
        ax = axes.flat[k]
        ax.plot([0, 1], [0, 1], color="#CCCCCC", lw=0.6, ls=(0, (3, 2)), zorder=1)
        for i_, lab in INPUTS.items():
            c = cur[(cur.cohort == co) & (cur.input == i_)]
            ax.plot(c.fpr, c.tpr, color=METHOD_COLORS[i_], lw=1.0, zorder=3 if i_ == "C" else 2)
            m = met[(met.cohort == co) & (met.input == i_)].iloc[0]
            ax.plot(1 - m.ref_spec, m.ref_sens, "o", ms=3.4, mfc=METHOD_COLORS[i_], mec="white", mew=0.5, zorder=4)
            a = auc[(auc.cohort == co) & (auc.input == i_)].iloc[0]
            pts.append({"cohort": co, "input": i_, "auroc": a.auroc, "auroc_lo": a.lo, "auroc_hi": a.hi,
                        "op_fpr": 1 - m.ref_spec, "op_tpr": m.ref_sens})
        txt = "\n".join(f"{INPUTS[i_].split()[0]} {auc[(auc.cohort == co) & (auc.input == i_)].auroc.iloc[0]:.3f}"
                        for i_ in INPUTS)
        ax.text(0.97, 0.05, "AUROC\n" + txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6,
                linespacing=1.25)
        ax.set_xlim(-0.02, 1.0); ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity")
        ax.set_title(title, fontsize=7, pad=3)
        ax.set_aspect("equal", adjustable="box")
        panel_label(ax, "abcde"[k])
    leg = axes.flat[5]
    leg.axis("off")
    handles = [Line2D([], [], color=METHOD_COLORS[i_], lw=1.2, label=lab) for i_, lab in INPUTS.items()]
    handles.append(Line2D([], [], ls="none", marker="o", ms=3.6, mfc="#777777", mec="white", label="Operating point\n(most probable grade)"))
    leg.legend(handles=handles, loc="center left", frameon=False, fontsize=6.4, handlelength=1.8, labelspacing=0.9)
    save_source(cur[cur.input.isin(list(INPUTS))], ROOT, "FigS4", "roc_curves")
    save_source(pd.DataFrame(pts), ROOT, "FigS4", "auroc_operating_points")
    save_fig(fig, ROOT, "FigS4")


if __name__ == "__main__":
    main()
