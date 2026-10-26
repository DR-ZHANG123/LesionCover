"""Supplementary Fig. S2：裁块分支辅助分级头的作用。
a 测试时把裁块换成空白图或换成另一张照片的裁块后 κ 的变化（DDR 开发区与 DeepDRiD；标准融合 vs 辅助头）；
b 标准融合与辅助头训练下，各规则在 DDR 测试区轻度 DR 被判为无 DR 的比例；c 同样两种训练下的 DeepDRiD 眼级 κ。
只读 results/07b_crop_reliance 与 results/08_stats。"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from figstyle import METHOD_COLORS, W2, apply_style, panel_label, save_fig, save_source  # noqa: E402

RULES = ["U", "R", "T", "N", "D", "C"]
SHORT = {"U": "Quadrants", "R": "Random", "T": "Heatmap", "N": "NMS", "D": "DPP", "C": "LesionCover"}


def main():
    apply_style(7)
    rel = pd.read_csv(ROOT / "results/07b_crop_reliance/reliance.tsv", sep="\t")
    rel = rel[~rel.tag.str.contains("gdrop")]
    rel["training"] = np.where(rel.tag.str.endswith("_aux"), "Auxiliary crop head", "Standard fusion")
    piv = rel.pivot_table(index=["tag", "set", "training"], columns="crops", values="qwk").reset_index()
    piv["Blank crops"] = piv.blank - piv.intact
    piv["Crops of another photograph"] = piv.shuffled - piv.intact
    summ = pd.read_csv(ROOT / "results/08_stats/summary.tsv", sep="\t")

    fig, axs = plt.subplots(1, 3, figsize=(W2, 0.34 * W2), gridspec_kw={"wspace": 0.42, "width_ratios": [1.15, 1, 1]})
    a = axs[0]
    rows = []
    groups = [("ddr_dev", "Blank crops"), ("ddr_dev", "Crops of another photograph"),
              ("deepdrid", "Blank crops"), ("deepdrid", "Crops of another photograph")]
    for i, (st, col) in enumerate(groups):
        for j, (tr, face) in enumerate((("Standard fusion", "white"), ("Auxiliary crop head", "#555555"))):
            v = piv[(piv.set == st) & (piv.training == tr)][col]
            x = i + (j - 0.5) * 0.36
            a.bar(x, v.mean(), width=0.34, color=face, edgecolor="#333333", lw=0.7)
            a.scatter(np.full(len(v), x), v, s=4, color="#B2182B", zorder=3)
            rows += [{"set": st, "replacement": col, "training": tr, "delta_kappa": d} for d in v]
    a.axhline(0, color="#888888", lw=0.6)
    a.set_xticks(range(4)); a.set_xticklabels(["Dev.\nblank", "Dev.\nswapped", "External\nblank", "External\nswapped"])
    a.set_ylabel("Change in $\\kappa_w$ when crops\nare replaced at test time")
    a.legend(handles=[Patch(fc="white", ec="#333333", label="Standard fusion"),
                      Patch(fc="#555555", ec="#333333", label="Auxiliary crop head")], fontsize=5.8, frameon=False,
             loc="upper left", ncol=2, borderaxespad=0.2, columnspacing=0.8, handlelength=1.0)
    lo = min(-0.27, piv[["Blank crops", "Crops of another photograph"]].min().min() - 0.02)
    a.set_ylim(lo, 0.12)
    save_source(pd.DataFrame(rows), ROOT, "FigS2", "a_reliance"); panel_label(a, "a")

    for ax, st, col, ylab, scale, lab in ((axs[1], "ddr_test", "mild_to_none_mean", "Mild DR graded as no DR (%)\nDDR test", 100, "b"),
                                          (axs[2], "deepdrid_eye", "qwk_mean", "Eye-level $\\kappa_w$, DeepDRiD", 1, "c")):
        rows = []
        for i, r in enumerate(RULES):
            for j, (sfx, face) in enumerate((("", "white"), ("_aux", METHOD_COLORS[r]))):
                v = summ[(summ.method == f"{r}_k4{sfx}") & (summ.set == st)]
                if not len(v):
                    continue
                v = v.iloc[0]
                ax.bar(i + (j - 0.5) * 0.36, v[col] * scale, width=0.34, color=face, edgecolor=METHOD_COLORS[r], lw=0.8,
                       yerr=v[col.replace("_mean", "_std")] * scale, error_kw={"elinewidth": 0.6, "capsize": 1.2})
                rows.append({"rule": r, "training": "auxiliary head" if sfx else "standard fusion", "value": v[col],
                             "sd": v[col.replace("_mean", "_std")]})
        ax.set_xticks(range(len(RULES))); ax.set_xticklabels([SHORT[r] for r in RULES], rotation=45, ha="right",
                                                             rotation_mode="anchor")
        ax.set_ylabel(ylab)
        if lab == "c":
            ax.set_ylim(0.65, 0.85)
        save_source(pd.DataFrame(rows), ROOT, "FigS2", f"{lab}_{st}"); panel_label(ax, lab)
    save_fig(fig, ROOT, "FigS2")


if __name__ == "__main__":
    main()
