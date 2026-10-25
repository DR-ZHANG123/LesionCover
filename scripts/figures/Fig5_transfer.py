"""Fig. 5：外部队列中的病灶样斑点与迁移。
a 参考等级为无 DR 的照片中，病灶覆盖裁块内预测微动脉瘤数与只看裁块判为有 DR 的比例（各评价队列；照片 bootstrap 区间）；
b 只看裁块时各规则在各评价队列中判对无 DR 的比例（主设置，ConvNeXt-Tiny）。
编码器容量与训练相机的比较见 Fig. 6。只读 results/08g_healthy_spots 与 08d_crop_only。"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from figstyle import METHOD_COLORS, W2, apply_style, panel_label, save_fig, save_source  # noqa: E402

RULES = ["U", "R", "T", "S", "N", "D", "Cmass", "C"]
SHORT = {"U": "Quadrants", "R": "Random", "T": "Heatmap", "S": "Score top-K", "N": "NMS", "D": "DPP",
         "Cmass": "Area-weighted", "C": "Coverage"}
COH_ALL = {"ddr_test": ("DDR test", "o", "#222222"), "deepdrid": ("DeepDRiD", "s", "#4C72B0"),
           "idrid": ("IDRiD", "^", "#DD8452"), "messidor2": ("Messidor-2", "D", "#55A868"),
           "eyepacs": ("EyePACS", "v", "#8172B3")}
BINS = ["0", "1", "2", "3-4", "≥5"]


def main():
    apply_style(7)
    hb = pd.read_csv(ROOT / "results/08g_healthy_spots/bins.tsv", sep="\t")
    co = pd.read_csv(ROOT / "results/08d_crop_only/summary.tsv", sep="\t")
    COH = {k: v for k, v in COH_ALL.items() if (hb.cohort == k).any() and (co.cohort == k).any()}
    off = (len(COH) - 1) / 2
    fig = plt.figure(figsize=(W2, 0.36 * W2))
    gs = fig.add_gridspec(1, 6, wspace=1.3, left=0.075, right=0.985, top=0.9, bottom=0.25)

    a = fig.add_subplot(gs[0, 0:3])
    for j, (coh, (lab, mk, col)) in enumerate(COH.items()):
        t = hb[(hb.cohort == coh) & (hb.grade == 0)].set_index("bin").loc[BINS]
        x = np.arange(len(BINS)) + (j - off) * 0.07
        a.errorbar(x, t.called_dr_pct, yerr=[t.called_dr_pct - t.lo, t.hi - t.called_dr_pct], fmt=f"{mk}-", ms=3.2,
                   lw=0.9, color=col, elinewidth=0.5, capsize=1.2, label=lab)
    a.set_xticks(range(len(BINS))); a.set_xticklabels(BINS)
    a.set_xlabel("Predicted microaneurysms inside the coverage crops")
    a.set_ylabel("No-DR photographs graded\nas DR from crops (%)"); a.set_ylim(0, 122); a.set_yticks(range(0, 101, 20))
    a.legend(fontsize=6, frameon=False, loc="upper center", ncol=3, columnspacing=0.8, handletextpad=0.3)   # 顶部留出图例带
    save_source(hb[hb.grade == 0], ROOT, "Fig5", "a_healthy_spots"); panel_label(a, "a")

    b = fig.add_subplot(gs[0, 3:6])
    rows = []
    for j, (coh, (lab, mk, col)) in enumerate(COH.items()):
        t = co[co.cohort == coh].set_index("rule").loc[RULES]
        x = np.arange(len(RULES)) + (j - off) * 0.15
        b.scatter(x, t.recall_0_mean * 100, marker=mk, s=11, color=col, edgecolor="none", label=lab, zorder=3)
        rows += [{"cohort": coh, "rule": r, "nodr_correct_pct": v * 100} for r, v in zip(RULES, t.recall_0_mean)]
    b.set_xticks(range(len(RULES))); b.set_xticklabels([SHORT[r] for r in RULES], rotation=40, ha="right",
                                                     rotation_mode="anchor")
    b.set_ylabel("No-DR cases graded\ncorrectly from crops (%)"); b.set_ylim(40, 103)
    save_source(pd.DataFrame(rows), ROOT, "Fig5", "b_crop_only_nodr"); panel_label(b, "b")

    save_fig(fig, ROOT, "Fig5")


if __name__ == "__main__":
    main()
