"""Fig. 4：分级结果（每种输入为三个种子模型的概率平均；点 = 单个种子模型）。
a 五个评价队列的二次加权 κ（DDR 测试区、IDRiD、Messidor-2、EyePACS 照片级，DeepDRiD 眼级）；
b DDR 测试区轻度 DR 被判为无 DR 的比例；c–f 四个外部队列中轻度漏判与无 DR 判对之间的权衡（箭头：全局 512 → 病灶覆盖）。
只读 results/08_stats。"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from figstyle import METHOD_COLORS, W2, align_panel_labels, apply_style, panel_label, save_fig, save_source  # noqa: E402

ORDER = ["G512", "G1024", "U", "R", "T", "N", "D", "C"]
KEY = {m: (m if m.startswith("G") else f"{m}_k4_aux") for m in ORDER}
SHORT = {"G512": "Global 512", "G1024": "Global 1024", "U": "Quadrants", "R": "Random", "T": "Heatmap",
         "N": "NMS", "D": "DPP", "C": "LesionCover"}
SETS = [("ddr_test", "DDR test"), ("deepdrid_eye", "DeepDRiD"), ("idrid", "IDRiD"), ("messidor2", "Messidor-2"),
        ("eyepacs", "EyePACS")]
EXT = {"deepdrid_eye": ("DeepDRiD", "s"), "idrid": ("IDRiD", "^"), "messidor2": ("Messidor-2", "D"), "eyepacs": ("EyePACS", "v")}


def main():
    apply_style(7)
    per = pd.read_csv(ROOT / "results/08_stats/per_run_metrics.tsv", sep="\t")
    summ = pd.read_csv(ROOT / "results/08_stats/summary.tsv", sep="\t")
    sets = [x for x in SETS if (summ.set == x[0]).any()]
    ext = {k: v for k, v in EXT.items() if (summ.set == k).any()}
    fig = plt.figure(figsize=(W2, 0.62 * W2))
    gs = fig.add_gridspec(2, 1 + len(ext), height_ratios=[1, 1.05], width_ratios=[1.45] + [1] * len(ext), hspace=0.62,
                          wspace=0.55, left=0.07, right=0.985, top=0.93, bottom=0.12)
    axa = fig.add_subplot(gs[0, :])
    w = 0.105
    rows = []
    for i, m in enumerate(ORDER):
        for j, (st, _) in enumerate(sets):
            v = summ[(summ.method == KEY[m]) & (summ.set == st)].iloc[0]
            x = j + (i - 3.5) * w
            axa.bar(x, v.qwk_mean, width=w * 0.92, color=METHOD_COLORS[m], edgecolor="none", zorder=2)
            sd = per[(per.method == KEY[m]) & (per.set == st)].qwk
            axa.scatter(np.full(len(sd), x), sd, s=2.2, color="#222222", zorder=3, linewidths=0)
            rows.append({"input": m, "cohort": st, "kappa_ensemble": v.qwk_mean, **{f"kappa_seed{k}": q for k, q in enumerate(sd)}})
    axa.set_xticks(range(len(sets))); axa.set_xticklabels([s for _, s in sets])
    lo = min(min(r["kappa_ensemble"], *[v for k, v in r.items() if k.startswith("kappa_seed")]) for r in rows)
    axa.set_ylim(np.floor((lo - 0.04) * 20) / 20, 0.92); axa.set_ylabel("Quadratic weighted $\\kappa$")
    axa.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[m]) for m in ORDER], labels=[SHORT[m] for m in ORDER],
               ncol=8, fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0,
               columnspacing=0.9, borderaxespad=0.2)
    save_source(pd.DataFrame(rows), ROOT, "Fig4", "a_kappa_cohorts"); panel_label(axa, "a")

    axb = fig.add_subplot(gs[1, 0])
    rows = []
    for i, m in enumerate(ORDER):
        v = summ[(summ.method == KEY[m]) & (summ.set == "ddr_test")].iloc[0]
        sd = per[(per.method == KEY[m]) & (per.set == "ddr_test")].mild_to_none * 100
        axb.bar(i, v.mild_to_none_mean * 100, width=0.72, color=METHOD_COLORS[m], edgecolor="none", zorder=2)
        axb.scatter(np.full(len(sd), i), sd, s=3, color="#222222", zorder=3, linewidths=0)
        rows.append({"input": m, "mild_missed_ensemble_pct": v.mild_to_none_mean * 100,
                     **{f"seed{k}_pct": q for k, q in enumerate(sd)}})
    axb.set_xticks(range(len(ORDER))); axb.set_xticklabels([SHORT[m] for m in ORDER], rotation=55, ha="right",
                                                          rotation_mode="anchor", fontsize=6)
    axb.set_ylabel("Mild DR graded as no DR (%)\nDDR test"); axb.set_ylim(0, 50)
    save_source(pd.DataFrame(rows), ROOT, "Fig4", "b_mild_missed"); panel_label(axb, "b")

    rows = []
    show = ["G512", "G1024", "N", "D", "C"]
    for k, (st, (lab, mk)) in enumerate(ext.items()):
        ax = fig.add_subplot(gs[1, k + 1])
        pts = {}
        for m in show:
            v = summ[(summ.method == KEY[m]) & (summ.set == st)].iloc[0]
            pts[m] = (v.mild_to_none_mean * 100, v.recall_0_mean * 100)
            ax.scatter(*pts[m], marker="o", s=18, color=METHOD_COLORS[m], edgecolor="#333333", linewidths=0.3, zorder=3)
            rows.append({"cohort": st, "input": m, "mild_missed_pct": pts[m][0], "nodr_correct_pct": pts[m][1]})
        ax.annotate("", xy=pts["C"], xytext=pts["G512"],
                    arrowprops={"arrowstyle": "-|>", "lw": 0.6, "color": "#555555", "shrinkA": 3.5, "shrinkB": 3.5,
                                "mutation_scale": 6}, zorder=2)
        xs, ys = [p[0] for p in pts.values()], [p[1] for p in pts.values()]
        ax.set_xlim(max(0, min(xs) - 6), max(xs) + 6); ax.set_ylim(min(ys) - 6, max(ys) + 6)
        ax.set_title(lab, fontsize=7, pad=3)
        ax.set_xlabel("Mild DR graded\nas no DR (%)")
        if k == 0:
            ax.set_ylabel("No-DR cases graded correctly (%)")            # 颜色与 a 的图例相同
        panel_label(ax, "cdef"[k])
    save_source(pd.DataFrame(rows), ROOT, "Fig4", "cf_external_tradeoff")
    save_fig(fig, ROOT, "Fig4")


if __name__ == "__main__":
    main()
