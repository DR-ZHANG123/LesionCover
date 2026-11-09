"""Figure 5：每级少量本地标注照片即可把输出对齐到当地分级标准（网络权重不变）。只读 results/08v_local_calibration/metrics.tsv。
a 五个外部队列 κ 的平均：原输出与每级 5、20 张照片校准（温度 + 每级偏置）后；评价集与校准集按单位分开，五次随机抽取平均
b LesionCover 与只用全局图的 DINOv2-L 在每个外部队列的 κ：原输出 → 每级 5 张校准
c IDRiD 上 LesionCover 校准前后：轻度判为 ≥2 级、可转诊特异度、可转诊敏感度"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from cohorts import COH_LABEL, EXTERNAL, LC_COL, MODEL_COLOR, MODEL_LABEL, MODEL_MARK  # noqa: E402
from figstyle import W2, apply_style, panel_label, save_fig, save_source  # noqa: E402

R = ROOT / "results"
SHOW = ["LesionCover", "DINOv2 global tokens only", "Global 1024", "Swin-T 1024", "EfficientNetV2-M 1024", "ResNet-50 1024",
        "GDRNet", "CABNet"]
METH = "bias_temp"


def panel_a(ax, m):
    rows = []
    for key in SHOW:
        d = m[m.input == key]
        none = d[(d.method == "none") & (d.per_grade == 5)].qwk.mean()
        c5 = d[(d.method == METH) & (d.per_grade == 5)].qwk.mean()
        c20 = d[(d.method == METH) & (d.per_grade == 20)].qwk.mean()
        rows.append({"model": MODEL_LABEL[key], "key": key, "uncalibrated": none, "calibrated_5": c5, "calibrated_20": c20})
    df = pd.DataFrame(rows).sort_values("calibrated_5", ascending=False).reset_index(drop=True)
    for i, r in df.iterrows():
        col = MODEL_COLOR[r.key]
        ax.plot([r.uncalibrated, r.calibrated_5], [i, i], color=col, lw=1.0, zorder=2)
        ax.scatter(r.uncalibrated, i, s=16, facecolor="white", edgecolor=col, lw=0.9, zorder=3)
        ax.scatter(r.calibrated_5, i, s=18, color=col, marker=MODEL_MARK[r.key], zorder=4)
        ax.scatter(r.calibrated_20, i, s=10, color=col, marker="|", lw=1.0, zorder=4)
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df.model, fontsize=6)
    for t, k in zip(ax.get_yticklabels(), df.key):
        if k == "LesionCover":
            t.set_fontweight("bold")
    ax.set_ylim(len(df) - 0.5, -0.5); ax.set_xlim(0.45, 0.82)
    ax.set_xlabel("Mean $\\kappa$, five external cohorts")
    h = [Line2D([], [], ls="", marker="o", mfc="white", mec="#555555", ms=3.6, label="DDR standard (as trained)"),
         Line2D([], [], ls="", marker="o", color="#555555", ms=3.6, label="Aligned, 5 photographs per grade"),
         Line2D([], [], ls="", marker="|", color="#555555", ms=5, label="Aligned, 20 photographs per grade")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=5.6, borderaxespad=0.1,
              handletextpad=0.3)
    return df.drop(columns="key")


def panel_b(ax, m):
    rows, keys = [], ["LesionCover", "DINOv2 global tokens only"]
    for i, (coh, lab) in enumerate(EXTERNAL):
        for j, key in enumerate(keys):
            d = m[(m.input == key) & (m.cohort == coh) & (m.per_grade == 5)].set_index("method")
            y = i + (j - 0.5) * 0.3; col = MODEL_COLOR[key]
            a, b, sd = d.loc["none", "qwk"], d.loc[METH, "qwk"], d.loc[METH, "qwk_sd"]
            ax.annotate("", xy=(b, y), xytext=(a, y), arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9,
                                                                      mutation_scale=6, shrinkA=0, shrinkB=0))
            ax.scatter(a, y, s=12, facecolor="white", edgecolor=col, lw=0.8, zorder=3)
            ax.errorbar(b, y, xerr=sd, fmt="none", ecolor=col, elinewidth=0.6, zorder=3)
            rows.append({"cohort": coh, "model": MODEL_LABEL[key], "uncalibrated": a, "calibrated_5": b, "calibrated_5_sd": sd})
        if i:
            ax.axhline(i - 0.5, color="#DDDDDD", lw=0.5, zorder=1)
    ax.set_yticks(range(len(EXTERNAL))); ax.set_yticklabels([l for _, l in EXTERNAL])
    ax.set_ylim(len(EXTERNAL) - 0.5, -0.5); ax.set_xlim(0.5, 0.95)
    ax.set_xlabel("$\\kappa$: DDR standard → aligned, 5 photographs per grade\n(bars, SD over five draws)")
    h = [Line2D([], [], color=MODEL_COLOR[k], lw=1, label=MODEL_LABEL[k]) for k in keys]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=5.6, ncol=2, borderaxespad=0.1,
              columnspacing=0.8)
    return pd.DataFrame(rows)


def panel_c(ax, m):
    d = m[(m.input == "LesionCover") & (m.cohort == "idrid")]
    sets = [("DDR standard", d[(d.method == "none") & (d.per_grade == 5)].iloc[0], "#BBBBBB"),
            ("Aligned, 5 per grade", d[(d.method == METH) & (d.per_grade == 5)].iloc[0], "#E8A39B"),
            ("Aligned, 20 per grade", d[(d.method == METH) & (d.per_grade == 20)].iloc[0], LC_COL)]
    stats = [("mild_to_moderate_plus", "Mild DR graded\nas moderate or worse"), ("ref_spec", "Referable DR\nspecificity"),
             ("ref_sens", "Referable DR\nsensitivity")]
    rows, w = [], 0.26
    for j, (lab, r, col) in enumerate(sets):
        for i, (st, _) in enumerate(stats):
            v = 100 * r[st]; x = i + (j - 1) * w
            ax.bar(x, v, width=w * 0.92, color=col, zorder=2)
            ax.text(x, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=5.4)
            rows.append({"setting": lab, "stat": st, "percent": v})
    ax.set_xticks(range(len(stats))); ax.set_xticklabels([l for _, l in stats], fontsize=6)
    ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("IDRiD, LesionCover (%)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in sets], [l for l, _, _ in sets], loc="upper left",
              frameon=False, fontsize=5.6, ncol=3, borderaxespad=0.1, bbox_to_anchor=(0, 1.04), columnspacing=0.8)
    return pd.DataFrame(rows)


def main():
    apply_style()
    m = pd.read_csv(R / "08v_local_calibration/metrics.tsv", sep="\t")
    fig = plt.figure(figsize=(W2, 118 / 25.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.9, 1.1], height_ratios=[1, 1])
    axa, axb, axc = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    src = {"a": panel_a(axa, m), "b": panel_b(axb, m), "c": panel_c(axc, m)}
    for ax, k in ((axa, "a"), (axb, "b"), (axc, "c")):
        panel_label(ax, k)
    for k, df in src.items():
        save_source(df, ROOT, "Fig5", k)
    print(save_fig(fig, ROOT, "Fig5"))


if __name__ == "__main__":
    main()
