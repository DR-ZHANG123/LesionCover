"""Figure 6：LesionCover 判级时依据哪些病灶。只读 results/08za_interpretability。
a 最后一层 [CLS] 注意力在全局 token、计数/规则 token 与各类病灶 token 上的占比（按参考等级；DDR 测试区与五个外部队列合并）
b 概念消融：测试时去掉一类病灶（或全部病灶证据）后期望等级的平均变化（各队列）
c 病灶负荷门控（融合输出与全局输出的权重）按参考等级的平均值（各队列）"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from cohorts import COH_COLOR, COH_LABEL, COHORTS, LES_COLOR, LESIONS  # noqa: E402
from figstyle import W2, apply_style, panel_label, save_fig, save_source  # noqa: E402

R = ROOT / "results" / "08za_interpretability"
GRADES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
EXT = [c for c, _ in COHORTS[1:]]


def panel_a(ax, at):
    parts = [("attention_global", "Global image tokens", "#BDBDBD"), ("attention_count_rule", "Count and rule tokens", "#636363")] + \
            [(f"attention_{k}", lab, LES_COLOR[k]) for k, lab in LESIONS]
    rows = []
    for gi, (grp, sel) in enumerate((("DDR test", ["ddr_test"]), ("External", EXT))):
        for g in range(5):
            d = at[at.cohort.isin(sel) & (at.grade == g)]
            w = d.n / d.n.sum()
            rows.append({"group": grp, "grade": g, **{c: float((d[c] * w).sum()) for c, _, _ in parts}})
    df = pd.DataFrame(rows)
    ys = [g + gi * 5.8 for gi in range(2) for g in range(5)]
    left = np.zeros(len(df))
    for c, lab, col in parts:
        ax.barh(ys, df[c], left=left, color=col, height=0.75, label=lab, edgecolor="white", lw=0.3)
        left += df[c].to_numpy()
    ax.set_yticks(ys); ax.set_yticklabels(GRADES * 2, fontsize=6)
    ax.set_ylim(max(ys) + 0.6, -0.6)
    for gi, grp in enumerate(("DDR test", "Five external cohorts")):
        ax.text(1.0, gi * 5.8 - 0.75, grp, ha="right", va="bottom", fontsize=6, fontweight="bold")
    ax.set_xlim(0, 1.0); ax.set_xlabel("Share of [CLS] attention, last layer")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=5.6, borderaxespad=0.1,
              handlelength=1.0, columnspacing=0.8)
    return df


def panel_b(ax, ab):
    conds = [(f"no_{k}", lab) for k, lab in LESIONS] + [("no_lesions", "All lesion evidence")]
    cols = [c for c, _ in COHORTS]
    M = np.array([[ab[(ab.condition == c) & (ab.cohort == k)].delta_expected_grade.iloc[0] for k in cols] for c, _ in conds])
    v = 0.25
    ax.imshow(M, cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, (f"{M[i, j]:+.2f}" if round(M[i, j], 2) != 0 else "0.00").replace("-", "−"), ha="center", va="center", fontsize=5.4,
                    color="white" if abs(M[i, j]) > 0.15 else "black")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([COH_LABEL[c] for c in cols], rotation=35, ha="right", fontsize=6)
    ax.set_yticks(range(len(conds))); ax.set_yticklabels([f"Without {l.lower()}" if i < 4 else "Without all lesion evidence"
                                                          for i, (_, l) in enumerate(conds)], fontsize=6)
    ax.set_title("Change in expected grade", fontsize=6.4, pad=3)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return pd.DataFrame(M, index=[c for c, _ in conds], columns=cols).reset_index(names="condition")


def panel_c(ax, comp):
    rows = []
    for coh, lab in COHORTS:
        d = comp[comp.cohort == coh].sort_values("grade")
        ax.plot(d.grade, d.gate_mean, "-o", color=COH_COLOR[coh], ms=2.6, lw=0.9, label=lab)
        rows.append(d)
    ax.set_xticks(range(5)); ax.set_xticklabels(GRADES, fontsize=6)
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("Weight on global output (gate)")
    ax.set_xlabel("Reference grade")
    ax.legend(loc="lower right", frameon=False, fontsize=5.6, ncol=2, handlelength=1.2, columnspacing=0.8)
    return pd.concat(rows)


def main():
    apply_style()
    at = pd.read_csv(R / "attention.tsv", sep="\t"); ab = pd.read_csv(R / "ablation.tsv", sep="\t")
    comp = pd.read_csv(R / "components.tsv", sep="\t")
    fig = plt.figure(figsize=(W2, 105 / 25.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.05], height_ratios=[1, 1])
    axa, axb, axc = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    src = {"a": panel_a(axa, at), "b": panel_b(axb, ab), "c": panel_c(axc, comp)}
    for ax, k in ((axa, "a"), (axb, "b"), (axc, "c")):
        panel_label(ax, k)
    for k, df in src.items():
        save_source(df, ROOT, "Fig6", k)
    print(save_fig(fig, ROOT, "Fig6"))


if __name__ == "__main__":
    main()
