"""Figure 2：LesionCover 学到的象限分级规则与 DDR 测试区分级。只读 results/08x_learned_rule、results/08s_lcnet_eval、
metadata/literature_ddr.tsv。
a 每类病灶学到的三个“每象限分量数”阈值（DINOv2-L 与 ConvNeXt-T 编码器，各三个种子）
b 规则有序分级中每个特征（病灶 × 阈值 × 至少几个象限）的非负权重（DINOv2-L，三个种子平均）
c DDR 测试区准确率与 κ：重新训练的模型（实心，95% 区间）与其他 DDR 研究的报告值（空心）
d 结构对比：DDR 测试区 κ（95% 区间）"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from cohorts import LC_COL, LESIONS, LES_COLOR, MODELS  # noqa: E402
from figstyle import W2, apply_style, panel_label, save_fig, save_source  # noqa: E402

R = ROOT / "results"
LIT_POS = {"Dual-SwinOrd": (-0.004, 0, "right"), "CRA-Net": (0.004, -0.004, "left"), "CABNet": (-0.004, 0, "right"),
           "Xception": (0, 0.007, "center"), "ResNet-50": (0.004, 0, "left"), "MobileNet": (0.004, -0.004, "left")}
ABL = [("Global tokens only", "Global view only"), ("Lesion tokens without rule/gate", "+ lesion tokens"),
       ("ConvNeXt-T, fixed thresholds", "+ rule (fixed thresholds) + gate"), ("ConvNeXt-T encoder", "+ rule (learned) + gate"),
       ("DINOv2 global tokens only", "Global view only"), ("Fixed rule thresholds", "+ lesion tokens, rule (fixed) + gate"),
       ("LesionCover", "+ lesion tokens, rule (learned) + gate")]


def panel_a(ax):
    t = pd.read_csv(R / "08x_learned_rule/thresholds.tsv", sep="\t")
    for i, (l, _) in enumerate(LESIONS):
        for e, (enc, mk, fc) in enumerate((("DINOv2-L", "o", None), ("ConvNeXt-T", "s", "white"))):
            d = t[(t.lesion == l) & (t.encoder == enc)]
            for k in (1, 2, 3):
                v = d[d.level == k].threshold_per_quadrant.to_numpy()
                x = i + (k - 2) * 0.24 + (e - 0.5) * 0.09
                ax.scatter(np.full(len(v), x), v, s=9, marker=mk, color=LES_COLOR[l], facecolor=fc or LES_COLOR[l],
                           lw=0.6, zorder=3)
    ax.set_yscale("log"); ax.set_ylim(0.3, 25)
    ax.set_yticks([0.5, 1, 2, 5, 10, 20]); ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.set_xticks(range(len(LESIONS))); ax.set_xticklabels([l for l, _ in LESIONS])
    ax.set_xlabel("Lesion type")
    ax.set_ylabel("Learned threshold\n(lesions per quadrant)")
    for y in (1, 2, 5, 10):
        ax.axhline(y, color="#EEEEEE", lw=0.5, zorder=0)
    h = [Line2D([], [], ls="", marker="o", color="#555555", ms=3, label="DINOv2-L encoder"),
         Line2D([], [], ls="", marker="s", mfc="white", mec="#555555", ms=3, label="ConvNeXt-T encoder")]
    ax.legend(handles=h, loc="upper left", frameon=False, fontsize=5.6, borderaxespad=0.1, handletextpad=0.2)
    return t


def panel_b(ax):
    w = pd.read_csv(R / "08x_learned_rule/weights.tsv", sep="\t")
    m = w[w.encoder == "DINOv2-L"].groupby(["lesion", "level", "quadrants_at_least"]).weight.mean().reset_index()
    rows, labels = [], []
    for l, name in LESIONS:
        for k in (1, 2, 3):
            rows.append([m[(m.lesion == l) & (m.level == k) & (m.quadrants_at_least == j)].weight.iloc[0] for j in (1, 2, 3, 4)])
            labels.append(f"{l}, threshold {k}")
    A = np.asarray(rows)
    im = ax.imshow(A, cmap="Reds", vmin=0, vmax=A.max(), aspect="auto")
    for i in range(A.shape[0]):
        for j in range(4):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=5,
                    color="white" if A[i, j] > 0.5 * A.max() else "#333333")
    ax.set_xticks(range(4)); ax.set_xticklabels(["≥1", "≥2", "≥3", "≥4"])
    ax.set_xlabel("Quadrants at or above threshold")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=5.6)
    for y in (2.5, 5.5, 8.5):
        ax.axhline(y, color="white", lw=1.2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02, aspect=25)
    cb.set_label("Weight in rule-based grade", fontsize=6); cb.ax.tick_params(labelsize=5.6, length=2)
    return m


def panel_c(ax, met, lit):
    lt = lit.dropna(subset=["qwk"])
    ax.scatter(lt.accuracy, lt.qwk, s=16, facecolor="white", edgecolor="#777777", lw=0.8, zorder=3)
    d = met[met.cohort == "ddr_test"].set_index("input")
    rows = []
    for key, lab, col, mk in MODELS:
        r = d.loc[key]
        ax.errorbar(r.accuracy, r.qwk, xerr=[[r.accuracy - r.accuracy_lo], [r.accuracy_hi - r.accuracy]],
                    yerr=[[r.qwk - r.qwk_lo], [r.qwk_hi - r.qwk]], fmt=mk, ms=5.2 if key == "LesionCover" else 4.0,
                    color=col, mec=col, mew=0.4, elinewidth=0.7, capsize=0, zorder=4)
        rows.append({"model": lab, "kind": "retrained on DDR training", "accuracy": r.accuracy, "accuracy_lo": r.accuracy_lo,
                     "accuracy_hi": r.accuracy_hi, "qwk": r.qwk, "qwk_lo": r.qwk_lo, "qwk_hi": r.qwk_hi})
    for _, r in lit.iterrows():
        rows.append({"model": r.method, "kind": "reported", "accuracy": r.accuracy, "qwk": r.qwk})
    ax.set_xlabel("Accuracy, DDR test"); ax.set_ylabel("Quadratic weighted $\\kappa$, DDR test")
    ax.set_xlim(0.70, 0.92); ax.set_ylim(0.68, 0.95)
    for _, r in lt.iterrows():                      # 文献点标签：固定偏移，避开重训模型的误差线
        dx, dy, ha = LIT_POS.get(r.method, (0.004, 0, "left"))
        ax.text(r.accuracy + dx, r.qwk + dy, r.method, fontsize=5.4, color="#555555", ha=ha, va="center")
    h = [Line2D([], [], ls="", marker="o", mfc="white", mec="#777777", ms=4, label="Reported (other DDR studies)")]
    h += [Line2D([], [], ls="", marker=mk, color=col, ms=4, label=lab) for _, lab, col, mk in MODELS]
    ax.legend(handles=h, loc="upper left", frameon=False, fontsize=5.4, handletextpad=0.3, borderaxespad=0.1,
              labelspacing=0.3)
    return pd.DataFrame(rows)


def panel_d(ax, met):
    d = met[met.cohort == "ddr_test"].set_index("input")
    rows = []
    for i, (key, lab) in enumerate(ABL):
        r = d.loc[key]
        c = LC_COL if key == "LesionCover" else ("#E8A39B" if i >= 4 else "#9DB4D6")
        ax.barh(i, r.qwk, height=0.7, color=c, zorder=2)
        ax.errorbar(r.qwk, i, xerr=[[r.qwk - r.qwk_lo], [r.qwk_hi - r.qwk]], fmt="none", ecolor="#222222", elinewidth=0.6, zorder=3)
        rows.append({"setting": key, "label": lab, "encoder": "DINOv2-L" if i >= 4 else "ConvNeXt-T", "qwk": r.qwk,
                     "qwk_lo": r.qwk_lo, "qwk_hi": r.qwk_hi})
    ax.set_yticks(range(len(ABL))); ax.set_yticklabels([l for _, l in ABL], fontsize=5.8)
    ax.set_ylim(len(ABL) - 0.5, -0.5); ax.set_xlim(0.85, 0.94)
    ax.axhline(3.5, color="#999999", lw=0.5, ls="--")
    ax.text(0.938, 1.5, "ConvNeXt-T", ha="right", va="center", fontsize=5.8, color="#4C72B0")
    ax.text(0.938, 4.6, "DINOv2-L", ha="right", va="center", fontsize=5.8, color=LC_COL)
    ax.set_xlabel("Quadratic weighted $\\kappa$, DDR test (95% CI)")
    return pd.DataFrame(rows)


def main():
    apply_style()
    met = pd.read_csv(R / "08s_lcnet_eval/metrics.tsv", sep="\t")
    lit = pd.read_csv(ROOT / "metadata/literature_ddr.tsv", sep="\t")
    fig = plt.figure(figsize=(W2, 140 / 25.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.05], height_ratios=[1, 1.1])
    axa, axb, axc, axd = (fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2))
    src = {"a": panel_a(axa), "b": panel_b(axb), "c": panel_c(axc, met, lit), "d": panel_d(axd, met)}
    for ax, k in ((axa, "a"), (axb, "b"), (axc, "c"), (axd, "d")):
        panel_label(ax, k)
    for k, df in src.items():
        save_source(df, ROOT, "Fig2", k)
    print(save_fig(fig, ROOT, "Fig2"))


if __name__ == "__main__":
    main()
