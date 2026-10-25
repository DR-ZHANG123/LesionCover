"""Supplementary Fig. S1：病灶分割器与选择参数。
a 开发区掩膜逐类像素 AUPR 随训练轮次；b 开发区掩膜上 NMS 阈值与 DPP θ 对参考病灶覆盖（四类平均）的影响，
虚线为病灶覆盖规则（无此参数）。只读 results/03_segmenter、05_selection、06_coverage。"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from figstyle import LESION_COLORS, LESION_LABELS, METHOD_COLORS, METHOD_LABELS, W2, apply_style, panel_label, save_fig, save_source  # noqa: E402


def main():
    apply_style(7)
    log = pd.read_csv(ROOT / "results/03_segmenter/train_log.tsv", sep="\t").dropna(subset=["dev_aupr_mean"])
    tun = pd.read_csv(ROOT / "results/05_selection/tuning.tsv", sep="\t")
    cov = pd.read_csv(ROOT / "results/06_coverage/summary.tsv", sep="\t")
    c_dev = cov[(cov.mask_role == "seg_dev") & (cov.method == "C_k4")].cov_macro.iloc[0]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(W2, 0.32 * W2), gridspec_kw={"wspace": 0.45})
    for les in ["MA", "HE", "EX", "SE"]:
        a.plot(log.epoch + 1, log[f"dev_aupr_{les}"], "-o", ms=2.5, lw=0.9, color=LESION_COLORS[les], label=LESION_LABELS[les])
    a.plot(log.epoch + 1, log.dev_aupr_mean, "-", lw=1.4, color="#222222", label="Mean")
    a.set_xlabel("Epoch"); a.set_ylabel("Pixel AUPR, development masks"); a.set_ylim(0, 0.8)
    a.legend(fontsize=5.6, frameon=False, ncol=2, loc="upper left", handlelength=1.2, columnspacing=0.6)
    save_source(log[["epoch", "dev_aupr_MA", "dev_aupr_HE", "dev_aupr_EX", "dev_aupr_SE", "dev_aupr_mean"]], ROOT, "FigS1", "a_aupr")
    panel_label(a, "a")
    for ax, m, xl in ((b, "N", "NMS overlap threshold (IoU)"), (c, "D", "DPP quality–diversity trade-off θ")):
        t = tun[tun.method == m]
        ax.plot(t.value, t.dev_macro_coverage * 100, "-o", ms=3, color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        ax.axhline(c_dev * 100, ls="--", lw=0.9, color=METHOD_COLORS["C"], label=METHOD_LABELS["C"])
        best = t.loc[t.dev_macro_coverage.idxmax()]
        ax.scatter([best.value], [best.dev_macro_coverage * 100], s=30, facecolor="none", edgecolor="#222222", lw=0.8, zorder=4)
        ax.set_xlabel(xl); ax.set_ylabel("Lesions retained, mean of types (%)")
        ax.set_ylim(30, 70)
        ax.legend(fontsize=5.6, frameon=False, loc="lower left")
        save_source(t.assign(coverage_rule_dev=c_dev), ROOT, "FigS1", f"{'b' if m == 'N' else 'c'}_tuning_{m}")
    panel_label(b, "b"); panel_label(c, "c")
    save_fig(fig, ROOT, "FigS1")


if __name__ == "__main__":
    main()
