"""Fig. S3：含与不含参考病灶的裁块所得注意力权重（DDR 测试区中两类裁块都有的照片；每种规则三个模型的均值与标准差）。
只读 results/08b_attention/summary.tsv。"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from figstyle import METHOD_COLORS, W15, apply_style, save_fig, save_source  # noqa: E402

SHORT = {"U": "Quadrants", "R": "Random", "T": "Heatmap", "N": "NMS", "D": "DPP", "C": "Coverage"}


def main():
    apply_style(7)
    fig, ax = plt.subplots(figsize=(W15, 0.55 * W15))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.13)
    at = pd.read_csv(ROOT / "results/08b_attention/summary.tsv", sep="\t")
    rules = ["U", "R", "T", "N", "D", "C"]
    rows = []
    for i, r in enumerate(rules):
        a = at[at.method == f"{r}_k4"]
        for j, (col, face) in enumerate((("attention_with_lesion", METHOD_COLORS[r]), ("attention_without_lesion", "white"))):
            ax.bar(i + (j - 0.5) * 0.36, a[col].mean(), width=0.34, yerr=a[col].std(), color=face,
                   edgecolor=METHOD_COLORS[r], lw=0.8, error_kw={"elinewidth": 0.6, "capsize": 1.2})
        rows += [{"rule": r, "seed": int(s), "with_lesion": w1, "without_lesion": w0, "n_images": n}
                 for s, w1, w0, n in zip(a.seed, a.attention_with_lesion, a.attention_without_lesion, a.n_images_mixed)]
    ax.set_xticks(range(len(rules))); ax.set_xticklabels([SHORT[r] for r in rules])
    ax.set_ylabel("Mean attention weight per crop"); ax.set_ylim(0, 0.72)
    ax.legend(handles=[Patch(fc="#777777", ec="#777777", label="Crop with annotated lesion"),
                       Patch(fc="white", ec="#777777", label="Crop without lesion")], fontsize=6, frameon=False,
              loc="upper left", ncol=2)
    save_source(pd.DataFrame(rows), ROOT, "FigS3", "attention")
    save_fig(fig, ROOT, "FigS3")


if __name__ == "__main__":
    main()
