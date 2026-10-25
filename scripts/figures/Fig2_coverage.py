"""Fig. 2：参考病灶覆盖（DDR 测试区 185 张掩膜图像，原始分辨率）。
a 同一张图上 NMS、DPP 与病灶覆盖三种规则所选裁块及参考病灶（实心 = 被某一裁块完整保留）
b 各规则按病灶类型的覆盖率；c 病灶覆盖相对 NMS、DPP 的差值（95% bootstrap 区间）；d 裁块数 K 与微动脉瘤覆盖。
只读 results/06_coverage 与 results/05_selection；面板输入写入 figures/source_data/。"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from figstyle import (HMAX, LESION_COLORS, LESION_LABELS, METHOD_COLORS, METHOD_LABELS, W2, apply_style,  # noqa: E402
                      panel_label, save_fig, save_source)

LES = ["MA", "HE", "EX", "SE"]
RULES = ["U", "R", "T", "N", "D", "C"]
CROP = 384


def pick_example(comp: pd.DataFrame) -> str:
    """示例图：微动脉瘤 ≥ 15 个、病灶覆盖比 NMS 多保留的参考病灶最多的一张。"""
    t = comp[comp.mask_role == "seg_test"]
    piv = t.pivot_table(index="image_key", columns="method", values="covered", aggfunc="sum")
    nma = t[(t.method == "C_k4") & (t.lesion == "MA")].groupby("image_key").size()
    cand = piv.loc[nma[nma >= 15].index]
    gain = (cand["C_k4"] - cand[["N_k4", "D_k4"]].max(1)).sort_values(ascending=False)
    return gain.index[0]


def panel_example(axs, comp, crops, key):
    img = cv2.imread(str(ROOT / "data/processed/canon" / f"{key}.jpg"))[:, :, ::-1]
    f = 640 / img.shape[0]                       # 示意图用 640 像素底图，坐标同比缩放
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
    rows = []
    for ax, m in zip(axs, ["N", "D", "C"]):
        ax.imshow(img, interpolation="lanczos")
        c = comp[(comp.image_key == key) & (comp.method == f"{m}_k4")]
        for les in LES:
            g = c[c.lesion == les]
            ax.scatter(g.cx_canon[g.covered == 1] * f, g.cy_canon[g.covered == 1] * f, s=7, marker="o",
                       facecolor=LESION_COLORS[les], edgecolor="white", linewidth=0.3, zorder=4)
            ax.scatter(g.cx_canon[g.covered == 0] * f, g.cy_canon[g.covered == 0] * f, s=7, marker="o",
                       facecolor="none", edgecolor=LESION_COLORS[les], linewidth=0.6, zorder=4)
        cr = crops[(crops.image_key == key) & (crops.method == f"{m}_k4")].sort_values("slot")
        for x, y in cr[["x0", "y0"]].to_numpy():
            ax.add_patch(Rectangle((x * f, y * f), CROP * f, CROP * f, fill=False, lw=1.1, ec=METHOD_COLORS[m], zorder=5))
        n_cov, n_all = int(c.covered.sum()), len(c)
        ax.set_title(f"{METHOD_LABELS[m]}\n{n_cov} of {n_all} lesions retained", fontsize=7, pad=2)
        ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        rows.append(c.assign(rule=m)[["rule", "lesion", "cx_canon", "cy_canon", "covered"]])
        rows.append(cr.assign(rule=m, lesion="crop")[["rule", "lesion", "x0", "y0"]])
    return pd.concat(rows)


def main():
    apply_style(7)
    comp = pd.read_csv(ROOT / "results/06_coverage/components.tsv.gz", sep="\t")
    summ = pd.read_csv(ROOT / "results/06_coverage/summary.tsv", sep="\t")
    con = pd.read_csv(ROOT / "results/06_coverage/contrasts.tsv", sep="\t")
    crops = pd.read_csv(ROOT / "results/05_selection/crops.tsv.gz", sep="\t")
    key = pick_example(comp)

    fig = plt.figure(figsize=(W2, 0.66 * W2))
    fig._col_tol = 0.12
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 0.10, 0.95], hspace=0.30, wspace=0.42,
                          left=0.065, right=0.99, top=0.95, bottom=0.085)
    gtop = gs[0, :].subgridspec(1, 3, wspace=0.04)
    axa = [fig.add_subplot(gtop[0, i]) for i in range(3)]
    src_a = panel_example(axa, comp, crops, key)
    save_source(src_a.assign(image_key=key), ROOT, "Fig2", "a_example")
    handles = [Line2D([], [], ls="", marker="o", ms=3.2, mfc=LESION_COLORS[l], mec="white", mew=0.3,
                      label=LESION_LABELS[l]) for l in LES]
    handles += [Line2D([], [], ls="", marker="o", ms=3.2, mfc="none", mec="#555555", label="Not retained")]
    axa[2].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=6,
                  handletextpad=0.2, borderaxespad=0.1)
    panel_label(axa[0], "a")

    def bars(ax, role, src_name):
        t = summ[(summ.mask_role == role) & summ.method.isin([f"{r}_k4" for r in RULES])].copy()
        t["rule"] = t.method.str.replace("_k4", "", regex=False)
        w, rows = 0.13, []
        for i, r in enumerate(RULES):
            v = t[t.rule == r].iloc[0]
            ys = [v[f"cov_{l}"] * 100 for l in LES]
            ax.bar(np.arange(4) + (i - 2.5) * w, ys, width=w, color=METHOD_COLORS[r], edgecolor="none")
            rows += [{"rule": r, "lesion": l, "coverage_pct": y} for l, y in zip(LES, ys)]
        save_source(pd.DataFrame(rows), ROOT, "Fig2", src_name)
        ax.set_xticks(range(4)); ax.set_xticklabels(["MA", "HE", "EX", "SE"])
        ax.set_ylim(0, 100)

    lax = fig.add_subplot(gs[1, :]); lax.axis("off")
    lax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[r]) for r in RULES],
               labels=[METHOD_LABELS[r] for r in RULES], loc="center", ncol=6, frameon=False, fontsize=6.5,
               handlelength=1.2, columnspacing=1.4)
    axb = fig.add_subplot(gs[2, 0])
    bars(axb, "seg_test", "b_coverage_ddr")
    axb.set_ylabel("Annotated lesions retained (%)"); axb.set_title("DDR test (185)", fontsize=7, pad=3)
    panel_label(axb, "b")
    axc = fig.add_subplot(gs[2, 1])
    bars(axc, "ext_seg", "c_coverage_idrid")
    axc.set_title("IDRiD, external (81)", fontsize=7, pad=3)
    panel_label(axc, "c")

    # d 差值森林图（两个队列）
    axd = fig.add_subplot(gs[2, 2])
    labs = ["MA", "HE", "EX", "SE", "macro"]
    names = ["MA", "HE", "EX", "SE", "Mean"]
    rows = []
    offs = {("seg_test", "N"): -0.27, ("ext_seg", "N"): -0.09, ("seg_test", "D"): 0.09, ("ext_seg", "D"): 0.27}
    for (role, alt), off in offs.items():
        c = con[(con.contrast == f"C-{alt}") & (con.small_only == 0) & (con.mask_role == role)].set_index("lesion").loc[labs]
        y = np.arange(len(labs))[::-1] + off
        axd.errorbar(c["diff"] * 100, y, xerr=[(c["diff"] - c.lo) * 100, (c.hi - c["diff"]) * 100],
                     fmt="o" if role == "seg_test" else "s", ms=2.6, color=METHOD_COLORS[alt],
                     mfc=METHOD_COLORS[alt] if role == "seg_test" else "white", elinewidth=0.7, capsize=1.2)
        rows += [{"cohort": role, "comparator": alt, "lesion": l, "diff_pct": d * 100, "lo_pct": lo * 100, "hi_pct": hi * 100}
                 for l, d, lo, hi in zip(labs, c["diff"], c.lo, c.hi)]
    save_source(pd.DataFrame(rows), ROOT, "Fig2", "d_differences")
    axd.plot([0, 0], [-0.6, len(labs) - 1 + 0.42], color="#888888", lw=0.6, ls="--")   # 零线止于图例带之下
    axd.set_yticks(range(len(labs))[::-1]); axd.set_yticklabels(names)
    axd.set_xlabel("Coverage rule minus\nNMS or DPP (points)")
    axd.legend(handles=[Line2D([], [], ls="", marker="o", ms=3, mfc="#555555", mec="#555555", label="DDR"),
                        Line2D([], [], ls="", marker="s", ms=3, mfc="white", mec="#555555", label="IDRiD")],
               fontsize=5.6, frameon=False, loc="upper right", ncol=2, handletextpad=0.2, columnspacing=0.8,
               borderaxespad=0.1)
    axd.set_ylim(-0.6, len(labs) - 1 + 1.15)            # 顶部留一行给图例，不压误差线
    panel_label(axd, "d")

    # e 裁块数
    axe = fig.add_subplot(gs[2, 3])
    rows = []
    for r in ["U", "T", "N", "D", "C"]:
        for role, ls in (("seg_test", "-"), ("ext_seg", "--")):
            ks, ys = [], []
            for k in (2, 4, 8):
                v = summ[(summ.mask_role == role) & (summ.method == f"{r}_k{k}")]
                if len(v):
                    ks.append(k); ys.append(v.cov_MA.iloc[0] * 100)
            axe.plot(ks, ys, ls, marker="o", ms=2.4, lw=0.9, color=METHOD_COLORS[r])
            rows += [{"rule": r, "cohort": role, "k": k, "ma_coverage_pct": y} for k, y in zip(ks, ys)]
    save_source(pd.DataFrame(rows), ROOT, "Fig2", "e_budget")
    axe.set_xticks([2, 4, 8]); axe.set_xlabel("Number of crops (K)")
    axe.set_ylabel("Microaneurysms retained (%)"); axe.set_ylim(0, 100)
    axe.legend(handles=[Line2D([], [], ls="-", color="#555555", label="DDR"),
                        Line2D([], [], ls="--", color="#555555", label="IDRiD")], fontsize=5.6, frameon=False,
               loc="upper left", borderaxespad=0.1, handlelength=1.6)
    panel_label(axe, "e")
    save_fig(fig, ROOT, "Fig2")


if __name__ == "__main__":
    main()
