"""Fig. 6：编码器容量与训练相机多样性。全部为三个模型集成的统计量，区间为两层 bootstrap（评价单位 + 模型）。
a、b 病灶覆盖与全局 512 的 κ 之差（C − G512）按编码器（ConvNeXt-Tiny / Small / Base）与队列；a 只用 DDR 训练，b 加入 APTOS。
c 融合模型（正式输出）把无 DR 的照片（DeepDRiD 为眼）判为 DR 的比例：病灶覆盖（柱）与全局 512（空心点），按训练设置与外部队列。
d 辅助头只看病灶覆盖裁块时同一比例（机制分析），按训练设置与外部队列。
只读 results/08j_capacity/{contrasts, clinical, healthy_crop}.tsv。"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from figstyle import W2, apply_style, panel_label, save_fig, save_source  # noqa: E402

COH_ALL = [("ddr_test", "DDR test"), ("deepdrid", "DeepDRiD\n(eyes)"), ("idrid", "IDRiD"), ("messidor2", "Messidor-2"),
           ("eyepacs", "EyePACS")]
ENC_ALL = [("tiny", "ConvNeXt-Tiny"), ("small", "ConvNeXt-Small"), ("base", "ConvNeXt-Base")]
ENC_COL = {"tiny": "#F4A582", "small": "#D6604D", "base": "#67001F"}
TRAIN = [("ddr", "DDR only"), ("multicamera", "DDR + APTOS")]


def present(df, coh):
    return [(c, l) for c, l in coh if (df.cohort == c).any()]


def advantage_panel(ax, con, train, encs, cohs, ylim):
    t = con[(con.training == train) & (con.contrast == "C-G512")]
    w = 0.22
    for j, (enc, _lab) in enumerate(encs):
        s = t[t.encoder == enc].set_index("cohort")
        for i, (coh, _) in enumerate(cohs):
            if coh not in s.index:
                continue
            r = s.loc[coh]
            x = i + (j - (len(encs) - 1) / 2) * w
            ax.errorbar(x, r["diff"], yerr=[[r["diff"] - r.lo], [r.hi - r["diff"]]], fmt="o", ms=3.4, color=ENC_COL[enc],
                        elinewidth=0.8, capsize=1.5, zorder=3)
    ax.plot([-0.5, len(cohs) - 0.5], [0, 0], color="#444444", lw=0.6, zorder=1)
    ax.set_xticks(range(len(cohs))); ax.set_xticklabels([c for _, c in cohs])
    ax.set_xlim(-0.5, len(cohs) - 0.5); ax.set_ylim(*ylim)
    ax.set_ylabel("$\\kappa$, LesionCover crops\nminus global view")
    return t[t.encoder.isin([e for e, _ in encs])]


def rate_panel(ax, df, cohs, cells, value, lo, hi, marker_df=None):
    w = 0.8 / len(cells)
    for j, (tr, e) in enumerate(cells):
        s = df[(df.training == tr) & (df.encoder == e)].set_index("cohort")
        m = None if marker_df is None else marker_df[(marker_df.training == tr) & (marker_df.encoder == e)].set_index("cohort")
        for i, (coh, _) in enumerate(cohs):
            if coh not in s.index:
                continue
            r = s.loc[coh]
            x = i + (j - (len(cells) - 1) / 2) * w
            ax.bar(x, 100 * r[value], width=w * 0.9, color=ENC_COL[e], hatch="////" if tr == "multicamera" else None,
                   edgecolor="white", linewidth=0.3,
                   yerr=[[100 * (r[value] - r[lo])], [100 * (r[hi] - r[value])]], error_kw={"elinewidth": 0.5, "capsize": 0.8})
            if m is not None and coh in m.index:
                ax.plot(x, 100 * m.loc[coh, "value"], marker="o", ms=2.6, mfc="white", mec="#222222", mew=0.6, ls="", zorder=4)
    ax.set_xticks(range(len(cohs))); ax.set_xticklabels([c for _, c in cohs]); ax.set_xlim(-0.5, len(cohs) - 0.5)


def main():
    apply_style(7)
    d = ROOT / "results/08j_capacity"
    con = pd.read_csv(d / "contrasts.tsv", sep="\t")
    clin = pd.read_csv(d / "clinical.tsv", sep="\t")
    hc = pd.read_csv(d / "healthy_crop.tsv", sep="\t")
    encs = [(e, lab) for e, lab in ENC_ALL if (con.encoder == e).any()]
    cohs = present(con, COH_ALL)
    ext = [c for c in cohs if c[0] != "ddr_test"]
    t = con[con.contrast == "C-G512"]
    ylim = (min(-0.11, t.lo.min() - 0.01), max(0.08, t.hi.max()) + 0.085)

    fig = plt.figure(figsize=(W2, 0.66 * W2))
    gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.3, left=0.085, right=0.985, top=0.93, bottom=0.1)
    for k, (train, title) in enumerate(TRAIN):
        ax = fig.add_subplot(gs[0, k])
        src = advantage_panel(ax, con, train, encs, cohs, ylim)
        ax.set_title(f"Trained on {title}", fontsize=7, pad=3)
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color=ENC_COL[e], ms=3.4, label=lab) for e, lab in encs],
                  fontsize=6, frameon=False, loc="upper center", ncol=len(encs), handletextpad=0.2, columnspacing=0.8)
        save_source(src, ROOT, "Fig6", f"{'ab'[k]}_advantage_{train}")
        panel_label(ax, "ab"[k])

    cells = [(tr, e) for tr, _ in TRAIN for e, _ in encs]
    fp = clin[(clin.stat == "healthy_fp") & (clin.input == "C") & clin.cohort.isin([c for c, _ in ext])]
    fpg = clin[(clin.stat == "healthy_fp") & (clin.input == "G512") & clin.cohort.isin([c for c, _ in ext])]
    hcx = hc[hc.cohort.isin([c for c, _ in ext])]
    top = 100 * max(fp.hi.max(), hcx.hi.max(), fpg.value.max())
    ymax = 10 * int(top / 0.74 / 10 + 1)
    legend = ([Patch(facecolor=ENC_COL[e], label=lab) for e, lab in encs] +
              [Patch(facecolor="white", edgecolor="#444444", hatch="////", label="DDR + APTOS")])
    axc = fig.add_subplot(gs[1, 0])
    rate_panel(axc, fp, ext, cells, "value", "lo", "hi", marker_df=fpg)
    axc.set_ylabel("No DR graded as DR (%)\nglobal view with crops"); axc.set_ylim(0, ymax)
    axc.legend(handles=legend + [Line2D([], [], marker="o", ls="", mfc="white", mec="#222222", ms=2.6, label="Global view")],
               fontsize=5.6, frameon=False, loc="upper left", ncol=3, columnspacing=0.6, handlelength=1.2)
    save_source(pd.concat([fp, fpg]), ROOT, "Fig6", "c_healthy_final_grader")
    panel_label(axc, "c")

    axd = fig.add_subplot(gs[1, 1])
    rate_panel(axd, hcx.rename(columns={"called_dr": "value"}), ext, cells, "value", "lo", "hi")
    axd.set_ylabel("No DR graded as DR (%)\nfrom LesionCover crops alone"); axd.set_ylim(0, ymax)
    axd.legend(handles=legend, fontsize=5.6, frameon=False, loc="upper left", ncol=2, columnspacing=0.6, handlelength=1.2)
    save_source(hcx, ROOT, "Fig6", "d_healthy_crops_alone")
    panel_label(axd, "d")
    save_fig(fig, ROOT, "Fig6")


if __name__ == "__main__":
    main()
