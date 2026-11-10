"""补充图 S1–S3（只读结果表）。
S2 病灶尺子：a 预测与人工掩膜分量数（IDRiD，按病灶类别）；b 同上，DDR 测试区；c MMRDR 逐图病灶有无的敏感度与特异度
S1 只用规则特征的有序分级在各队列的 κ（两种编码器，三个模型平均），以及学到的四个切点
S3 输出层校准方法比较：五个外部队列 κ 平均，按模型与每级校准照片数"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from cohorts import COH_COLOR, COH_LABEL, COHORTS, LESIONS, LES_COLOR, MODEL_COLOR, MODEL_LABEL  # noqa: E402
from figstyle import W2, apply_style, panel_label, save_fig, save_source  # noqa: E402

R = ROOT / "results"


def fig_ruler():
    d = pd.read_csv(R / "08w_ruler_check/ruler_per_image.tsv", sep="\t")
    rc = pd.read_csv(R / "08w_ruler_check/ruler_counts.tsv", sep="\t")
    fig = plt.figure(figsize=(W2, 105 / 25.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.3])
    src = []
    for r, g in enumerate(("IDRiD", "DDR test")):
        x = d[d.group == g]
        for i, (l, name) in enumerate(LESIONS):
            ax = fig.add_subplot(gs[r, i])
            h, p = x[f"human_{l}"].to_numpy(), x[f"pred_{l}"].to_numpy()
            ax.scatter(h + 1, p + 1, s=5, color=LES_COLOR[l], alpha=0.6, lw=0)
            lim = max(h.max(), p.max()) + 2
            ax.plot([1, lim], [1, lim], color="#999999", lw=0.6, ls="--")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.8, lim * 1.2); ax.set_ylim(0.8, lim * 1.2)
            s = rc[(rc.group == g) & (rc.lesion == l)].iloc[0]
            ax.set_title(f"{name}\nρ = {s.spearman:.2f}", fontsize=6.2)
            ax.tick_params(labelsize=5.6)
            if i == 0:
                ax.set_ylabel(f"{g}\npredicted count + 1", fontsize=6.2)
                panel_label(ax, "ab"[r])
            if r == 1:
                ax.set_xlabel("annotated count + 1", fontsize=6.2)
            src.append(x[["group", "image_key", f"human_{l}", f"pred_{l}"]].rename(columns={f"human_{l}": "human", f"pred_{l}": "pred"})
                       .assign(lesion=l))
    ax = fig.add_subplot(gs[:, 4])
    pr = pd.read_csv(R / "08w_ruler_check/ruler_presence_mmrdr.tsv", sep="\t")
    w = 0.38
    for j, (st, col) in enumerate((("sensitivity", "#C0392B"), ("specificity", "#4C72B0"))):
        ax.bar(np.arange(len(pr)) + (j - 0.5) * w, 100 * pr[st], width=w * 0.92, color=col, label=st.capitalize())
    ax.set_xticks(range(len(pr))); ax.set_xticklabels(pr.lesion)
    ax.set_ylim(0, 112); ax.set_ylabel("MMRDR, lesion present vs\nper-photograph label (%)")
    ax.legend(loc="upper right", frameon=False, fontsize=5.6, ncol=1)
    panel_label(ax, "c")
    save_source(pd.concat(src), ROOT, "FigS2", "ab"); save_source(pr, ROOT, "FigS2", "c")
    print(save_fig(fig, ROOT, "FigS2"))


def fig_rule_only():
    ro = pd.read_csv(R / "08x_learned_rule/rule_only.tsv", sep="\t")
    cp = pd.read_csv(R / "08x_learned_rule/cutpoints.tsv", sep="\t")
    met = pd.read_csv(R / "08s_lcnet_eval/metrics.tsv", sep="\t")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 62 / 25.4), constrained_layout=True, gridspec_kw={"width_ratios": [1.5, 1]})
    cohs = [c for c, _ in COHORTS]; w = 0.27; rows = []
    for j, (lab, col) in enumerate((("Rule only, DINOv2-L models", "#E6AB02"), ("Rule only, ConvNeXt-T models", "#F3D27A"),
                                    ("LesionCover (full network)", "#C0392B"))):
        for i, c in enumerate(cohs):
            if j < 2:
                v = ro[(ro.encoder == ("DINOv2-L" if j == 0 else "ConvNeXt-T")) & (ro.cohort == c)].qwk.iloc[0]
            else:
                v = met[(met.input == "LesionCover") & (met.cohort == c)].qwk.iloc[0]
            a.bar(i + (j - 1) * w, v, width=w * 0.92, color=col, label=lab if i == 0 else None)
            rows.append({"setting": lab, "cohort": c, "qwk": v})
    a.set_xticks(range(len(cohs))); a.set_xticklabels([COH_LABEL[c] for c in cohs], fontsize=6, rotation=25, ha="right")
    a.set_ylabel("Quadratic weighted $\\kappa$"); a.set_ylim(0, 1.12)
    a.legend(loc="upper left", frameon=False, fontsize=5.6, ncol=3, bbox_to_anchor=(0, 1.04), columnspacing=0.8)
    panel_label(a, "a")
    for e, (enc, mk) in enumerate((("DINOv2-L", "o"), ("ConvNeXt-T", "s"))):
        d = cp[cp.encoder == enc]
        for k in range(4):
            v = d[f"cut_{k + 1}"].to_numpy()
            b.scatter(v, np.full(len(v), k) + (e - 0.5) * 0.25, marker=mk, s=12, color="#8A6100",
                      facecolor="#8A6100" if e == 0 else "white", lw=0.7)
    b.set_yticks(range(4)); b.set_yticklabels(["No DR | DR", "Mild | moderate", "Moderate | severe", "Severe | PDR"])
    b.set_ylim(3.6, -0.6); b.set_xlabel("Learned cut point on the rule score")
    b.legend(handles=[Line2D([], [], ls="", marker="o", color="#8A6100", ms=3.4, label="DINOv2-L"),
                      Line2D([], [], ls="", marker="s", mfc="white", mec="#8A6100", ms=3.4, label="ConvNeXt-T")],
             loc="upper right", frameon=False, fontsize=5.6)
    panel_label(b, "b")
    save_source(pd.DataFrame(rows), ROOT, "FigS1", "a"); save_source(cp, ROOT, "FigS1", "b")
    print(save_fig(fig, ROOT, "FigS1"))


def fig_calibration():
    m = pd.read_csv(R / "08v_local_calibration/metrics.tsv", sep="\t"); m = m[m.scheme == "balanced"]
    keys = ["LesionCover", "DINOv2 global tokens only", "Global 1024", "Swin-T 1024", "EfficientNetV2-M 1024", "ResNet-50 1024",
            "GDRNet", "CABNet"]
    meths = [("none", "Trained on DDR", "#BBBBBB"), ("prior_em", "Label-free prior adjustment", "#8172B3"),
             ("cutpoints", "Cut points on expected grade", "#55A868"), ("temp_only", "Temperature only", "#E6B85C"),
             ("bias_only", "Grade intercepts only", "#4C72B0"), ("bias_temp", "Temperature + grade intercepts", "#C0392B")]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 66 / 25.4), constrained_layout=True, sharey=True)
    rows = []
    for ax, k in zip(axs, (5, 20)):
        for j, (mt, lab, col) in enumerate(meths):
            for i, key in enumerate(keys):
                v = m[(m.input == key) & (m.method == mt) & (m.per_grade == k)].qwk.mean()
                ax.bar(i + (j - (len(meths) - 1) / 2) * 0.135, v, width=0.13, color=col, label=lab if i == 0 else None)
                rows.append({"per_grade": k, "method": mt, "model": MODEL_LABEL[key], "mean_external_qwk": v})
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([MODEL_LABEL[x] for x in keys], rotation=35, ha="right", fontsize=5.8)
        ax.set_ylim(0.3, 0.86); ax.set_title(f"{k} labelled photographs per grade", fontsize=6.6)
    axs[0].set_ylabel("Mean $\\kappa$, five external cohorts")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False, fontsize=5.8)
    panel_label(axs[0], "a"); panel_label(axs[1], "b")
    save_source(pd.DataFrame(rows), ROOT, "FigS3", "ab")
    print(save_fig(fig, ROOT, "FigS3"))


if __name__ == "__main__":
    apply_style()
    fig_rule_only(); fig_ruler(); fig_calibration()
