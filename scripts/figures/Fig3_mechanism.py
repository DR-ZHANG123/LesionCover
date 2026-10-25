"""Fig. 3：进入分级器的微动脉瘤决定轻度 DR 是否被识别。
a 裁块内预测微动脉瘤数与轻度照片被判为无 DR 的比例（六种规则汇总，整体输入与只看裁块两种预测；照片 bootstrap 区间）；
b 固定的病灶覆盖分级器（三个模型集成）：裁块原样、去除裁块内微动脉瘤、在同一裁块无病灶处去除同样形状的区域时，
  轻度照片被判为无 DR 的比例（08k）；c 示例：DDR 测试区一张只有微动脉瘤的轻度照片（选择规则见 08f），病灶覆盖裁块（彩色框）
  与固定象限（白色虚线），参考微动脉瘤以圆圈标出；d 四个病灶覆盖裁块的放大图与注意力权重。
只读 results/08e_dose_response、08f_cases、08k_occlusion 及示例照片的 canon 图与参考掩膜。"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Patch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from figstyle import METHOD_COLORS, W2, apply_style, panel_label, save_fig, save_source  # noqa: E402

SHORT = {"U": "Quadrants", "R": "Random", "T": "Heatmap", "N": "NMS", "D": "DPP", "C": "Coverage"}
BINS = ["0", "1", "2", "3-4", "≥5"]
CROP = 384


def ma_components(mask: np.ndarray):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return cen[1:]


def main():
    apply_style(7)
    case = pd.read_csv(ROOT / "results/08f_cases/cases.tsv", sep="\t").set_index("case").loc["internal_mild"]
    key = case.image_key
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False).set_index("image_key")
    img = cv2.imread(str(ROOT / "data/processed/canon" / f"{key}.jpg"))[:, :, ::-1]
    mask = np.load(ROOT / "data/processed/canon_masks" / f"{man.loc[key, 'seg_key']}.npy")[0]
    cen = ma_components(mask)
    crops_c = np.asarray(ast.literal_eval(case.crops_C)); crops_u = np.asarray(ast.literal_eval(case.crops_U))
    att = np.array([case[f"attn{j}"] for j in range(4)])

    fig = plt.figure(figsize=(W2, 0.70 * W2))
    fig._col_tol = 0.2
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1, 1], hspace=0.55, wspace=0.22,
                          left=0.07, right=0.985, top=0.95, bottom=0.10)
    axa = fig.add_subplot(gs[1, 0])
    f = 600 / img.shape[0]
    axa.imshow(cv2.resize(img, (600, 600), interpolation=cv2.INTER_AREA))
    for x, y in crops_u:
        axa.add_patch(Rectangle((x * f, y * f), CROP * f, CROP * f, fill=False, lw=0.7, ec="white", ls=(0, (3, 2))))
    for x, y in crops_c:
        axa.add_patch(Rectangle((x * f, y * f), CROP * f, CROP * f, fill=False, lw=1.0, ec=METHOD_COLORS["C"]))
    for x, y in cen:
        axa.add_patch(Circle((x * f, y * f), 6, fill=False, lw=0.6, ec="#7CFC00"))
    axa.set_xticks([]); axa.set_yticks([])
    axa.set_title(f"Reference grade: mild. Global 512 view: no DR (P = {case.global_p0:.2f});\n"
                  f"with coverage crops: mild (P = {case.cover_p1:.2f})", fontsize=6.3, pad=3)
    panel_label(axa, "c")

    axb = fig.add_subplot(gs[1, 1])
    tiles = []
    for (x, y) in crops_c:
        t = img[y:y + CROP, x:x + CROP].copy()
        pad = np.zeros((CROP, CROP, 3), np.uint8)
        y0, x0 = max(0, -y), max(0, -x)
        pad[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
        tiles.append(pad)
    gap = 12
    mos = np.full((2 * CROP + gap, 2 * CROP + gap, 3), 255, np.uint8)
    offs = [(0, 0), (CROP + gap, 0), (0, CROP + gap), (CROP + gap, CROP + gap)]
    for t, (ox, oy) in zip(tiles, offs):
        mos[oy:oy + CROP, ox:ox + CROP] = t
    axb.imshow(mos)
    rows = []
    for j, ((x, y), (ox, oy)) in enumerate(zip(crops_c, offs)):
        inside = [(cx - x, cy - y) for cx, cy in cen if x <= cx < x + CROP and y <= cy < y + CROP]
        for cx, cy in inside:
            axb.add_patch(Circle((ox + cx, oy + cy), 11, fill=False, lw=0.6, ec="#7CFC00"))
        axb.add_patch(Rectangle((ox, oy), CROP, CROP, fill=False, lw=1.0, ec=METHOD_COLORS["C"]))
        axb.text(ox + 10, oy + 14, f"attention {att[j]:.2f}", color="white", fontsize=6, va="top",
                 bbox={"boxstyle": "round,pad=0.15", "fc": "black", "ec": "none", "alpha": 0.55})
        rows.append({"crop": j, "x0": x, "y0": y, "attention": att[j], "annotated_MA_inside": len(inside)})
    axb.set_xticks([]); axb.set_yticks([]); axb.axis("off")
    axb.set_title("Coverage crops at full resolution (circles, annotated microaneurysms)", fontsize=6.3, pad=3)
    save_source(pd.DataFrame(rows).assign(image_key=key, global_p0=case.global_p0, cover_p1=case.cover_p1), ROOT,
                "Fig3", "cd_case")
    panel_label(axb, "d")

    axc = fig.add_subplot(gs[0, 0])
    b = pd.read_csv(ROOT / "results/08e_dose_response/bins.tsv", sep="\t")
    b = b[b.rule == "pooled"]
    for kind, lab, col, mk, dx in (("full", "Global view with crops", "#333333", "o", -0.07),
                                   ("croponly", "Crops alone", METHOD_COLORS["C"], "s", 0.07)):
        t = b[b.kind == kind].set_index("bin").loc[BINS]
        x = np.arange(len(BINS)) + dx
        axc.errorbar(x, t.none_pct, yerr=[t.none_pct - t.lo, t.hi - t.none_pct], fmt=f"{mk}-", ms=3.5, lw=0.9,
                     color=col, elinewidth=0.6, capsize=1.5, label=lab)
    axc.set_xticks(range(len(BINS))); axc.set_xticklabels(BINS)
    axc.set_xlabel("Predicted microaneurysms inside the four crops")
    axc.set_ylabel("Mild DR graded as no DR (%)\nDDR test")
    axc.set_ylim(0, 75)
    axc.legend(fontsize=6, frameon=False, loc="upper right")
    save_source(b, ROOT, "Fig3", "a_dose_response"); panel_label(axc, "a")

    axd = fig.add_subplot(gs[0, 1])                # b：固定分级器，去除裁块内的微动脉瘤（与匹配对照）
    oc = pd.read_csv(ROOT / "results/08k_occlusion/summary.tsv", sep="\t")
    oc = oc[oc.grade_group == "mild DR"]
    groups = [("predicted", "1-4", "1–4 predicted\nmicroaneurysms"), ("predicted", "5+", "≥5 predicted\nmicroaneurysms"),
              ("annotated", "all", "Annotated\nmicroaneurysms")]
    conds = [("none_intact", "Intact crops", "#555555", None), ("none_occ", "Microaneurysms removed", METHOD_COLORS["C"], None),
             ("none_ctrl", "Same patches elsewhere", "white", "////")]
    rows = []
    for i, (occ, b_, lab) in enumerate(groups):
        r = oc[(oc.occluded == occ) & (oc.ma_in_crops == b_)].iloc[0]
        for j, (col, clab, face, hatch) in enumerate(conds):
            v, lo, hi = 100 * r[col], 100 * r[f"{col}_lo"], 100 * r[f"{col}_hi"]
            axd.bar(i + (j - 1) * 0.26, v, width=0.24, color=face, hatch=hatch, edgecolor="#555555" if hatch else face,
                    lw=0.6, yerr=[[v - lo], [hi - v]], error_kw={"elinewidth": 0.6, "capsize": 1.2})
            rows.append({"microaneurysms": occ, "in_crops": b_, "n": int(r.n), "condition": clab, "pct": v, "lo": lo, "hi": hi})
    axd.set_xticks(range(len(groups)))
    axd.set_xticklabels([f"{lab}\n(n = {int(oc[(oc.occluded == o) & (oc.ma_in_crops == b_)].n.iloc[0])})"
                         for o, b_, lab in groups])
    axd.set_ylabel("Mild DR graded as no DR (%)\nDDR test, fixed grader")
    axd.set_ylim(0, 118); axd.set_yticks(range(0, 101, 20))
    axd.legend(handles=[Patch(fc=f, ec="#555555" if h else f, hatch=h, label=l) for _, l, f, h in conds], fontsize=6,
               frameon=False, loc="upper center", ncol=3, handlelength=1.4, columnspacing=0.8)
    save_source(pd.DataFrame(rows), ROOT, "Fig3", "b_occlusion"); panel_label(axd, "b")
    save_fig(fig, ROOT, "Fig3")


if __name__ == "__main__":
    main()
