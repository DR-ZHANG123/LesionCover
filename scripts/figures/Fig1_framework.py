"""Fig. 1：方法与研究设计框架（矢量图；照片、病灶概率与裁块为真实数据；示意图标见 vector_icons.py：
Health Icons（MIT 许可证）与代码绘制的网络、橡皮擦）。
a 眼底照片 → 病灶分割器 → 校准后的四类病灶概率 → 在共享候选窗口中选出 K=4 个裁块；
b 分级网络：全局图与裁块共享 ConvNeXt 编码，门控注意力汇聚裁块，与全局特征拼接后输出五级概率；
c 选择原理：按像素显著性排序时窗口堆在最强病灶上；按病灶分量覆盖时每个窗口只为新带入的病灶得分；
d 研究设计：DDR 开发队列与四个外部队列（人数取自 results/02_manifest），以及四项分析。
示例图与 Fig. 2a 相同。整张图在毫米坐标系中排版，保证元素不重叠、对齐。"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
sys.path.insert(0, str(ROOT))
from figstyle import LESION_COLORS, LESION_LABELS, METHOD_COLORS, MM, apply_style, save_fig, save_source  # noqa: E402
from Fig2_coverage import pick_example  # noqa: E402
from vector_icons import draw_icon  # noqa: E402

W, H = 174.0, 157.0
CROP = 384
GREY = "#5A5A5A"


def box(ax, x, y, w, h, text, fs=6.5, fc="#F2F2F2"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", lw=0.7, ec=GREY, fc=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.9, color="#444444",
                                 shrinkA=0, shrinkB=0))


def image(ax, img, x, y, s, edge=GREY, lw=0.5):
    ax.imshow(img, extent=(x, x + s, y, y + s), interpolation="lanczos", zorder=1)
    ax.add_patch(Rectangle((x, y), s, s, fill=False, lw=lw, ec=edge, zorder=2))


def caption(ax, x, y, text, fs=6.8, bold=False):
    ax.text(x, y, text, ha="center", va="bottom", fontsize=fs, linespacing=1.2, fontweight="bold" if bold else "normal")


def main():
    apply_style(7)
    from scripts.lib.calib import calibrated_map, load_calibration
    from scripts.lib.repro import load_config
    comp = pd.read_csv(ROOT / "results/06_coverage/components.tsv.gz", sep="\t")
    crops = pd.read_csv(ROOT / "results/05_selection/crops.tsv.gz", sep="\t")
    key = pick_example(comp)
    canon = cv2.imread(str(ROOT / "data/processed/canon" / f"{key}.jpg"))[:, :, ::-1]
    prob, _, tau = calibrated_map(key, load_config(), load_calibration())
    n = 520
    small = cv2.resize(canon, (n, n), interpolation=cv2.INTER_AREA)
    f = n / canon.shape[0]
    over = small.astype(np.float32) / 255 * 0.6
    for k, les in enumerate(["MA", "HE", "EX", "SE"]):
        p = cv2.resize(prob[k], (n, n), interpolation=cv2.INTER_LINEAR)
        a = np.clip(p / max(tau[k], 1e-3) * 0.5, 0, 1)[..., None]          # 阈值处半透明
        col = np.array([int(LESION_COLORS[les][i:i + 2], 16) / 255 for i in (1, 3, 5)])
        over = over * (1 - a) + col * a
    over = np.clip(over, 0, 1)
    sel = crops[(crops.image_key == key) & (crops.method == "C_k4")].sort_values("slot")[["x0", "y0"]].to_numpy()

    import json
    summ = json.loads((ROOT / "results/02_manifest/summary.json").read_text())
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False, dtype=str)
    man = man[man.included == "1"]
    nr, mk = summ["n_by_role"], summ["masks_by_role"]
    cohort = {"ddr_train": nr["ddr_train"], "ddr_dev": nr["ddr_dev"], "ddr_test": nr["ddr_test"],
              "seg_train": mk["seg_train"], "seg_dev": mk["seg_dev"], "seg_test": mk["seg_test"],
              "drid_eyes": summ["deepdrid_eyes"], "drid_patients": summ["deepdrid_patients"], "idrid": nr["idrid"],
              "idrid_seg": nr["idrid_seg"], "mes": nr["messidor2"], "mes_exams": man[man.role == "messidor2"].patient_id.nunique(),
              "eyp": nr["eyepacs"], "eyp_patients": summ["eyepacs_patients"]}

    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

    # ---------------- a：病灶引导的裁块选择
    s, ya = 34.0, 112.0
    ax.text(1.5, H - 1.5, "a", fontsize=11, fontweight="bold", va="top")
    draw_icon(ax, "fundus_camera", 1.5, ya + 10, 13)
    arrow(ax, 15.0, ya + s / 2, 18.5, ya + s / 2)
    image(ax, small, 19, ya, s)
    caption(ax, 19 + s / 2, ya + s + 1.5, "Fundus photograph\n(1536 × 1536 pixels)")
    arrow(ax, 53.5, ya + s / 2, 56.5, ya + s / 2)
    draw_icon(ax, "segmenter", 57, ya + 13, 15)
    ax.text(64.5, ya + 12, "Lesion\nsegmenter\n(U-Net)", ha="center", va="top", fontsize=6.0, linespacing=1.2)
    arrow(ax, 72.5, ya + s / 2, 75.5, ya + s / 2)
    image(ax, over, 76, ya, s)
    caption(ax, 76 + s / 2, ya + s + 1.5, "Calibrated lesion\nprobability maps")
    for i, les in enumerate(["MA", "HE", "EX", "SE"]):
        xx, yy = 77 + (i % 2) * 18, ya - 3.5 - (i // 2) * 4.0
        ax.add_patch(Circle((xx, yy), 1.0, fc=LESION_COLORS[les], ec="none"))
        ax.text(xx + 2, yy, LESION_LABELS[les], va="center", fontsize=6.0)
    arrow(ax, 110.5, ya + s / 2, 135.5, ya + s / 2)
    draw_icon(ax, "magnifier", 117.5, ya + s / 2 + 1.5, 11)
    ax.text(123, ya + s / 2 - 1.5, "Greedy coverage\nof lesion\ncomponents\n(K = 4)", ha="center", va="top", fontsize=6.0,
            linespacing=1.2)
    image(ax, small, 136, ya, s)
    for x, y in sel:
        ax.add_patch(Rectangle((136 + x * f / n * s, ya + s - (y + CROP) * f / n * s), CROP * f / n * s,
                               CROP * f / n * s, fill=False, lw=1.0, ec=METHOD_COLORS["C"], zorder=3))
    caption(ax, 136 + s / 2, ya + s + 1.5, "Four crops chosen to\ncover distinct lesions")

    # ---------------- b：分级网络
    yb = 60.0
    ax.text(1.5, 107.0, "b", fontsize=11, fontweight="bold", va="top")
    g = cv2.resize(canon, (200, 200), interpolation=cv2.INTER_AREA)
    image(ax, g, 6, yb + 22, 20)
    caption(ax, 16, yb + 43, "Global view\n512 × 512")
    t = 9.6
    for i, (x, y) in enumerate(sel):
        tile = cv2.resize(canon[y:y + CROP, x:x + CROP], (140, 140), interpolation=cv2.INTER_AREA)
        image(ax, tile, 6 + (i % 2) * (t + 0.8), yb + 1 + (1 - i // 2) * (t + 0.8), t, edge=METHOD_COLORS["C"], lw=0.8)
    caption(ax, 16, yb - 5.0, "Four crops 384 × 384")
    draw_icon(ax, "encoder", 32, yb + 13, 22)
    ax.text(43, yb + 12, "Shared ConvNeXt\nencoder", ha="center", va="top", fontsize=6.2, linespacing=1.2)
    arrow(ax, 27, yb + 32, 31.5, yb + 27)
    arrow(ax, 27, yb + 11, 31.5, yb + 18)
    box(ax, 61, yb + 1, 21, 16, "Gated\nattention\npooling", fs=6.3)
    box(ax, 61, yb + 24, 21, 12, "Global\nfeature", fs=6.3)
    arrow(ax, 54.5, yb + 20, 60.5, yb + 10)
    arrow(ax, 54.5, yb + 27, 60.5, yb + 30)
    box(ax, 88, yb + 12, 14, 14, "Concatenate\n+ linear", fs=6.0)
    arrow(ax, 82.5, yb + 9, 87.5, yb + 16)
    arrow(ax, 82.5, yb + 30, 87.5, yb + 22)
    arrow(ax, 102.5, yb + 19, 106.0, yb + 19)
    draw_icon(ax, "grade_report", 106.5, yb + 13, 12)
    ax.text(112.5, yb + 12, "Grade 0–4", ha="center", va="top", fontsize=6.2)
    # 训练时的辅助裁块头与损失（只在训练中使用）
    ax.add_patch(FancyBboxPatch((88, yb - 2.5), 16, 9, boxstyle="round,pad=0,rounding_size=1.2", lw=0.7, ec=GREY,
                                fc="#FFFFFF", ls=(0, (3, 2))))
    ax.text(96, yb + 2, "Crop-only\nhead", ha="center", va="center", fontsize=5.8, linespacing=1.15)
    arrow(ax, 82.5, yb + 5, 87.5, yb + 2)
    aux_w = load_config()["classifier"]["crop_aux_weight"]
    ax.text(105.5, yb + 2, "Training loss\n$\\mathcal{L}=\\mathrm{CE}_{\\mathrm{grade}}$\n$+\\," + f"{aux_w:g}" + "\\,\\mathrm{CE}_{\\mathrm{crops}}$",
            ha="left", va="center", fontsize=5.6, linespacing=1.25)

    # ---------------- c：选择原理
    yc = 60.0
    ax.text(124, 107.0, "c", fontsize=11, fontweight="bold", va="top")
    lesions = [("HE", 0.36, 0.55, 0.14), ("HE", 0.47, 0.44, 0.06)] + \
              [("MA", x, y, 0.028) for x, y in [(0.73, 0.30), (0.80, 0.38), (0.70, 0.20), (0.22, 0.20), (0.62, 0.80), (0.82, 0.70)]]
    panels = [("Heatmap\nranking", [(0.18, 0.37, 0.34), (0.30, 0.28, 0.34)], METHOD_COLORS["T"]),
              ("Lesion\ncoverage", [(0.19, 0.38, 0.34), (0.58, 0.12, 0.34)], METHOD_COLORS["C"])]
    for j, (title, wins, col) in enumerate(panels):
        x0, d = 127 + j * 24, 21.0
        y0 = yc + 14
        ax.add_patch(Circle((x0 + d / 2, y0 + d / 2), d / 2, fc="#F6E3D0", ec="#B98B67", lw=0.6))
        for les, lx, ly, r in lesions:
            ax.add_patch(Circle((x0 + lx * d, y0 + (1 - ly) * d), r * d, fc=LESION_COLORS[les], ec="none"))
        for wx, wy, ww in wins:
            ax.add_patch(Rectangle((x0 + wx * d, y0 + (1 - wy - ww) * d), ww * d, ww * d, fill=False, lw=1.0, ec=col))
        caption(ax, x0 + d / 2, y0 + d + 1.5, title, fs=6.5, bold=True)
    ax.text(147.5, yc + 8.0, "Each lesion counts once:", ha="center", fontsize=6.2)
    ax.text(147.5, yc + 1.5, r"$F(S)=\sum_{\ell}\, q_\ell \max_{j\in S}\, a_{\ell j}$", ha="center", fontsize=7)

    # ---------------- d：研究设计（图形化：分区比例条、外部队列卡片、四项分析图标；细节在图注与表 1）
    ax.text(1.5, 51.0, "d", fontsize=11, fontweight="bold", va="top")
    ty = 44.0                                                   # 两个区块标题的基线
    draw_icon(ax, "hospital", 3, ty - 3.6, 7.2)
    ax.text(11.5, ty, "DDR development cohort", fontsize=6.6, fontweight="bold", va="center")
    parts = [("Training", cohort["ddr_train"], "#2B4C7E", "white"), ("Development", cohort["ddr_dev"], "#2A9D8F", "white"),
             ("Test", cohort["ddr_test"], "#E9C46A", "#222222")]
    x, x1, by, bh = 3.0, 72.0, 31.0, 6.0
    tot = sum(p[1] for p in parts)
    for lab, n_, fc, tc in parts:
        w_ = (x1 - 3.0) * n_ / tot
        ax.add_patch(Rectangle((x, by), w_, bh, fc=fc, ec="white", lw=0.6))
        ax.text(x + w_ / 2, by + bh / 2, f"{n_:,}", ha="center", va="center", fontsize=6.0, color=tc)
        ax.text(x + w_ / 2, by - 1.2, lab, ha="center", va="top", fontsize=5.6, color="#444444")
        x += w_
    ax.text(3.0, by - 6.2, f"{cohort['seg_train'] + cohort['seg_dev'] + cohort['seg_test']} photographs with lesion masks",
            fontsize=5.6, color="#444444", va="top")
    arrow(ax, 74.0, by + bh / 2, 84.5, by + bh / 2)
    ax.text(79.3, by + bh / 2 + 1.4, "applied\nunchanged", ha="center", va="bottom", fontsize=5.6, linespacing=1.1)
    draw_icon(ax, "globe", 86, ty - 3.6, 7.2)
    ax.text(94.5, ty, "External test cohorts", fontsize=6.6, fontweight="bold", va="center")
    tiles = [("DeepDRiD", cohort["drid_eyes"], "eyes"), ("IDRiD", cohort["idrid"], "photographs"),
             ("Messidor-2", cohort["mes"], "photographs"), ("EyePACS", cohort["eyp"], "photographs")]
    tw, gap, t0, th = 20.6, 1.4, 86.0, 13.5
    tb = by + bh / 2 - th / 2                                  # 卡片与分区条垂直居中对齐
    for k, (nm, n_, unit) in enumerate(tiles):
        tx = t0 + k * (tw + gap)
        ax.add_patch(FancyBboxPatch((tx, tb), tw, th, boxstyle="round,pad=0,rounding_size=1.2", lw=0.6, ec=GREY, fc="#F7F7F7"))
        ax.text(tx + tw / 2, tb + th - 3.0, nm, ha="center", va="center", fontsize=6.2, fontweight="bold")
        ax.text(tx + tw / 2, tb + th / 2 - 1.0, f"{n_:,}", ha="center", va="center", fontsize=6.4)
        ax.text(tx + tw / 2, tb + 2.6, unit, ha="center", va="center", fontsize=5.4, color="#555555")
    ax.plot([3, 172.6], [20.5, 20.5], color="#D0D0D0", lw=0.5)
    items = [("magnifier", "Lesion retention"), ("eraser", "Microaneurysm removal"),
             ("stethoscope", "Screening accuracy"), ("encoder", "Encoder and cameras")]
    for k, (icon, label) in enumerate(items):
        x = 3 + k * 43
        draw_icon(ax, icon, x, 5.0, 11.5)
        ax.text(x + 14.0, 10.75, label, fontsize=6.2, va="center")
    save_source(pd.DataFrame([cohort]), ROOT, "Fig1", "d_cohorts")

    save_source(pd.DataFrame(sel, columns=["x0_canon", "y0_canon"]).assign(image_key=key), ROOT, "Fig1", "a_crops")
    fig.savefig(ROOT / "figures" / "Fig1.pdf", bbox_inches=None)
    fig.savefig(ROOT / "figures" / "Fig1.png", bbox_inches=None, dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
