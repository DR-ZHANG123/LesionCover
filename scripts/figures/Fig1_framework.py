"""Figure 1：LesionCover 网络与研究设计（矢量示意；照片、病灶 token、计数与概率为真实数据）。
a 网络：眼底照片 → (1) DINOv2-L（LoRA）全局 token；(2) 分割器 → 每个预测病灶一个 token；(3) 逐类逐象限病灶计数 →
  象限规则（可学习阈值、“至少 j 个象限”）→ 规则 token 与只看规则的有序分级；Transformer 融合，按病灶负荷门控后输出五级概率。
b 研究设计：DDR 训练与开发区确定全部设置，DDR 测试区与五个外部队列各评价一次；同一病灶尺子测量各队列的划级位置。
c 部署步骤：每级少量本地标注照片只重估输出层（温度与每级偏置），网络不变。
示例照片：DDR 测试区参考 2 级、6–14 个病灶 token、四个象限均有病灶、LesionCover 三模型平均判为 2 级者中，
概率最高（相同时 token 最少、image_key 最小）的一张。整张图在毫米坐标中排版。"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from cohorts import LC_COL, LES_COLOR  # noqa: E402
from figstyle import MM, apply_style, save_fig, save_source  # noqa: E402

W, H = 174.0, 122.0
GREY, NAVY, TEAL = "#5A5A5A", "#2B4C7E", "#2A9D8F"
GRADE_COL = ["#4C9F70", "#C9B33A", "#E08A2C", "#C8553D", "#7B2D26"]
LES = ["MA", "HE", "EX", "SE"]
P = [f"p{i}" for i in range(5)]


def pick_example():
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", low_memory=False)
    keys = man[(man.role == "ddr_test") & (man.grade == 2) & (man.included == 1)].image_key.sort_values()
    runs = [pd.read_csv(ROOT / f"results/07n_lcnet/LC_fmrglearn_s{s}/predictions.tsv.gz", sep="\t").set_index("image_key")
            for s in (11, 22, 33)]
    best = None
    for k in keys:
        f = ROOT / "data/processed/lesion_tokens" / f"{k}.npz"
        if not f.exists():
            continue
        z = np.load(f); c = z["counts"].reshape(4, 4)
        if not (6 <= len(z["tokens"]) <= 14 and (c.sum(0) > 0).all()):
            continue
        p = np.mean([r.loc[k, P].to_numpy(float) for r in runs], 0)
        if p.argmax() != 2:
            continue
        score = (round(p.max(), 3), -len(z["tokens"]))
        if best is None or score > best[0]:
            best = (score, k, p, z["tokens"], c)
    return best[1:]


def box(ax, x, y, w, h, title, body="", fc="#F4F4F4", ec=GREY, fs=6.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4", lw=0.7, ec=ec, fc=fc, zorder=2))
    if body:
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=fs, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h - 6.0, body, ha="center", va="top", fontsize=fs - 0.8, linespacing=1.25, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, fontweight="bold", zorder=3,
                linespacing=1.2)


def arrow(ax, p0, p1, color="#444444", lw=0.8, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6.5, lw=lw, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=4))


def photo(ax, key, tokens, x, y, s):
    im = cv2.imread(str(ROOT / "data/processed/canon" / f"{key}.jpg"))[:, :, ::-1]
    n = 520; im = cv2.resize(im, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)
    yy, xx = np.mgrid[:n, :n]
    fov = (xx - n / 2 + 0.5) ** 2 + (yy - n / 2 + 0.5) ** 2 <= (n / 2 - 2) ** 2
    gain = 215.0 / np.percentile(im.mean(2)[fov], 99)            # 仅显示：按视野内亮度统一增益
    im = np.clip(im * gain, 0, 255); im[~fov] = 0; im = im.astype(np.uint8)
    ax.imshow(im, extent=(x, x + s, y, y + s), interpolation="lanczos", zorder=1)
    ax.plot([x + s / 2] * 2, [y, y + s], color="white", lw=0.5, ls=(0, (2, 1.5)), zorder=2)
    ax.plot([x, x + s], [y + s / 2] * 2, color="white", lw=0.5, ls=(0, (2, 1.5)), zorder=2)
    for q, (qx, qy) in enumerate(((0.06, 0.94), (0.94, 0.94), (0.06, 0.06), (0.94, 0.06))):
        ax.text(x + qx * s, y + qy * s, str(q + 1), color="white", fontsize=5.4, ha="center", va="center", zorder=3)
    f = s / 768.0
    for t in tokens:                                     # (cls, q, log_area, x, y, w, h)，概率图坐标（768）
        c, cx, cy, w, h = int(t[0]), t[3], t[4], max(t[5], 10), max(t[6], 10)
        ax.add_patch(Rectangle((x + (cx - w / 2) * f, y + s - (cy + h / 2) * f), w * f, h * f, fill=False, lw=0.6,
                               ec=LES_COLOR[LES[c]], zorder=3))


def panel_a(ax, key, p, tokens, counts):
    ax.text(2, H - 2, "a", fontsize=9, fontweight="bold", va="top")
    s = 36; px, py = 6, 70
    photo(ax, key, tokens, px, py, s)
    ax.text(px + s / 2, py - 2.0, "Colour fundus photograph\n(DDR test, reference grade 2)", ha="center", va="top", fontsize=5.6)
    for i, l in enumerate(LES):
        ax.add_patch(Rectangle((px + i * 9.2, py + s + 2.2), 2.2, 2.2, fill=False, ec=LES_COLOR[l], lw=0.8))
        ax.text(px + i * 9.2 + 3.0, py + s + 3.3, l, fontsize=5.4, va="center")
    # 三条通路
    bx, bw = 50, 50
    rows = [(99, "Foundation-model encoder", "DINOv2-L with LoRA, 1024-pixel view\n→ 8 × 8 global tokens", "#EAF1FB", NAVY),
            (82, "Lesion tokens", "segmenter → one token per lesion:\ntype, confidence, area, position, appearance", "#FBEDEC", LC_COL),
            (65, "Quadrant rule", "learned thresholds per lesion type;\n'at least j quadrants' → rule token", "#FFF5E0", "#B07A00")]
    for y, t, b, fc, ec in rows:
        box(ax, bx, y, bw, 14, t, b, fc=fc, ec=ec)
        arrow(ax, (px + s + 1.5, py + s / 2), (bx - 0.8, y + 7))
    # 计数表（真实）：象限 1 左上、2 右上、3 左下、4 右下
    cx, cy = bx + 12, 52.6
    ax.text(cx + 2, cy + 10.2, "Quadrant", fontsize=5.0, va="center", ha="right", color=GREY)
    for q in range(4):
        ax.text(cx + 9 + q * 6.5, cy + 10.2, str(q + 1), fontsize=5.0, ha="center", va="center", color=GREY)
    for i, l in enumerate(LES):
        ax.text(cx, cy + 7.2 - i * 2.8, l, fontsize=5.2, va="center", color=LES_COLOR[l])
        for q in range(4):
            ax.text(cx + 9 + q * 6.5, cy + 7.2 - i * 2.8, f"{int(counts[i, q])}", fontsize=5.2, ha="center", va="center")
    arrow(ax, (bx + 3, cy + 4), (bx + 3, 64.6), lw=0.6)
    # Transformer、门控、输出
    tx = 108
    box(ax, tx, 70, 22, 38, "Transformer", "4 layers\n\nclass token +\nglobal, lesion,\ncount and\nrule tokens", fc="#F0F0F0", fs=5.8)
    for y, *_ in rows:
        arrow(ax, (bx + bw + 0.8, y + 7), (tx - 0.8, min(max(y + 7, 74), 104)))
    gx = 134
    box(ax, gx, 80, 14, 18, "Burden\ngate", fc="#F0F0F0", fs=5.8)
    ax.text(gx + 7, 78.0, "weights fused vs\nglobal output by\nlesion counts", ha="center", va="top", fontsize=4.9)
    arrow(ax, (tx + 22.8, 89), (gx - 0.8, 89))
    # 五级概率（真实）
    ox, oy, ow = 152, 78, 19
    arrow(ax, (gx + 14.8, 89), (ox - 0.8, 89))
    for g in range(5):
        y = oy + 18 - g * 4.2
        ax.add_patch(Rectangle((ox + 6, y), (ow - 7) * p[g], 3.0, fc=GRADE_COL[g], ec="none"))
        ax.text(ox + 5.2, y + 1.5, str(g), fontsize=5.2, ha="right", va="center")
        ax.text(ox + 6.5 + (ow - 7) * p[g], y + 1.5, f"{p[g]:.2f}", fontsize=4.8, va="center")
    ax.text(ox + ow / 2 + 1, oy + 23.2, "Grade\nprobabilities", ha="center", fontsize=5.6, fontweight="bold")
    ax.text(ox + ow / 2 + 1, oy - 1.5, "via the standard layer\n(temperature T,\ngrade biases b$_m$)", ha="center", va="top", fontsize=4.9,
            color=LC_COL)
    # 只看规则的有序分级
    ax.text(bx + bw + 3, 60.5, "rule-only ordinal grade:\nP(grade > k) = σ(s − c$_k$)", fontsize=5.0, va="center", color="#8A6100")
    return pd.DataFrame({"image_key": [key] * 5, "grade": range(5), "probability": p})


def panel_b(ax):
    ax.text(2, 47, "b", fontsize=9, fontweight="bold", va="top")
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", low_memory=False)
    mm = pd.read_csv(ROOT / "results/01b_mmrdr/images.tsv", sep="\t", low_memory=False)
    man = man[man.included == 1]; mm = mm[mm.included == 1]
    n = man.role.value_counts().to_dict(); n["mmrdr"] = len(mm)
    box(ax, 6, 28, 30, 13, "DDR training", f"{n['ddr_train']:,} photographs\nall weights learned", fc="#EEEEEE")
    box(ax, 6, 11, 30, 13, "DDR development", f"{n['ddr_dev']:,} photographs\nall settings chosen", fc="#EEEEEE")
    arrow(ax, (21, 27.4), (21, 24.6))
    cohorts = [("DDR test", n["ddr_test"]), ("DeepDRiD", n["deepdrid"]), ("IDRiD", n["idrid"]), ("Messidor-2", n["messidor2"]),
               ("EyePACS", n["eyepacs"]), ("MMRDR", n["mmrdr"])]
    x0 = 44
    ax.text(x0 + 21, 43.2, "Evaluated once, model unchanged", ha="center", fontsize=5.8, fontweight="bold")
    for i, (c, k) in enumerate(cohorts):
        y = 36 - i * 5.4
        ax.add_patch(FancyBboxPatch((x0, y), 42, 4.4, boxstyle="round,pad=0,rounding_size=0.8", lw=0.5, ec=GREY,
                                    fc="#FFFFFF" if i else "#EEEEEE"))
        ax.text(x0 + 2, y + 2.2, c, fontsize=5.6, va="center")
        ax.text(x0 + 40, y + 2.2, f"{k:,}", fontsize=5.6, va="center", ha="right")
    arrow(ax, (36.8, 17.5), (x0 - 0.8, 20))
    ax.text(x0 + 21, 2.6, "photographs (DeepDRiD graded per eye)", ha="center", fontsize=5.0, color=GREY)
    # 同一把病灶尺子
    rx = 100
    ax.text(rx + 15, 43.2, "Same lesion ruler in every cohort", ha="center", fontsize=5.8, fontweight="bold")
    for i, (c, sh, col) in enumerate([("DDR", 0.0, "#424242"), ("External", 1.0, "#1B9E77")]):
        y = 33 - i * 10
        ax.plot([rx, rx + 30], [y, y], color=GREY, lw=0.7)
        ax.text(rx - 1, y, c, fontsize=5.4, ha="right", va="center")
        for j, b in enumerate([4, 9, 18, 24]):
            x = rx + b + (sh * 4 if j == 1 else 0)
            ax.plot([x, x], [y - 1.6, y + 1.6], color=LC_COL if j == 1 else col, lw=1.1 if j == 1 else 0.8)
    ax.annotate("", xy=(rx + 13, 20.6), xytext=(rx + 9, 20.6),
                arrowprops=dict(arrowstyle="-|>", color=LC_COL, lw=0.8, mutation_scale=6))
    ax.text(rx + 15, 14.5, "lesion burden index\n(mild | moderate boundary in red)", ha="center", va="top", fontsize=5.0, color=GREY)
    return pd.DataFrame(cohorts, columns=["cohort", "n"])


def panel_c(ax):
    ax.text(136, 47, "c", fontsize=9, fontweight="bold", va="top")
    box(ax, 139, 26, 33, 15, "New site: align standard layer", "5 labelled photographs per grade\nre-estimate T and 4 grade biases\nall other weights unchanged",
        fc="#FBEDEC", ec=LC_COL)
    ax.text(155.5, 22.5, "Evaluated on photographs of\nother patients in the same cohort", ha="center", va="top", fontsize=5.0,
            color=GREY)
    return None


def main():
    apply_style(7)
    key, p, tokens, counts = pick_example()
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    src_a = panel_a(ax, key, p, tokens, counts)
    src_b = panel_b(ax)
    panel_c(ax)
    ax.plot([2, W - 2], [49.5, 49.5], color="#DDDDDD", lw=0.6)
    save_source(src_a, ROOT, "Fig1", "a")
    save_source(src_b, ROOT, "Fig1", "b")
    tok = pd.DataFrame(tokens, columns=["cls", "q", "log_area", "x", "y", "w", "h"]); tok.insert(0, "image_key", key)
    save_source(tok, ROOT, "Fig1", "a_tokens")
    fig.savefig(ROOT / "figures/Fig1.pdf"); fig.savefig(ROOT / "figures/Fig1.png", dpi=600)
    print(key)


if __name__ == "__main__":
    main()
