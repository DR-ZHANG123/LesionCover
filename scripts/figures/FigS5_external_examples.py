"""Supplementary Fig. S5：DeepDRiD 中被裁块输入误判为有 DR、而全局图判对的无 DR 眼。
每例一张照片（同眼两张中病灶覆盖规则预测等级较高者），叠加病灶覆盖规则选出的四个裁块与其放大图；
标题给出参考等级与两种输入（三个种子平均概率）的预测等级。只读 results/07_classify、05_selection、02_manifest。"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from figstyle import METHOD_COLORS, W2, apply_style, save_fig, save_source  # noqa: E402

P = [f"p{i}" for i in range(5)]
N_EX = 4


def mean_pred(prefix: str) -> pd.DataFrame:
    fs = sorted((ROOT / "results/07_classify").glob(f"{prefix}/predictions.tsv.gz"))
    d = pd.concat([pd.read_csv(f, sep="\t") for f in fs])
    d = d[d.role == "deepdrid"]
    return d.groupby("image_key")[P].mean()


def main():
    apply_style(7)
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    man = man[(man.role == "deepdrid") & (man.included.astype(int) == 1)].copy()
    man["grade"] = man.grade.astype(int)
    c, g = mean_pred("C_k4_s[0-9][0-9]_aux"), mean_pred("G512_s[0-9][0-9]")
    man = man.join(c.add_prefix("c_"), on="image_key").join(g.add_prefix("g_"), on="image_key")
    eye = man.groupby(["patient_id", "eye"]).agg(grade=("grade", "first"),
                                                 **{f"c{i}": (f"c_p{i}", "mean") for i in range(5)},
                                                 **{f"g{i}": (f"g_p{i}", "mean") for i in range(5)}).reset_index()
    eye["pc"] = eye[[f"c{i}" for i in range(5)]].to_numpy().argmax(1)
    eye["pg"] = eye[[f"g{i}" for i in range(5)]].to_numpy().argmax(1)
    cand = eye[(eye.grade == 0) & (eye.pg == 0) & (eye.pc >= 1)].copy()
    cand["conf"] = 1 - cand.c0
    cand = cand.sort_values("conf", ascending=False).head(N_EX)
    crops = pd.read_csv(ROOT / "results/05_selection/crops.tsv.gz", sep="\t")
    crops = crops[crops.method == "C_k4"]
    fig, axs = plt.subplots(2, N_EX, figsize=(W2, 0.53 * W2), gridspec_kw={"height_ratios": [1, 1], "hspace": 0.06,
                                                                           "wspace": 0.08})
    rows = []
    for j, r in enumerate(cand.itertuples()):
        imgs = man[(man.patient_id == r.patient_id) & (man.eye == r.eye)]
        k = imgs.sort_values("c_p0").image_key.iloc[0]          # 预测为“无 DR”概率最低的那张
        img = cv2.imread(str(ROOT / "data/processed/canon" / f"{k}.jpg"))[:, :, ::-1]
        f = 480 / img.shape[0]
        small = cv2.resize(img, (480, 480), interpolation=cv2.INTER_AREA)
        ax = axs[0, j]
        ax.imshow(small); ax.set_xticks([]); ax.set_yticks([])
        cr = crops[crops.image_key == k].sort_values("slot")[["x0", "y0"]].to_numpy()
        for x, y in cr:
            ax.add_patch(Rectangle((x * f, y * f), 384 * f, 384 * f, fill=False, lw=1.0, ec=METHOD_COLORS["C"]))
        ax.set_title(f"Reference 0; global view {r.pg};\nwith LesionCover crops {r.pc}", fontsize=6.3, pad=2)
        tiles = [cv2.resize(img[y:y + 384, x:x + 384], (192, 192), interpolation=cv2.INTER_AREA) for x, y in cr]
        mosaic = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]) if len(tiles) == 4 else np.hstack(tiles)
        axs[1, j].imshow(mosaic); axs[1, j].set_xticks([]); axs[1, j].set_yticks([])
        for sp in axs[1, j].spines.values():
            sp.set_color(METHOD_COLORS["C"])
        rows.append({"image_key": k, "reference": 0, "pred_global": int(r.pg), "pred_coverage": int(r.pc),
                     "p_nodr_coverage": float(r.c0), "p_nodr_global": float(r.g0)})
    for ax in axs.ravel():
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
    save_source(pd.DataFrame(rows), ROOT, "FigS5", "examples")
    fig.savefig(ROOT / "figures" / "FigS5.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(ROOT / "figures" / "FigS5.png", bbox_inches="tight", pad_inches=0.03, dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
