"""病灶“尺子”的外部有效性与不依赖尺子的分级标准比较（只做描述，不参与选择）。

(a) 尺子有效性：人工病灶掩膜（DDR 测试区 185 张、IDRiD 81 张）上，分割器分量数与人工掩膜分量数的一致性
    （按类别：Spearman ρ、预测/人工的中位比值、出现与否的一致率）。人工掩膜降采样到概率图边长（最大池化）后
    用与预测相同的 8 连通、最小面积规则计数。若 IDRiD 与 DDR 测试区的偏差相近，说明尺子在外部照片上没有系统性少数或多数。
    MMRDR 只有逐图的病灶有无人工标注：给出分割器“有无”相对人工标注的敏感度/特异度。
(b) 不依赖尺子：只用人工标注比较 DDR（病灶掩膜，751 张 1–4 级）与 MMRDR（逐图病灶有无）在各级中出现出血/渗出的比例，
    以及出现出血时判为 ≥2 级的比例（国际分级中轻度 = 仅微动脉瘤）。
输出 results/08w_ruler_check/{ruler_counts.tsv, ruler_presence_mmrdr.tsv, human_grade_rule.tsv}。"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402

STAGE = "08w_ruler_check"
LES = ["MA", "HE", "EX", "SE"]


def human_counts(mask: np.ndarray, size: int, min_px: int) -> np.ndarray:
    f = mask.shape[-1] // size
    out = np.zeros(len(LES), int)
    for k in range(len(LES)):
        m = mask[k].reshape(size, f, size, f).max((1, 3)).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        out[k] = int((st[1:, cv2.CC_STAT_AREA] >= min_px).sum())
    return out


def pred_counts(tdir: Path, key: str) -> np.ndarray | None:
    f = tdir / f"{key}.npz"
    return np.load(f)["counts"].reshape(len(LES), 4).sum(1) if f.exists() else None


def main():
    cfg = load_config(); proc = ROOT / cfg["paths"]["processed"]
    size, min_px = cfg["segmenter"]["prob_map_size"], \
        cfg["selection"]["comp_min_pixels"]
    tdir, mdir = proc / "lesion_tokens", proc / "canon_masks"
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", low_memory=False)
    seg = man[man.seg_key.notna() & (man.seg_key.astype(str) != "")]
    rows, hum = [], []
    for r in seg.itertuples():
        key = r.image_key
        pc = pred_counts(tdir, key)
        mf = mdir / f"{r.seg_key}.npy"
        if pc is None or not mf.exists():
            continue
        hc = human_counts(np.load(mf), size, min_px)
        grp = "DDR test" if r.role == "ddr_test" else ("IDRiD" if r.dataset == "IDRiDseg" else "DDR train/dev")
        rows.append({"group": grp, "image_key": key, "grade": r.grade, **{f"pred_{l}": pc[i] for i, l in enumerate(LES)},
                     **{f"human_{l}": hc[i] for i, l in enumerate(LES)}})
    d = pd.DataFrame(rows)
    out = ROOT / "results" / STAGE; out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "ruler_per_image.tsv", sep="\t", index=False)
    summ = []
    for g, x in d.groupby("group"):
        for l in LES:
            p, h = x[f"pred_{l}"].to_numpy(), x[f"human_{l}"].to_numpy()
            pos = h > 0
            summ.append({"group": g, "lesion": l, "n": len(x), "spearman": spearmanr(p, h)[0],
                         "median_ratio_pred_human": np.median((p[pos] + 1) / (h[pos] + 1)) if pos.any() else np.nan,
                         "presence_agreement": np.mean((p > 0) == pos), "human_median": np.median(h), "pred_median": np.median(p)})
    pd.DataFrame(summ).to_csv(out / "ruler_counts.tsv", sep="\t", index=False)

    mm = pd.read_csv(ROOT / "results/01b_mmrdr/images.tsv", sep="\t", low_memory=False)
    mm = mm[mm.included == 1]
    pres = []
    pairs = [("MA", "lesion_MA"), ("HE", "lesion_IRH"), ("EX", "lesion_HEX")]
    pm = {k: pred_counts(tdir, k) for k in mm.image_key}
    mm = mm[[pm[k] is not None for k in mm.image_key]]
    P = np.stack([pm[k] for k in mm.image_key])
    for l, col in pairs:
        h = mm[col].to_numpy() > 0; p = P[:, LES.index(l)] > 0
        pres.append({"lesion": l, "n": len(mm), "sensitivity": p[h].mean(), "specificity": (~p[~h]).mean(),
                     "human_prevalence": h.mean(), "pred_prevalence": p.mean()})
    pd.DataFrame(pres).to_csv(out / "ruler_presence_mmrdr.tsv", sep="\t", index=False)

    ddr = d[d.group != "IDRiD"].copy(); ddr["grade"] = ddr.grade.astype(int)
    hr = []
    for name, x, he, ex in (("DDR (lesion masks)", ddr, ddr.human_HE > 0, ddr.human_EX > 0),
                            ("MMRDR (lesion labels)", mm[mm.grade >= 1], mm[mm.grade >= 1].lesion_IRH > 0, mm[mm.grade >= 1].lesion_HEX > 0)):
        y = x.grade.astype(int).to_numpy(); he, ex = np.asarray(he), np.asarray(ex)
        for g in range(1, 5):
            hr.append({"cohort": name, "grade": g, "n": int((y == g).sum()), "haemorrhage_present": he[y == g].mean(),
                       "exudate_present": ex[y == g].mean()})
        hr.append({"cohort": name, "grade": "grade>=2 | haemorrhage present", "n": int(he.sum()), "share": np.mean(y[he] >= 2)})
        hr.append({"cohort": name, "grade": "grade>=2 | exudate without haemorrhage", "n": int((ex & ~he).sum()),
                   "share": np.mean(y[ex & ~he] >= 2) if (ex & ~he).any() else np.nan})
    pd.DataFrame(hr).to_csv(out / "human_grade_rule.tsv", sep="\t", index=False)
    write_provenance(STAGE, [ROOT / "results/02_manifest/images.tsv", ROOT / "results/01b_mmrdr/images.tsv"],
                     [out / "ruler_counts.tsv", out / "ruler_presence_mmrdr.tsv", out / "human_grade_rule.tsv",
                      out / "ruler_per_image.tsv"])


if __name__ == "__main__":
    main()
