"""参考病灶覆盖评价（DDR 测试区掩膜图像，原始分辨率）：每个参考分量是否被某一裁块保留 ≥ retain_frac；
裁块是否含参考病灶；裁块间冗余。开发区掩膜同样计算（仅供补充）。
输出 results/06_coverage/components.tsv.gz（分量 × 方法）、crops.tsv（裁块级）、summary.tsv、contrasts.tsv。"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.refcov import (CLASSES, crop_has_lesion, crops_to_orig, overlap_fraction,  # noqa: E402
                                 ref_components_for, retained_fraction)
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402

CFG = load_config()


def one(args):
    row, crops = args
    comps = ref_components_for(row, CFG)
    box = (row["box_x0"], row["box_y0"], row["box_side"])
    scale2 = (CFG["preprocess"]["canon_size"] / row["box_side"]) ** 2
    area = np.array([len(c[1]) for c in comps]) * scale2
    cls = np.array([c[0] for c in comps], int)
    # 分量质心（canon 坐标，仅用于示意图）
    sc = CFG["preprocess"]["canon_size"] / row["box_side"]
    cen = [((c[1].mean() - row["box_x0"]) * sc, (c[2].mean() - row["box_y0"]) * sc) for c in comps]
    comp_rows, crop_rows = [], []
    for tag, xy in crops.items():
        rects = crops_to_orig(xy, box, CFG["preprocess"]["canon_size"], CFG["selection"]["crop"])
        F = retained_fraction(comps, rects) if comps else np.zeros((0, len(rects)))
        best = F.max(1) if len(F) else np.zeros(0)
        for l in range(len(comps)):
            comp_rows.append((row["image_key"], row["mask_role"], tag, l, CLASSES[cls[l]], float(area[l]), float(best[l]),
                              round(cen[l][0], 1), round(cen[l][1], 1)))
        has = crop_has_lesion(comps, rects) if comps else np.zeros(len(rects), bool)
        crop_rows.append((row["image_key"], row["mask_role"], tag, float(has.mean()), overlap_fraction(rects)))
    return comp_rows, crop_rows


def main():
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    man = man[(man.included.astype(int) == 1) & man.mask_role.isin(["seg_dev", "seg_test", "ext_seg"])]
    sel = pd.read_csv(ROOT / "results/05_selection/crops.tsv.gz", sep="\t")
    sel = sel[sel.image_key.isin(man.image_key)]
    by = {k: {t: g2.sort_values("slot")[["x0", "y0"]].to_numpy() for t, g2 in g.groupby("method")}
          for k, g in sel.groupby("image_key")}
    jobs = [(r, by[r["image_key"]]) for r in man.to_dict("records")]
    with Pool(64) as pool:
        res = pool.map(one, jobs, chunksize=1)
    comp = pd.DataFrame([x for a, _ in res for x in a],
                        columns=["image_key", "mask_role", "method", "comp_id", "lesion", "area_canon", "retained",
                                 "cx_canon", "cy_canon"])
    crop = pd.DataFrame([x for _, b in res for x in b],
                        columns=["image_key", "mask_role", "method", "frac_crops_with_lesion", "redundancy"])
    rf = CFG["coverage_eval"]["retain_frac"]
    comp["covered"] = (comp.retained >= rf).astype(int)
    # 小病灶：在该类别（测试区、任一方法相同分量集合）的面积中位数以下
    ref = comp[comp.method == comp.method.iloc[0]]
    med = ref.groupby(["mask_role", "lesion"]).area_canon.median().rename("med").reset_index()
    comp = comp.merge(med, on=["mask_role", "lesion"])
    comp["small"] = (comp.area_canon <= comp.med).astype(int)
    out = ROOT / "results/06_coverage"; out.mkdir(parents=True, exist_ok=True)
    comp.drop(columns="med").to_csv(out / "components.tsv.gz", sep="\t", index=False)
    crop.to_csv(out / "crops.tsv", sep="\t", index=False)
    rows = []
    for (role, m), g in comp.groupby(["mask_role", "method"]):
        r = {"mask_role": role, "method": m}
        for les in CLASSES:
            gl = g[g.lesion == les]
            r[f"cov_{les}"] = gl.covered.mean()
            r[f"cov_small_{les}"] = gl[gl.small == 1].covered.mean()
        r["cov_macro"] = np.mean([r[f"cov_{les}"] for les in CLASSES])
        r["cov_all"] = g.covered.mean()
        c = crop[(crop.mask_role == role) & (crop.method == m)]
        r["frac_crops_with_lesion"] = c.frac_crops_with_lesion.mean()
        r["redundancy"] = c.redundancy.mean()
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "summary.tsv", sep="\t", index=False)
    contrasts(comp, out)
    print(summ[summ.mask_role == "seg_test"].round(3).to_string())
    write_provenance("06_coverage", [ROOT / "results/05_selection/crops.tsv.gz"],
                     [out / "summary.tsv", out / "components.tsv.gz"], seed=CFG["seed"])


def contrasts(comp: pd.DataFrame, out: Path):
    """按图像整群 bootstrap：C 与其他方法的覆盖率差（逐类、小病灶、四类平均）；DDR 测试区与 IDRiD 外部掩膜分别计算。"""
    allrows = []
    for role in ("seg_test", "ext_seg"):
        if (comp.mask_role == role).any():
            allrows.append(_contrasts_role(comp[comp.mask_role == role]).assign(mask_role=role))
    pd.concat(allrows).to_csv(out / "contrasts.tsv", sep="\t", index=False)


def _contrasts_role(t: pd.DataFrame) -> pd.DataFrame:
    keys = t.image_key.unique()
    rng = np.random.default_rng(CFG["seed"])
    B = CFG["stats"]["n_boot"]
    piv = {m: g for m, g in t.groupby("method")}
    rows = []
    pairs = [("C_k4", m) for m in piv if m != "C_k4" and m.endswith("_k4")]
    pairs += [(f"C_k{k}", f"{m}_k{k}") for k in (2, 8) for m in ("N", "D", "T", "U") if f"{m}_k{k}" in piv]
    for ref, alt in pairs:
        a, b = piv[ref], piv[alt]
        name = f"C-{alt.split('_')[0]}" + ("" if alt.endswith("_k4") else f"_{alt.split('_')[1]}")
        for les in CLASSES + ["macro"]:
            for small in (0, 1):
                vals = []
                for L in (CLASSES if les == "macro" else [les]):
                    sa = a[(a.lesion == L) & ((a.small == 1) | (small == 0))]
                    sb = b[(b.lesion == L) & ((b.small == 1) | (small == 0))]
                    vals.append(sa.covered.mean() - sb.covered.mean())
                obs = float(np.nanmean(vals))
                bs = _boot(a, b, les, small, keys, rng, B)
                rows.append({"contrast": name, "lesion": les, "small_only": small, "diff": obs,
                             "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)})
    return pd.DataFrame(rows)


def _boot(a, b, les, small, keys, rng, B):
    idx = {k: i for i, k in enumerate(keys)}
    Ls = CLASSES if les == "macro" else [les]
    mats = []
    for L in Ls:
        sa = a[(a.lesion == L) & ((a.small == 1) | (small == 0))]
        sb = b[(b.lesion == L) & ((b.small == 1) | (small == 0))]
        ca = np.bincount(sa.image_key.map(idx), weights=sa.covered, minlength=len(keys))
        na = np.bincount(sa.image_key.map(idx), minlength=len(keys)).astype(float)
        cb = np.bincount(sb.image_key.map(idx), weights=sb.covered, minlength=len(keys))
        mats.append((ca, cb, na))
    out = np.empty(B)
    for i in range(B):
        w = np.bincount(rng.integers(len(keys), size=len(keys)), minlength=len(keys))
        out[i] = np.nanmean([((ca - cb) * w).sum() / max((na * w).sum(), 1) for ca, cb, na in mats])
    return out


if __name__ == "__main__":
    main()
