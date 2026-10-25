"""分级网络的注意力是否落在含病灶的裁块上：DDR 测试区 185 张掩膜图像，每个裁块的参考病灶像素数与
门控注意力权重；同一图像内比较（含病灶裁块 vs 不含病灶裁块的注意力），按种子与规则汇总。
输出 results/08b_attention/crop_attention.tsv 与 summary.tsv。"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.refcov import CLASSES, crops_to_orig, ref_components_for  # noqa: E402
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402

CFG = load_config()
RULES = ["U", "R", "T", "N", "D", "C"]


def lesion_pixels(args):
    row, crops = args
    comps = ref_components_for(row, CFG)
    out = []
    for tag, xy in crops.items():
        rects = crops_to_orig(xy, (row["box_x0"], row["box_y0"], row["box_side"]), CFG["preprocess"]["canon_size"],
                              CFG["selection"]["crop"])
        for slot, r in enumerate(rects):
            n = {c: 0 for c in CLASSES}
            for ci, xx, yy in comps:
                n[CLASSES[ci]] += int(((xx >= r[0]) & (xx < r[2]) & (yy >= r[1]) & (yy < r[3])).sum())
            out.append({"image_key": row["image_key"], "method": tag, "slot": slot, **{f"px_{c}": v for c, v in n.items()}})
    return out


def main():
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    man = man[(man.included.astype(int) == 1) & (man.mask_role == "seg_test")]
    for c in ["box_x0", "box_y0", "box_side"]:
        man[c] = man[c].astype(int)
    sel = pd.read_csv(ROOT / "results/05_selection/crops.tsv.gz", sep="\t")
    sel = sel[sel.image_key.isin(man.image_key) & sel.method.isin([f"{r}_k4" for r in RULES])]
    by = {k: {t: g2.sort_values("slot")[["x0", "y0"]].to_numpy() for t, g2 in g.groupby("method")}
          for k, g in sel.groupby("image_key")}
    with Pool(32) as pool:
        res = pool.map(lesion_pixels, [(r, by[r["image_key"]]) for r in man.to_dict("records")])
    px = pd.DataFrame([x for r in res for x in r])
    px["px_all"] = px[[f"px_{c}" for c in CLASSES]].sum(1)
    rows = []
    for r in RULES:
        sfx = f"_{CFG['classifier']['main_variant']}" if CFG["classifier"]["main_variant"] else ""
        for d in sorted((ROOT / "results/07_classify").glob(f"{r}_k4_s*{sfx}/predictions.tsv.gz")):
            name = d.parent.name[:-len(sfx)] if sfx else d.parent.name
            if not name.rsplit("_s", 1)[1].isdigit():
                continue
            seed = int(name.rsplit("_s", 1)[1])
            p = pd.read_csv(d, sep="\t")
            p = p[p.image_key.isin(man.image_key)]
            att = p.melt(id_vars="image_key", value_vars=[f"attn{j}" for j in range(4)], var_name="slot", value_name="attention")
            att["slot"] = att.slot.str[-1].astype(int)
            rows.append(att.assign(method=f"{r}_k4", seed=seed))
    att = pd.concat(rows).merge(px, on=["image_key", "method", "slot"])
    out = ROOT / "results/08b_attention"; out.mkdir(parents=True, exist_ok=True)
    att.to_csv(out / "crop_attention.tsv", sep="\t", index=False)
    summ = []
    for (m, s), g in att.groupby(["method", "seed"]):
        g = g.assign(has=g.px_all > 0)
        mixed = g.groupby("image_key").has.transform(lambda v: v.any() and not v.all())
        h = g[mixed]
        # 同一图像内：含病灶裁块的平均注意力 / 不含病灶裁块的平均注意力
        ratio = h.groupby(["image_key", "has"]).attention.mean().unstack().dropna()
        summ.append({"method": m, "seed": s, "n_images_mixed": len(ratio),
                     "attention_with_lesion": ratio[True].mean(), "attention_without_lesion": ratio[False].mean(),
                     "share_images_lesion_crop_higher": float((ratio[True] > ratio[False]).mean())})
    pd.DataFrame(summ).to_csv(out / "summary.tsv", sep="\t", index=False)
    print(pd.DataFrame(summ).round(3).to_string())
    write_provenance("08b_attention", [ROOT / "results/05_selection/crops.tsv.gz"], [out / "summary.tsv"])


if __name__ == "__main__":
    main()
