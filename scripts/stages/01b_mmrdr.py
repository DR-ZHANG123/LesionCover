"""MMRDR 彩色眼底子集（外部队列，只做评价）：逐张解码、视野外接正方形、canon / 视野 / 全局图，与 01_preprocess 相同处理。
同时整理分级与逐图病灶标签（七类：MA、硬性渗出、视网膜内出血、静脉串珠/IRMA、新生血管、玻璃体积血、视网膜脱离）。
输出 results/01b_mmrdr/images.tsv（role = mmrdr；split = 原始 tr/ts）。"""
from __future__ import annotations

import ast
import hashlib
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.fundus import crop_square, foreground, fundus_box, to_canon  # noqa: E402
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402

CFG = load_config()
PROC = ROOT / CFG["paths"]["processed"]
LESIONS = ["MA", "HEX", "IRH", "VB_IRMA", "NV", "VH", "RD"]


def process(row: dict) -> dict:
    p = ROOT / row["path"]
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    out = {"image_key": row["image_key"], "file_md5": hashlib.md5(p.read_bytes()).hexdigest()}
    if img is None:
        return {**out, "decode_ok": 0}
    img = img[:, :, ::-1]
    pc = CFG["preprocess"]
    fg = foreground(img, pc["fundus_threshold"])
    box = fundus_box(fg)
    out.update(decode_ok=1, orig_h=img.shape[0], orig_w=img.shape[1], box_x0=box[0], box_y0=box[1], box_side=box[2])
    cp, fp = PROC / "canon" / f"{row['image_key']}.jpg", PROC / "fov" / f"{row['image_key']}.png"
    if not cp.exists():
        canon = to_canon(img, box, pc["canon_size"])
        fov = cv2.resize(crop_square(fg, *box), (pc["canon_size"],) * 2, interpolation=cv2.INTER_NEAREST)
        canon[fov == 0] = 0
        cv2.imwrite(str(cp), canon[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, pc["jpeg_quality"]])
        cv2.imwrite(str(fp), fov * 255)
    canon = cv2.imread(str(cp))[:, :, ::-1]
    for gs in (pc["global_size"], pc["global_size_large"]):
        gp = PROC / f"global{gs}" / f"{row['image_key']}.jpg"
        if not gp.exists():
            g = cv2.resize(canon, (gs, gs), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(gp), g[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, pc["jpeg_quality"]])
    out["fov_frac"] = float((cv2.imread(str(fp), 0) > 0).mean())
    return out


def main():
    raw = ROOT / CFG["paths"]["raw_mmrdr"] / "MMRDR-CFP"
    lab = pd.read_csv(raw / "FP.csv")
    les = np.array([ast.literal_eval(v) for v in lab.lesion], int)
    df = pd.DataFrame({"image_key": "mmrdr_" + lab.image.str.extract(r"(\w+)\.jpg")[0],
                       "dataset": "MMRDR", "role": "mmrdr", "split": lab.image.str.extract(r"img/(tr|ts)")[0],
                       "grade": lab.grade.astype(int), "eye": lab.lr.map({0: "L", 1: "R"}),
                       "path": [str((raw / v).relative_to(ROOT)) if (raw / v).is_relative_to(ROOT) else str(raw / v)
                                for v in lab.image]})
    df["path"] = [str(Path(CFG["paths"]["raw_mmrdr"]) / "MMRDR-CFP" / v) for v in lab.image]
    for j, c in enumerate(LESIONS):
        df[f"lesion_{c}"] = les[:, j]
    with Pool(64) as pool:
        res = pd.DataFrame(pool.map(process, df.to_dict("records"), chunksize=16))
    df = df.merge(res, on="image_key")
    df["included"] = ((df.decode_ok == 1) & (df.fov_frac > 0.2)).astype(int)
    out = ROOT / "results/01b_mmrdr"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "images.tsv", sep="\t", index=False)
    print(df.groupby(["split", "included"]).size(), df.groupby("split").box_side.describe(), sep="\n")
    write_provenance("01b_mmrdr", [raw / "FP.csv"], [out / "images.tsv"], seed=CFG["seed"])


if __name__ == "__main__":
    main()
