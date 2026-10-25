"""逐张解码全部原图：文件 MD5、解码像素哈希、感知哈希、视野外接正方形；写 canon 图与视野掩膜；
分割图像额外把四类参考掩膜映射到 canon 坐标。输出 results/01_preprocess/images_raw.tsv。"""
from __future__ import annotations

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
from scripts.lib.segnet import resize_mask  # noqa: E402

CFG = load_config()
PROC = ROOT / CFG["paths"]["processed"]
CLASSES = ["MA", "HE", "EX", "SE"]
REBUILD = "--rebuild" in sys.argv


def list_sources() -> pd.DataFrame:
    rows = []
    ddr = ROOT / CFG["paths"]["raw_ddr"] / "DDR-dataset"
    for split in ["train", "valid", "test"]:
        for line in (ddr / "DR_grading" / f"{split}.txt").read_text().split("\n"):
            if not line.strip():
                continue
            f, g = line.split()
            rows.append({"image_key": f"ddr_{Path(f).stem}", "dataset": "DDR", "source_split": split, "grade_raw": int(g),
                         "path": str(ddr / "DR_grading" / split / f)})
    seg = ddr / "lesion_segmentation"
    for split in ["train", "valid", "test"]:
        for f in sorted((seg / split / "image").glob("*.jpg")):
            masks = {c: str(seg / split / "label" / c / f"{f.stem}.tif") for c in CLASSES}
            rows.append({"image_key": f"ddrseg_{f.stem}", "dataset": "DDRseg", "source_split": split, "grade_raw": -1,
                         "path": str(f), **{f"mask_{c}": masks[c] if Path(masks[c]).exists() else "" for c in CLASSES}})
    drid = ROOT / CFG["paths"]["raw_deepdrid"]
    for sub in ["regular-fundus-training", "regular-fundus-validation", "Online-Challenge1&2-Evaluation"]:
        for f in sorted((drid / sub / "Images").rglob("*.jpg")):
            rows.append({"image_key": f"drid_{f.stem}", "dataset": "DeepDRiD", "source_split": sub, "grade_raw": -1,
                         "path": str(f)})
    idr = ROOT / CFG["paths"]["raw_idrid"]
    for split, sub in (("train", "a. Training Set"), ("test", "b. Testing Set")):
        for f in sorted((idr / "B. Disease Grading" / "1. Original Images" / sub).glob("*.jpg")):
            rows.append({"image_key": f"idrid_{split}_{f.stem}", "dataset": "IDRiD", "source_split": split, "grade_raw": -1,
                         "path": str(f)})
        folders = {"MA": "1. Microaneurysms", "HE": "2. Haemorrhages", "EX": "3. Hard Exudates", "SE": "4. Soft Exudates"}
        for f in sorted((idr / "A. Segmentation" / "1. Original Images" / sub).glob("*.jpg")):
            gt = idr / "A. Segmentation" / "2. All Segmentation Groundtruths" / sub
            masks = {c: gt / folders[c] / f"{f.stem}_{c}.tif" for c in CLASSES}
            rows.append({"image_key": f"idridseg_{f.stem}", "dataset": "IDRiDseg", "source_split": split, "grade_raw": -1,
                         "path": str(f), **{f"mask_{c}": str(m) if m.exists() else "" for c, m in masks.items()}})
    mes = ROOT / CFG["paths"]["raw_messidor2"]
    for f in sorted((mes / "images").iterdir()):
        rows.append({"image_key": f"mes2_{f.stem}", "dataset": "Messidor2", "source_split": "all", "grade_raw": -1,
                     "path": str(f)})
    apt = ROOT / CFG["paths"]["raw_aptos"]
    for f in sorted((apt / "images").iterdir()):
        rows.append({"image_key": f"aptos_{f.stem}", "dataset": "APTOS", "source_split": "train", "grade_raw": -1,
                     "path": str(f)})
    eyp = ROOT / CFG["paths"]["raw_eyepacs"]          # 第四外部队列（只做外部评价）
    if (eyp / "images").exists():
        for f in sorted((eyp / "images").iterdir()):
            rows.append({"image_key": f"eyp_{f.stem}", "dataset": "EyePACS", "source_split": "all", "grade_raw": -1,
                         "path": str(f)})
    return pd.DataFrame(rows)


def dhash(img: np.ndarray) -> str:
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (17, 16), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):064x}"


def process(row: dict) -> dict:
    p = ROOT / row["path"] if not Path(row["path"]).is_absolute() else Path(row["path"])
    if not p.exists():
        return {"image_key": row["image_key"], "decode_ok": 0}
    out = {"image_key": row["image_key"], "file_md5": hashlib.md5(p.read_bytes()).hexdigest()}
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        out["decode_ok"] = 0
        return out
    img = img[:, :, ::-1]
    out.update(decode_ok=1, orig_h=img.shape[0], orig_w=img.shape[1],
               pixel_md5=hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest())
    pc = CFG["preprocess"]
    fg = foreground(img, pc["fundus_threshold"])
    box = fundus_box(fg)
    out.update(box_x0=box[0], box_y0=box[1], box_side=box[2])
    canon_path = PROC / "canon" / f"{row['image_key']}.jpg"
    fov_path = PROC / "fov" / f"{row['image_key']}.png"
    canon = None
    if not canon_path.exists() or REBUILD:
        canon = to_canon(img, box, pc["canon_size"])
        fov = cv2.resize(crop_square(fg, *box), (pc["canon_size"],) * 2, interpolation=cv2.INTER_NEAREST)
        canon[fov == 0] = 0
        cv2.imwrite(str(canon_path), canon[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, pc["jpeg_quality"]])
        cv2.imwrite(str(fov_path), fov * 255)
    stored = cv2.imread(str(canon_path))[:, :, ::-1]       # 感知哈希一律由存盘的 canon（JPEG 解码后）计算，与是否首次运行无关
    canon = stored if canon is None else canon
    for gs in (pc["global_size"], pc["global_size_large"]):
        gp = PROC / f"global{gs}" / f"{row['image_key']}.jpg"
        if not gp.exists() or REBUILD:
            g = cv2.resize(canon, (gs, gs), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(gp), g[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, pc["jpeg_quality"]])
    out["dhash"] = dhash(stored)
    out["fov_frac"] = float((cv2.imread(str(fov_path), 0) > 0).mean())
    if row["dataset"] in ("DDRseg", "IDRiDseg"):
        ms = []
        for c in CLASSES:
            mp = row.get(f"mask_{c}", "")
            m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED) if isinstance(mp, str) and mp else None
            if m is None:
                m = np.zeros(img.shape[:2], np.uint8)
            m = (m.max(axis=2) if m.ndim == 3 else m) > 0
            out[f"mask_shape_ok_{c}"] = int(m.shape == img.shape[:2])
            out[f"npix_{c}"] = int(m.sum())
            ms.append(resize_mask(crop_square(m.astype(np.uint8), *box), pc["canon_size"])
                      if m.shape == img.shape[:2] else np.zeros((pc["canon_size"],) * 2, np.uint8))
        np.save(PROC / "canon_masks" / f"{row['image_key']}.npy", np.stack(ms).astype(np.uint8))
    return out


def main():
    for d in ["canon", "fov", "canon_masks", f"global{CFG['preprocess']['global_size']}",
              f"global{CFG['preprocess']['global_size_large']}"]:
        (PROC / d).mkdir(parents=True, exist_ok=True)
    src = list_sources()
    print(src.groupby(["dataset", "source_split"]).size(), flush=True)
    with Pool(64) as pool:
        res = pool.map(process, src.to_dict("records"), chunksize=8)
    df = src.merge(pd.DataFrame(res), on="image_key")
    # 结果表只记录相对项目根目录的路径
    for c in ["path"] + [f"mask_{c}" for c in CLASSES]:
        df[c] = df[c].fillna("").map(lambda v: str(Path(v).relative_to(ROOT)) if v else "")
    out = ROOT / "results/01_preprocess"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "images_raw.tsv", sep="\t", index=False)
    write_provenance("01_preprocess", [], [out / "images_raw.tsv"])


if __name__ == "__main__":
    main()
