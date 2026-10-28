"""相机复现先验：真实病灶因人而异、位置随机；镜头灰尘、反光等伪影由相机决定、固定在画面同一位置。
对每个评价队列（只用照片，不用任何标签），统计每个预测病灶分量在同队列其他照片中同类分量落在同一位置（中心相距
≤radius 格）的比例（复现率）；复现率 ≥threshold 的分量视为相机伪影：不进入候选库与覆盖目标（NMS、DPP 与病灶覆盖
重新选裁块），推理时在图像中修补（07h）。DDR 开发与训练区只用来报告真实病灶复现率的范围（阈值的依据）。
输出 results/05d_camera_prior/{components.tsv.gz, summary.tsv, recurrence_maps.npz, crops.tsv.gz}。"""
from __future__ import annotations

import json
import os
import sys
import zlib

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # 与 05_selection 相同（覆盖矩阵求和顺序）
    os.environ[_v] = "64"
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.calib import calibrated_map, load_calibration  # noqa: E402
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402
from scripts.lib.selection import extract_components, select_all  # noqa: E402

CFG = load_config()
CAL = load_calibration()
CP = CFG["camera_prior"]
PAR = json.loads((ROOT / "results/05_selection/chosen_parameters.json").read_text())
TYPES = CFG["segmenter"]["classes"]


def comps_of(key: str):
    prob, _, tau = calibrated_map(key, CFG, CAL)
    c = extract_components(prob, tau, CFG["selection"]["comp_min_pixels"])
    return c.cls.astype(np.int8), c.centre.astype(np.float32), c.q.astype(np.float32)


def recurrence(cls_list, cen_list, t: int, size: int, r: int):
    """返回 (计数图 H, 每张照片该类分量的复现率列表)。H[y,x]：有该类分量中心落在 (y,x)±r 内的照片数。"""
    H = np.zeros((size, size), np.float32)
    for cls, cen in zip(cls_list, cen_list):
        m = np.zeros((size, size), np.uint8)
        for x, y in cen[cls == t].astype(int):
            m[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = 1
        H += m
    n = len(cls_list)
    rec = [np.array([(H[y, x] - 1) / max(n - 1, 1) for x, y in cen[cls == t].astype(int)], np.float32)
           for cls, cen in zip(cls_list, cen_list)]
    return H, rec


def reselect(args):
    key, keep = args
    prob, fov, tau = calibrated_map(key, CFG, CAL)
    scale = CFG["preprocess"]["canon_size"] / prob.shape[-1]
    rng = np.random.default_rng((CFG["seed"], zlib.crc32(key.encode())))
    r = select_all(prob, fov, CFG, 4, rng, PAR["nms_iou"], PAR["dpp_theta"], tau, methods=tuple(CP["methods"]),
                   keep=keep)
    return [(key, f"{m}_k4", s, int(round(x)), int(round(y))) for m in CP["methods"] for s, (x, y) in enumerate(r[m] * scale)]


def main():
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False, low_memory=False)
    man = man[man.included.astype(int) == 1]
    out = ROOT / "results/05d_camera_prior"; out.mkdir(parents=True, exist_ok=True)
    size, r, thr = CFG["segmenter"]["prob_map_size"], CP["radius"], CP["threshold"]
    comp_rows, summ, maps = [], [], {}
    for role in CP["reference_roles"] + CP["roles"]:
        keys = sorted(man[man.role == role].image_key)
        with Pool(64) as pool:
            res = pool.map(comps_of, keys, chunksize=16)
        cls_list = [c for c, _, _ in res]; cen_list = [e for _, e, _ in res]
        rec = [np.zeros(len(c), np.float32) for c in cls_list]
        for t, name in enumerate(TYPES):
            H, rt = recurrence(cls_list, cen_list, t, size, r)
            maps[f"{role}_{name}"] = H / max(len(keys) - 1, 1)
            for i, (cls, v) in enumerate(zip(cls_list, rt)):
                rec[i][cls == t] = v
            allv = np.concatenate(rt + [np.zeros(0, np.float32)])
            summ.append({"role": role, "lesion": name, "photos": len(keys), "components": len(allv),
                         "max_recurrence": float(allv.max()) if len(allv) else 0.0,
                         "flagged_components": int((allv >= thr).sum()),
                         "flagged_share": float((allv >= thr).mean()) if len(allv) else 0.0,
                         "photos_with_flag": int(sum(((cls == t) & (rv >= thr)).any() for cls, rv in zip(cls_list, rec)))})
        if role in CP["roles"]:
            for key, (cls, cen, q), rv in zip(keys, res, rec):
                comp_rows += [(role, key, j, TYPES[c], round(float(x), 1), round(float(y), 1), round(float(qq), 4),
                               round(float(v), 4), int(v >= thr)) for j, (c, (x, y), qq, v) in enumerate(zip(cls, cen, q, rv))]
        print(role, len(keys), flush=True)
    comp = pd.DataFrame(comp_rows, columns=["role", "image_key", "component", "lesion", "x", "y", "q", "recurrence", "flagged"])
    comp.to_csv(out / "components.tsv.gz", sep="\t", index=False)
    summary = pd.DataFrame(summ); summary.to_csv(out / "summary.tsv", sep="\t", index=False)
    np.savez_compressed(out / "recurrence_maps.npz", **maps)
    print(summary.to_string(), flush=True)
    # 有复现分量的照片重新选裁块，其余照片沿用 05_selection 的裁块
    flagged = comp.groupby("image_key").flagged.max()
    fkeys = sorted(flagged.index[flagged == 1])
    jobs = [(k, (comp[comp.image_key == k].sort_values("component").flagged.to_numpy() == 0)) for k in fkeys]
    with Pool(32) as pool:
        new = [x for rows in pool.map(reselect, jobs, chunksize=4) for x in rows]
    base = pd.read_csv(ROOT / "results/05_selection/crops.tsv.gz", sep="\t")
    keys = set(man[man.role.isin(CP["roles"])].image_key)
    tags = [f"{m}_k4" for m in CP["methods"]]
    base = base[base.image_key.isin(keys) & base.method.isin(tags) & ~base.image_key.isin(set(fkeys))]
    crops = pd.concat([base, pd.DataFrame(new, columns=base.columns)]).sort_values(["image_key", "method", "slot"])
    crops.to_csv(out / "crops.tsv.gz", sep="\t", index=False)
    print("photos re-selected", len(fkeys), flush=True)
    write_provenance("05d_camera_prior", [ROOT / "results/02_manifest/images.tsv", ROOT / "results/05_selection/crops.tsv.gz"],
                     [out / "components.tsv.gz", out / "summary.tsv", out / "recurrence_maps.npz", out / "crops.tsv.gz"],
                     seed=CFG["seed"], extra={"threshold": thr, "radius": r, "photos_reselected": len(fkeys)})


if __name__ == "__main__":
    main()
