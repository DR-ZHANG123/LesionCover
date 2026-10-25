"""裁块选择。
(1) 在 DDR 开发区掩膜图像上，以参考病灶覆盖（四类分量覆盖率的平均）为准则，为 N 的 IoU 阈值与 D 的 θ 选参；
(2) 用选定参数为全部纳入图像生成各方法的裁块（K=4 全部方法；K=2/8 主要方法）；
(3) 掩膜图像上额外生成分量阈值扰动版本，供 06 评价。输出 results/05_selection/crops.tsv.gz（canon 坐标）。"""
from __future__ import annotations

import json
import os
import sys
import zlib

# 覆盖矩阵（ww @ inside）的浮点求和顺序随 BLAS 线程数而变，会改变得分并列的窗口的取舍；
# 线程数固定为 64（OpenBLAS 在本机的默认值，即发表结果所用的设置），裁块与之逐行一致。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "64"
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.refcov import crops_to_orig, ref_components_for, retained_fraction  # noqa: E402
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402
from scripts.lib.calib import calibrated_map, load_calibration  # noqa: E402
from scripts.lib.selection import select_all, shift_threshold  # noqa: E402

CFG = load_config()
PROC = ROOT / CFG["paths"]["processed"]
MAIN = ("U", "R", "T", "N", "D", "C", "Cmass", "S")
BUDGET = ("U", "T", "N", "D", "C")


CAL = load_calibration()


def load_prob(key):
    return calibrated_map(key, CFG, CAL)


def rng_for(key):
    return np.random.default_rng((CFG["seed"], zlib.crc32(key.encode())))


def ref_comps(row):
    return ref_components_for(row, CFG)


def covered(comps, crops_canon, row):
    """每个参考分量是否被某一裁块保留 ≥ retain_frac。"""
    if not comps:
        return np.zeros(0, bool)
    rects = crops_to_orig(crops_canon, (row["box_x0"], row["box_y0"], row["box_side"]),
                          CFG["preprocess"]["canon_size"], CFG["selection"]["crop"])
    return retained_fraction(comps, rects).max(1) >= CFG["coverage_eval"]["retain_frac"]


def tune_one(row):
    prob, fov, tau = load_prob(row["image_key"])
    comps = ref_comps(row)
    cls = np.array([c[0] for c in comps], int)
    scale = CFG["preprocess"]["canon_size"] / prob.shape[-1]
    sc = CFG["selection"]
    out = []
    for thr in sc["nms_iou_grid"]:
        r = select_all(prob, fov, CFG, 4, rng_for(row["image_key"]), thr, 0.5, tau, methods=("N",))
        out.append(("N", thr, cls, covered(comps, r["N"] * scale, row)))
    for th in sc["dpp_theta_grid"]:
        r = select_all(prob, fov, CFG, 4, rng_for(row["image_key"]), 0.3, th, tau, methods=("D",))
        out.append(("D", th, cls, covered(comps, r["D"] * scale, row)))
    return out


def macro_cov(results):
    cls = np.concatenate([r[0] for r in results]); cov = np.concatenate([r[1] for r in results])
    return float(np.mean([cov[cls == c].mean() for c in range(4) if (cls == c).any()]))


def crops_one(args):
    row, nms_thr, theta = args
    key = row["image_key"]
    prob, fov, tau = load_prob(key)
    scale = CFG["preprocess"]["canon_size"] / prob.shape[-1]
    recs = []
    r = select_all(prob, fov, CFG, 4, rng_for(key), nms_thr, theta, tau, methods=MAIN)
    n_comp = r["_n_comp"]
    for m in MAIN:
        recs.append((f"{m}_k4", r[m]))
    for k in CFG["selection"]["k_ablation"]:
        if k == 4:
            continue
        r = select_all(prob, fov, CFG, k, rng_for(key), nms_thr, theta, tau, methods=BUDGET)
        recs += [(f"{m}_k{k}", r[m]) for m in BUDGET]
    if row["mask_role"] in ("seg_dev", "seg_test", "ext_seg"):
        for delta in CFG["selection"]["threshold_shift"]:
            r = select_all(prob, fov, CFG, 4, rng_for(key), nms_thr, theta, shift_threshold(tau, delta),
                           methods=("N", "D", "C"))
            recs += [(f"{m}_k4_t{'low' if delta < 0 else 'high'}", r[m]) for m in ("N", "D", "C")]
    rows = []
    for tag, xy in recs:
        for s, (x, y) in enumerate(np.asarray(xy) * scale):
            rows.append((key, tag, s, int(round(x)), int(round(y))))
    return rows, (key, n_comp)


def main():
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    man = man[man.included.astype(int) == 1]
    out = ROOT / "results/05_selection"; out.mkdir(parents=True, exist_ok=True)
    dev = man[man.mask_role == "seg_dev"].to_dict("records")
    with Pool(32) as pool:
        tuned = pool.map(tune_one, dev, chunksize=1)
    grid = {}
    for res in tuned:
        for m, v, cls, cov in res:
            grid.setdefault((m, v), []).append((cls, cov))
    tab = pd.DataFrame([{"method": m, "value": v, "dev_macro_coverage": macro_cov(r)} for (m, v), r in grid.items()])
    tab.to_csv(out / "tuning.tsv", sep="\t", index=False)
    best = {m: float(g.sort_values(["dev_macro_coverage", "value"], ascending=[False, True]).value.iloc[0])
            for m, g in tab.groupby("method")}
    (out / "chosen_parameters.json").write_text(json.dumps({"nms_iou": best["N"], "dpp_theta": best["D"]}, indent=2))
    print(tab, best, flush=True)
    jobs = [(r, best["N"], best["D"]) for r in man.to_dict("records")]
    with Pool(32) as pool:            # 结果与进程数无关；进程数取 32，避免 32×64 以外的线程超额
        res = pool.map(crops_one, jobs, chunksize=16)
    rows = [x for r, _ in res for x in r]
    pd.DataFrame(rows, columns=["image_key", "method", "slot", "x0", "y0"]).to_csv(
        out / "crops.tsv.gz", sep="\t", index=False)
    pd.DataFrame([nc for _, nc in res], columns=["image_key", "n_pred_components"]).to_csv(
        out / "pred_components.tsv", sep="\t", index=False)
    write_provenance("05_selection", [ROOT / "results/02_manifest/images.tsv"],
                     [out / "crops.tsv.gz", out / "tuning.tsv"], seed=CFG["seed"], extra=best)


if __name__ == "__main__":
    main()
