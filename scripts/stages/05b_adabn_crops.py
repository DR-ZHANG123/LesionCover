"""用按相机适配的病灶概率图（04c）与相应的校准（04b_calibration_adabn）重新选择裁块：固定象限、NMS、DPP 与病灶覆盖（K=4），参数与 05_selection 相同。
同时记录每张照片的预测病灶分量数。输出 results/05b_adabn_crops/{crops.tsv.gz, pred_components.tsv}。
--subdir/--cal/--out/--all-roles：换用另一组概率图（如健康照片阴性监督的分割器 probmaps_neg），并为全部纳入照片
（含分级训练区与 APTOS）选裁块，供重新训练分级器。"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.calib import calibrated_map, load_calibration  # noqa: E402
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402
from scripts.lib.selection import extract_components, select_all  # noqa: E402

CFG = load_config()
ap = argparse.ArgumentParser(); ap.add_argument("--subdir", default="probmaps_adabn"); ap.add_argument("--cal", default="04b_calibration_adabn")
ap.add_argument("--out", default="05b_adabn_crops"); ap.add_argument("--all-roles", action="store_true")
ARGS = ap.parse_args()
CAL = load_calibration(ROOT / "results" / ARGS.cal / "calibration.json")
METHODS = ("U", "N", "D", "C")
PAR = json.loads((ROOT / "results/05_selection/chosen_parameters.json").read_text())


def one(key):
    prob, fov, tau = calibrated_map(key, CFG, CAL, ARGS.subdir)
    scale = CFG["preprocess"]["canon_size"] / prob.shape[-1]
    rng = np.random.default_rng((CFG["seed"], zlib.crc32(key.encode())))
    r = select_all(prob, fov, CFG, 4, rng, PAR["nms_iou"], PAR["dpp_theta"], tau, methods=METHODS)
    rows = [(key, f"{m}_k4", s, int(round(x)), int(round(y))) for m in METHODS for s, (x, y) in enumerate(r[m] * scale)]
    comps = extract_components(prob, tau, CFG["selection"]["comp_min_pixels"])
    return rows, (key, *[int((comps.cls == c).sum()) for c in range(4)])


def main():
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    inc = man.included.astype(int) == 1
    keys = man.image_key[inc if ARGS.all_roles else inc & man.role.isin(["ddr_test", "deepdrid", "idrid", "messidor2"])].tolist()
    with Pool(64) as pool:
        res = pool.map(one, keys, chunksize=8)
    out = ROOT / "results" / ARGS.out; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([x for r, _ in res for x in r], columns=["image_key", "method", "slot", "x0", "y0"]).to_csv(
        out / "crops.tsv.gz", sep="\t", index=False)
    pd.DataFrame([c for _, c in res], columns=["image_key", "MA", "HE", "EX", "SE"]).to_csv(
        out / "pred_components.tsv", sep="\t", index=False)
    write_provenance(ARGS.out, [], [out / "crops.tsv.gz"], seed=CFG["seed"], extra=PAR)


if __name__ == "__main__":
    main()
