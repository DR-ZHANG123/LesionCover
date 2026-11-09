"""为全部纳入照片提取病灶 token（scripts/lib/lesiontokens.py），写到 data/processed/lesion_tokens/<key>.npz。
只用分割器概率图与开发区校准，不用任何分级标签。"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.calib import calibrated_map, load_calibration  # noqa: E402
from scripts.lib.lesiontokens import image_tokens  # noqa: E402
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402

CFG = load_config()
CAL = load_calibration()
OUT = ROOT / CFG["paths"]["processed"] / "lesion_tokens"


def one(key: str):
    f = OUT / f"{key}.npz"
    lc = CFG["lcnet"]
    if not f.exists():
        prob, _, tau = calibrated_map(key, CFG, CAL)
        tok, cnt = image_tokens(prob, tau, CFG["selection"]["comp_min_pixels"], lc["max_tokens"], lc["cover_cell"])
        np.savez_compressed(f, tokens=tok, counts=cnt)
    z = np.load(f)
    return key, len(z["tokens"]), float(z["counts"].sum())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results/02_manifest/images.tsv")
    ap.add_argument("--stage", default="05e_lesion_tokens")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ROOT / args.manifest, sep="\t", low_memory=False)
    man = man[man.included.astype(int) == 1]
    keys = [k for k in man.image_key if (ROOT / CFG["paths"]["processed"] / "probmaps" / f"{k}.npz").exists()]
    with Pool(64) as pool:
        rows = pool.map(one, keys, chunksize=32)
    out = ROOT / "results" / args.stage; out.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows, columns=["image_key", "n_tokens", "n_components"]).merge(man[["image_key", "role"]])
    d.to_csv(out / "tokens_per_image.tsv", sep="\t", index=False)
    print(d.groupby("role")[["n_tokens", "n_components"]].describe().round(1).to_string())
    write_provenance(args.stage, [ROOT / args.manifest, ROOT / "results/04b_calibration/calibration.json"],
                     [out / "tokens_per_image.tsv"], seed=CFG["seed"])


if __name__ == "__main__":
    main()
