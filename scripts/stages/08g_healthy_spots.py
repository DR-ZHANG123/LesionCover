"""无 DR 照片中的预测微动脉瘤与误判（评价队列：DDR 测试区、DeepDRiD、IDRiD、Messidor-2、EyePACS）。
每张参考等级为 0 或 1 的照片数出全图与病灶覆盖裁块（K=4）内的预测微动脉瘤分量；结局为病灶覆盖模型（辅助头主设置，
三个种子概率平均）只看裁块时判为有 DR（等级 ≥1）。按裁块内分量数分箱，照片 bootstrap 区间。
输出 results/08g_healthy_spots/{per_photo.tsv.gz, bins.tsv, counts.tsv}。
--var：换用另一训练设置的病灶覆盖模型（如 auxmc、auxmchn），输出到 results/08g_healthy_spots_<var>/。"""
from __future__ import annotations

import argparse
import importlib
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402

dose = importlib.import_module("scripts.stages.08e_dose_response")
CFG = load_config()
P = [f"p{i}" for i in range(5)]
COHORTS = ("ddr_test", "deepdrid", "idrid", "messidor2", "eyepacs")
OWN_FILE = ("messidor2", "eyepacs")      # 这些队列的只看裁块预测单独成文件


def crop_only_ensemble(role: str, var: str = "aux") -> pd.DataFrame | None:
    ds = []
    dirs = [d for d in sorted((ROOT / "results/07_classify").glob(f"C_k4_s??_{var}/")) if (d / "PROVENANCE.json").exists()]
    assert len(dirs) == 3, f"{var}: 需要三个已完成的模型，现有 {len(dirs)}"
    for d in dirs:
        f = d / (f"predictions_croponly_{role}.tsv.gz" if role in OWN_FILE else "predictions_croponly.tsv.gz")
        if not f.exists():
            return None
        t = pd.read_csv(f, sep="\t")
        ds.append(t[t.role == role].set_index("image_key")[P].sort_index())
    return pd.DataFrame(np.mean([x.to_numpy() for x in ds], 0), index=ds[0].index, columns=P)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--var", default="aux"); var = ap.parse_args().var
    stage = "08g_healthy_spots" if var == "aux" else f"08g_healthy_spots_{var}"
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    ens = {r: crop_only_ensemble(r, var) for r in COHORTS}
    roles = [r for r in COHORTS if ens[r] is not None]
    man = man[(man.included.astype(int) == 1) & man.role.isin(roles)].copy()
    man["grade"] = man.grade.astype(int)
    man = man[man.grade <= 1]
    sel = pd.read_csv(ROOT / "results/05_selection/crops.tsv.gz", sep="\t")
    sel = sel[(sel.method == "C_k4") & sel.image_key.isin(man.image_key)].sort_values(["image_key", "slot"])
    crops = {k: {"C": g[["x0", "y0"]].to_numpy()} for k, g in sel.groupby("image_key")}
    with Pool(64) as pool:
        cnt = pd.DataFrame(pool.map(dose.count_one, [(k, crops[k]) for k in man.image_key], chunksize=8))
    d = man[["image_key", "role", "grade"]].merge(cnt.rename(columns={"n_in_C": "n_in"}), on="image_key")
    pr = pd.concat([ens[r] for r in roles])
    d["called_dr"] = (pr.loc[d.image_key, P].to_numpy().argmax(1) > 0).astype(float)
    out = ROOT / "results" / stage; out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "per_photo.tsv.gz", sep="\t", index=False)
    bins = []
    for (role, g), s0 in d.groupby(["role", "grade"]):
        rng = np.random.default_rng((CFG["seed"], COHORTS.index(role), int(g)))   # 每组独立的随机数，增删队列不影响其他组
        for lo, hi, lab in dose.BINS:
            s = s0[(s0.n_in >= lo) & (s0.n_in <= hi)]
            v = s.called_dr.to_numpy()
            bt = [v[rng.integers(len(v), size=len(v))].mean() for _ in range(1000)] if len(v) else [np.nan]
            bins.append({"cohort": role, "grade": g, "bin": lab, "n": len(v), "called_dr_pct": 100 * v.mean() if len(v) else np.nan,
                         "lo": 100 * np.nanpercentile(bt, 2.5), "hi": 100 * np.nanpercentile(bt, 97.5)})
    pd.DataFrame(bins).to_csv(out / "bins.tsv", sep="\t", index=False)
    c = d.groupby(["role", "grade"]).agg(n=("image_key", "size"), ma_median=("n_ma_total", "median"),
                                         ma_q1=("n_ma_total", lambda x: x.quantile(0.25)),
                                         ma_q3=("n_ma_total", lambda x: x.quantile(0.75)),
                                         in_median=("n_in", "median"), share_in5=("n_in", lambda x: (x >= 5).mean()),
                                         called_dr=("called_dr", "mean")).reset_index()
    c.to_csv(out / "counts.tsv", sep="\t", index=False)
    print(pd.DataFrame(bins).round(1).to_string()); print(c.round(3).to_string())
    write_provenance(stage, [ROOT / "results/05_selection/crops.tsv.gz"],
                     [out / "bins.tsv", out / "counts.tsv"], seed=CFG["seed"])


if __name__ == "__main__":
    main()
