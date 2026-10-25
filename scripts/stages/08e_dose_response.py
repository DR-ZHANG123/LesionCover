"""裁块内微动脉瘤数量与轻度 DR 判为无 DR 的关系（DDR 测试区轻度照片）。
对每张照片和每种规则，数出校准概率图上中心落在 K=4 个裁块内的预测微动脉瘤分量，以及全图的分量总数；
结局为该规则模型（辅助头主设置，三个种子）把照片判为无 DR 的比例（整体预测与只看裁块的预测各一套）。
按裁块内微动脉瘤数分箱汇总；并以照片为单位 bootstrap，估计在全图数量相同时裁块内数量每增加一倍的比值比。
输出 results/08e_dose_response/。"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.calib import calibrated_map, load_calibration  # noqa: E402
from scripts.lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from scripts.lib.selection import extract_components  # noqa: E402

CFG = load_config()
CAL = load_calibration()
RULES = ["U", "R", "T", "N", "D", "C"]
BINS = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 10**6, "≥5")]
P = [f"p{i}" for i in range(5)]


def count_one(args):
    key, crops = args
    prob, _, tau = calibrated_map(key, CFG, CAL)
    comps = extract_components(prob[:1], tau[:1], CFG["selection"]["comp_min_pixels"])
    scale = CFG["preprocess"]["canon_size"] / prob.shape[-1]
    c = CFG["selection"]["crop"] / scale
    cen = comps.centre
    out = {"image_key": key, "n_ma_total": len(cen)}
    for r, xy in crops.items():
        xy = np.asarray(xy, float) / scale
        inside = np.zeros(len(cen), bool)
        for x0, y0 in xy:
            inside |= (cen[:, 0] >= x0) & (cen[:, 0] < x0 + c) & (cen[:, 1] >= y0) & (cen[:, 1] < y0 + c)
        out[f"n_in_{r}"] = int(inside.sum())
    return out


def called_none(rule, keys, kind):
    fn = "predictions.tsv.gz" if kind == "full" else "predictions_croponly.tsv.gz"
    vals = []
    for f in sorted((ROOT / "results/07_classify").glob(f"{rule}_k4_s*_aux/{fn}")):
        d = pd.read_csv(f, sep="\t").set_index("image_key").loc[keys]
        vals.append((d[P].to_numpy().argmax(1) == 0).astype(float))
    return np.mean(vals, 0), len(vals)


def odds_ratio(df, rng, B):
    """结局为判为无 DR 的种子比例（按种子数加权的二项），协变量 log2(1+裁块内) 与 log2(1+全图)。"""
    def fit(d):
        X = np.c_[np.log2(1 + d.n_in), np.log2(1 + d.n_ma_total)]
        X2, y = np.r_[X, X], np.r_[np.ones(len(d)), np.zeros(len(d))]
        w = np.r_[d.none * d.n_seed, (1 - d.none) * d.n_seed]
        m = LogisticRegression(penalty=None, max_iter=1000).fit(X2, y, sample_weight=w + 1e-9)
        return m.coef_[0]
    obs = fit(df)
    keys = df.image_key.unique()
    groups = {k: g.index.to_numpy() for k, g in df.groupby("image_key")}
    boot = []
    for _ in range(B):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        boot.append(fit(df.loc[idx].reset_index(drop=True)))
    boot = np.asarray(boot)
    return {"or_in": np.exp(obs[0]), "or_in_lo": np.exp(np.percentile(boot[:, 0], 2.5)),
            "or_in_hi": np.exp(np.percentile(boot[:, 0], 97.5)), "or_total": np.exp(obs[1]),
            "or_total_lo": np.exp(np.percentile(boot[:, 1], 2.5)), "or_total_hi": np.exp(np.percentile(boot[:, 1], 97.5))}


def main():
    set_global_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    man = man[(man.included.astype(int) == 1) & (man.role == "ddr_test")]
    keys = man[man.grade.astype(int) == 1].image_key.tolist()
    sel = pd.read_csv(ROOT / "results/05_selection/crops.tsv.gz", sep="\t")
    sel = sel[sel.method.isin([f"{r}_k4" for r in RULES]) & sel.image_key.isin(keys)].sort_values(["image_key", "method", "slot"])
    crops = {k: {m[:-3]: g[["x0", "y0"]].to_numpy() for m, g in d.groupby("method")} for k, d in sel.groupby("image_key")}
    with Pool(48) as pool:
        cnt = pd.DataFrame(pool.map(count_one, [(k, crops[k]) for k in keys], chunksize=4))
    out = ROOT / "results/08e_dose_response"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind in ("full", "croponly"):
        for r in RULES:
            none, ns = called_none(r, keys, kind)
            rows.append(pd.DataFrame({"image_key": keys, "kind": kind, "rule": r, "none": none, "n_seed": ns,
                                      "n_in": cnt.set_index("image_key").loc[keys, f"n_in_{r}"].to_numpy(),
                                      "n_ma_total": cnt.set_index("image_key").loc[keys, "n_ma_total"].to_numpy()}))
    long = pd.concat(rows, ignore_index=True)
    long.to_csv(out / "per_image_rule.tsv.gz", sep="\t", index=False)
    bins = []
    for (kind, r), g in long.groupby(["kind", "rule"]):
        for lo, hi, lab in BINS:
            s = g[(g.n_in >= lo) & (g.n_in <= hi)]
            bins.append({"kind": kind, "rule": r, "bin": lab, "n": len(s), "none_pct": 100 * s.none.mean() if len(s) else np.nan})
    for kind, g in long.groupby("kind"):
        for lo, hi, lab in BINS:
            s = g[(g.n_in >= lo) & (g.n_in <= hi)]
            u = s.image_key.unique()
            bt = [s.set_index("image_key").loc[rng.choice(u, len(u))].none.mean() for _ in range(1000)] if len(u) else [np.nan]
            bins.append({"kind": kind, "rule": "pooled", "bin": lab, "n": len(s), "none_pct": 100 * s.none.mean(),
                         "lo": 100 * np.nanpercentile(bt, 2.5), "hi": 100 * np.nanpercentile(bt, 97.5)})
    pd.DataFrame(bins).to_csv(out / "bins.tsv", sep="\t", index=False)
    ors = [{"kind": kind, **odds_ratio(g.reset_index(drop=True), rng, 1000)} for kind, g in long.groupby("kind")]
    pd.DataFrame(ors).to_csv(out / "odds_ratios.tsv", sep="\t", index=False)
    mean_in = long[long.kind == "full"].groupby("rule").n_in.agg(["mean", "median"]).reset_index()
    mean_in.to_csv(out / "ma_in_crops_by_rule.tsv", sep="\t", index=False)
    print(pd.DataFrame(bins).round(1).to_string()); print(pd.DataFrame(ors).round(3)); print(mean_in, cnt.n_ma_total.describe())
    write_provenance("08e_dose_response", [ROOT / "results/05_selection/crops.tsv.gz"],
                     [out / "bins.tsv", out / "odds_ratios.tsv"], seed=CFG["seed"])


if __name__ == "__main__":
    main()
