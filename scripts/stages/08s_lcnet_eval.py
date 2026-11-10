"""LesionCover 网络与对照（三个模型概率平均）在 DDR 测试区与五个外部队列上的分级与筛查指标。
指标：κ、准确率、宏 F1、可转诊 DR（≥2 级）敏感度/特异度、无 DR 判为 DR、轻度 DR 判为无 DR；按单位重抽样的 95% 区间，
以及 LesionCover 与各对照之差。DeepDRiD 为眼级（同眼照片概率平均）、按患者重抽样；EyePACS 按患者；其余按照片。
设置全部在 DDR 开发区确定后运行一次。输出 results/08s_lcnet_eval/{metrics.tsv, contrasts.tsv}。"""
from __future__ import annotations

import importlib
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.boot2 import Resampler, summarize, wkappa, wshare  # noqa: E402
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402
from scripts.lib.stats import eye_level  # noqa: E402

v8c = importlib.import_module("scripts.stages.08c_variants")
P = [f"p{i}" for i in range(5)]
COHORTS = ("ddr_test", "deepdrid", "idrid", "messidor2", "eyepacs", "mmrdr")
PATIENT_UNITS = ("deepdrid", "eyepacs", "messidor2")   # Messidor-2 的 patient_id 为检查编号（每次检查两张照片）
REF = "LesionCover"
# 输入名 → (来源, 键)：lc 为 07n_lcnet 的运行标签，cls 为 07_classify 的方法_variant
INPUTS = {"LesionCover": ("lc", "fmrglearn"), "Fixed rule thresholds": ("lc", "fmrg"),
          "DINOv2 global tokens only": ("lc", "fmnolesion"), "ConvNeXt-T encoder": ("lc", "rglearn"),
          "ConvNeXt-T, fixed thresholds": ("lc", "rulegate"), "Lesion tokens without rule/gate": ("lc", "base"),
          "Gate only": ("lc", "gate"), "Rule only": ("lc", "rule"), "Global tokens only": ("lc", "nolesion"),
          "Global 1024": ("cls", "G1024"), "Global 512": ("cls", "G512"), "Global 512 + crops": ("cls", "C_k4_aux"),
          "ResNet-50 1024": ("cls", "G1024_r50"), "Swin-T 1024": ("cls", "G1024_swin"),
          "EfficientNetV2-M 1024": ("cls", "G1024_effm"), "GDRNet": ("cls", "GDRNet"), "CABNet": ("cls", "CABNet"),
          "GDRNet 512": ("cls", "GDRNet_512"), "CABNet 512": ("cls", "CABNet_512"),
          "LesionCover, RETFound encoder": ("lc", "rfrglearn"), "RETFound global tokens only": ("lc", "rfnolesion")}
STATS = {"ref_sens": (lambda y: y >= 2, lambda p: p >= 2), "ref_spec": (lambda y: y < 2, lambda p: p < 2),
         "healthy_fp": (lambda y: y == 0, lambda p: p >= 1), "mild_missed": (lambda y: y == 1, lambda p: p == 0),
         "accuracy": (lambda y: np.ones(len(y), bool), None)}


def load_runs(src: str, key: str) -> dict:
    if src == "cls":
        return v8c.load(key)
    out = {}
    for d in sorted((ROOT / "results/07n_lcnet").glob(f"LC_{key}_s*/")):
        if (d / "PROVENANCE.json").exists() and (d / "predictions.tsv.gz").exists():
            out[int(d.name.rsplit("_s", 1)[1])] = pd.read_csv(d / "predictions.tsv.gz", sep="\t")
    return out


def ensemble(runs: dict, cohort: str, man: pd.DataFrame, drid: pd.DataFrame):
    lst, y, units, keys = [], None, None, None
    if not runs:
        return None
    for sd in sorted(runs):
        pr = runs[sd][runs[sd].role == cohort]
        if not len(pr):
            return None
        if cohort == "deepdrid":
            e = eye_level(pr, drid).sort_values(["patient_id", "eye"]).reset_index(drop=True)
            lst.append(e[P].to_numpy()); y, units = e.grade.to_numpy(), e.patient_id.to_numpy()
            keys = (e.patient_id.astype(str) + "_" + e.eye.astype(str)).to_numpy()
        else:
            d = pr.merge(man[["image_key", "grade", "patient_id"]], on="image_key").sort_values("image_key")
            lst.append(d[P].to_numpy()); y, keys = d.grade.to_numpy(), d.image_key.to_numpy()
            units = d.patient_id.to_numpy() if cohort in PATIENT_UNITS else keys
    return np.mean(lst, 0), y.astype(int), units, keys, len(lst)


def dev_thresholds(prob: np.ndarray, y: np.ndarray) -> np.ndarray:
    """期望等级解码：期望等级 Σ k·p_k 按四个阈值切分为五级；阈值在 DDR 开发区上最大化 κ（Nelder–Mead，起点 0.5/1.5/2.5/3.5）。"""
    from scipy.optimize import minimize
    from sklearn.metrics import cohen_kappa_score
    e = prob @ np.arange(5)
    f = lambda t: -cohen_kappa_score(y, np.digitize(e, np.sort(t)), weights="quadratic")
    return np.sort(minimize(f, np.array([0.5, 1.5, 2.5, 3.5]), method="Nelder-Mead",
                            options={"xatol": 1e-3, "maxiter": 400}).x)


def decode(prob: np.ndarray, thr) -> np.ndarray:
    return prob.argmax(1) if thr is None else np.digitize(prob @ np.arange(5), thr)


def stat_value(name, y, p, w):
    sel, hit = STATS[name]
    return wshare(sel(y), (p == y) if hit is None else hit(p), w)


def one_cohort(args):
    cohort, labs, y, units, B, seed = args
    def stats(w):
        out = {}
        for inp, p in labs.items():
            out[f"qwk|{inp}"] = wkappa(y, p, w)
            for name in STATS:
                out[f"{name}|{inp}"] = stat_value(name, y, p, w)
        return out
    obs = stats(None)
    res = Resampler(units, np.random.default_rng(seed))
    draws = {k: [] for k in obs}
    for _ in range(B):
        for k, v in stats(res.weights()).items():
            draws[k].append(v)
    rows, cons = [], []
    for inp, p in labs.items():
        r = {"cohort": cohort, "input": inp, "n_units": len(np.unique(units)), "n": len(y),
             "macro_f1": float(f1_score(y, p, average="macro", labels=range(5), zero_division=0))}
        for name in ["qwk", *STATS]:
            v = summarize(obs[f"{name}|{inp}"], draws[f"{name}|{inp}"])
            r.update({name: v[0], f"{name}_lo": v[1], f"{name}_hi": v[2]})
        rows.append(r)
    for alt in [a for a in labs if a != REF]:
        for name in ["qwk", *STATS]:
            d = np.asarray(draws[f"{name}|{REF}"]) - np.asarray(draws[f"{name}|{alt}"])
            v = summarize(obs[f"{name}|{REF}"] - obs[f"{name}|{alt}"], list(d))
            cons.append({"cohort": cohort, "contrast": f"{REF} vs {alt}", "stat": name, "diff": v[0], "lo": v[1],
                         "hi": v[2], "p_boot": v[3]})
    return rows, cons


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--decode", choices=["argmax", "expected"], default="argmax")
    args = ap.parse_args()
    cfg = load_config(); B = cfg["stats"]["n_boot"]
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False, dtype={"patient_id": str})
    mm = pd.read_csv(ROOT / "results/01b_mmrdr/images.tsv", sep="\t", keep_default_na=False)
    mm["patient_id"] = ""
    man = pd.concat([man, mm[["image_key", "role", "grade", "included", "patient_id"]]])
    man = man[man.included.astype(int) == 1].copy(); man["grade"] = man.grade.astype(int)
    drid = man[man.role == "deepdrid"].copy(); drid["patient_id"] = drid.patient_id.astype(int)
    runs = {inp: load_runs(*src) for inp, src in INPUTS.items()}
    thr = {inp: None for inp in runs}
    if args.decode == "expected":                      # 阈值只在 DDR 开发区确定
        for inp, r in runs.items():
            g = ensemble(r, "ddr_dev", man, drid)
            thr[inp] = None if g is None else dev_thresholds(g[0], g[1])
            print("thresholds", inp, None if thr[inp] is None else np.round(thr[inp], 3), flush=True)
    jobs = []
    for i, coh in enumerate(COHORTS):
        got = {inp: ensemble(r, coh, man, drid) for inp, r in runs.items()}
        got = {k: v for k, v in got.items() if v is not None}
        if REF not in got:
            print("skip", coh); continue
        keys = got[REF][3]
        assert all((g[3] == keys).all() for g in got.values())
        print(coh, {k: g[4] for k, g in got.items()}, flush=True)
        jobs.append((coh, {inp: decode(g[0], thr[inp]) for inp, g in got.items()}, got[REF][1], got[REF][2], B, cfg["seed"] + i))
    with Pool(len(jobs)) as pool:
        res = pool.map(one_cohort, jobs)
    out = ROOT / ("results/08s_lcnet_eval" + ("_expected" if args.decode == "expected" else "")); out.mkdir(parents=True, exist_ok=True)
    met = pd.DataFrame([r for rows, _ in res for r in rows]); con = pd.DataFrame([c for _, cs in res for c in cs])
    met.to_csv(out / "metrics.tsv", sep="\t", index=False); con.to_csv(out / "contrasts.tsv", sep="\t", index=False)
    print(met[["cohort", "input", "qwk", "accuracy", "macro_f1", "ref_sens", "ref_spec", "healthy_fp", "mild_missed"]]
          .round(3).to_string())
    write_provenance(out.name, [ROOT / "results/02_manifest/images.tsv", ROOT / "results/01b_mmrdr/images.tsv"],
                     [out / "metrics.tsv", out / "contrasts.tsv"], seed=cfg["seed"], extra={"n_boot": B, "inputs": INPUTS, "decode": args.decode,
                            "thresholds": {k: None if v is None else v.tolist() for k, v in thr.items()}})


if __name__ == "__main__":
    main()
