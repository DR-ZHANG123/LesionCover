"""筛查层面的指标：主设置（ConvNeXt-Tiny、只用 DDR 训练）的全局 512、NMS、DPP 与病灶覆盖（三个模型概率平均）在
DDR 测试区与四个外部队列中的 κ、可转诊 DR（≥2 级）的敏感度与特异度、无 DR 被判为 DR 的比例、轻度 DR 被判为无 DR
的比例，附分子与分母；以及病灶覆盖与其他输入之差。区间来自按单位重抽样的 bootstrap（DeepDRiD 与 EyePACS 按患者，
其余按照片），与 08_stats 的主比较相同；模型固定为三个模型的集成。
EyePACS 为第四外部队列：全部模型与设置确定后预测一次。输出 results/08l_screening/{metrics.tsv, contrasts.tsv}。"""
from __future__ import annotations

import importlib
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.boot2 import Resampler, summarize, wkappa, wshare  # noqa: E402
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402
from scripts.lib.stats import eye_level  # noqa: E402

v8c = importlib.import_module("scripts.stages.08c_variants")
P = [f"p{i}" for i in range(5)]
COHORTS = ("ddr_test", "deepdrid", "idrid", "messidor2", "eyepacs")
PATIENT_UNITS = ("deepdrid", "eyepacs")
INPUTS = {"G512": "G512", "N": "N_k4_aux", "D": "D_k4_aux", "C": "C_k4_aux"}
STATS = {"ref_sens": (lambda y: y >= 2, lambda p: p >= 2), "ref_spec": (lambda y: y < 2, lambda p: p < 2),
         "healthy_fp": (lambda y: y == 0, lambda p: p >= 1), "mild_missed": (lambda y: y == 1, lambda p: p == 0)}


def ensemble_labels(key: str, cohort: str, man: pd.DataFrame, drid: pd.DataFrame):
    runs = v8c.load(key)
    lst, y, units, keys = [], None, None, None
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
    return np.mean(lst, 0).argmax(1), y.astype(int), units, keys, len(lst)


def one_cohort(args):
    cohort, labs, y, units, B, seed = args
    def stats(w):
        out = {}
        for inp, p in labs.items():
            out[f"qwk_{inp}"] = wkappa(y, p, w)
            for name, (sel, hit) in STATS.items():
                out[f"{name}_{inp}"] = wshare(sel(y), hit(p), w)
        return out
    obs = stats(None)
    res = Resampler(units, np.random.default_rng(seed))
    draws = {k: [] for k in obs}
    for _ in range(B):
        for k, v in stats(res.weights()).items():
            draws[k].append(v)
    rows, cons = [], []
    for inp, p in labs.items():
        r = {"cohort": cohort, "input": inp, "n_units": len(np.unique(units)), "n": len(y)}
        v = summarize(obs[f"qwk_{inp}"], draws[f"qwk_{inp}"]); r.update(qwk=v[0], qwk_lo=v[1], qwk_hi=v[2])
        for name, (sel, hit) in STATS.items():
            v = summarize(obs[f"{name}_{inp}"], draws[f"{name}_{inp}"])
            r.update({name: v[0], f"{name}_lo": v[1], f"{name}_hi": v[2], f"{name}_k": int((sel(y) & hit(p)).sum()),
                      f"{name}_n": int(sel(y).sum())})
        rows.append(r)
    for alt in [a for a in labs if a != "C"]:
        for name in ["qwk", *STATS]:
            d = np.asarray(draws[f"{name}_C"]) - np.asarray(draws[f"{name}_{alt}"])
            v = summarize(obs[f"{name}_C"] - obs[f"{name}_{alt}"], list(d))
            cons.append({"cohort": cohort, "contrast": f"C-{alt}", "stat": name, "diff": v[0], "lo": v[1], "hi": v[2],
                         "p_boot": v[3]})
    return rows, cons


def main():
    cfg = load_config(); B = cfg["stats"]["n_boot"]
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False, dtype={"patient_id": str})
    man = man[man.included.astype(int) == 1].copy(); man["grade"] = man.grade.astype(int)
    drid = man[man.role == "deepdrid"].copy(); drid["patient_id"] = drid.patient_id.astype(int)
    jobs = []
    for i, coh in enumerate(COHORTS):
        got = {inp: ensemble_labels(k, coh, man, drid) for inp, k in INPUTS.items()}
        if any(g is None for g in got.values()):
            print("skip", coh); continue
        keys = got["C"][3]
        assert all((g[3] == keys).all() for g in got.values()) and all(g[4] == 3 for g in got.values())
        jobs.append((coh, {inp: g[0] for inp, g in got.items()}, got["C"][1], got["C"][2], B, cfg["seed"] + i))
    with Pool(len(jobs)) as pool:
        res = pool.map(one_cohort, jobs)
    out = ROOT / "results/08l_screening"; out.mkdir(parents=True, exist_ok=True)
    met = pd.DataFrame([r for rows, _ in res for r in rows]); con = pd.DataFrame([c for _, cons in res for c in cons])
    met.to_csv(out / "metrics.tsv", sep="\t", index=False); con.to_csv(out / "contrasts.tsv", sep="\t", index=False)
    print(met[["cohort", "input", "qwk", "ref_sens", "ref_spec", "healthy_fp", "mild_missed"]].round(3).to_string())
    print(con[con.contrast == "C-G512"].round(3).to_string())
    write_provenance("08l_screening", [ROOT / "results/02_manifest/images.tsv"], [out / "metrics.tsv", out / "contrasts.tsv"],
                     seed=cfg["seed"], extra={"n_boot": B})


if __name__ == "__main__":
    main()
