"""四种训练设置下的稳健性比较：主设置（ConvNeXt-Tiny + 辅助裁块头）、跨相机伪影增强、更大的编码器（ConvNeXt-Small）、
多相机训练集（DDR 训练区 + APTOS 2019）。
每种设置内比较病灶覆盖与全局 512 图、与 NMS：DDR 测试区（照片级 κ 与轻度漏判）、DeepDRiD（眼级 κ，患者整群
bootstrap）、IDRiD 与 Messidor-2（照片级 κ，照片 bootstrap）。各输入取三个种子的概率平均；summary.tsv 中 *_mean 为集成指标，
*_std 为单种子模型间的标准差。输出 results/08c_variants/{summary.tsv, contrasts.tsv}。
--extra：改为比较多相机训练集上的三种附加设置（伪影增强、更大编码器、易误读健康照片加权抽样），输出 results/08c_variants_mc/。"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.clsnet import quadratic_kappa  # noqa: E402
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402
from scripts.lib.stats import eye_level, metrics  # noqa: E402

P = [f"p{i}" for i in range(5)]
SETTINGS = {"main": {"G512": "G512", "C": "C_k4_aux", "N": "N_k4_aux"},
            "artifact": {"G512": "G512_art", "C": "C_k4_auxart", "N": "N_k4_auxart"},
            "larger": {"G512": "G512_small", "C": "C_k4_auxsmall", "N": "N_k4_auxsmall"},
            "multicamera": {"G512": "G512_mc", "C": "C_k4_auxmc", "N": "N_k4_auxmc"}}
EXTRA = {"multicamera": SETTINGS["multicamera"],
         "multicamera_artifact": {"G512": "G512_mcart", "C": "C_k4_auxmcart", "N": "N_k4_auxmcart"},
         "multicamera_larger": {"G512": "G512_mcsmall", "C": "C_k4_auxmcsmall", "N": "N_k4_auxmcsmall"},
         "multicamera_hardneg": {"G512": "G512_mchn", "C": "C_k4_auxmchn", "N": "N_k4_auxmchn"}}
COHORTS = ("ddr_test", "deepdrid", "idrid", "messidor2")


def load(key: str) -> dict:
    out = {}
    for d in sorted((ROOT / "results/07_classify").glob("*/")):
        m = re.match(r"(?P<method>.+?)_s(?P<seed>\d+)(?:_(?P<var>[a-z]+))?$", d.name)
        if not m or not (d / "PROVENANCE.json").exists():
            continue
        k = m["method"] + (f"_{m['var']}" if m["var"] else "")
        if k != key:
            continue
        pr = pd.read_csv(d / "predictions.tsv.gz", sep="\t")
        for role in ("idrid", "messidor2", "eyepacs"):
            fi = d / f"predictions_{role}.tsv.gz"
            if fi.exists() and not (pr.role == role).any():
                pr = pd.concat([pr, pd.read_csv(fi, sep="\t")])
        out[int(m["seed"])] = pr
    return out


def boot(units: np.ndarray, ys, preds_a, preds_b, B, rng):
    """units：每行所属的重抽样单位；统计量 = 各种子 κ 均值之差（a − b）。"""
    uu, inv = np.unique(units, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uu))]
    obs = np.mean([quadratic_kappa(ys, p) for p in preds_a]) - np.mean([quadratic_kappa(ys, p) for p in preds_b])
    d = []
    for _ in range(B):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        d.append(np.mean([quadratic_kappa(ys[idx], p[idx]) for p in preds_a]) -
                 np.mean([quadratic_kappa(ys[idx], p[idx]) for p in preds_b]))
    d = np.asarray(d)
    return obs, np.percentile(d, 2.5), np.percentile(d, 97.5), min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / B))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--extra", action="store_true"); extra = ap.parse_args().extra
    settings, stage = (EXTRA, "08c_variants_mc") if extra else (SETTINGS, "08c_variants")
    cfg = load_config()
    B = cfg["stats"]["n_boot"]
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False, dtype={"patient_id": str})
    man = man[man.included.astype(int) == 1].copy(); man["grade"] = man.grade.astype(int)
    drid = man[man.role == "deepdrid"].copy(); drid["patient_id"] = drid.patient_id.astype(int)
    drid["quality"] = pd.to_numeric(drid.quality)
    rows, cons = [], []
    for setting, keys in settings.items():
        runs = {k: load(v) for k, v in keys.items()}
        if any(len(v) < 3 for v in runs.values()):
            print("skip", setting, {k: len(v) for k, v in runs.items()}); continue
        preds = {}
        for k, by in runs.items():
            for cohort in COHORTS:
                lst, y, units = [], None, None
                for sd in sorted(by):
                    pr = by[sd][by[sd].role == cohort]
                    if cohort == "deepdrid":
                        e = eye_level(pr, drid).sort_values(["patient_id", "eye"]).reset_index(drop=True)
                        lst.append(e[P].to_numpy()); y, units = e.grade.to_numpy(), e.patient_id.to_numpy()
                    else:
                        d = pr.merge(man[["image_key", "grade", "patient_id"]], on="image_key").sort_values("image_key")
                        lst.append(d[P].to_numpy()); y = d.grade.to_numpy()
                        units = d.patient_id.to_numpy() if cohort == "messidor2" else d.image_key.to_numpy()   # Messidor-2 按检查
                    rows.append({"setting": setting, "input": k, "seed": sd, "cohort": cohort, **metrics(y, lst[-1].argmax(1))})
                ens = np.mean(lst, 0).argmax(1)
                rows.append({"setting": setting, "input": k, "seed": "ensemble", "cohort": cohort, **metrics(y, ens)})
                preds[(k, cohort)] = ([ens], y, units)
        rng = np.random.default_rng(cfg["seed"])
        for cohort in COHORTS:
            for alt in ("G512", "N"):
                a, y, units = preds[("C", cohort)]
                b = preds[(alt, cohort)][0]
                obs, lo, hi, p = boot(units, y, a, b, B, rng)
                cons.append({"setting": setting, "cohort": cohort, "contrast": f"C-{alt}", "diff": obs, "lo": lo, "hi": hi,
                             "p_boot": p})
    out = ROOT / "results" / stage; out.mkdir(parents=True, exist_ok=True)
    per = pd.DataFrame(rows)
    per.to_csv(out / "per_run.tsv", sep="\t", index=False)
    num = [c for c in per.columns if c not in ("setting", "input", "seed", "cohort")]
    key = ["setting", "input", "cohort"]
    single = per[per.seed != "ensemble"]
    summ = per[per.seed == "ensemble"].set_index(key)[num].add_suffix("_mean").join(
        single.groupby(key)[num].std().add_suffix("_std"))
    summ.reset_index().to_csv(out / "summary.tsv", sep="\t", index=False)
    pd.DataFrame(cons).to_csv(out / "contrasts.tsv", sep="\t", index=False)
    print(summ.reset_index()[["setting", "input", "cohort", "qwk_mean", "mild_to_none_mean", "recall_0_mean"]].round(3).to_string())
    print(pd.DataFrame(cons).round(3).to_string())
    write_provenance(stage, [], [out / "summary.tsv", out / "contrasts.tsv"], seed=cfg["seed"])


if __name__ == "__main__":
    main()
