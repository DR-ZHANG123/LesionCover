"""不依赖阈值的任意 DR 判别：以三个模型集成的 P(任意 DR)=1−p0 为评分，计算全局 512、全局 1024、NMS、DPP 与病灶覆盖在每个
评价队列中区分有无 DR（≥1 级 对 0 级）的 AUC，以及病灶覆盖与其他输入之差（按单位重抽样的 bootstrap，同 08m；对数概率来自 07g）。
输出 results/08r_any_dr_auc/auroc.tsv。"""
from __future__ import annotations

import importlib
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.boot2 import Resampler, summarize  # noqa: E402
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402

m8 = importlib.import_module("scripts.stages.08m_referable_roc")
INPUTS = {"G512": "G512_s{}", "G1024": "G1024_s{}", "N": "N_k4_s{}_aux", "D": "D_k4_s{}_aux", "C": "C_k4_s{}_aux"}


def one_cohort(args):
    cohort, probs, y, units, B, seed = args
    pos = y >= 1
    sc = {k: 1 - p[:, 0] for k, p in probs.items()}
    obs = {k: roc_auc_score(pos, s) for k, s in sc.items()}
    res = Resampler(units, np.random.default_rng(seed))
    draws = {k: [] for k in sc}
    for _ in range(B):
        w = res.weights()
        if w[pos].sum() == 0 or w[~pos].sum() == 0:
            continue
        for k, s in sc.items():
            draws[k].append(roc_auc_score(pos, s, sample_weight=w))
    rows = []
    for k in sc:
        v = summarize(obs[k], draws[k])
        rows.append({"cohort": cohort, "input": k, "auroc": v[0], "lo": v[1], "hi": v[2], "n_dr": int(pos.sum()),
                     "n_no_dr": int((~pos).sum())})
        if k != "C":
            v = summarize(obs["C"] - obs[k], list(np.asarray(draws["C"]) - np.asarray(draws[k])))
            rows.append({"cohort": cohort, "input": f"C-{k}", "auroc": v[0], "lo": v[1], "hi": v[2], "p_boot": v[3]})
    return rows


def main():
    cfg = load_config(); B = cfg["stats"]["n_boot"]
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False, dtype={"patient_id": str},
                      low_memory=False)
    man = man[man.included.astype(int) == 1].copy(); man["grade"] = man.grade.astype(int)
    drid = man[man.role == "deepdrid"].copy(); drid["patient_id"] = drid.patient_id.astype(int)
    jobs = []
    for i, coh in enumerate(m8.COHORTS):
        got = {k: m8.ensemble_probs(v, coh, man, drid) for k, v in INPUTS.items()}
        keys = got["C"][3]
        assert all((g[3] == keys).all() for g in got.values())
        jobs.append((coh, {k: g[0] for k, g in got.items()}, got["C"][1], got["C"][2], B, cfg["seed"] + 100 + i))
    with Pool(len(jobs)) as pool:
        res = pool.map(one_cohort, jobs)
    out = ROOT / "results/08r_any_dr_auc"; out.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame([r for rows in res for r in rows]); t.to_csv(out / "auroc.tsv", sep="\t", index=False)
    print(t.round(3).to_string())
    write_provenance("08r_any_dr_auc", sorted((ROOT / "results/07g_referable_scores").glob("*.tsv.gz")), [out / "auroc.tsv"],
                     seed=cfg["seed"], extra={"n_boot": B})


if __name__ == "__main__":
    main()
