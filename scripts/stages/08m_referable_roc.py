"""可转诊 DR（≥2 级）的连续评分评价：以三个模型集成的 P(≥2 级) 为评分，计算每个队列中全局 512、全局 1024、NMS、DPP 与
病灶覆盖的 AUROC，以及在全局 512 的特异度（其最可能等级给出的可转诊判定）下，其他输入的敏感度（匹配特异度的工作点）。
概率取自 07g_referable_scores 的全精度对数概率；按 1 − P(<2 级) 排序（与 P(≥2 级) 同序，但高置信照片不因舍入并列）。
区间来自按单位重抽样的 bootstrap（DeepDRiD、EyePACS 按患者，Messidor-2 按检查，其余按照片），与 08_stats 相同。
输出 results/08m_referable_roc/{auroc.tsv, matched.tsv, roc_curves.tsv.gz}。"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.boot2 import Resampler, summarize  # noqa: E402
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402
from scripts.lib.stats import eye_level  # noqa: E402

P = [f"p{i}" for i in range(5)]
SEEDS = (11, 22, 33)
COHORTS = ("ddr_test", "deepdrid", "idrid", "messidor2", "eyepacs")
CLUSTER = ("deepdrid", "eyepacs", "messidor2")
INPUTS = {"G512": "G512_s{}", "G1024": "G1024_s{}", "N": "N_k4_s{}_aux", "D": "D_k4_s{}_aux", "C": "C_k4_s{}_aux"}
SC = ROOT / "results/07g_referable_scores"


def ensemble_probs(key, cohort, man, drid):
    lst, y, units, keys = [], None, None, None
    for sd in SEEDS:
        f = SC / f"{key.format(sd)}.tsv.gz"
        if not f.exists():
            return None
        pr = pd.read_csv(f, sep="\t")
        pr = pr[pr.role == cohort].copy()
        for i in range(5):
            pr[f"p{i}"] = np.exp(pr[f"lp{i}"].to_numpy(np.float64))
        if cohort == "deepdrid":
            e = eye_level(pr, drid).sort_values(["patient_id", "eye"]).reset_index(drop=True)
            lst.append(e[P].to_numpy()); y, units = e.grade.to_numpy(), e.patient_id.to_numpy()
            keys = (e.patient_id.astype(str) + "_" + e.eye.astype(str)).to_numpy()
        else:
            d = pr.merge(man[["image_key", "grade", "patient_id"]], on="image_key").sort_values("image_key")
            lst.append(d[P].to_numpy()); y, keys = d.grade.to_numpy(), d.image_key.to_numpy()
            units = d.patient_id.to_numpy() if cohort in CLUSTER else keys
    return np.mean(lst, 0), y.astype(int), units, keys


def matched_sens(score: np.ndarray, ref: np.ndarray, spec: float, w: np.ndarray) -> float:
    """在非可转诊单位中取加权分位数作阈值，使特异度等于 spec，返回可转诊单位中评分高于阈值的加权比例。"""
    neg, pos = ~ref, ref
    s, ww = score[neg], w[neg]
    o = np.argsort(s); cs = np.cumsum(ww[o]) / ww.sum()
    t = s[o][min(np.searchsorted(cs, spec), len(o) - 1)]
    return float(w[pos][score[pos] > t].sum() / w[pos].sum())


def one_cohort(args):
    cohort, probs, y, units, B, seed = args
    ref = y >= 2
    sc = {k: -p[:, :2].sum(1) for k, p in probs.items()}       # 与 P(≥2) 同序
    lab = {k: p.argmax(1) for k, p in probs.items()}

    def stats(w):
        ww = np.ones(len(y)) if w is None else w
        out = {f"auc_{k}": roc_auc_score(ref, s, sample_weight=ww) for k, s in sc.items()}
        spec_g = ww[~ref][lab["G512"][~ref] < 2].sum() / ww[~ref].sum()
        out["spec_G512"] = spec_g
        out["sens_G512"] = ww[ref][lab["G512"][ref] >= 2].sum() / ww[ref].sum()
        for k in sc:
            if k != "G512":
                out[f"msens_{k}"] = matched_sens(sc[k], ref, spec_g, ww)
        return out

    obs = stats(None)
    res = Resampler(units, np.random.default_rng(seed))
    draws = {k: [] for k in obs}
    for _ in range(B):
        w = res.weights()
        if w[ref].sum() == 0 or w[~ref].sum() == 0:
            continue
        for k, v in stats(w).items():
            draws[k].append(v)
    d = {k: np.asarray(v) for k, v in draws.items()}
    auc = []
    for k in sc:
        v = summarize(obs[f"auc_{k}"], list(d[f"auc_{k}"]))
        auc.append({"cohort": cohort, "input": k, "auroc": v[0], "lo": v[1], "hi": v[2], "n_ref": int(ref.sum()), "n_nonref": int((~ref).sum())})
        if k != "C":
            v = summarize(obs["auc_C"] - obs[f"auc_{k}"], list(d["auc_C"] - d[f"auc_{k}"]))
            auc.append({"cohort": cohort, "input": f"C-{k}", "auroc": v[0], "lo": v[1], "hi": v[2], "p_boot": v[3]})
    mat = [{"cohort": cohort, "input": "G512", "specificity": obs["spec_G512"], "sensitivity": obs["sens_G512"]}]
    for k in sc:
        if k == "G512":
            continue
        v = summarize(obs[f"msens_{k}"] - obs["sens_G512"], list(d[f"msens_{k}"] - d["sens_G512"]))
        s_ = summarize(obs[f"msens_{k}"], list(d[f"msens_{k}"]))
        mat.append({"cohort": cohort, "input": k, "specificity": obs["spec_G512"], "sensitivity": s_[0], "sens_lo": s_[1],
                    "sens_hi": s_[2], "diff_vs_G512": v[0], "diff_lo": v[1], "diff_hi": v[2], "p_boot": v[3]})
    curves = []
    for k in ("G512", "D", "C"):
        fpr, tpr, _ = roc_curve(ref, sc[k])
        keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 400)).astype(int))
        curves += [{"cohort": cohort, "input": k, "fpr": float(a), "tpr": float(b)} for a, b in zip(fpr[keep], tpr[keep])]
    return auc, mat, curves


def main():
    cfg = load_config(); B = cfg["stats"]["n_boot"]
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False, dtype={"patient_id": str})
    man = man[man.included.astype(int) == 1].copy(); man["grade"] = man.grade.astype(int)
    drid = man[man.role == "deepdrid"].copy(); drid["patient_id"] = drid.patient_id.astype(int)
    jobs = []
    for i, coh in enumerate(COHORTS):
        got = {k: ensemble_probs(v, coh, man, drid) for k, v in INPUTS.items()}
        if any(g is None for g in got.values()):
            continue
        keys = got["C"][3]
        assert all((g[3] == keys).all() for g in got.values())
        jobs.append((coh, {k: g[0] for k, g in got.items()}, got["C"][1], got["C"][2], B, cfg["seed"] + 100 + i))
    with Pool(len(jobs)) as pool:
        res = pool.map(one_cohort, jobs)
    out = ROOT / "results/08m_referable_roc"; out.mkdir(parents=True, exist_ok=True)
    auc = pd.DataFrame([r for a, _, _ in res for r in a]); mat = pd.DataFrame([r for _, m, _ in res for r in m])
    pd.DataFrame([r for _, _, c in res for r in c]).to_csv(out / "roc_curves.tsv.gz", sep="\t", index=False)
    auc.to_csv(out / "auroc.tsv", sep="\t", index=False); mat.to_csv(out / "matched.tsv", sep="\t", index=False)
    print(auc.round(3).to_string()); print(mat.round(3).to_string())
    write_provenance("08m_referable_roc", [ROOT / "results/02_manifest/images.tsv"] + sorted(SC.glob("*.tsv.gz")),
                     [out / "auroc.tsv", out / "matched.tsv"],
                     seed=cfg["seed"], extra={"n_boot": B})


if __name__ == "__main__":
    main()
