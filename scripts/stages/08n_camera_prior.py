"""相机复现先验的效果：主设置（ConvNeXt-Tiny、只用 DDR 训练、三个模型概率平均）在五个评价队列中，比较原输入
（全局 512、病灶覆盖）与去掉相机复现分量后的输入（全局 512、NMS、DPP、病灶覆盖；07h）的 κ、无 DR 被判为 DR 的比例、
轻度 DR 被判为无 DR 的比例、可转诊 DR（≥2 级）的敏感度与特异度、AUROC，以及在原全局 512 特异度下的敏感度。
DeepDRiD 为眼级（同眼照片概率平均）；区间来自按单位重抽样的 bootstrap（DeepDRiD、EyePACS 按患者，Messidor-2 按检查，
其余按照片）。输出 results/08n_camera_prior/{metrics.tsv, contrasts.tsv}。"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.boot2 import Resampler, summarize, wkappa, wshare  # noqa: E402
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402
from scripts.lib.stats import eye_level  # noqa: E402

P = [f"p{i}" for i in range(5)]
SEEDS = (11, 22, 33)
CLUSTER = ("deepdrid", "eyepacs", "messidor2")
SRC = {"orig": ROOT / "results/07g_referable_scores", "cam": ROOT / "results/07h_predict_camera"}
INPUTS = {"G512": ("orig", "G512_s{}"), "C": ("orig", "C_k4_s{}_aux"), "G512cam": ("cam", "G512_s{}"),
          "Ncam": ("cam", "N_k4_s{}_aux"), "Dcam": ("cam", "D_k4_s{}_aux"), "Ccam": ("cam", "C_k4_s{}_aux")}
CONTRASTS = [("Ccam", "C"), ("Ccam", "G512"), ("Ccam", "G512cam"), ("Ccam", "Ncam"), ("Ccam", "Dcam"), ("G512cam", "G512")]
STATS = {"ref_sens": (lambda y: y >= 2, lambda p: p >= 2), "ref_spec": (lambda y: y < 2, lambda p: p < 2),
         "healthy_fp": (lambda y: y == 0, lambda p: p >= 1), "mild_missed": (lambda y: y == 1, lambda p: p == 0)}


def ensemble(src: str, key: str, cohort: str, man, drid):
    lst, y, units, keys = [], None, None, None
    for sd in SEEDS:
        pr = pd.read_csv(SRC[src] / f"{key.format(sd)}.tsv.gz", sep="\t")
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


def matched_sens(score, ref, spec, w):
    s, ww = score[~ref], w[~ref]
    o = np.argsort(s); cs = np.cumsum(ww[o]) / ww.sum()
    t = s[o][min(np.searchsorted(cs, spec), len(o) - 1)]
    return float(w[ref][score[ref] > t].sum() / w[ref].sum())


def one_cohort(args):
    cohort, probs, y, units, B, seed = args
    ref = y >= 2
    lab = {k: p.argmax(1) for k, p in probs.items()}
    sc = {k: -p[:, :2].sum(1) for k, p in probs.items()}

    def stats(w):
        ww = np.ones(len(y)) if w is None else w
        out = {}
        spec_g = ww[~ref][lab["G512"][~ref] < 2].sum() / ww[~ref].sum()
        for k, p in lab.items():
            out[f"qwk_{k}"] = wkappa(y, p, w)
            for name, (sel, hit) in STATS.items():
                out[f"{name}_{k}"] = wshare(sel(y), hit(p), w)
            out[f"auc_{k}"] = roc_auc_score(ref, sc[k], sample_weight=ww)
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
    names = ["qwk", *STATS, "auc", "msens"]
    rows, cons = [], []
    for k in probs:
        r = {"cohort": cohort, "input": k, "n_units": len(np.unique(units)), "n": len(y)}
        for nm in names:
            v = summarize(obs[f"{nm}_{k}"], draws[f"{nm}_{k}"])
            r.update({nm: v[0], f"{nm}_lo": v[1], f"{nm}_hi": v[2]})
        for nm, (sel, hit) in STATS.items():
            r.update({f"{nm}_k": int((sel(y) & hit(lab[k])).sum()), f"{nm}_n": int(sel(y).sum())})
        rows.append(r)
    for a, b in CONTRASTS:
        for nm in names:
            d = np.asarray(draws[f"{nm}_{a}"]) - np.asarray(draws[f"{nm}_{b}"])
            v = summarize(obs[f"{nm}_{a}"] - obs[f"{nm}_{b}"], list(d))
            cons.append({"cohort": cohort, "contrast": f"{a}-{b}", "stat": nm, "diff": v[0], "lo": v[1], "hi": v[2],
                         "p_boot": v[3]})
    return rows, cons


def main():
    cfg = load_config(); B = cfg["stats"]["n_boot"]
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False, dtype={"patient_id": str},
                      low_memory=False)
    man = man[man.included.astype(int) == 1].copy(); man["grade"] = man.grade.astype(int)
    drid = man[man.role == "deepdrid"].copy(); drid["patient_id"] = drid.patient_id.astype(int)
    jobs = []
    for i, coh in enumerate(cfg["camera_prior"]["roles"]):
        got = {k: ensemble(s, key, coh, man, drid) for k, (s, key) in INPUTS.items()}
        keys = got["C"][3]
        assert all((g[3] == keys).all() for g in got.values())
        jobs.append((coh, {k: g[0] for k, g in got.items()}, got["C"][1], got["C"][2], B, cfg["seed"] + 300 + i))
    with Pool(len(jobs)) as pool:
        res = pool.map(one_cohort, jobs)
    out = ROOT / "results/08n_camera_prior"; out.mkdir(parents=True, exist_ok=True)
    met = pd.DataFrame([r for rows, _ in res for r in rows]); con = pd.DataFrame([c for _, cs in res for c in cs])
    met.to_csv(out / "metrics.tsv", sep="\t", index=False); con.to_csv(out / "contrasts.tsv", sep="\t", index=False)
    print(met[["cohort", "input", "qwk", "healthy_fp", "mild_missed", "ref_sens", "ref_spec", "auc", "msens"]].round(3).to_string())
    print(con[con.contrast.isin(["Ccam-C", "Ccam-G512", "Ccam-G512cam"])].round(3).to_string())
    write_provenance("08n_camera_prior", sorted(SRC["cam"].glob("*.tsv.gz")), [out / "metrics.tsv", out / "contrasts.tsv"],
                     seed=cfg["seed"], extra={"n_boot": B})


if __name__ == "__main__":
    main()
