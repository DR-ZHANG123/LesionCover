"""更高分辨率全局图上加病灶覆盖裁块（variant hg：1024 像素全局图 + 4 个 384 像素裁块，其余训练设置与主设置相同，轮次按 DDR
开发区 κ 选择）：与只用 1024 全局图、512 全局图及 512 全局图 + 病灶覆盖裁块比较。三个模型概率平均；DeepDRiD 为眼级。
指标同 08n：κ、无 DR 判为 DR、轻度判为无 DR、可转诊敏感度/特异度、AUC，以及在 1024 全局图特异度下的敏感度；
Messidor-2 与 IDRiD 另给去除相机复现斑点后（07h）的结果。区间为按单位重抽样的 bootstrap（同 08n）。
输出 results/08q_highres_global/{metrics.tsv, contrasts.tsv}。"""
from __future__ import annotations

import importlib
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.boot2 import Resampler, summarize, wkappa, wshare  # noqa: E402
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402

n8 = importlib.import_module("scripts.stages.08n_camera_prior")
INPUTS = {"G512": ("orig", "G512_s{}"), "G1024": ("orig", "G1024_s{}"), "C": ("orig", "C_k4_s{}_aux"),
          "Chg": ("orig", "C_k4_s{}_auxhg")}
INPUTS_CAM = {"G1024cam": ("cam", "G1024_s{}"), "Chgcam": ("cam", "C_k4_s{}_auxhg")}
CONTRASTS = [("Chg", "G1024"), ("Chg", "C"), ("Chg", "G512"), ("Chgcam", "G1024cam")]
STATS = n8.STATS
REF = "G1024"                     # 匹配特异度的参照：1024 全局图


def one_cohort(args):
    cohort, probs, y, units, B, seed = args
    ref = y >= 2
    lab = {k: p.argmax(1) for k, p in probs.items()}
    sc = {k: -p[:, :2].sum(1) for k, p in probs.items()}

    def stats(w):
        ww = np.ones(len(y)) if w is None else w
        spec = ww[~ref][lab[REF][~ref] < 2].sum() / ww[~ref].sum()
        out = {}
        for k, p in lab.items():
            out[f"qwk_{k}"] = wkappa(y, p, w)
            for name, (sel, hit) in STATS.items():
                out[f"{name}_{k}"] = wshare(sel(y), hit(p), w)
            out[f"auc_{k}"] = roc_auc_score(ref, sc[k], sample_weight=ww)
            out[f"msens_{k}"] = n8.matched_sens(sc[k], ref, spec, ww)
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
        if a not in probs or b not in probs:
            continue
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
        spec = dict(INPUTS)
        if coh in ("messidor2", "idrid"):
            spec.update(INPUTS_CAM)
        got = {k: n8.ensemble(s, key, coh, man, drid) for k, (s, key) in spec.items()}
        keys = got["C"][3]
        assert all((g[3] == keys).all() for g in got.values())
        jobs.append((coh, {k: g[0] for k, g in got.items()}, got["C"][1], got["C"][2], B, cfg["seed"] + 300 + i))
    with Pool(len(jobs)) as pool:
        res = pool.map(one_cohort, jobs)
    out = ROOT / "results/08q_highres_global"; out.mkdir(parents=True, exist_ok=True)
    met = pd.DataFrame([r for rows, _ in res for r in rows]); con = pd.DataFrame([c for _, cs in res for c in cs])
    dev = []
    for sd in cfg["seeds"]:
        for tag in (f"G1024_s{sd}", f"C_k4_s{sd}_aux", f"C_k4_s{sd}_auxhg", f"G512_s{sd}"):
            lg = pd.read_csv(ROOT / "results/07_classify" / tag / "train_log.tsv", sep="\t")
            dev.append({"tag": tag, "best_dev_qwk": lg.dev_qwk.max(), "best_epoch": int(lg.dev_qwk.idxmax()),
                        "train_minutes": lg.minutes.iloc[-1]})
    met.to_csv(out / "metrics.tsv", sep="\t", index=False); con.to_csv(out / "contrasts.tsv", sep="\t", index=False)
    pd.DataFrame(dev).to_csv(out / "development.tsv", sep="\t", index=False)
    print(met[["cohort", "input", "qwk", "healthy_fp", "mild_missed", "ref_sens", "ref_spec", "auc", "msens"]].round(3).to_string())
    print(con[con.stat.isin(["qwk", "mild_missed", "healthy_fp", "auc"])].round(3).to_string())
    print(pd.DataFrame(dev).round(3).to_string())
    srcs = [ROOT / "results/07g_referable_scores" / f"C_k4_s{s}_auxhg.tsv.gz" for s in cfg["seeds"]]
    write_provenance("08q_highres_global", srcs, [out / "metrics.tsv", out / "contrasts.tsv", out / "development.tsv"],
                     seed=cfg["seed"], extra={"n_boot": B})


if __name__ == "__main__":
    main()
