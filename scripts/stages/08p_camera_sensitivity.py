"""相机复现斑点去除的敏感性分析（补充表）：主设置（阈值 5%、半径 4 格）与阈值 3%/10%、半径 2/8 格下，
Messidor-2 与 IDRiD 中被标记的分量与照片数，以及去除后全局 512 与病灶覆盖裁块（三个模型集成）的 κ、无 DR 判为 DR、
轻度判为无 DR、可转诊 AUC 和两者之差（bootstrap 95% 区间；Messidor-2 按检查、IDRiD 按照片重抽样，同 08n）。
主设置来自 05d_camera_prior / 07h_predict_camera，其余来自 *_<设置名>（scripts/run_camera_sensitivity.sh）。
输出 results/08p_camera_sensitivity/summary.tsv。"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.boot2 import Resampler, summarize, wkappa, wshare  # noqa: E402
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402

n8 = importlib.import_module("scripts.stages.08n_camera_prior")
SETTINGS = {"main": (0.05, 4, ""), "t03": (0.03, 4, "_t03"), "t10": (0.10, 4, "_t10"), "r2": (0.05, 2, "_r2"),
            "r8": (0.05, 8, "_r8")}
COHORTS = ("messidor2", "idrid")


def stats(y, probs, w):
    ww = np.ones(len(y)) if w is None else w
    out = {}
    for k, p in probs.items():
        lab = p.argmax(1)
        out[f"qwk_{k}"] = wkappa(y, lab, w)
        out[f"healthy_fp_{k}"] = wshare(y == 0, lab >= 1, w)
        out[f"mild_missed_{k}"] = wshare(y == 1, lab == 0, w)
        out[f"auc_{k}"] = roc_auc_score(y >= 2, -p[:, :2].sum(1), sample_weight=ww)
    return out


def main():
    cfg = load_config(); B = cfg["stats"]["n_boot"]
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False, dtype={"patient_id": str},
                      low_memory=False)
    man = man[man.included.astype(int) == 1].copy(); man["grade"] = man.grade.astype(int)
    rows = []
    for name, (thr, rad, sfx) in SETTINGS.items():
        n8.SRC["cam"] = ROOT / "results" / f"07h_predict_camera{sfx}"
        comp = pd.read_csv(ROOT / "results" / f"05d_camera_prior{sfx}" / "components.tsv.gz", sep="\t")
        for i, coh in enumerate(COHORTS):
            got = {k: n8.ensemble(s, key, coh, man, None) for k, (s, key) in
                   (("G512cam", ("cam", "G512_s{}")), ("Ccam", ("cam", "C_k4_s{}_aux")))}
            y, units = got["Ccam"][1], got["Ccam"][2]
            probs = {k: g[0] for k, g in got.items()}
            obs = stats(y, probs, None)
            # 与 08n 相同的重抽样种子（同一队列同一组权重），主设置的区间与 08n 一致
            res = Resampler(units, np.random.default_rng(cfg["seed"] + 300 + cfg["camera_prior"]["roles"].index(coh)))
            draws = {k: [] for k in obs}
            for _ in range(B):
                for k, v in stats(y, probs, res.weights()).items():
                    draws[k].append(v)
            c = comp[comp.role == coh]
            r = {"setting": name, "threshold": thr, "radius_cells": rad, "cohort": coh,
                 "flagged_components": int(c.flagged.sum()), "flagged_share": float(c.flagged.mean()),
                 "photos_flagged": int(c[c.flagged == 1].image_key.nunique())}
            for st in ("qwk", "healthy_fp", "mild_missed", "auc"):
                for k in probs:
                    r[f"{st}_{k}"] = obs[f"{st}_{k}"]
                d = np.asarray(draws[f"{st}_Ccam"]) - np.asarray(draws[f"{st}_G512cam"])
                v = summarize(obs[f"{st}_Ccam"] - obs[f"{st}_G512cam"], list(d))
                r.update({f"{st}_diff": v[0], f"{st}_diff_lo": v[1], f"{st}_diff_hi": v[2]})
            rows.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    out = ROOT / "results/08p_camera_sensitivity"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "summary.tsv", sep="\t", index=False)
    write_provenance("08p_camera_sensitivity", [ROOT / "results/02_manifest/images.tsv"], [out / "summary.tsv"],
                     seed=cfg["seed"], extra={"n_boot": B})


if __name__ == "__main__":
    main()
