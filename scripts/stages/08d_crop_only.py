"""只用裁块分级（辅助头）的统计：各规则在 DDR 测试区、DeepDRiD（眼级）、IDRiD 与 Messidor-2 的 κ、轻度漏判与无 DR 召回；
病灶覆盖与其他规则之差的 bootstrap 区间（照片级队列按照片、DeepDRiD 按患者）。各规则取三个种子的概率平均；
summary.tsv 中 *_mean 为集成指标，*_std 为单种子间标准差。输出 results/08d_crop_only/。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402
from scripts.lib.stats import eye_level, metrics  # noqa: E402

P = [f"p{i}" for i in range(5)]
RULES = ["U", "R", "T", "S", "N", "D", "Cmass", "C"]
COHORTS = ("ddr_test", "deepdrid", "idrid", "messidor2", "eyepacs")
CLUSTERED = ("messidor2", "eyepacs")      # 按检查或患者重抽样


def no_dr_recall(y, p):
    m = y == 0
    return float((p[m] == 0).mean())


def mild_none(y, p):
    m = y == 1
    return float((p[m] == 0).mean())


def main():
    cfg = load_config()
    B, rng = cfg["stats"]["n_boot"], np.random.default_rng(cfg["seed"])
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False, dtype={"patient_id": str})
    man = man[man.included.astype(int) == 1].copy(); man["grade"] = man.grade.astype(int)
    drid = man[man.role == "deepdrid"].copy(); drid["patient_id"] = drid.patient_id.astype(int)
    drid["quality"] = pd.to_numeric(drid.quality)
    data, rows = {}, []
    for r in RULES + ["C_art", "N_art", "C_mc", "N_mc"]:   # *_art 伪影增强；*_mc 多相机
        base, var = (r.rsplit("_", 1)[0], "aux" + r.rsplit("_", 1)[1]) if r.endswith(("_art", "_mc", "_rf")) else (r, "aux")
        for f in sorted((ROOT / "results/07_classify").glob(f"{base}_k4_s*_{var}/predictions_croponly.tsv.gz")):
            seed = int(re.search(r"_s(\d+)_aux", f.parent.name)[1])
            pr = pd.read_csv(f, sep="\t")
            for role in CLUSTERED:
                fm = f.parent / f"predictions_croponly_{role}.tsv.gz"
                if fm.exists():
                    pr = pd.concat([pr, pd.read_csv(fm, sep="\t")])
            for cohort in COHORTS:
                q = pr[pr.role == cohort]
                if not len(q):
                    continue
                if cohort == "deepdrid":
                    e = eye_level(q, drid).sort_values(["patient_id", "eye"]).reset_index(drop=True)
                    y, p, u = e.grade.to_numpy(), e[P].to_numpy(), e.patient_id.to_numpy()
                else:
                    d = q.merge(man[["image_key", "grade", "patient_id"]], on="image_key").sort_values("image_key")
                    y, p = d.grade.to_numpy(), d[P].to_numpy()
                    u = d.patient_id.to_numpy() if cohort in CLUSTERED else d.image_key.to_numpy()
                data.setdefault((r, cohort), []).append(p)
                data[("_y", cohort)], data[("_u", cohort)] = y, u
                rows.append({"rule": r, "seed": seed, "cohort": cohort, **metrics(y, p.argmax(1))})
    for (r, cohort), lst in list(data.items()):
        if r.startswith("_"):
            continue
        data[(r, cohort)] = [np.mean(lst, 0).argmax(1)]
        rows.append({"rule": r, "seed": "ensemble", "cohort": cohort, **metrics(data[("_y", cohort)], data[(r, cohort)][0])})
    per = pd.DataFrame(rows)
    out = ROOT / "results/08d_crop_only"; out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_run.tsv", sep="\t", index=False)
    num = [c for c in per.columns if c not in ("rule", "seed", "cohort")]
    s = per[per.seed == "ensemble"].set_index(["rule", "cohort"])[num].add_suffix("_mean").join(
        per[per.seed != "ensemble"].groupby(["rule", "cohort"])[num].std().add_suffix("_std"))
    s.reset_index().to_csv(out / "summary.tsv", sep="\t", index=False)
    cons = []
    for cohort, fn, name in (("ddr_test", mild_none, "mild_to_none"), ("deepdrid", no_dr_recall, "nodr_recall"),
                             ("idrid", no_dr_recall, "nodr_recall"), ("ddr_test", no_dr_recall, "nodr_recall"),
                             ("messidor2", no_dr_recall, "nodr_recall"), ("messidor2", mild_none, "mild_to_none"),
                             ("deepdrid", mild_none, "mild_to_none"), ("eyepacs", no_dr_recall, "nodr_recall"),
                             ("eyepacs", mild_none, "mild_to_none")):
        if any((r, cohort) not in data for r in RULES):
            continue
        y, u = data[("_y", cohort)], data[("_u", cohort)]
        uu, inv = np.unique(u, return_inverse=True)
        groups = [np.flatnonzero(inv == i) for i in range(len(uu))]
        stat = {}
        for r in RULES:
            stat[r] = []
        for _ in range(B):
            idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
            for r in RULES:
                stat[r].append(np.mean([fn(y[idx], p[idx]) for p in data[(r, cohort)]]))
        obs = {r: np.mean([fn(y, p) for p in data[(r, cohort)]]) for r in RULES}
        for r in RULES:
            if r == "C":
                continue
            d = np.asarray(stat["C"]) - np.asarray(stat[r])
            cons.append({"cohort": cohort, "metric": name, "contrast": f"C-{r}", "diff": obs["C"] - obs[r], "C": obs["C"],
                         "alt": obs[r], "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5)})
    pd.DataFrame(cons).to_csv(out / "contrasts.tsv", sep="\t", index=False)
    print(s.reset_index()[["rule", "cohort", "qwk_mean", "mild_to_none_mean", "recall_0_mean"]].round(3).to_string())
    print(pd.DataFrame(cons).round(3).to_string())
    write_provenance("08d_crop_only", [], [out / "summary.tsv", out / "contrasts.tsv"], seed=cfg["seed"])


if __name__ == "__main__":
    main()
