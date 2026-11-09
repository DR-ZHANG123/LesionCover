"""分级错误的方向（描述分析）：各模型在每个队列中判高（预测 > 参考）与判低的比例、参考轻度判为 ≥2 级、
参考无 DR 判为 DR 的比例，并与该队列在 ≥2 级边界上的分级标准偏移（08u）并列。
三个模型概率平均后 argmax；DeepDRiD 为眼级、EyePACS 按患者重抽样，其余按照片；95% 区间为按单位重抽样。
输出 results/08y_error_direction/error_direction.tsv。"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402

v8s = importlib.import_module("scripts.stages.08s_lcnet_eval")
STAGE = "08y_error_direction"
INPUTS = ["LesionCover", "DINOv2 global tokens only", "Global 1024", "Swin-T 1024", "GDRNet", "CABNet"]
STATS = {"over_graded": lambda y, p: (np.ones(len(y), bool), p > y), "under_graded": lambda y, p: (np.ones(len(y), bool), p < y),
         "mild_to_moderate_plus": lambda y, p: (y == 1, p >= 2), "no_dr_to_dr": lambda y, p: (y == 0, p >= 1)}


def main():
    cfg = load_config(); rng = np.random.default_rng(cfg["seed"]); B = cfg["stats"]["n_boot"] if "n_boot" in cfg["stats"] else 1000
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False, dtype={"patient_id": str})
    mm = pd.read_csv(ROOT / "results/01b_mmrdr/images.tsv", sep="\t", keep_default_na=False); mm["patient_id"] = ""
    man = pd.concat([man, mm[["image_key", "role", "grade", "included", "patient_id"]]])
    man = man[man.included.astype(int) == 1].copy(); man["grade"] = man.grade.astype(int)
    drid = man[man.role == "deepdrid"].copy(); drid["patient_id"] = drid.patient_id.astype(int)
    cp = pd.read_csv(ROOT / "results/08u_grading_standard/cutpoints.tsv", sep="\t")
    cp = cp[cp.boundary == "grade>=2"].set_index("cohort")
    rows = []
    for inp in INPUTS:
        runs = v8s.load_runs(*v8s.INPUTS[inp])
        for coh in v8s.COHORTS:
            e = v8s.ensemble(runs, coh, man, drid)
            if e is None:
                continue
            p, y, units = e[0].argmax(1), e[1], np.asarray(e[2]).astype(str)
            uu, inv = np.unique(units, return_inverse=True)
            W = np.stack([np.bincount(rng.integers(0, len(uu), len(uu)), minlength=len(uu))[inv] for _ in range(B)]).astype(float)
            for name, f in STATS.items():
                sel, hit = f(y, p)
                est = hit[sel].mean() if sel.any() else np.nan
                bs = (W[:, sel] * hit[sel]).sum(1) / np.maximum(W[:, sel].sum(1), 1e-9)
                rows.append({"input": inp, "cohort": coh, "stat": name, "n": int(sel.sum()), "value": est,
                             "lo": np.quantile(bs, 0.025), "hi": np.quantile(bs, 0.975),
                             "standard_shift_grade2": cp.loc[coh, "shift"]})
        print(inp, flush=True)
    out = ROOT / "results" / STAGE; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "error_direction.tsv", sep="\t", index=False)
    write_provenance(STAGE, [ROOT / "results/08u_grading_standard/cutpoints.tsv"], [out / "error_direction.tsv"],
                     seed=cfg["seed"], extra={"n_boot": B, "inputs": INPUTS})


if __name__ == "__main__":
    main()
