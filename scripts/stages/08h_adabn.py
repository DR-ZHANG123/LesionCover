"""病灶图按相机适配归一化统计（04c/04b_calibration_adabn/05b/07e）后的分级：U/N/D/C 四种规则（辅助头主设置，三个模型概率平均）
在四个评价队列的 κ、无 DR 判对、轻度漏判与只看裁块的无 DR 判对，并与原始病灶图并列；以及无 DR 与轻度照片的预测微动脉瘤中位数。
输出 results/08h_adabn/{summary.tsv, ma_counts.tsv}。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402
from scripts.lib.stats import eye_level, metrics  # noqa: E402

P = [f"p{i}" for i in range(5)]
C = [f"c{i}" for i in range(5)]
ROLES = ("ddr_test", "deepdrid", "idrid", "messidor2")


def score(e, role, cols, man, drid):
    q = e[e.role == role][["image_key"] + cols].rename(columns=dict(zip(cols, P)))
    if role == "deepdrid":
        q = eye_level(q, drid)
    else:
        q = q.merge(man[["image_key", "grade"]], on="image_key")
    return metrics(q.grade, q[P].to_numpy().argmax(1))


def ensemble(rule, fn, extra=None):
    ds = []
    for d in sorted((ROOT / "results/07_classify").glob(f"{rule}_k4_s??_aux/")):
        t = pd.read_csv(d / fn, sep="\t")
        if extra is not None:
            t = pd.concat([t] + [pd.read_csv(d / x, sep="\t") for x in extra if (d / x).exists()])
        ds.append(t.drop_duplicates(["role", "image_key"]).set_index(["role", "image_key"]).sort_index())
    cols = [c for c in ds[0].columns if c in P + C]
    e = pd.DataFrame(np.mean([d[cols].to_numpy() for d in ds], 0), index=ds[0].index, columns=cols)
    return e.reset_index()


def main():
    cfg = load_config()
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    man = man[man.included.astype(int) == 1].copy(); man["grade"] = man.grade.astype(int)
    drid = man[man.role == "deepdrid"].copy(); drid["patient_id"] = drid.patient_id.astype(int)
    drid["quality"] = pd.to_numeric(drid.quality)
    rows = []
    for r in "UNDC":
        ad = ensemble(r, "predictions_adabn.tsv.gz")
        og = ensemble(r, "predictions.tsv.gz", ["predictions_idrid.tsv.gz", "predictions_messidor2.tsv.gz"])
        co = ensemble(r, "predictions_croponly.tsv.gz", ["predictions_croponly_messidor2.tsv.gz"]).rename(columns=dict(zip(P, C)))
        for role in ROLES:
            for maps, e, ec in (("original", og, co), ("adapted", ad, ad)):
                full, crop = score(e, role, P, man, drid), score(ec, role, C, man, drid)
                rows.append({"rule": r, "cohort": role, "maps": maps, "qwk": full["qwk"], "nodr_correct": full["recall_0"],
                             "mild_to_none": full["mild_to_none"], "croponly_nodr_correct": crop["recall_0"]})
    out = ROOT / "results/08h_adabn"; out.mkdir(parents=True, exist_ok=True)
    s = pd.DataFrame(rows); s.to_csv(out / "summary.tsv", sep="\t", index=False)
    a = pd.read_csv(ROOT / "results/05b_adabn_crops/pred_components.tsv", sep="\t").merge(man[["image_key", "role", "grade"]])
    o = pd.read_csv(ROOT / "results/08g_healthy_spots/per_photo.tsv.gz", sep="\t")[["image_key", "n_ma_total"]]
    a = a.merge(o, on="image_key")
    c = a[a.grade <= 1].groupby(["role", "grade"]).agg(ma_original=("n_ma_total", "median"), ma_adapted=("MA", "median")).reset_index()
    c.to_csv(out / "ma_counts.tsv", sep="\t", index=False)
    print(s.round(3).to_string()); print(c)
    write_provenance("08h_adabn", [ROOT / "results/05b_adabn_crops/crops.tsv.gz"], [out / "summary.tsv"], seed=cfg["seed"])


if __name__ == "__main__":
    main()
