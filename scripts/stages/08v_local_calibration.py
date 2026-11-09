"""部署步骤：用新队列的少量标注照片对齐当地分级标准（单独评价，主表不受影响）。

对每个外部队列，按单位（DeepDRiD/EyePACS 按患者，其余按照片）随机抽取校准集：每个等级抽 k 个样本，
其所属单位的全部样本一并划入校准集；其余单位为评价集。每种 k 重复 R 次（不同随机种子），只在评价集上计算指标。
网络权重全部不变，比较以下输出层处理：
  none        原输出（argmax）
  prior_em    无标签先验校正（Saerens EM，在评价集上估计等级先验；不用任何标签）
  bias_temp   校准集上拟合温度 + 每级偏置（5 个参数，对数似然）
  cutpoints   校准集上只重估 4 个划级切点：期望等级 s = Σk·p_k 的比例优势模型 P(y>k)=σ(a·(s−c_k))，
              斜率 a 在 DDR 开发区估计并固定，只估 c_k；按 s 落在哪两个切点之间定级（对应当地分级标准的移动）
输出 results/08v_local_calibration/{metrics.tsv, cutpoints.tsv}。"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402

v8s = importlib.import_module("scripts.stages.08s_lcnet_eval")
STAGE = "08v_local_calibration"


def ord_nll(s, y, a, cut, w=None):
    cum = np.c_[np.ones(len(s)), expit(a * (s[:, None] - cut[None])), np.zeros(len(s))]
    p = np.clip(cum[np.arange(len(s)), y] - cum[np.arange(len(s)), y + 1], 1e-12, 1)
    return -np.log(p).sum()


def to_cut(t):
    return t[0] + np.r_[0, np.cumsum(np.exp(t[1:]))]


def fit_ordinal(s, y, a=None, init=(0.5, 1.5, 2.5, 3.5)):
    """a 为 None 时同时估计斜率（开发区）；否则固定斜率只估切点。小样本时对切点间隔加弱先验（以 init 为中心），避免空等级导致发散。"""
    init = np.asarray(init, float); t0 = np.r_[init[0], np.log(np.diff(init))]
    if a is None:
        r = minimize(lambda t: ord_nll(s, y, np.exp(t[0]), to_cut(t[1:])), np.r_[np.log(3.0), t0], method="L-BFGS-B")
        return np.exp(r.x[0]), to_cut(r.x[1:])
    pen = lambda t: 0.5 * ((t - t0) ** 2).sum()
    r = minimize(lambda t: ord_nll(s, y, a, to_cut(t)) + pen(t), t0, method="L-BFGS-B")
    return a, to_cut(r.x)


def prior_em(p, src_prior, it=100):
    """Saerens EM：在目标样本上估计等级先验并重加权概率（不用标签）。"""
    q = src_prior.copy()
    for _ in range(it):
        w = p * (q / src_prior); w /= w.sum(1, keepdims=True)
        q_new = w.mean(0)
        if np.abs(q_new - q).max() < 1e-6:
            break
        q = q_new
    return w


def fit_bias_temp(p, y):
    lp = np.log(np.clip(p, 1e-8, 1))
    def f(t):
        z = log_softmax(lp / np.exp(t[0]) + np.r_[0, t[1:]], 1)
        return -z[np.arange(len(y)), y].sum() + 0.5 * (t[1:] ** 2).sum()
    t = minimize(f, np.zeros(5), method="L-BFGS-B").x
    return lambda q: log_softmax(np.log(np.clip(q, 1e-8, 1)) / np.exp(t[0]) + np.r_[0, t[1:]], 1)


def metrics(y, pred):
    from sklearn.metrics import cohen_kappa_score
    m = y == 1
    return {"qwk": cohen_kappa_score(y, pred, weights="quadratic"), "accuracy": np.mean(y == pred),
            "mild_to_moderate_plus": np.mean(pred[m] >= 2) if m.any() else np.nan,
            "mild_to_none": np.mean(pred[m] == 0) if m.any() else np.nan,
            "ref_sens": np.mean(pred[y >= 2] >= 2), "ref_spec": np.mean(pred[y < 2] < 2)}


def split(y, units, k, rng):
    cal_units = set()
    for g in range(5):
        idx = np.flatnonzero(y == g)
        if len(idx):
            cal_units |= set(units[rng.choice(idx, min(k, len(idx)), replace=False)])
    cal = np.isin(units, list(cal_units))
    return cal, ~cal


def main():
    cfg = load_config(); lcfg = cfg["local_calibration"]
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False, dtype={"patient_id": str})
    mm = pd.read_csv(ROOT / "results/01b_mmrdr/images.tsv", sep="\t", keep_default_na=False); mm["patient_id"] = ""
    man = pd.concat([man, mm[["image_key", "role", "grade", "included", "patient_id"]]])
    man = man[man.included.astype(int) == 1].copy(); man["grade"] = man.grade.astype(int)
    drid = man[man.role == "deepdrid"].copy(); drid["patient_id"] = drid.patient_id.astype(int)
    rows, cps = [], []
    for inp in lcfg["inputs"]:
        runs = v8s.load_runs(*v8s.INPUTS[inp])
        dev = v8s.ensemble(runs, "ddr_dev", man, drid)
        if dev is None:
            print("skip", inp); continue
        a, cut_dev = fit_ordinal(dev[0] @ np.arange(5), dev[1])
        src_prior = dev[0].mean(0)
        for coh in lcfg["cohorts"]:
            e = v8s.ensemble(runs, coh, man, drid)
            if e is None:
                continue
            p, y, units = e[0], e[1], np.asarray(e[2]).astype(str)
            if coh not in v8s.PATIENT_UNITS:
                units = np.asarray(e[3]).astype(str)
            s = p @ np.arange(5)
            for k in lcfg["per_grade"]:
                for r in range(lcfg["repeats"]):
                    rng = np.random.default_rng(1000 * k + r)
                    cal, ev = split(y, units, k, rng)
                    out = {"none": p[ev].argmax(1), "prior_em": prior_em(p[ev], src_prior).argmax(1)}
                    out["bias_temp"] = fit_bias_temp(p[cal], y[cal])(p[ev]).argmax(1)
                    _, c = fit_ordinal(s[cal], y[cal], a=a, init=cut_dev)
                    out["cutpoints"] = np.digitize(s[ev], c)
                    cps.append({"input": inp, "cohort": coh, "per_grade": k, "repeat": r,
                                **{f"cut_{j + 1}": c[j] for j in range(4)}, **{f"dev_cut_{j + 1}": cut_dev[j] for j in range(4)}})
                    for meth, pred in out.items():
                        rows.append({"input": inp, "cohort": coh, "per_grade": k, "repeat": r, "method": meth,
                                     "n_cal": int(cal.sum()), "n_eval": int(ev.sum()), **metrics(y[ev], pred)})
            print(inp, coh, flush=True)
    out = ROOT / "results" / STAGE; out.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows); d.to_csv(out / "metrics_by_repeat.tsv", sep="\t", index=False)
    g = d.groupby(["input", "cohort", "per_grade", "method"])
    summ = g.mean(numeric_only=True).drop(columns="repeat").join(g.qwk.std().rename("qwk_sd")).reset_index()
    summ.to_csv(out / "metrics.tsv", sep="\t", index=False)
    pd.DataFrame(cps).to_csv(out / "cutpoints.tsv", sep="\t", index=False)
    write_provenance(STAGE, [ROOT / "results/02_manifest/images.tsv"],
                     [out / "metrics.tsv", out / "metrics_by_repeat.tsv", out / "cutpoints.tsv"],
                     extra={"per_grade": lcfg["per_grade"], "repeats": lcfg["repeats"]})


if __name__ == "__main__":
    main()
