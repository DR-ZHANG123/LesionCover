"""部署步骤：用新队列的少量标注照片对齐当地分级标准（标准层重估；网络权重不变，单独评价）。

设计（每个外部队列）：
  1. 固定划分：按单位（DeepDRiD/EyePACS 按患者，Messidor-2 按检查，其余按照片）以单位内最高等级分层，
     一半单位为“校准池”，另一半为固定评价集；所有模型、所有 k、所有抽样都在同一评价集上计算指标。
  2. 每种 k（每级标注数）重复 R 次：从校准池按等级各抽 k 个样本（不足时取全部），其所属单位全部划入校准集；
     另设“自然比例”抽样：从校准池随机抽 5k 个样本（不按等级平衡）。
  3. 输出层处理：none（DDR 标准层原样）、prior_em（无标签先验校正）、temp_only、bias_only、bias_temp（标准层五参数）、
     cutpoints（期望等级上的比例优势切点，斜率取 DDR 开发区）。
  4. 配对比较：参照模型与每个对照在同一抽样、同一评价集上的 κ 差；不确定性同时包含校准抽样与评价集单位重抽样
     （每次自助随机取一次抽样，再按单位重抽样评价集），五个队列的平均差用同一自助序号合成。
输出 results/08v_local_calibration/{metrics.tsv, metrics_by_repeat.tsv, cutpoints.tsv, paired.tsv, eval_sizes.tsv}。"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import expit, log_softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402

v8s = importlib.import_module("scripts.stages.08s_lcnet_eval")
STAGE = "08v_local_calibration"
W_Q = (np.arange(5)[:, None] - np.arange(5)[None]) ** 2 / 16.0


def ord_nll(s, y, a, cut):
    cum = np.c_[np.ones(len(s)), expit(a * (s[:, None] - cut[None])), np.zeros(len(s))]
    p = np.clip(cum[np.arange(len(s)), y] - cum[np.arange(len(s)), y + 1], 1e-12, 1)
    return -np.log(p).sum()


def to_cut(t):
    return t[0] + np.r_[0, np.cumsum(np.exp(t[1:]))]


def fit_ordinal(s, y, a=None, init=(0.5, 1.5, 2.5, 3.5)):
    """a 为 None 时同时估计斜率（开发区）；否则固定斜率只估切点，切点间隔加弱先验（以 init 为中心），避免空等级导致发散。"""
    init = np.asarray(init, float); t0 = np.r_[init[0], np.log(np.diff(init))]
    if a is None:
        r = minimize(lambda t: ord_nll(s, y, np.exp(t[0]), to_cut(t[1:])), np.r_[np.log(3.0), t0], method="L-BFGS-B")
        return np.exp(r.x[0]), to_cut(r.x[1:])
    r = minimize(lambda t: ord_nll(s, y, a, to_cut(t)) + 0.5 * ((t - t0) ** 2).sum(), t0, method="L-BFGS-B")
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


def fit_layer(p, y, temp=True, bias=True):
    """标准层：log p / T + b（b_0 = 0）；对偏置加弱高斯先验。temp/bias 为 False 时固定 T = 1 或 b = 0。"""
    lp = np.log(np.clip(p, 1e-8, 1))
    def unpack(t):
        return (np.exp(t[0]) if temp else 1.0), (np.r_[0, t[1:]] if bias else np.zeros(5))
    def f(t):
        T, b = unpack(t)
        z = log_softmax(lp / T + b, 1)
        return -z[np.arange(len(y)), y].sum() + 0.5 * (t[1:] ** 2).sum()
    T, b = unpack(minimize(f, np.zeros(5), method="L-BFGS-B").x)
    return lambda q: (np.log(np.clip(q, 1e-8, 1)) / T + b).argmax(1)


def wkappa(y, pred, w=None):
    """二次加权 κ（可带样本权重，用于单位重抽样）。"""
    w = np.ones(len(y)) if w is None else w
    O = np.bincount(y * 5 + pred, weights=w, minlength=25).reshape(5, 5)
    n = O.sum()
    E = np.outer(O.sum(1), O.sum(0)) / max(n, 1e-12)
    den = (W_Q * E).sum()
    return 1 - (W_Q * O).sum() / den if den > 0 else np.nan


def metrics(y, pred):
    m = y == 1
    return {"qwk": wkappa(y, pred), "accuracy": np.mean(y == pred),
            "mild_to_moderate_plus": np.mean(pred[m] >= 2) if m.any() else np.nan,
            "mild_to_none": np.mean(pred[m] == 0) if m.any() else np.nan,
            "ref_sens": np.mean(pred[y >= 2] >= 2), "ref_spec": np.mean(pred[y < 2] < 2)}


def fixed_split(y, units, seed):
    """以单位内最高等级分层，一半单位为校准池、一半为固定评价集。"""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"u": units, "y": y}).groupby("u").y.max()
    pool = set()
    for g in range(5):
        uu = d.index[d.values == g].to_numpy()
        rng.shuffle(uu); pool |= set(uu[: len(uu) // 2])
    inpool = np.isin(units, list(pool))
    return inpool, ~inpool


def draw(y, units, inpool, k, scheme, rng):
    idx_pool = np.flatnonzero(inpool); cal_units = set()
    if scheme == "balanced":
        for g in range(5):
            idx = idx_pool[y[idx_pool] == g]
            if len(idx):
                cal_units |= set(units[rng.choice(idx, min(k, len(idx)), replace=False)])
    else:
        cal_units = set(units[rng.choice(idx_pool, min(5 * k, len(idx_pool)), replace=False)])
    return np.isin(units, list(cal_units))


def one_draw(p, y, s, units, inpool, ev, k, scheme, r, a, cut_dev, src_prior, base_seed):
    rng = np.random.default_rng(base_seed + 100003 * k + 7919 * r + (0 if scheme == "balanced" else 50021))
    cal = draw(y, units, inpool, k, scheme, rng)
    pc, yc, pe = p[cal], y[cal], p[ev]
    out = {"none": pe.argmax(1), "prior_em": prior_em(pe, src_prior).argmax(1),
           "bias_temp": fit_layer(pc, yc)(pe)}
    if scheme == "balanced":
        out["temp_only"] = fit_layer(pc, yc, bias=False)(pe)
        out["bias_only"] = fit_layer(pc, yc, temp=False)(pe)
    _, c = fit_ordinal(s[cal], yc, a=a, init=cut_dev)
    out["cutpoints"] = np.digitize(s[ev], c)
    return int(cal.sum()), out, c


def run_input(inp, cohorts, man, drid, lcfg, seed):
    runs = v8s.load_runs(*v8s.INPUTS[inp])
    dev = v8s.ensemble(runs, "ddr_dev", man, drid)
    if dev is None:
        return None
    a, cut_dev = fit_ordinal(dev[0] @ np.arange(5), dev[1])
    src_prior = dev[0].mean(0)
    rows, cps, preds = [], [], {}
    for ci, coh in enumerate(cohorts):
        e = v8s.ensemble(runs, coh, man, drid)
        if e is None:
            continue
        p, y, units, keys = e[0], e[1], np.asarray(e[2]).astype(str), np.asarray(e[3]).astype(str)
        if coh not in v8s.PATIENT_UNITS:
            units = keys
        inpool, ev = fixed_split(y, units, seed + 31 * ci)
        s = p @ np.arange(5)
        for scheme in lcfg["schemes"]:
            for k in lcfg["per_grade"]:
                for r in range(lcfg["repeats"]):
                    n_cal, out, c = one_draw(p, y, s, units, inpool, ev, k, scheme, r, a, cut_dev, src_prior, seed + 31 * ci)
                    if scheme == "balanced":
                        cps.append({"input": inp, "cohort": coh, "per_grade": k, "repeat": r,
                                    **{f"cut_{j + 1}": c[j] for j in range(4)}, **{f"dev_cut_{j + 1}": cut_dev[j] for j in range(4)}})
                    for meth, pred in out.items():
                        rows.append({"input": inp, "cohort": coh, "scheme": scheme, "per_grade": k, "repeat": r, "method": meth,
                                     "n_cal": n_cal, "n_eval": int(ev.sum()), **metrics(y[ev], pred)})
                        if scheme == "balanced" and meth in ("none", "bias_temp"):
                            preds[(coh, k, r, meth)] = pred.astype(np.int8)
        preds[(coh, "eval")] = (keys[ev], y[ev], units[ev])
    print(inp, flush=True)
    return rows, cps, preds


def paired(allp, ref, inputs, cohorts, per_grade, R, B, seed):
    """参照模型 − 对照的 κ 差：每次自助随机取一次抽样，并按单位重抽样固定评价集；平均差按同一自助序号跨队列合成。"""
    rng = np.random.default_rng(seed)
    rows = []
    for k in per_grade:
        for meth in ("none", "bias_temp"):
            draws = rng.integers(0, R, B) if meth == "bias_temp" else np.zeros(B, int)
            for comp in inputs:
                if comp == ref or comp not in allp:
                    continue
                per_coh = []
                for coh in cohorts:
                    if (coh, "eval") not in allp[ref] or (coh, "eval") not in allp[comp]:
                        continue
                    keys, y, units = allp[ref][(coh, "eval")]
                    assert np.array_equal(keys, allp[comp][(coh, "eval")][0])
                    uu, inv = np.unique(units, return_inverse=True)
                    crng = np.random.default_rng(seed + 101 * cohorts.index(coh))
                    d = np.empty(B)
                    for b in range(B):
                        w = np.bincount(crng.integers(0, len(uu), len(uu)), minlength=len(uu))[inv].astype(float)
                        rr = draws[b] if meth == "bias_temp" else 0
                        pa, pb = allp[ref][(coh, k, rr, meth)], allp[comp][(coh, k, rr, meth)]
                        d[b] = wkappa(y, pa, w) - wkappa(y, pb, w)
                    est = np.mean([wkappa(y, allp[ref][(coh, k, r, meth)]) - wkappa(y, allp[comp][(coh, k, r, meth)])
                                   for r in (range(R) if meth == "bias_temp" else [0])])
                    per_coh.append(d)
                    rows.append({"reference": ref, "comparator": comp, "cohort": coh, "per_grade": k, "method": meth,
                                 "diff": est, "lo": np.quantile(d, 0.025), "hi": np.quantile(d, 0.975),
                                 "p_two_sided": min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))})
                if len(per_coh) == len(cohorts):
                    m = np.mean(per_coh, 0)
                    est = np.mean([r_["diff"] for r_ in rows[-len(cohorts):]])
                    rows.append({"reference": ref, "comparator": comp, "cohort": "mean", "per_grade": k, "method": meth,
                                 "diff": est, "lo": np.quantile(m, 0.025), "hi": np.quantile(m, 0.975),
                                 "p_two_sided": min(1.0, 2 * min(np.mean(m <= 0), np.mean(m >= 0)))})
    return pd.DataFrame(rows)


def main():
    cfg = load_config(); lcfg = cfg["local_calibration"]; seed = cfg["seed"]
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False, dtype={"patient_id": str})
    mm = pd.read_csv(ROOT / "results/01b_mmrdr/images.tsv", sep="\t", keep_default_na=False); mm["patient_id"] = ""
    man = pd.concat([man, mm[["image_key", "role", "grade", "included", "patient_id"]]])
    man = man[man.included.astype(int) == 1].copy(); man["grade"] = man.grade.astype(int)
    drid = man[man.role == "deepdrid"].copy(); drid["patient_id"] = drid.patient_id.astype(int)
    res = Parallel(n_jobs=lcfg["n_jobs"], backend="multiprocessing")(delayed(run_input)(inp, lcfg["cohorts"], man, drid, lcfg, seed) for inp in lcfg["inputs"])
    rows, cps, allp = [], [], {}
    for inp, r in zip(lcfg["inputs"], res):
        if r is None:
            print("skip", inp); continue
        rows += r[0]; cps += r[1]; allp[inp] = r[2]
    out = ROOT / "results" / STAGE; out.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows); d.to_csv(out / "metrics_by_repeat.tsv.gz", sep="\t", index=False, float_format="%.5f")
    g = d.groupby(["input", "cohort", "scheme", "per_grade", "method"])
    summ = g.mean(numeric_only=True).drop(columns="repeat").join(g.qwk.std().rename("qwk_sd")).join(
        g.qwk.quantile(0.025).rename("qwk_lo")).join(g.qwk.quantile(0.975).rename("qwk_hi")).reset_index()
    summ.to_csv(out / "metrics.tsv", sep="\t", index=False)
    pd.DataFrame(cps).to_csv(out / "cutpoints.tsv", sep="\t", index=False)
    ev = [{"cohort": coh, "grade": gg, "n_eval": int((allp[v8s.REF][(coh, "eval")][1] == gg).sum())}
          for coh in lcfg["cohorts"] for gg in range(5)]
    pd.DataFrame(ev).to_csv(out / "eval_sizes.tsv", sep="\t", index=False)
    pr = paired(allp, v8s.REF, lcfg["inputs"], lcfg["cohorts"], lcfg["per_grade"], lcfg["repeats"], lcfg["n_boot"], seed)
    pr.to_csv(out / "paired.tsv", sep="\t", index=False)
    outs = [out / f for f in ("metrics.tsv", "metrics_by_repeat.tsv.gz", "cutpoints.tsv", "paired.tsv", "eval_sizes.tsv")]
    write_provenance(STAGE, [ROOT / "results/02_manifest/images.tsv"], outs, seed=seed,
                     extra={k: lcfg[k] for k in ("per_grade", "repeats", "schemes", "n_boot")})


if __name__ == "__main__":
    main()
