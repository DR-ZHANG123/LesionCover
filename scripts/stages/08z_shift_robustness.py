"""分级标准偏移的稳健性（只做描述，不参与选择；主分析见 08u）。

(a) 参照与系数不确定性：以 DDR 训练区、开发区、测试区、开发+测试区分别作参照；联合自助（DDR 训练区单位与各队列单位同时重抽样，
    每次重新拟合负荷系数 β 与参照切点，再拟合各队列切点），给出 mild|moderate 边界偏移的 95% 区间。
(b) DDR 内部尺度：DDR 开发区与测试区照片合并后随机对半分，两半切点之差的分布（同一分级团队内部的波动）；
    各外部队列偏移与该分布第 97.5 百分位之比。
(c) 比例优势假设：每个队列分别拟合 (i) 只估切点（位置模型）与 (ii) 切点 + 尺度 σ（位置–尺度模型，P(y>k)=σ((s−c_k)/σ)）
    与 (iii) 每条边界各自斜率（部分比例优势），给出似然比检验与位置–尺度模型下的 mild|moderate 偏移。
（IDRiD 的病灶掩膜照片没有分级，人工计数无法用于切点比较；尺子与人工掩膜的一致性见 08w。）
输出 results/08z_shift_robustness/{reference.tsv, within_ddr.tsv, model_checks.tsv}。"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

u = importlib.import_module("scripts.stages.08u_grading_standard")
STAGE = "08z_shift_robustness"
EXT = ["deepdrid", "idrid", "messidor2", "eyepacs", "mmrdr"]
REFS = {"DDR training": ["ddr_train"], "DDR development": ["ddr_dev"], "DDR test": ["ddr_test"],
        "DDR development + test": ["ddr_dev", "ddr_test"]}


def unit_weights(units, rng):
    uu, inv = np.unique(units, return_inverse=True)
    return np.bincount(rng.integers(0, len(uu), len(uu)), minlength=len(uu))[inv].astype(float)


def fit_full_w(X, y, w, init):
    p = X.shape[1]
    r = minimize(lambda t: u.nll(X @ t[:p], y, u.to_cut(t[p:]), w), init, method="L-BFGS-B")
    return r.x


def boot_once(b, Z, y, units, roles, theta0, seed):
    rng = np.random.default_rng(seed + b)
    tr = roles == "ddr_train"
    w = unit_weights(units[tr], rng)
    th = fit_full_w(Z[tr], y[tr], w, theta0)
    p = Z.shape[1]; beta, cut_tr = th[:p], u.to_cut(th[p:])
    s = Z @ beta
    out = {"DDR training": cut_tr}; W = np.zeros(len(y))
    for role in ["ddr_dev", "ddr_test"] + EXT:
        m = roles == role
        ww = unit_weights(units[m], rng); W[m] = ww
        if not all(ww[y[m] == k].sum() > 0 for k in range(5)):
            return None
        out[role] = u.fit_cuts(s[m], y[m], cut_tr, ww)
    m = np.isin(roles, ["ddr_dev", "ddr_test"])
    out["DDR development + test"] = u.fit_cuts(s[m], y[m], cut_tr, W[m])
    return out


def reference_table(man, Z, cfg, seed):
    y, units, roles = man.grade.to_numpy(), man.unit.to_numpy(), man.role.to_numpy()
    tr = roles == "ddr_train"
    beta, cut0 = u.fit_full(Z[tr], y[tr])
    theta0 = np.r_[beta, cut0[0], np.log(np.diff(cut0))]
    s = Z @ beta
    point = {"DDR training": cut0}
    for role in ["ddr_dev", "ddr_test"] + EXT:
        m = roles == role; point[role] = u.fit_cuts(s[m], y[m], cut0)
    m = np.isin(roles, ["ddr_dev", "ddr_test"]); point["DDR development + test"] = u.fit_cuts(s[m], y[m], cut0)
    point["DDR development"], point["DDR test"] = point["ddr_dev"], point["ddr_test"]
    nb = cfg["shift_robustness"]["n_boot_joint"]
    bs = [r for r in Parallel(n_jobs=cfg["shift_robustness"]["n_jobs"], backend="multiprocessing")(
        delayed(boot_once)(b, Z, y, units, roles, theta0, seed) for b in range(nb)) if r is not None]
    for r in bs:
        r["DDR development"], r["DDR test"] = r["ddr_dev"], r["ddr_test"]
    rows = []
    for ref in REFS:
        for coh in EXT:
            for k in range(4):
                d = np.array([r[coh][k] - r[ref][k] for r in bs])
                rows.append({"reference": ref, "cohort": coh, "boundary": f"grade>={k + 1}",
                             "shift": point[coh][k] - point[ref][k], "lo": np.quantile(d, 0.025), "hi": np.quantile(d, 0.975),
                             "n_boot": len(bs)})
    return pd.DataFrame(rows), s


def within_ddr(man, s, cfg, seed):
    rng = np.random.default_rng(seed + 7)
    m = man.role.isin(["ddr_dev", "ddr_test"]).to_numpy()
    ss, yy = s[m], man.grade.to_numpy()[m]
    init = np.array([-0.3, 0.05, 1.5, 2.5])
    diffs = []
    for _ in range(cfg["shift_robustness"]["n_split"]):
        h = rng.permutation(len(ss)) < len(ss) // 2
        diffs.append(u.fit_cuts(ss[h], yy[h], init) - u.fit_cuts(ss[~h], yy[~h], init))
    diffs = np.abs(np.asarray(diffs))
    return pd.DataFrame([{"boundary": f"grade>={k + 1}", "median_abs_diff": np.median(diffs[:, k]),
                          "q975_abs_diff": np.quantile(diffs[:, k], 0.975), "n_split": len(diffs)} for k in range(4)])


def nll_ls(s, y, cut, scale):
    cum = np.c_[np.ones(len(s)), expit((s[:, None] - cut[None]) / scale), np.zeros(len(s))]
    p = np.clip(cum[np.arange(len(s)), y] - cum[np.arange(len(s)), y + 1], 1e-12, 1)
    return -np.log(p).sum()


def nll_npo(s, y, a, cut):
    """部分比例优势：每条边界各自斜率 a_k；累计概率不单调时截断。"""
    cum = expit(a[None] * s[:, None] - cut[None])
    cum = np.minimum.accumulate(cum, axis=1)
    cum = np.c_[np.ones(len(s)), cum, np.zeros(len(s))]
    p = np.clip(cum[np.arange(len(s)), y] - cum[np.arange(len(s)), y + 1], 1e-12, 1)
    return -np.log(p).sum()


def model_checks(man, s, cut0):
    rows = []
    for role in ["ddr_dev", "ddr_test"] + EXT:
        m = (man.role == role).to_numpy(); ss, yy = s[m], man.grade.to_numpy()[m]
        c_loc = u.fit_cuts(ss, yy, cut0); l_loc = u.nll(ss, yy, c_loc)
        th0 = np.r_[cut0[0], np.log(np.diff(cut0)), 0.0]
        r = minimize(lambda t: nll_ls(ss, yy, u.to_cut(t[:4]), np.exp(t[4])), th0, method="L-BFGS-B")
        c_ls, sc = u.to_cut(r.x[:4]), np.exp(r.x[4]); l_ls = r.fun
        r2 = minimize(lambda t: nll_npo(ss, yy, t[:4], t[4:]), np.r_[np.ones(4), c_loc], method="L-BFGS-B")
        l_npo = r2.fun
        rows.append({"cohort": role, "n": int(m.sum()), "scale": sc,
                     "shift_grade2_location": c_loc[1] - cut0[1], "shift_grade2_location_scale": c_ls[1] - cut0[1],
                     "lr_scale": 2 * (l_loc - l_ls), "p_scale": chi2.sf(2 * (l_loc - l_ls), 1),
                     "lr_partial_po": 2 * (l_loc - l_npo), "p_partial_po": chi2.sf(max(2 * (l_loc - l_npo), 0), 3),
                     **{f"slope_grade{k + 1}": r2.x[k] for k in range(4)}})
    return pd.DataFrame(rows)


def main():
    cfg = load_config(); seed = cfg.get("seed", 11); set_global_seed(seed)
    man, X = u.load(cfg)
    tr = (man.role == "ddr_train").to_numpy()
    Z = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-9)
    out = ROOT / "results" / STAGE; out.mkdir(parents=True, exist_ok=True)
    ref, s = reference_table(man, Z, cfg, seed); ref.to_csv(out / "reference.tsv", sep="\t", index=False)
    print(ref[ref.boundary == "grade>=2"].round(3).to_string(), flush=True)
    wd = within_ddr(man, s, cfg, seed); wd.to_csv(out / "within_ddr.tsv", sep="\t", index=False); print(wd, flush=True)
    _, cut0 = u.fit_full(Z[tr], man.grade.to_numpy()[tr])
    mc = model_checks(man, s, cut0); mc.to_csv(out / "model_checks.tsv", sep="\t", index=False); print(mc.round(3).to_string(), flush=True)
    write_provenance(STAGE, [ROOT / "results/02_manifest/images.tsv"],
                     [out / f for f in ("reference.tsv", "within_ddr.tsv", "model_checks.tsv")], seed=seed,
                     extra=cfg["shift_robustness"])


if __name__ == "__main__":
    main()
