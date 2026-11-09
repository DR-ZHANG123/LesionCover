"""各队列隐含的分级标准：用同一病灶分割器的计数（同一把“尺子”）比较不同队列在多大病灶负荷处划分等级。

做法：
1. 每张照片的病灶特征 x（各类病灶分量总数 log1p，以及各类病灶出现的象限数），按 DDR 训练区均值/标准差标准化。
2. 在 DDR 训练区拟合比例优势有序回归：P(级别 > k) = sigmoid(β·x − c_k)，得到病灶负荷指数 s = β·x。
3. 每个队列固定 β，只拟合四个切点 c_k（该队列在负荷指数上的划级位置）；与 DDR 训练区切点之差即该队列分级标准的偏移
   （正值 = 需要更多病灶才判为更高等级）。按患者（无患者编号时按照片）重抽样给 95% 区间。
4. 只做描述分析，不参与任何模型或设置的选择。
输出 results/08u_grading_standard/{cutpoints.tsv, burden_by_grade.tsv, model.tsv}。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "08u_grading_standard"
ROLES = ["ddr_train", "ddr_dev", "ddr_test", "eyepacs", "messidor2", "deepdrid", "idrid", "mmrdr"]
LES = ["MA", "HE", "EX", "SE"]


def load(cfg):
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", low_memory=False, dtype={"patient_id": str})
    mm = pd.read_csv(ROOT / "results/01b_mmrdr/images.tsv", sep="\t", low_memory=False)
    mm["patient_id"] = np.nan
    man = pd.concat([man[["image_key", "role", "grade", "included", "patient_id"]],
                     mm[["image_key", "role", "grade", "included", "patient_id"]]])
    man = man[(man.included == 1) & man.grade.notna() & man.role.isin(ROLES)].copy()
    tdir = ROOT / cfg["paths"]["processed"] / "lesion_tokens"
    X, keep = [], []
    for i, k in enumerate(man.image_key):
        f = tdir / f"{k}.npz"
        if f.exists():
            c = np.load(f)["counts"].reshape(len(LES), 4)
            X.append(np.r_[np.log1p(c.sum(1)), (c >= 1).sum(1)]); keep.append(i)
    man = man.iloc[keep].reset_index(drop=True)
    man["grade"] = man.grade.astype(int)
    man["unit"] = np.where(man.patient_id.notna() & (man.patient_id.astype(str) != ""), man.role + "_" + man.patient_id.astype(str),
                           man.image_key)
    return man, np.asarray(X, np.float64)


def nll(s, y, cut, w=None):
    """比例优势模型负对数似然；s 为负荷指数，cut 递增的 4 个切点。"""
    cum = np.c_[np.ones(len(s)), expit(s[:, None] - cut[None]), np.zeros(len(s))]   # P(y>k−1)，k=0..5
    p = np.clip(cum[np.arange(len(s)), y] - cum[np.arange(len(s)), y + 1], 1e-12, 1)
    return -(np.log(p) if w is None else w * np.log(p)).sum()


def to_cut(theta):
    return theta[0] + np.r_[0, np.cumsum(np.exp(theta[1:]))]


def fit_cuts(s, y, init, w=None):
    th0 = np.r_[init[0], np.log(np.diff(init))]
    r = minimize(lambda t: nll(s, y, to_cut(t), w), th0, method="L-BFGS-B")
    return to_cut(r.x)


def fit_full(X, y):
    p = X.shape[1]
    def f(t):
        return nll(X @ t[:p], y, to_cut(t[p:]))
    r = minimize(f, np.r_[np.zeros(p), -1.0, 0.0, 0.0, 0.0], method="L-BFGS-B")
    return r.x[:p], to_cut(r.x[p:])


def main():
    cfg = load_config(); seed = cfg.get("seed", 11); set_global_seed(seed)
    rng = np.random.default_rng(seed); nb = cfg["grading_standard"]["n_boot"]
    man, X = load(cfg)
    tr = (man.role == "ddr_train").to_numpy()
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Z = (X - mu) / sd
    beta, cut0 = fit_full(Z[tr], man.grade.to_numpy()[tr])
    man["burden_index"] = Z @ beta
    names = [f"log1p_total_{l}" for l in LES] + [f"quadrants_with_{l}" for l in LES]
    out = ROOT / "results" / STAGE; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": names + [f"cut_{k}" for k in range(1, 5)], "value": np.r_[beta, cut0],
                  "train_mean": np.r_[mu, [np.nan] * 4], "train_sd": np.r_[sd, [np.nan] * 4]}).to_csv(out / "model.tsv", sep="\t", index=False)
    rows = []
    for role in ROLES:
        d = man[man.role == role]
        s, y, u = d.burden_index.to_numpy(), d.grade.to_numpy(), d.unit.to_numpy()
        c = fit_cuts(s, y, cut0)
        uu, inv = np.unique(u, return_inverse=True)
        bs = []
        for _ in range(nb):                                  # 按单位重抽样（以抽中次数作权重）
            w = np.bincount(rng.integers(0, len(uu), len(uu)), minlength=len(uu))[inv].astype(float)
            if all(w[y == k].sum() > 0 for k in range(5)):
                bs.append(fit_cuts(s, y, c, w))
        bs = np.asarray(bs)
        for k in range(4):
            rows.append({"cohort": role, "boundary": f"grade>={k + 1}", "n": len(d), "n_units": len(uu), "cut": c[k],
                         "cut_lo": np.quantile(bs[:, k], 0.025), "cut_hi": np.quantile(bs[:, k], 0.975),
                         "shift": c[k] - cut0[k], "shift_lo": np.quantile(bs[:, k] - cut0[k], 0.025),
                         "shift_hi": np.quantile(bs[:, k] - cut0[k], 0.975)})
        print(role, len(d), np.round(c - cut0, 2), flush=True)
    cp = pd.DataFrame(rows); cp.to_csv(out / "cutpoints.tsv", sep="\t", index=False)
    bg = man.groupby(["role", "grade"]).agg(n=("grade", "size"), burden_median=("burden_index", "median"),
                                            burden_q25=("burden_index", lambda x: x.quantile(0.25)),
                                            burden_q75=("burden_index", lambda x: x.quantile(0.75))).reset_index()
    for i, l in enumerate(LES):
        bg[f"{l}_median"] = man.groupby(["role", "grade"]).apply(lambda d: np.median(np.expm1(X[d.index, i]))).to_numpy()
    bg.rename(columns={"role": "cohort"}).to_csv(out / "burden_by_grade.tsv", sep="\t", index=False)
    write_provenance(STAGE, [ROOT / "results/02_manifest/images.tsv", ROOT / "results/01b_mmrdr/images.tsv"],
                     [out / "cutpoints.tsv", out / "burden_by_grade.tsv", out / "model.tsv"], seed=seed,
                     extra={"bootstrap": nb, "unit": "patient where available, otherwise image"})


if __name__ == "__main__":
    main()
