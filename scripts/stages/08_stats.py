"""汇总分级结果。每种输入的最终分级器是三个种子模型的概率平均（开发区上对所有输入都优于单个种子）。
summary.tsv 中 *_mean 为该集成的指标，*_std 为三个单种子模型指标的标准差（逐种子指标见 per_run_metrics.tsv）。
外部眼级患者整群 bootstrap 主比较（C−N、C−D，Holm 校正）与其他比较、DDR 测试区与 IDRiD、Messidor-2 的照片级 bootstrap、
质量分层、混淆矩阵。输出 results/08_stats/。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.clsnet import quadratic_kappa  # noqa: E402
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402
from scripts.lib.boot2 import Resampler, summarize, wkappa, wshare  # noqa: E402
from scripts.lib.stats import cluster_bootstrap, eye_level, holm, metrics  # noqa: E402

P = [f"p{i}" for i in range(5)]
EXT = ("idrid", "messidor2", "eyepacs")   # 照片级评价的外部队列
CLUSTERED = ("messidor2", "eyepacs")       # 按检查（Messidor-2）或患者（EyePACS）重抽样；IDRiD 无患者编号，按照片


def load_runs(cfg):
    """{方法键: {种子: 预测表}}；方法键 = 规则_K[_训练方式]，如 C_k4_aux、N_k4（标准融合）、G512。"""
    import re
    runs = {}
    for d in sorted((ROOT / "results/07_classify").glob("*/")):
        f = d / "predictions.tsv.gz"
        if not (d / "PROVENANCE.json").exists():
            continue
        m = re.match(r"(?P<method>.+)_s(?P<seed>\d+)(?:_(?P<var>[a-z]+))?$", d.name)
        if m is None or d.name.rsplit("_", 1)[-1] in cfg.get("recipes", {}):   # 超参筛选运行只用于开发区比较
            continue
        key = m["method"] + (f"_{m['var']}" if m["var"] else "")
        pr = pd.read_csv(f, sep="\t")
        for role in EXT:
            fi = d / f"predictions_{role}.tsv.gz"
            if fi.exists() and not (pr.role == role).any():
                pr = pd.concat([pr, pd.read_csv(fi, sep="\t")])
        runs.setdefault(key, {})[int(m["seed"])] = pr
    return runs


def ensemble(by_seed: dict) -> pd.DataFrame:
    """各种子概率按照片平均；只保留所有种子都有预测的照片与队列。"""
    ds = [v.drop_duplicates(["role", "image_key"]).set_index(["role", "image_key"])[P] for _, v in sorted(by_seed.items())]
    idx = ds[0].index
    for d in ds[1:]:
        idx = idx.intersection(d.index)
    e = pd.DataFrame(np.mean([d.loc[idx].to_numpy() for d in ds], 0), index=idx, columns=P)
    return e.reset_index()


def mild_to_none(y, p):
    m = y == 1
    return float((p[m] == 0).mean())


def image_bootstrap_metric(preds: dict, y: np.ndarray, ref: str, alts: list, B: int, seed: int, fn) -> pd.DataFrame:
    """按照片重抽样的任意指标（各种子均值）之差。"""
    rng = np.random.default_rng(seed)
    n = len(y)
    arg = {m: [p.argmax(1) for p in v] for m, v in preds.items()}
    st = {m: [] for m in [ref] + alts}
    for _ in range(B):
        i = rng.integers(n, size=n)
        for m in st:
            st[m].append(np.mean([fn(y[i], a[i]) for a in arg[m]]))
    obs = {m: np.mean([fn(y, a) for a in arg[m]]) for m in st}
    return pd.DataFrame([{"contrast": f"{ref}-{a}", "diff": obs[ref] - obs[a], "ref": obs[ref], "alt": obs[a],
                          "lo": np.percentile(np.asarray(st[ref]) - st[a], 2.5),
                          "hi": np.percentile(np.asarray(st[ref]) - st[a], 97.5)} for a in alts])


def image_bootstrap(preds: dict, y: np.ndarray, ref: str, alts: list, B: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(y)
    arg = {m: [p.argmax(1) for p in v] for m, v in preds.items()}
    st = {m: [] for m in [ref] + alts}
    for _ in range(B):
        i = rng.integers(n, size=n)
        for m in st:
            st[m].append(np.mean([quadratic_kappa(y[i], a[i]) for a in arg[m]]))
    obs = {m: np.mean([quadratic_kappa(y, a) for a in arg[m]]) for m in st}
    rows = []
    for a in alts:
        d = np.asarray(st[ref]) - np.asarray(st[a])
        rows.append({"contrast": f"{ref}-{a}", "diff": obs[ref] - obs[a], "lo": np.percentile(d, 2.5),
                     "hi": np.percentile(d, 97.5), "p_boot": min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / B))})
    return pd.DataFrame(rows)


def unit_bootstrap(preds: dict, y: np.ndarray, units: np.ndarray, ref: str, alts: list, B: int, seed: int,
                   stat: str = "qwk") -> pd.DataFrame:
    """按单位（检查或患者）重抽样的集成指标之差；stat 为 qwk 或 mild_to_none。"""
    lab = {m: v[0].argmax(1) for m, v in preds.items()}
    f = (lambda p, w: wkappa(y, p, w)) if stat == "qwk" else (lambda p, w: wshare(y == 1, p == 0, w))
    res = Resampler(units, np.random.default_rng(seed))
    obs = {m: f(lab[m], None) for m in lab}
    draws = {m: [] for m in lab}
    for _ in range(B):
        w = res.weights()
        for m in lab:
            draws[m].append(f(lab[m], w))
    rows = []
    for a in alts:
        v = summarize(obs[ref] - obs[a], list(np.asarray(draws[ref]) - np.asarray(draws[a])))
        r = {"contrast": f"{ref}-{a}", "diff": v[0], "lo": v[1], "hi": v[2], "p_boot": v[3]}
        if stat != "qwk":
            r.update(ref=obs[ref], alt=obs[a])
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    out = ROOT / "results/08_stats"; out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False, dtype={"patient_id": str})
    man = man[man.included.astype(int) == 1].copy()
    man["grade"] = man.grade.astype(int)
    drid = man[man.role == "deepdrid"].copy()
    drid["patient_id"] = drid.patient_id.astype(int)
    drid["quality"] = pd.to_numeric(drid.quality, errors="coerce")
    runs = load_runs(cfg)
    rows, eyes = [], {}
    for m, by_seed in runs.items():
        for s, pr in sorted(by_seed.items()):
            for role in ["ddr_dev", "ddr_test", *EXT]:
                d = pr[pr.role == role].merge(man[["image_key", "grade"]], on="image_key")
                if not len(d):
                    continue
                rows.append({"method": m, "seed": s, "set": role, **metrics(d.grade, d[P].to_numpy().argmax(1))})
            e = eye_level(pr[pr.role == "deepdrid"], drid).sort_values(["patient_id", "eye"]).reset_index(drop=True)
            eyes.setdefault(m, []).append(e)
            rows.append({"method": m, "seed": s, "set": "deepdrid_eye", **metrics(e.grade, e[P].to_numpy().argmax(1))})
            for qv, lab in [(1, "deepdrid_eye_good_quality"), (0, "deepdrid_eye_poor_quality")]:
                sub = e[(e.quality >= 1) if qv == 1 else (e.quality < 1)]
                rows.append({"method": m, "seed": s, "set": lab, "n": len(sub),
                             **metrics(sub.grade, sub[P].to_numpy().argmax(1))})
    per = pd.DataFrame(rows)
    per.to_csv(out / "per_run_metrics.tsv", sep="\t", index=False)
    num = [c for c in per.columns if c not in ("method", "seed", "set")]
    sd = per.groupby(["method", "set"])[num].std()
    nseed = {m: len(v) for m, v in runs.items()}
    runs = {m: {0: ensemble(v)} for m, v in runs.items()}
    rows, eyes = [], {}
    for m, by_seed in runs.items():
        pr = by_seed[0]
        for role in ["ddr_dev", "ddr_test", *EXT]:
            d = pr[pr.role == role].merge(man[["image_key", "grade"]], on="image_key")
            if len(d):
                rows.append({"method": m, "set": role, **metrics(d.grade, d[P].to_numpy().argmax(1))})
        e = eye_level(pr[pr.role == "deepdrid"], drid).sort_values(["patient_id", "eye"]).reset_index(drop=True)
        eyes[m] = [e]
        rows.append({"method": m, "set": "deepdrid_eye", **metrics(e.grade, e[P].to_numpy().argmax(1))})
        for qv, lab in [(1, "deepdrid_eye_good_quality"), (0, "deepdrid_eye_poor_quality")]:
            sub = e[(e.quality >= 1) if qv == 1 else (e.quality < 1)]
            rows.append({"method": m, "set": lab, "n": len(sub), **metrics(sub.grade, sub[P].to_numpy().argmax(1))})
    ens = pd.DataFrame(rows).set_index(["method", "set"])
    summ = ens[num].add_suffix("_mean").join(sd[num].add_suffix("_std"))
    summ["n_seeds"] = [nseed[m] for m, _ in summ.index]
    summ = summ.reset_index()
    summ.to_csv(out / "summary.tsv", sep="\t", index=False)

    B, seed = cfg["stats"]["n_boot"], cfg["seed"]
    var = cfg["classifier"]["main_variant"]
    sfx = f"_{var}" if var else ""
    ref = f"C_k4{sfx}"
    if ref in eyes:
        main_alts = [a for a in [f"N_k4{sfx}", f"D_k4{sfx}"] if a in eyes]
        prim = cluster_bootstrap({k: eyes[k] for k in [ref] + main_alts}, ref, main_alts, B, seed)
        prim["p_holm"] = holm(prim.p_boot.to_numpy())
        prim.to_csv(out / "primary_contrasts_deepdrid.tsv", sep="\t", index=False)
        other = [a for a in eyes if a not in [ref] + main_alts and (a.endswith(sfx) or a.startswith("G"))
                 and nseed[a] == nseed[ref]]
        sec = cluster_bootstrap({k: eyes[k] for k in [ref] + other}, ref, other, B, seed)
        sec.to_csv(out / "other_contrasts_deepdrid.tsv", sep="\t", index=False)
        t = man[man.role == "ddr_test"][["image_key", "grade"]]
        keep = [ref] + main_alts + other
        preds = {m: [v[s][v[s].role == "ddr_test"].merge(t, on="image_key")[P].to_numpy() for s in sorted(v)]
                 for m, v in runs.items() if m in keep}
        y = runs[ref][sorted(runs[ref])[0]].query("role == 'ddr_test'").merge(t, on="image_key").grade.to_numpy()
        image_bootstrap(preds, y, ref, [m for m in keep if m != ref], B, seed).to_csv(
            out / "contrasts_ddr_test.tsv", sep="\t", index=False)
        # 照片级外部队列 IDRiD 与 Messidor-2（无患者编号）：κ 与轻度漏判的照片重抽样比较
        for role in EXT:
            ti = man[man.role == role][["image_key", "grade"]]
            have = [m for m in keep if (runs[m][0].role == role).any()]
            if ref not in have:
                continue
            pi = {m: [runs[m][0][runs[m][0].role == role].merge(ti, on="image_key").sort_values("image_key")[P].to_numpy()]
                  for m in have}
            sub = runs[ref][0].query("role == @role").merge(man[["image_key", "grade", "patient_id"]], on="image_key")
            sub = sub.sort_values("image_key")
            yi = sub.grade.to_numpy()
            alts_i = [m for m in have if m != ref]
            if role in CLUSTERED:
                ui = sub.patient_id.to_numpy()
                unit_bootstrap(pi, yi, ui, ref, alts_i, B, seed).to_csv(out / f"contrasts_{role}.tsv", sep="\t", index=False)
                unit_bootstrap(pi, yi, ui, ref, alts_i, B, seed, "mild_to_none").to_csv(
                    out / f"mild_to_none_{role}.tsv", sep="\t", index=False)
                if "G512" in pi:
                    unit_bootstrap(pi, yi, ui, "G512", [m for m in have if m != "G512"], B, seed).to_csv(
                        out / f"contrasts_{role}_vs_global.tsv", sep="\t", index=False)
                continue
            image_bootstrap(pi, yi, ref, alts_i, B, seed).to_csv(
                out / f"contrasts_{role}.tsv", sep="\t", index=False)
            image_bootstrap_metric(pi, yi, ref, alts_i, B, seed, mild_to_none).to_csv(
                out / f"mild_to_none_{role}.tsv", sep="\t", index=False)
            if "G512" in pi:
                image_bootstrap(pi, yi, "G512", [m for m in have if m != "G512"], B, seed).to_csv(
                    out / f"contrasts_{role}_vs_global.tsv", sep="\t", index=False)
        # 轻度 DR 被判为无 DR：DDR 测试区（照片）与 DeepDRiD（眼），病灶覆盖规则与各输入之比较
        image_bootstrap_metric(preds, y, ref, [m for m in keep if m != ref], B, seed, mild_to_none).to_csv(
            out / "mild_to_none_ddr_test.tsv", sep="\t", index=False)
        g512 = "G512"
        if g512 in preds:
            image_bootstrap_metric(preds, y, g512, [m for m in keep if m != g512], B, seed, mild_to_none).to_csv(
                out / "mild_to_none_ddr_test_vs_global.tsv", sep="\t", index=False)
        if "G512" in eyes:
            gl = [a for a in eyes if a != "G512" and nseed[a] == nseed["G512"] and (a.endswith(sfx) or a.startswith("G"))]
            cluster_bootstrap({k: eyes[k] for k in ["G512"] + gl}, "G512", gl, B, seed).to_csv(
                out / "contrasts_deepdrid_vs_global.tsv", sep="\t", index=False)
        print(prim.round(4).to_string()); print(sec.round(4).to_string())
    # 混淆矩阵：种子平均概率后取最大
    cms = []
    for m, lst in eyes.items():
        pm = np.mean([e[P].to_numpy() for e in lst], 0).argmax(1)
        y = lst[0].grade.to_numpy()
        cm = np.zeros((5, 5), int); np.add.at(cm, (y, pm), 1)
        cms += [{"method": m, "true": i, "pred": j, "n": int(cm[i, j])} for i in range(5) for j in range(5)]
    pd.DataFrame(cms).to_csv(out / "confusion_deepdrid_eye.tsv", sep="\t", index=False)
    print(summ[summ.set.isin(["ddr_test", "deepdrid_eye"])][["method", "set", "qwk_mean", "qwk_std", "macro_f1_mean",
                                                             "mild_to_none_mean", "moderate_plus_under_mean"]].round(4).to_string())
    write_provenance("08_stats", [ROOT / "results/02_manifest/images.tsv"], [out / "summary.tsv"], seed=seed,
                     extra={"seeds_per_input": nseed})


if __name__ == "__main__":
    main()
