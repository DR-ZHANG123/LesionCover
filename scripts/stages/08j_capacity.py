"""编码器容量 × 训练相机多样性：全局 512、NMS 与病灶覆盖（辅助头）在 ConvNeXt-Tiny / Small / Base 编码器、
只用 DDR 或 DDR + APTOS 2019 训练时的分级。

统计量一律是每个设置三个模型概率平均（集成）后的指标，与正文其余部分一致；κ 之差即两个集成 κ 之差。
区间来自两层 bootstrap：每次重抽评价单位（DeepDRiD、EyePACS 按患者，其余按照片），并在每个设置内有放回地
重抽三个模型再平均其概率，使区间同时包含评价样本与模型训练的变异（lib/boot2.py）。同一次重抽的单位权重在各设置
间共享，故设置之间的差（编码器带来的变化）为配对比较。外部队列的差中差在每种编码器内对外部队列 × 训练集做 Holm 校正。

除 κ 外，对融合模型（正式输出）报告：无 DR 被判为 DR 的比例、轻度 DR 被判为无 DR 的比例、可转诊 DR（≥2 级）
的敏感度与特异度；对辅助头只看裁块的预测报告无 DR 被判为 DR 的比例（机制分析），并给出每个模型各自的比例。
输出 results/08j_capacity/{summary, contrasts, did, clinical, healthy_crop, healthy_crop_models}.tsv。"""
from __future__ import annotations

import importlib
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.boot2 import COMBO_INDEX, Resampler, combo_labels, full_labels, summarize, wkappa, wshare  # noqa: E402
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402
from scripts.lib.stats import eye_level, holm, metrics  # noqa: E402

v8c = importlib.import_module("scripts.stages.08c_variants")
P = [f"p{i}" for i in range(5)]
COHORTS = ("ddr_test", "deepdrid", "idrid", "messidor2", "eyepacs")
PATIENT_UNITS = ("deepdrid", "eyepacs")
TRAIN = {"ddr": "", "multicamera": "mc"}
ENC = {"tiny": "", "small": "small", "base": "base"}
INPUTS = ("G512", "N", "C")
CLIN = {"healthy_fp": (lambda y: y == 0, lambda p: p >= 1), "mild_missed": (lambda y: y == 1, lambda p: p == 0),
        "ref_sens": (lambda y: y >= 2, lambda p: p >= 2), "ref_spec": (lambda y: y < 2, lambda p: p < 2)}


def family(inp: str, train: str, enc: str) -> str:
    v = TRAIN[train] + ENC[enc]
    if inp == "G512":
        return "G512" + (f"_{v}" if v else "")
    return f"{inp}_k4_aux{v}"


def align(frames: list[pd.DataFrame], cohort: str, man: pd.DataFrame, drid: pd.DataFrame):
    """每个模型的概率表 → （对齐的概率数组列表, 参考等级, 重抽样单位, 行键）。DeepDRiD 先平均同眼两张照片。"""
    lst, y, units, keys = [], None, None, None
    for pr in frames:
        pr = pr[pr.role == cohort]
        if not len(pr):
            return None
        if cohort == "deepdrid":
            e = eye_level(pr, drid).sort_values(["patient_id", "eye"]).reset_index(drop=True)
            lst.append(e[P].to_numpy()); y, units = e.grade.to_numpy(), e.patient_id.to_numpy()
            keys = (e.patient_id.astype(str) + "_" + e.eye.astype(str)).to_numpy()
        else:
            d = pr.merge(man[["image_key", "grade", "patient_id"]], on="image_key").sort_values("image_key")
            lst.append(d[P].to_numpy()); y = d.grade.to_numpy(); keys = d.image_key.to_numpy()
            units = d.patient_id.to_numpy() if cohort in PATIENT_UNITS else keys
    return lst, y.astype(int), units, keys


def croponly_frames(var: str) -> list[pd.DataFrame] | None:
    fr = []
    for sd in (11, 22, 33):
        d = ROOT / "results/07_classify" / f"C_k4_s{sd}_{var}"
        fs = [d / "predictions_croponly.tsv.gz"] + [d / f"predictions_croponly_{r}.tsv.gz" for r in ("messidor2", "eyepacs")]
        if not ((d / "PROVENANCE.json").exists() and fs[0].exists()):
            return None
        fr.append(pd.concat([pd.read_csv(f, sep="\t") for f in fs if f.exists()]))
    return fr


def build_cells(man, drid):
    """{(训练, 编码器, 队列): {"y","units","keys", 输入: (三个模型概率, 10 种组合的预测等级), "Ccrop": ...}}。"""
    cells, rows, per_model = {}, [], []
    for train in TRAIN:
        for enc in ENC:
            runs = {inp: v8c.load(family(inp, train, enc)) for inp in INPUTS}
            if any(len(v) < 3 for v in runs.values()):
                print("skip", train, enc, {k: len(v) for k, v in runs.items()}); continue
            crop = croponly_frames("aux" + TRAIN[train] + ENC[enc])
            for coh in COHORTS:
                al = {inp: align([r[s] for s in sorted(r)], coh, man, drid) for inp, r in runs.items()}
                if any(a is None for a in al.values()):
                    continue
                y, units, keys = al["C"][1], al["C"][2], al["C"][3]
                assert all((a[3] == keys).all() for a in al.values())
                cell = {"y": y, "units": units, "keys": keys}
                for inp, (lst, *_rest) in al.items():
                    cell[inp] = combo_labels(lst)
                    rows.append({"training": train, "encoder": enc, "input": inp, "cohort": coh, "n": len(y),
                                 **metrics(y, full_labels(lst))})
                ca = align(crop, coh, man, drid) if crop is not None else None
                if ca is not None:
                    assert (ca[3] == keys).all()
                    cell["Ccrop"] = combo_labels(ca[0])
                    per_model += [{"training": train, "encoder": enc, "cohort": coh, "seed": sd,
                                   "called_dr": float((p.argmax(1)[y == 0] > 0).mean())}
                                  for sd, p in zip((11, 22, 33), ca[0])]
                cells[(train, enc, coh)] = cell
    return cells, pd.DataFrame(rows), pd.DataFrame(per_model)


def stats_of(cell: dict, pick: dict, w, n_in) -> dict:
    """一个设置在一次重抽中的全部统计量。pick：输入 → 组合序号（10 = 全部三个模型）。"""
    y, out = cell["y"], {}
    for inp in INPUTS:
        p = cell[inp][pick[inp]]
        out[f"qwk_{inp}"] = wkappa(y, p, w)
        for name, (sel, hit) in CLIN.items():
            out[f"{name}_{inp}"] = wshare(sel(y), hit(p), w)
    if "Ccrop" in cell:
        p = cell["Ccrop"][pick["Ccrop"]]
        out["crop_fp"] = wshare(y == 0, p >= 1, w)
        if n_in is not None:
            out["crop_fp_five"] = wshare((y == 0) & (n_in >= 5), p >= 1, w)
    return out


def run_cohort(args):
    coh, cells, n_in_map, B, seed = args
    rng = np.random.default_rng(seed)
    sets = sorted(k for k in cells if k[2] == coh)
    if not sets:
        return coh, {}, {}
    units = cells[sets[0]]["units"]
    n_in = None
    if n_in_map is not None and coh != "deepdrid":
        n_in = pd.Series(n_in_map).reindex(cells[sets[0]]["keys"]).to_numpy()
    allm = COMBO_INDEX[(0, 1, 2)]                          # 三个模型全部 = 观测的集成
    full = {"G512": allm, "N": allm, "C": allm, "Ccrop": allm}
    obs = {s: stats_of(cells[s], full, None, n_in) for s in sets}
    res = Resampler(units, rng)
    draws = {s: {k: [] for k in obs[s]} for s in sets}
    for _ in range(B):
        w = res.weights()
        for s in sets:
            pick = {k: res.combo() for k in INPUTS}
            pick["Ccrop"] = pick["C"]                      # 辅助头与融合输出来自同一组病灶覆盖模型
            st = stats_of(cells[s], pick, w, n_in)
            for k, v in st.items():
                draws[s][k].append(v)
    return coh, obs, {s: {k: np.asarray(v) for k, v in d.items()} for s, d in draws.items()}


def tables(obs, draws, cells):
    cons, dids, clin, hc = [], [], [], []
    for s, o in obs.items():
        train, enc, coh = s
        d = draws[s]
        for alt in ("G512", "N"):
            v = summarize(o["qwk_C"] - o[f"qwk_{alt}"], list(d["qwk_C"] - d[f"qwk_{alt}"]))
            cons.append({"training": train, "encoder": enc, "cohort": coh, "contrast": f"C-{alt}", "diff": v[0],
                         "lo": v[1], "hi": v[2], "p_boot": v[3]})
        y = cells[s]["y"]
        for name, (sel, _h) in CLIN.items():
            for inp in ("G512", "C"):
                v = summarize(o[f"{name}_{inp}"], list(d[f"{name}_{inp}"]))
                clin.append({"training": train, "encoder": enc, "cohort": coh, "stat": name, "input": inp,
                             "value": v[0], "lo": v[1], "hi": v[2], "n_eligible": int(sel(y).sum()),
                             "n_events": int(round(v[0] * sel(y).sum())) if v[0] == v[0] else 0})
            v = summarize(o[f"{name}_C"] - o[f"{name}_G512"], list(d[f"{name}_C"] - d[f"{name}_G512"]))
            clin.append({"training": train, "encoder": enc, "cohort": coh, "stat": name, "input": "C-G512",
                         "value": v[0], "lo": v[1], "hi": v[2], "p_boot": v[3], "n_eligible": int(sel(y).sum())})
        if "crop_fp" in o:
            v = summarize(o["crop_fp"], list(d["crop_fp"]))
            five = o.get("crop_fp_five", np.nan)
            hc.append({"training": train, "encoder": enc, "cohort": coh,
                       "unit": "eye" if coh in PATIENT_UNITS else "photograph", "n": int((y == 0).sum()),
                       "called_dr": v[0], "lo": v[1], "hi": v[2], "called_dr_five_plus": five})
    for (train, enc, coh), o in obs.items():
        if enc == "tiny" or (train, "tiny", coh) not in obs:
            continue
        t, dt, db = obs[(train, "tiny", coh)], draws[(train, "tiny", coh)], draws[(train, enc, coh)]
        spec = {"did_C-G512": lambda a, b: (b["qwk_C"] - b["qwk_G512"]) - (a["qwk_C"] - a["qwk_G512"]),
                "gain_C": lambda a, b: b["qwk_C"] - a["qwk_C"], "gain_G512": lambda a, b: b["qwk_G512"] - a["qwk_G512"]}
        if "crop_fp" in o and "crop_fp" in t:
            spec["change_crop_fp"] = lambda a, b: b["crop_fp"] - a["crop_fp"]
        spec["change_healthy_fp_C"] = lambda a, b: b["healthy_fp_C"] - a["healthy_fp_C"]
        for name, f in spec.items():
            v = summarize(f(t, o), list(f(dt, db)))
            dids.append({"training": train, "encoder": enc, "cohort": coh, "stat": name, "value": v[0], "lo": v[1],
                         "hi": v[2], "p_boot": v[3]})
    did = pd.DataFrame(dids)
    did["p_holm"] = np.nan
    for enc, g in did[(did.stat == "did_C-G512") & (did.cohort != "ddr_test")].groupby("encoder"):
        did.loc[g.index, "p_holm"] = holm(g.p_boot.to_numpy())
    return pd.DataFrame(cons), did, pd.DataFrame(clin), pd.DataFrame(hc)


def main():
    cfg = load_config(); B = cfg["stats"]["n_boot"]
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False, dtype={"patient_id": str})
    man = man[man.included.astype(int) == 1].copy(); man["grade"] = man.grade.astype(int)
    drid = man[man.role == "deepdrid"].copy(); drid["patient_id"] = drid.patient_id.astype(int)
    cells, summ, per_model = build_cells(man, drid)
    pp = ROOT / "results/08g_healthy_spots/per_photo.tsv.gz"     # 病灶覆盖裁块内的预测微动脉瘤数（各设置裁块相同）
    n_in = pd.read_csv(pp, sep="\t").set_index("image_key").n_in.to_dict() if pp.exists() else None
    jobs = [(coh, {k: v for k, v in cells.items() if k[2] == coh}, n_in, B, cfg["seed"] + i) for i, coh in enumerate(COHORTS)]
    with Pool(len(jobs)) as pool:
        res = pool.map(run_cohort, jobs)
    obs, draws = {}, {}
    for _coh, o, d in res:
        obs.update(o); draws.update(d)
    cons, did, clin, hc = tables(obs, draws, cells)
    out = ROOT / "results/08j_capacity"; out.mkdir(parents=True, exist_ok=True)
    for name, df in (("summary", summ), ("contrasts", cons), ("did", did), ("clinical", clin), ("healthy_crop", hc),
                     ("healthy_crop_models", per_model)):
        df.to_csv(out / f"{name}.tsv", sep="\t", index=False)
    print(cons.round(3).to_string()); print(did[did.stat == "did_C-G512"].round(3).to_string()); print(hc.round(3).to_string())
    write_provenance("08j_capacity", [ROOT / "results/02_manifest/images.tsv"],
                     [out / f"{n}.tsv" for n in ("summary", "contrasts", "did", "clinical", "healthy_crop", "healthy_crop_models")],
                     seed=cfg["seed"], extra={"n_boot": B, "estimand": "ensemble of three models"})


if __name__ == "__main__":
    main()
