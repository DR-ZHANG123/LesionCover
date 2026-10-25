"""示例照片的选择（规则固定、不靠目视挑选），供病例图使用。
(1) 内部轻度 DR：DDR 测试区带病灶掩膜的轻度照片中，病灶覆盖模型（三种子平均）判对而全局 512 模型判为无 DR，
    且两者无 DR 概率之差最大者；
(2) 外部健康眼误判：IDRiD 无 DR 照片中，病灶覆盖裁块的只看裁块预测在三个种子上都判为有 DR、而固定象限裁块判为无 DR，
    且病灶覆盖裁块的有 DR 概率最大者。
输出 results/08f_cases/cases.tsv（照片、两种输入的等级概率、裁块坐标、注意力权重）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402

P = [f"p{i}" for i in range(5)]


def mean_pred(pattern, role, fn="predictions.tsv.gz"):
    ds = []
    for f in sorted((ROOT / "results/07_classify").glob(f"{pattern}/{fn}")):
        d = pd.read_csv(f, sep="\t")
        alt = f.parent / f"predictions_{role}.tsv.gz"
        if not (d.role == role).any() and fn == "predictions.tsv.gz" and alt.exists():
            d = pd.read_csv(alt, sep="\t")
        ds.append(d[d.role == role].set_index("image_key").sort_index())
    m = ds[0].copy()
    num = [c for c in m.columns if c.startswith("p") or c.startswith("attn")]
    m[num] = np.mean([d[num].to_numpy() for d in ds], 0)
    m["n_dr_votes"] = np.sum([(d[P].to_numpy().argmax(1) > 0) for d in ds], 0)
    return m


def crops_of(key, method):
    sel = pd.read_csv(ROOT / "results/05_selection/crops.tsv.gz", sep="\t")
    s = sel[(sel.image_key == key) & (sel.method == method)].sort_values("slot")
    return s[["x0", "y0"]].to_numpy()


def main():
    cfg = load_config()
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    man = man[man.included.astype(int) == 1]
    rows = []
    mild = man[(man.role == "ddr_test") & (man.grade.astype(int) == 1) & (man.mask_role == "seg_test")].image_key
    g, c = mean_pred("G512_s??", "ddr_test").loc[mild], mean_pred("C_k4_s??_aux", "ddr_test").loc[mild]
    ok = (c[P].to_numpy().argmax(1) == 1) & (g[P].to_numpy().argmax(1) == 0)
    cand = (g.p0 - c.p0)[ok].sort_values(ascending=False)
    k = cand.index[0]
    rows.append({"case": "internal_mild", "image_key": k, "grade": 1, "n_candidates": int(ok.sum()),
                 **{f"global_{p}": g.loc[k, p] for p in P}, **{f"cover_{p}": c.loc[k, p] for p in P},
                 **{f"attn{j}": c.loc[k, f"attn{j}"] for j in range(4)},
                 "crops_C": crops_of(k, "C_k4").tolist(), "crops_U": crops_of(k, "U_k4").tolist()})
    healthy = man[(man.role == "idrid") & (man.grade.astype(int) == 0)].image_key
    cc = mean_pred("C_k4_s??_aux", "idrid", "predictions_croponly.tsv.gz").loc[healthy]
    cu = mean_pred("U_k4_s??_aux", "idrid", "predictions_croponly.tsv.gz").loc[healthy]
    ok = (cc.n_dr_votes == 3) & (cu[P].to_numpy().argmax(1) == 0)
    k = (1 - cc.p0)[ok].sort_values(ascending=False).index[0]
    full = mean_pred("C_k4_s??_aux", "idrid").loc[k]
    rows.append({"case": "external_healthy", "image_key": k, "grade": 0, "n_candidates": int(ok.sum()),
                 **{f"croponlyU_{p}": cu.loc[k, p] for p in P}, **{f"croponlyC_{p}": cc.loc[k, p] for p in P},
                 **{f"cover_{p}": full[p] for p in P}, **{f"attn{j}": full[f"attn{j}"] for j in range(4)},
                 "crops_C": crops_of(k, "C_k4").tolist(), "crops_U": crops_of(k, "U_k4").tolist()})
    out = ROOT / "results/08f_cases"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "cases.tsv", sep="\t", index=False)
    print(pd.DataFrame(rows).T)
    write_provenance("08f_cases", [ROOT / "results/05_selection/crops.tsv.gz"], [out / "cases.tsv"], seed=cfg["seed"])


if __name__ == "__main__":
    main()
