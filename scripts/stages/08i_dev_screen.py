"""开发区比较：每个训练设置（输入 × variant，可含 recipe）在 DDR 开发区上的二次加权 κ（三个种子概率平均与单种子均值、
标准差），以及可用时的测试时增强结果。训练设置的取舍只依据本表。输出 results/08i_dev_screen/summary.tsv。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.clsnet import quadratic_kappa  # noqa: E402
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402

P = [f"p{i}" for i in range(5)]
PAT = re.compile(r"(?P<method>[A-Za-z0-9]+?)(?:_k(?P<k>\d+))?_s(?P<seed>\d+)(?:_(?P<rest>.+))?$")


def dev_probs(f: Path) -> pd.DataFrame | None:
    if not f.exists():
        return None
    d = pd.read_csv(f, sep="\t")
    d = d[d.role == "ddr_dev"]
    return d.set_index("image_key")[P].sort_index() if len(d) else None


def main():
    cfg = load_config()
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False).set_index("image_key")
    fam: dict[str, dict] = {}
    for d in sorted((ROOT / "results/07_classify").glob("*/")):
        m = PAT.match(d.name)
        if not m or not (d / "PROVENANCE.json").exists():
            continue
        key = m["method"] + (f"_k{m['k']}" if m["k"] else "") + (f"_{m['rest']}" if m["rest"] else "")
        fam.setdefault(key, {})[int(m["seed"])] = d
    rows = []
    for key, runs in sorted(fam.items()):
        r = {"setting": key, "n_seeds": len(runs)}
        for src, fname in (("single", "predictions.tsv.gz"), ("tta", "predictions_tta.tsv.gz")):
            ps = [p for p in (dev_probs(d / fname) for d in runs.values()) if p is not None]
            if len(ps) != len(runs):
                continue
            y = man.grade.loc[ps[0].index].astype(int).to_numpy()
            ks = [quadratic_kappa(y, p.to_numpy().argmax(1)) for p in ps]
            r[f"{src}_qwk_avg"] = quadratic_kappa(y, np.mean([p.to_numpy() for p in ps], 0).argmax(1))
            r[f"{src}_qwk_seed_mean"], r[f"{src}_qwk_seed_sd"] = float(np.mean(ks)), float(np.std(ks, ddof=1)) if len(ks) > 1 else np.nan
        rows.append(r)
    out = ROOT / "results/08i_dev_screen"; out.mkdir(parents=True, exist_ok=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "summary.tsv", sep="\t", index=False, float_format="%.4f")
    print(tab.round(3).to_string(index=False))
    write_provenance("08i_dev_screen", [], [out / "summary.tsv"], seed=cfg["seed"])


if __name__ == "__main__":
    main()
