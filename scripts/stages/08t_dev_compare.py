"""DDR 开发区上比较 LesionCover 网络各设置与常规骨干网络（只用开发区，用于确定最终设置）。
输出 results/08t_dev_compare/summary.tsv：每个运行与每个设置（多种子时另给概率平均）的 κ、准确率、宏 F1、轻度判为无 DR、可转诊敏感度/特异度。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, write_provenance  # noqa: E402
from scripts.lib.stats import metrics  # noqa: E402

KEEP = ["qwk", "accuracy", "macro_f1", "mild_to_none", "referable_sens", "referable_spec"]


def main():
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", low_memory=False)[["image_key", "grade"]]
    runs = {}
    for d in sorted((ROOT / "results/07n_lcnet").glob("LC_*_s*/")):
        f = d / "predictions_dev.tsv.gz"
        if not f.exists():
            f = d / "predictions.tsv.gz"
        if f.exists() and (d / "PROVENANCE.json").exists():
            runs[d.name] = pd.read_csv(f, sep="\t").query("role == 'ddr_dev'")
    for pat in ("G1024_s*/", "GDRNet_s*/", "CABNet_s*/"):          # 常规骨干网络与已发表方法（官方代码重训）
        for d in sorted((ROOT / "results/07_classify").glob(pat)):
            if (d / "PROVENANCE.json").exists():
                m, rest = d.name.split("_s", 1)
                runs[f"{m}|{rest}"] = pd.read_csv(d / "predictions.tsv.gz", sep="\t").query("role == 'ddr_dev'")
    rows, groups = [], {}
    for name, d in runs.items():
        d = d.merge(man).sort_values("image_key")
        y, Pm = d.grade.astype(int).to_numpy(), d[[f"p{i}" for i in range(5)]].to_numpy()
        r = metrics(y, Pm.argmax(1))
        if name.startswith("LC_"):
            setting, seed = name[3:].rsplit("_s", 1)
        else:
            m, rest = name.split("|"); parts = rest.split("_")
            seed, setting = parts[0], m + ("_" + "_".join(parts[1:]) if len(parts) > 1 else "")
        rows.append({"setting": setting, "seed": seed, **{k: r[k] for k in KEEP}})
        groups.setdefault(setting, []).append(Pm)
    for setting, ps in groups.items():
        if len(ps) > 1:
            r = metrics(y, np.mean(ps, 0).argmax(1))
            rows.append({"setting": setting, "seed": f"ensemble{len(ps)}", **{k: r[k] for k in KEEP}})
    t = pd.DataFrame(rows).sort_values(["setting", "seed"])
    out = ROOT / "results/08t_dev_compare"; out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "summary.tsv", sep="\t", index=False, float_format="%.4f")
    print(t.round(4).to_string(index=False))
    write_provenance("08t_dev_compare", [ROOT / "results/02_manifest/images.tsv"], [out / "summary.tsv"])


if __name__ == "__main__":
    main()
