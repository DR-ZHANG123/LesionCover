"""LesionCover 学到的分级规则（只读取训练好的权重，不再训练）。

规则模块（scripts/lib/rulehead.py）对每类病灶 c 学 3 个“每象限分量数”阈值 θ_ck（及斜率），
软指示按象限排序得到“至少 j 个象限达到阈值 k”的特征 f；只看规则特征的单调有序分级为
s = Σ softplus(w)·f，P(级别 > k) = σ(s − c_k)。本 stage 输出：
  thresholds.tsv  每个编码器 × 种子 × 病灶 × 阈值序号的 θ 与斜率
  weights.tsv     每个特征（病灶 × 阈值 × 象限数）的非负权重 softplus(w)
  cutpoints.tsv   有序分级的四个切点
  rule_only.tsv   只用规则（分割器计数 → 规则特征 → 有序分级，三个模型 P(y>k) 平均后按 0.5 判级）在各队列的 κ 与准确率
编码器：DINOv2-L（最终设计 fmrglearn）与 ConvNeXt-T（rglearn），各三个种子。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402

STAGE = "08x_learned_rule"
LES = ["MA", "HE", "EX", "SE"]
TAGS = {"DINOv2-L": "fmrglearn", "ConvNeXt-T": "rglearn"}
ROLES = ["ddr_dev", "ddr_test", "deepdrid", "idrid", "messidor2", "eyepacs", "mmrdr"]


def rule_params(sd):
    log_t, log_b, ow, oc = (sd[f"rule.{k}"].float() for k in ("log_t", "log_b", "ow", "oc"))
    cut = oc[:1] + torch.cat([oc.new_zeros(1), torch.cumsum(F.softplus(oc[1:]) + 0.5, 0)])
    return log_t, log_b, F.softplus(ow), cut


def rule_features(counts, log_t, log_b):
    c = torch.log1p(torch.as_tensor(counts, dtype=torch.float32).view(-1, len(LES), 1, 4))
    s = torch.sigmoid(log_b.exp()[None, :, :, None] * (c - log_t[None, :, :, None]))
    return s.sort(-1, descending=True).values.flatten(1)


def main():
    cfg = load_config()
    ck = ROOT / cfg["paths"]["checkpoints"] / "lcnet"
    tdir = ROOT / cfg["paths"]["processed"] / "lesion_tokens"
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", low_memory=False)
    mm = pd.read_csv(ROOT / "results/01b_mmrdr/images.tsv", sep="\t", low_memory=False)
    man = pd.concat([man[["image_key", "role", "grade", "included"]], mm[["image_key", "role", "grade", "included"]]])
    man = man[(man.included.astype(int) == 1) & man.role.isin(ROLES)].reset_index(drop=True)
    keep = [i for i, k in enumerate(man.image_key) if (tdir / f"{k}.npz").exists()]
    man = man.iloc[keep].reset_index(drop=True)
    C = np.stack([np.load(tdir / f"{k}.npz")["counts"] for k in man.image_key])
    th, wt, cp, ro, ins = [], [], [], [], []
    for enc, tag in TAGS.items():
        cum = []
        for f in sorted(ck.glob(f"LC_{tag}_s*.pt")):
            seed = int(f.stem.rsplit("_s", 1)[1]); ins.append(f)
            log_t, log_b, w, cut = rule_params(torch.load(f, map_location="cpu", weights_only=False))
            for i, l in enumerate(LES):
                for k in range(log_t.shape[1]):
                    th.append({"encoder": enc, "seed": seed, "lesion": l, "level": k + 1,
                               "threshold_per_quadrant": float(log_t[i, k].expm1()), "slope": float(log_b[i, k].exp())})
                    for j in range(4):
                        wt.append({"encoder": enc, "seed": seed, "lesion": l, "level": k + 1, "quadrants_at_least": j + 1,
                                   "weight": float(w.view(len(LES), -1, 4)[i, k, j])})
            cp.append({"encoder": enc, "seed": seed, **{f"cut_{k + 1}": float(cut[k]) for k in range(4)}})
            with torch.no_grad():
                s = (rule_features(C, log_t, log_b) * w).sum(1, keepdim=True)
                cum.append(torch.sigmoid(s - cut[None]).numpy())
        pgt = np.mean(cum, 0); pred = (pgt > 0.5).sum(1)
        for r in ROLES:
            m = (man.role == r).to_numpy(); y = man.grade.to_numpy()[m].astype(int)
            ro.append({"encoder": enc, "cohort": r, "n": int(m.sum()), "qwk": cohen_kappa_score(y, pred[m], weights="quadratic"),
                       "accuracy": float(np.mean(y == pred[m]))})
    out = ROOT / "results" / STAGE; out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("thresholds", th), ("weights", wt), ("cutpoints", cp), ("rule_only", ro)):
        pd.DataFrame(rows).to_csv(out / f"{name}.tsv", sep="\t", index=False)
    print(pd.DataFrame(ro).round(3).to_string(index=False))
    write_provenance(STAGE, ins + [ROOT / "results/02_manifest/images.tsv"],
                     [out / f"{n}.tsv" for n in ("thresholds", "weights", "cutpoints", "rule_only")])


if __name__ == "__main__":
    main()
