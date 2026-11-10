"""LesionCover 的可解释性分析（只读取训练好的权重，不再训练；DDR 测试区与五个外部队列）。

每个种子的模型对每张照片只算一次编码器特征，再在融合部分做以下操作：
(a) 概念消融（concept ablation）：测试时去掉一类病灶（MA、HE、EX、SE）的全部 token 与计数，或去掉全部病灶证据；
    比较三模型平均概率下的期望等级、判级改变比例与 κ。
(b) 输出分解：最终 logit = (1 − a)·(Transformer logit + 规则 logit) + a·全局 logit（a 为病灶负荷门控）；
    去掉规则 logit、或固定门控为 0/1 后的判级改变比例与 κ，以及各真实等级的平均门控值。
(c) 注意力分配：最后一层 [CLS] 对全局 token、计数/规则 token 与各类病灶 token 的注意力占比（按真实等级）。
(d) 反事实计数（partial dependence）：DDR 测试区无 DR 照片上，把某类病灶在 k 个象限各设为 n 个
    （其余病灶计数与 token 不变），记录 P(级别 ≥ 2)、P(≥ 3)、P(≥ 4) 与期望等级。
用法：python scripts/stages/08za_interpretability.py --seed 11（单个种子写 parts/），--merge 汇总三个种子。
输出 results/08za_interpretability/{ablation.tsv, components.tsv, attention.tsv, counterfactual.tsv}。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.lcdata import make_lc_loader  # noqa: E402
from scripts.lib.lcnet import LesionCoverNet, fourier  # noqa: E402
from scripts.lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "08za_interpretability"
LES = ["MA", "HE", "EX", "SE"]
ROLES = ["ddr_test", "deepdrid", "idrid", "messidor2", "eyepacs", "mmrdr"]
CONDS = ["full"] + [f"no_{c}" for c in LES] + ["no_lesions", "no_rule_logits", "gate_0", "gate_1"]
OUT = ROOT / "results" / STAGE


def encode(model, g):
    x = (g.float() - model.mean) / model.std
    f1, f2, f3 = model.enc(x)
    gt = torch.nn.functional.avg_pool2d(f3, f3.shape[-1] // model.G).flatten(2).transpose(1, 2)
    gt = model.gproj(gt) + model.pos(fourier(model.gxy))[None] + model.type_emb.weight[1]
    aux_g = model.fc_global(f3.mean((2, 3)))
    return f1, f2, gt, aux_g


def head(model, enc, tok, valid, counts, rule_on=True, gate=None, attn=False):
    """lcnet.forward 编码器之后的部分（本设计无超图、原型、文字头、双编码器）；可选返回最后一层 [CLS] 注意力。"""
    f1, f2, gt, aux_g = enc
    B, S = gt.shape[0], model._S
    lt = model._lesion_tokens(f1, f2, tok, S) if tok.shape[1] else gt.new_zeros(B, 0, gt.shape[-1])
    tot = torch.log1p(counts.sum(1, keepdim=True))
    ct = (model.count(torch.cat([torch.log1p(counts), tot], 1)) + model.type_emb.weight[3])[:, None]
    rlog, rtok = model.rule(counts)
    ct = torch.cat([ct, (rtok + model.type_emb.weight[3])[:, None]], 1)
    cl = model.cls_tok.expand(B, -1, -1) + model.type_emb.weight[0]
    seq = torch.cat([cl, gt, ct, lt], 1)
    pad = torch.cat([torch.zeros(B, 1 + gt.shape[1] + ct.shape[1], dtype=torch.bool, device=gt.device), ~valid], 1)
    x, w = seq, None
    for i, layer in enumerate(model.tf.layers):
        if attn and i == len(model.tf.layers) - 1:
            h = layer.norm1(x)
            _, w = layer.self_attn(h, h, h, key_padding_mask=pad, need_weights=True, average_attn_weights=True)
            w = w[:, 0].float()                                               # (B, L) [CLS] 的注意力
        x = layer(x, src_key_padding_mask=pad)
    out_t = model.fc(model.norm(x)[:, 0])
    out = out_t + (rlog.to(out_t.dtype) if rule_on else 0)
    a = model.gate(counts).to(out.dtype) if gate is None else torch.full_like(out[:, :1], float(gate))
    out = (1 - a) * out + a * aux_g.to(out.dtype)
    return out, a.float(), w


def build(cfg, seed, device):
    prov = json.loads((ROOT / f"results/07n_lcnet/LC_fmrglearn_s{seed}/PROVENANCE.json").read_text())
    lc = prov["lcnet"]
    assert not lc.get("hyper_layers") and not lc.get("proto") and not lc.get("fm_tokens") and not lc.get("text_proto")
    model = LesionCoverNet(cfg["classifier"]["backbone"], lc, prob_size=cfg["segmenter"]["prob_map_size"]).to(device)
    model.load_state_dict(torch.load(ROOT / cfg["paths"]["checkpoints"] / "lcnet" / f"LC_fmrglearn_s{seed}.pt",
                                     map_location=device))
    return model.eval(), lc


def manifest():
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", low_memory=False)
    mm = pd.read_csv(ROOT / "results/01b_mmrdr/images.tsv", sep="\t")
    man = pd.concat([man[["image_key", "role", "grade", "included"]], mm[["image_key", "role", "grade", "included"]]])
    return man[(man.included.astype(int) == 1) & man.role.isin(ROLES) & (man.grade.fillna(-1) >= 0)].reset_index(drop=True)


@torch.no_grad()
def run_seed(seed, cfg):
    device = torch.device("cuda")
    set_global_seed(seed); torch.backends.mha.set_fastpath_enabled(False)
    model, lc = build(cfg, seed, device)
    man = manifest()
    tdir = ROOT / cfg["paths"]["processed"] / "lesion_tokens"
    cf = cfg["interpretability"]
    probs = {c: np.zeros((len(man), 5), np.float32) for c in CONDS}
    gate = np.zeros(len(man), np.float32); att = np.zeros((len(man), 3 + len(LES)), np.float32)
    cf_rows = []
    for role in ROLES:
        idx = np.flatnonzero(man.role.to_numpy() == role)
        _, dl = make_lc_loader(man.image_key.to_numpy()[idx], man.grade.astype(int).to_numpy()[idx], cfg, False, seed,
                               2 * lc["batch_size"], tdir)
        for g, tok, val, cnt, y, ii in dl:
            g, tok, val, cnt = (t.to(device, non_blocking=True) for t in (g, tok, val, cnt))
            rows = idx[ii.numpy()]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model._S = g.shape[-1]
                enc = encode(model, g)
                for c in CONDS:
                    t, v, n = tok, val, cnt
                    if c.startswith("no_") and c != "no_rule_logits":
                        kill = torch.ones_like(val) if c == "no_lesions" else (tok[..., 0].long() == LES.index(c[3:]))
                        v = val & ~kill
                        n = cnt.view(len(g), len(LES), 4).clone()
                        if c == "no_lesions":
                            n[:] = 0
                        else:
                            n[:, LES.index(c[3:])] = 0
                        n = n.view(len(g), -1)
                    out, a, w = head(model, enc, t, v, n, rule_on=(c != "no_rule_logits"),
                                     gate={"gate_0": 0.0, "gate_1": 1.0}.get(c), attn=(c == "full"))
                    probs[c][rows] = torch.softmax(out.float(), 1).cpu().numpy()
                    if c == "full":
                        gate[rows] = a[:, 0].cpu().numpy()
                        L0 = 1 + enc[2].shape[1]
                        cls_ = tok[..., 0].long()
                        parts = [w[:, 1:L0].sum(1), w[:, L0:L0 + 2].sum(1)]
                        lw = w[:, L0 + 2:]
                        parts += [(lw * ((cls_ == k) & val)).sum(1) for k in range(len(LES))]
                        att[rows, 1:] = torch.stack(parts, 1).cpu().numpy()
                        att[rows, 0] = val.sum(1).cpu().numpy()
                if role == "ddr_test" and (y == 0).any():               # 反事实计数：只在无 DR 照片上
                    m0 = (y == 0).to(device)
                    e0 = tuple(e[m0] for e in enc)
                    for c in cf["lesions"]:
                        for k in cf["quadrants"]:
                            for n_ in cf["counts"]:
                                n = cnt[m0].view(-1, len(LES), 4).clone()
                                n[:, LES.index(c), :k] = float(n_)
                                n[:, LES.index(c), k:] = 0.0
                                out, _, _ = head(model, e0, tok[m0], val[m0], n.view(int(m0.sum()), -1))
                                p = torch.softmax(out.float(), 1).cpu().numpy()
                                cf_rows.append({"seed": seed, "lesion": c, "quadrants": k, "per_quadrant": n_,
                                                "n": len(p), "p_ge2": p[:, 2:].sum(1).sum(), "p_ge3": p[:, 3:].sum(1).sum(),
                                                "p_ge4": p[:, 4].sum(), "expected_grade": (p * np.arange(5)).sum()})
        print(role, len(idx), flush=True)
    part = OUT / "parts"; part.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(part / f"s{seed}.npz", keys=man.image_key.to_numpy(), gate=gate, att=att,
                        **{c: probs[c] for c in CONDS})
    pd.DataFrame(cf_rows).to_csv(part / f"counterfactual_s{seed}.tsv", sep="\t", index=False)


def merge(cfg):
    man = manifest(); y = man.grade.astype(int).to_numpy(); role = man.role.to_numpy()
    seeds = cfg["interpretability"]["seeds"]
    zs = [np.load(OUT / "parts" / f"s{s}.npz", allow_pickle=True) for s in seeds]
    for z in zs:
        assert (z["keys"] == man.image_key.to_numpy()).all()
    P = {c: np.mean([z[c] for z in zs], 0) for c in CONDS}
    E = {c: P[c] @ np.arange(5) for c in CONDS}; G = {c: P[c].argmax(1) for c in CONDS}
    rows = []
    for r in ROLES:
        m = role == r
        k0 = cohen_kappa_score(y[m], G["full"][m], weights="quadratic")
        for c in CONDS:
            k = cohen_kappa_score(y[m], G[c][m], weights="quadratic")
            rows.append({"cohort": r, "condition": c, "n": int(m.sum()), "qwk": k, "delta_qwk": k - k0,
                         "delta_expected_grade": float((E[c][m] - E["full"][m]).mean()),
                         "changed": float((G[c][m] != G["full"][m]).mean()),
                         "lowered": float((G[c][m] < G["full"][m]).mean()),
                         **{f"delta_expected_grade_g{g}": float((E[c][m & (y == g)] - E["full"][m & (y == g)]).mean())
                            for g in range(5)}})
    pd.DataFrame(rows).to_csv(OUT / "ablation.tsv", sep="\t", index=False)
    gate = np.mean([z["gate"] for z in zs], 0); att = np.mean([z["att"] for z in zs], 0)
    comp, at = [], []
    for r in ROLES:
        for g in range(5):
            m = (role == r) & (y == g)
            if not m.any():
                continue
            comp.append({"cohort": r, "grade": g, "n": int(m.sum()), "gate_mean": gate[m].mean(),
                         "gate_q25": np.quantile(gate[m], 0.25), "gate_q75": np.quantile(gate[m], 0.75)})
            at.append({"cohort": r, "grade": g, "n": int(m.sum()), "lesion_tokens": att[m, 0].mean(),
                       "attention_global": att[m, 1].mean(), "attention_count_rule": att[m, 2].mean(),
                       **{f"attention_{c}": att[m, 3 + i].mean() for i, c in enumerate(LES)}})
    pd.DataFrame(comp).to_csv(OUT / "components.tsv", sep="\t", index=False)
    pd.DataFrame(at).to_csv(OUT / "attention.tsv", sep="\t", index=False)
    cf = pd.concat([pd.read_csv(OUT / "parts" / f"counterfactual_s{s}.tsv", sep="\t") for s in seeds])
    cf = cf.groupby(["lesion", "quadrants", "per_quadrant"]).sum(numeric_only=True).reset_index()
    for k in ("p_ge2", "p_ge3", "p_ge4", "expected_grade"):
        cf[k] = cf[k] / cf["n"]
    cf["n"] = cf["n"] // len(seeds)
    cf.drop(columns="seed").to_csv(OUT / "counterfactual.tsv", sep="\t", index=False)
    write_provenance(STAGE, [OUT / "parts" / f"s{s}.npz" for s in seeds],
                     [OUT / f for f in ("ablation.tsv", "components.tsv", "attention.tsv", "counterfactual.tsv")],
                     seed=cfg.get("seed", 11), extra=cfg["interpretability"])
    print(pd.DataFrame(rows).round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    cfg = load_config(); cfg["_proc"] = ROOT / cfg["paths"]["processed"]
    merge(cfg) if a.merge else run_seed(a.seed, cfg)


if __name__ == "__main__":
    main()
