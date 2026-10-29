"""用已训练好的分级模型（不再训练）对评价队列重新前向一次，以全精度保存五个等级的对数概率（log-softmax），
供可转诊 DR 的连续评分（08m_referable_roc）使用：各运行目录下的 predictions*.tsv.gz 只保留 5 位小数，
高置信照片的 P(≥2 级) 会被截成 1.0 而并列，连续评分需要未截断的概率。最可能等级与原预测一致（同一权重、同一输入）。
输出 results/07g_referable_scores/<tag>.tsv.gz（image_key, role, lp0..lp4）。
用法：CUDA_VISIBLE_DEVICES=<gpu> python scripts/stages/07g_referable_scores.py --tags G512_s11 C_k4_s11_aux ..."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.clsnet import Grader, backbone_for  # noqa: E402
from scripts.lib.gradedata import load_crops, make_loader  # noqa: E402
from scripts.lib.variants import global_size  # noqa: E402
from scripts.lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

ROLES = ("ddr_test", "deepdrid", "idrid", "messidor2", "eyepacs")


@torch.no_grad()
def log_probs(model, loader, device) -> np.ndarray:
    model.eval()
    L = np.zeros((len(loader.dataset), 5), np.float64)
    for g, c, _, idx in loader:
        g = g.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True) if c.numel() else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit, _ = model(g, c)
        L[idx.numpy()] = torch.log_softmax(logit.double(), 1).cpu().numpy()
    return L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    cfg = load_config(); set_global_seed(cfg["seed"]); torch.set_num_threads(8)
    dev = torch.device("cuda")
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False, dtype={"patient_id": str})
    man = man[man.included.astype(int) == 1].reset_index(drop=True)
    man["grade"] = man.grade.astype(int)
    man["canon_path"] = [str(ROOT / cfg["paths"]["processed"] / "canon" / f"{k}.jpg") for k in man.image_key]
    out_dir = ROOT / "results/07g_referable_scores"; out_dir.mkdir(parents=True, exist_ok=True)
    for tag in args.tags:
        out = out_dir / f"{tag}.tsv.gz"
        if out.exists():
            continue
        m = re.match(r"(?P<method>[A-Za-z0-9]+?)(?:_k(?P<k>\d+))?_s(?P<seed>\d+)(?:_(?P<var>[a-z0-9_]+))?$", tag)
        method, k, var = m["method"], int(m["k"] or 4), (m["var"] or "").split("_")[0]
        gsize = global_size(method, var, cfg)
        model = Grader(backbone_for(tag, cfg, ROOT), 5, cfg["classifier"]["attn_dim"], not method.startswith("G"),
                       crop_head="aux" in var).to(dev)
        model.load_state_dict(torch.load(ROOT / cfg["paths"]["checkpoints"] / "classify" / f"{tag}.pt", map_location=dev))
        rows = []
        for role in ROLES:
            sub = man[man.role == role].reset_index(drop=True)
            _, dl = make_loader(sub, load_crops(method, k, sub, var=var), cfg, gsize, False, int(m["seed"]), 32)
            d = pd.DataFrame(log_probs(model, dl, dev), columns=[f"lp{i}" for i in range(5)])
            d.insert(0, "image_key", sub.image_key); d.insert(1, "role", role)
            rows.append(d)
            print(tag, role, len(d), flush=True)
        pd.concat(rows).to_csv(out, sep="\t", index=False, float_format="%.8g")
        write_provenance(f"07g_referable_scores/{tag}", [ROOT / "results/02_manifest/images.tsv"], [out], seed=int(m["seed"]))
    if all((out_dir / f"{t}.tsv.gz").exists() for t in args.tags):
        write_provenance("07g_referable_scores", [ROOT / "results/02_manifest/images.tsv"], sorted(out_dir.glob("*.tsv.gz")))


if __name__ == "__main__":
    main()
