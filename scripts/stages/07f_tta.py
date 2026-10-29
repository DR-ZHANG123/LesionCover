"""测试时增强：对已训练的分级模型，全局图与裁块同时施加 8 种二面体变换（4 个 90° 旋转 × 是否水平翻转，
与训练时的增强一致），平均 8 次的 softmax 概率。先只在 DDR 开发区上比较是否优于单次预测，再决定是否用于全部队列。
输出各运行目录下 predictions_tta.tsv.gz（逐队列追加，已有的队列跳过）。
用法：python scripts/stages/07f_tta.py --tags T1 T2 ... --roles ddr_dev [ddr_test deepdrid idrid messidor2]"""
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

P = [f"p{i}" for i in range(5)]


def dihedral(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    x = torch.rot90(x, k, dims=(-2, -1))
    return x.flip(-1) if flip else x


@torch.no_grad()
def predict_tta(model, loader, device):
    model.eval()
    out = np.zeros((len(loader.dataset), 5), np.float32)
    for g, c, _, idx in loader:
        g = g.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True) if c.numel() else None
        acc = 0
        for k in range(4):
            for flip in (False, True):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit, _ = model(dihedral(g, k, flip), dihedral(c, k, flip) if c is not None else None)
                acc = acc + torch.softmax(logit.float(), 1)
        out[idx.numpy()] = (acc / 8).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--roles", nargs="+", default=["ddr_dev"])
    args = ap.parse_args()
    cfg = load_config(); set_global_seed(cfg["seed"]); torch.set_num_threads(8)
    dev = torch.device("cuda")
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    man = man[man.included.astype(int) == 1].copy(); man["grade"] = man.grade.astype(int)
    man["canon_path"] = [str(ROOT / cfg["paths"]["processed"] / "canon" / f"{k}.jpg") for k in man.image_key]
    for tag in args.tags:
        f = ROOT / "results/07_classify" / tag / "predictions_tta.tsv.gz"
        have = pd.read_csv(f, sep="\t") if f.exists() else pd.DataFrame(columns=["image_key", "role", *P])
        roles = [r for r in args.roles if r not in set(have.role)]
        if not roles:
            continue
        m = re.match(r"(?P<method>[A-Za-z0-9]+?)(?:_k(?P<k>\d+))?_s(?P<seed>\d+)(?:_(?P<var>[a-z0-9_]+))?$", tag)
        method, k, var = m["method"], int(m["k"] or 4), (m["var"] or "").split("_")[0]
        gsize = global_size(method, var, cfg)
        model = Grader(backbone_for(tag, cfg, ROOT), 5, cfg["classifier"]["attn_dim"], not method.startswith("G"),
                       crop_head="aux" in var).to(dev).to(memory_format=torch.channels_last)
        model.load_state_dict(torch.load(ROOT / cfg["paths"]["checkpoints"] / "classify" / f"{tag}.pt", map_location=dev,
                                         weights_only=True))
        parts = [have]
        for role in roles:
            sub = man[man.role == role].reset_index(drop=True)
            _, dl = make_loader(sub, load_crops(method, k, sub, var=var), cfg, gsize, False, int(m["seed"]), 32)
            d = pd.DataFrame(predict_tta(model, dl, dev), columns=P)
            d.insert(0, "image_key", sub.image_key); d.insert(1, "role", role)
            parts.append(d)
        pd.concat(parts).to_csv(f, sep="\t", index=False, float_format="%.5f")
        print("tta", tag, roles, flush=True)
    write_provenance("07f_tta", [ROOT / "results/02_manifest/images.tsv"], [], extra={"roles": args.roles})


if __name__ == "__main__":
    main()
