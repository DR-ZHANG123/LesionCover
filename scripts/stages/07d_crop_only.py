"""只用裁块分级：带辅助头的模型中，辅助头只看四个裁块的汇聚特征。用它对 DDR 测试区、DeepDRiD 与 IDRiD 预测，
比较各选择规则所选裁块本身携带的分级信息。输出各运行目录下 predictions_croponly.tsv.gz。"""
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
from scripts.lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402


@torch.no_grad()
def crop_only(model, loader, device):
    model.eval()
    P = np.zeros((len(loader.dataset), 5), np.float32)
    for g, c, _, idx in loader:
        g, c = g.to(device, non_blocking=True), c.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, aux = model(g, c, return_aux=True)
        P[idx.numpy()] = torch.softmax(aux.float(), 1).cpu().numpy()
    return P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="*")
    args = ap.parse_args()
    cfg = load_config(); set_global_seed(cfg["seed"]); torch.set_num_threads(8)
    dev = torch.device("cuda")
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    man = man[man.included.astype(int) == 1].copy()
    man["grade"] = man.grade.astype(int)
    man["canon_path"] = [str(ROOT / cfg["paths"]["processed"] / "canon" / f"{k}.jpg") for k in man.image_key]
    tags = args.tags or [d.name for d in sorted((ROOT / "results/07_classify").glob("*_aux*/"))
                         if (d / "PROVENANCE.json").exists() and "aux" in d.name.rsplit("_", 1)[-1]]
    for tag in tags:
        out = ROOT / "results/07_classify" / tag / "predictions_croponly.tsv.gz"
        if out.exists():
            continue
        m = re.match(r"(?P<method>[A-Za-z0-9]+?)_k(?P<k>\d+)_s(?P<seed>\d+)_(?P<var>[a-z]+)$", tag)
        method, k = m["method"], int(m["k"])
        model = Grader(backbone_for(tag, cfg, ROOT), 5, cfg["classifier"]["attn_dim"], True, crop_head=True).to(dev)
        model.load_state_dict(torch.load(ROOT / cfg["paths"]["checkpoints"] / "classify" / f"{tag}.pt", map_location=dev))
        parts = []
        for role in ["ddr_test", "deepdrid", "idrid"]:
            sub = man[man.role == role].reset_index(drop=True)
            _, dl = make_loader(sub, load_crops(method, k, sub, var=m["var"]), cfg, cfg["preprocess"]["global_size"], False,
                                int(m["seed"]), 32)
            d = pd.DataFrame(crop_only(model, dl, dev), columns=[f"p{i}" for i in range(5)])
            d.insert(0, "image_key", sub.image_key); d.insert(1, "role", role)
            parts.append(d)
        pd.concat(parts).to_csv(out, sep="\t", index=False, float_format="%.5f")
        print("crop-only", tag, flush=True)


if __name__ == "__main__":
    main()
    write_provenance("07d_crop_only", [], [])
