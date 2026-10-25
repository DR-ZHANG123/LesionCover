"""用已训练好的分级模型（不再训练）在按相机适配的病灶图所选裁块（05b）上预测四个评价队列。
每个运行目录写 predictions_adabn.tsv.gz：p0..p4 为整体输入（全局图 + 裁块），c0..c4 为只看裁块的辅助头。
用法：python scripts/stages/07e_predict_adabn.py --tags C_k4_s11_aux ...（缺省：U/N/D/C 的全部辅助头运行）"""
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

ROLES = ("ddr_test", "deepdrid", "idrid", "messidor2")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P = np.zeros((len(loader.dataset), 5), np.float32); Q = np.zeros_like(P)
    for g, c, _, idx in loader:
        g, c = g.to(device, non_blocking=True), c.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit, _, aux = model(g, c, return_aux=True)
        P[idx.numpy()] = torch.softmax(logit.float(), 1).cpu().numpy()
        Q[idx.numpy()] = torch.softmax(aux.float(), 1).cpu().numpy()
    return P, Q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="*")
    args = ap.parse_args()
    cfg = load_config(); set_global_seed(cfg["seed"]); torch.set_num_threads(8)
    dev = torch.device("cuda")
    crops = ROOT / "results/05b_adabn_crops/crops.tsv.gz"
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    man = man[(man.included.astype(int) == 1) & man.role.isin(ROLES)].copy()
    man["grade"] = man.grade.astype(int)
    man["canon_path"] = [str(ROOT / cfg["paths"]["processed"] / "canon" / f"{k}.jpg") for k in man.image_key]
    tags = args.tags or [d.name for d in sorted((ROOT / "results/07_classify").glob("[UNDC]_k4_s??_aux*/"))
                         if re.fullmatch(r"[UNDC]_k4_s\d+_aux(art|small)?", d.name)]
    for tag in tags:
        out = ROOT / "results/07_classify" / tag / "predictions_adabn.tsv.gz"
        if out.exists():
            continue
        m = re.match(r"(?P<method>[UNDC])_k4_s(?P<seed>\d+)_(?P<var>[a-z]+)$", tag)
        model = Grader(backbone_for(tag, cfg, ROOT), 5,
                       cfg["classifier"]["attn_dim"], True, crop_head=True).to(dev)
        model.load_state_dict(torch.load(ROOT / cfg["paths"]["checkpoints"] / "classify" / f"{tag}.pt", map_location=dev))
        parts = []
        for role in ROLES:
            sub = man[man.role == role].reset_index(drop=True)
            _, dl = make_loader(sub, load_crops(m["method"], 4, sub, crops), cfg, cfg["preprocess"]["global_size"], False,
                                int(m["seed"]), 32)
            P, Q = predict(model, dl, dev)
            d = pd.DataFrame(np.c_[P, Q], columns=[f"p{i}" for i in range(5)] + [f"c{i}" for i in range(5)])
            d.insert(0, "image_key", sub.image_key); d.insert(1, "role", role)
            parts.append(d)
        pd.concat(parts).to_csv(out, sep="\t", index=False, float_format="%.5f")
        print("adabn", tag, flush=True)
    write_provenance("07e_predict_adabn", [crops], [])


if __name__ == "__main__":
    main()
