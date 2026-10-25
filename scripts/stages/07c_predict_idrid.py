"""用已训练好的分级模型（不再训练）对外部队列（IDRiD 516 张分级照片；Messidor-2 1,744 张照片）做预测，
写入各运行目录下的 predictions_<role>.tsv.gz（训练时已生成的 predictions.tsv.gz 不改动）。--croponly 时同时写辅助头只看裁块的预测
predictions_croponly_<role>.tsv.gz。Messidor-2 只在训练设置全部确定后对最终模型预测一次。
用法：python scripts/stages/07c_predict_idrid.py [--role idrid|messidor2] [--tags tag1 tag2 ...]（缺省：所有尚无该队列预测的运行）。"""
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
def predict(model, loader, device, want_aux: bool = False):
    """一次前向同时得到融合输出、注意力，以及（want_aux 时）辅助头只看裁块的输出。"""
    model.eval()
    P, A = np.zeros((len(loader.dataset), 5), np.float32), None
    C = np.zeros((len(loader.dataset), 5), np.float32) if want_aux else None
    for g, c, _, idx in loader:
        g = g.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True) if c.numel() else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if want_aux:
                logit, a, aux = model(g, c, return_aux=True)
                C[idx.numpy()] = torch.softmax(aux.float(), 1).cpu().numpy()
            else:
                logit, a = model(g, c)
        P[idx.numpy()] = torch.softmax(logit.float(), 1).cpu().numpy()
        if a is not None:
            if A is None:
                A = np.zeros((len(loader.dataset), a.shape[1]), np.float32)
            A[idx.numpy()] = a.float().cpu().numpy()
    return P, A, C


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="*"); ap.add_argument("--role", default="idrid")
    ap.add_argument("--croponly", action="store_true")
    args = ap.parse_args()
    cfg = load_config(); set_global_seed(cfg["seed"]); torch.set_num_threads(8)
    dev = torch.device("cuda")
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    sub = man[(man.role == args.role) & (man.included.astype(int) == 1)].reset_index(drop=True)
    sub["grade"] = sub.grade.astype(int)
    sub["canon_path"] = [str(ROOT / cfg["paths"]["processed"] / "canon" / f"{k}.jpg") for k in sub.image_key]
    tags = args.tags or [d.name for d in sorted((ROOT / "results/07_classify").glob("*/"))
                         if (d / "PROVENANCE.json").exists()]
    for tag in tags:
        f = ROOT / "results/07_classify" / tag / "predictions.tsv.gz"
        out = f.parent / f"predictions_{args.role}.tsv.gz"
        if out.exists() or (pd.read_csv(f, sep="\t").role == args.role).any():
            continue
        m = re.match(r"(?P<method>[A-Za-z0-9]+?)(?:_k(?P<k>\d+))?_s(?P<seed>\d+)(?:_(?P<var>[a-z0-9_]+))?$", tag)
        method, k, var = m["method"], int(m["k"] or 4), (m["var"] or "").split("_")[0]
        gsize = cfg["preprocess"]["global_size_large"] if method == "G1024" else cfg["preprocess"]["global_size"]
        model = Grader(backbone_for(tag, cfg, ROOT), 5, cfg["classifier"]["attn_dim"], not method.startswith("G"),
                       crop_head="aux" in var).to(dev)
        model.load_state_dict(torch.load(ROOT / cfg["paths"]["checkpoints"] / "classify" / f"{tag}.pt", map_location=dev))
        _, dl = make_loader(sub, load_crops(method, k, sub, var=var), cfg, gsize, False, int(m["seed"]), 32)
        want_aux = args.croponly and "aux" in var
        P, A, C = predict(model, dl, dev, want_aux)
        d = pd.DataFrame(P, columns=[f"p{i}" for i in range(5)]); d.insert(0, "image_key", sub.image_key); d.insert(1, "role", args.role)
        if A is not None:
            for j in range(A.shape[1]):
                d[f"attn{j}"] = A[:, j]
        d.to_csv(out, sep="\t", index=False, float_format="%.5f")
        if want_aux:
            c = pd.DataFrame(C, columns=[f"p{i}" for i in range(5)])
            c.insert(0, "image_key", sub.image_key); c.insert(1, "role", args.role)
            c.to_csv(f.parent / f"predictions_croponly_{args.role}.tsv.gz", sep="\t", index=False, float_format="%.5f")
        print("predicted", tag, args.role, flush=True)


if __name__ == "__main__":
    main()
    write_provenance("07c_predict_idrid", [ROOT / "results/02_manifest/images.tsv"], [])
