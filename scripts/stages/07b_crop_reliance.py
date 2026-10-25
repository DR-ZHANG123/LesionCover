"""分级网络对裁块的依赖程度：用训练好的模型在 DDR 开发区与 DeepDRiD 上推断三次——裁块不变、
裁块换成随机另一张照片的裁块（整个数据集范围内置换，跨等级）、裁块换成全零图——比较 κ。依赖越小，换掉裁块后 κ 下降越少。
用法：python scripts/stages/07b_crop_reliance.py --tags C_k4_s11 N_k4_s11 ...；输出 results/07b_crop_reliance/reliance.tsv。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.clsnet import Grader, backbone_for, quadratic_kappa  # noqa: E402
from scripts.lib.repro import ROOT, load_config, set_global_seed  # noqa: E402
from scripts.lib.stats import eye_level  # noqa: E402
from scripts.lib.gradedata import load_crops, make_loader  # noqa: E402


@torch.no_grad()
def run(model, loader, device, mode, rng):
    model.eval()
    P = np.zeros((len(loader.dataset), 5), np.float32)
    for g, c, _, idx in loader:
        g, c = g.to(device), c.to(device)
        if mode == "blank":
            c = torch.zeros_like(c)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit, _ = model(g, c)
        P[idx.numpy()] = torch.softmax(logit.float(), 1).cpu().numpy()
    return P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    cfg = load_config(); set_global_seed(cfg["seed"]); torch.set_num_threads(8)
    dev = torch.device("cuda")
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    man = man[man.included.astype(int) == 1].copy(); man["grade"] = man.grade.astype(int)
    man["canon_path"] = [str(ROOT / cfg["paths"]["processed"] / "canon" / f"{k}.jpg") for k in man.image_key]
    drid = man[man.role == "deepdrid"].copy(); drid["patient_id"] = drid.patient_id.astype(int)
    drid["quality"] = pd.to_numeric(drid.quality)
    rows = []
    for tag in args.tags:
        method, k = tag.split("_")[:2]
        k = int(k[1:])
        model = Grader(backbone_for(tag, cfg, ROOT), 5, cfg["classifier"]["attn_dim"], True, crop_head="aux" in tag).to(dev)
        model.load_state_dict(torch.load(ROOT / cfg["paths"]["checkpoints"] / "classify" / f"{tag}.pt", map_location=dev))
        for role in ["ddr_dev", "deepdrid"]:
            sub = man[man.role == role].reset_index(drop=True)
            crops = load_crops(method, k, sub)
            perm = np.random.default_rng(cfg["seed"]).permutation(len(sub))
            for mode in ["intact", "shuffled", "blank"]:
                _, dl = make_loader(sub, crops[perm] if mode == "shuffled" else crops, cfg, cfg["preprocess"]["global_size"],
                                    False, 0, 32)
                P = run(model, dl, dev, mode, None)
                if role == "deepdrid":
                    d = pd.DataFrame(P, columns=[f"p{i}" for i in range(5)]); d.insert(0, "image_key", sub.image_key)
                    e = eye_level(d, drid); q = quadratic_kappa(e.grade, e[[f"p{i}" for i in range(5)]].to_numpy().argmax(1))
                else:
                    q = quadratic_kappa(sub.grade, P.argmax(1))
                rows.append({"tag": tag, "set": role, "crops": mode, "qwk": q})
                print(rows[-1], flush=True)
    out = ROOT / "results/07b_crop_reliance"; out.mkdir(parents=True, exist_ok=True)
    f = out / "reliance.tsv"
    old = pd.read_csv(f, sep="\t") if f.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    pd.concat([old[~old.tag.isin(new.tag)] if len(old) else old, new]).to_csv(f, sep="\t", index=False)


if __name__ == "__main__":
    main()
