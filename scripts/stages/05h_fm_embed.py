"""冻结的通用图像基础模型（DINOv2 ViT-L/14，自然图像自监督预训练，未见眼底数据）提取每张照片的嵌入：
[CLS] 与 patch 平均拼接（2×1024 维），输入 518×518（由 1024 全局图缩放）。用于检验大模型表示能否辅助分级。
用法：python scripts/stages/05h_fm_embed.py --roles ddr_train ddr_dev ddr_test → data/processed/fm_embed/<model>/<role>.npz"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

MODELS = {"dinov2_l": ("vit_large_patch14_dinov2.lvd142m", 518), "retfound": ("vit_large_patch14_dinov2.lvd142m", 518)}


class DS(torch.utils.data.Dataset):
    def __init__(self, keys, gdir, size):
        self.keys, self.gdir, self.size = list(keys), gdir, size

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        g = cv2.imread(str(self.gdir / f"{self.keys[i]}.jpg"))[:, :, ::-1]
        g = cv2.resize(g, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(g)).permute(2, 0, 1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--roles", nargs="+", required=True)
    ap.add_argument("--model", default="dinov2_l"); ap.add_argument("--manifest", default="results/02_manifest/images.tsv")
    args = ap.parse_args()
    cfg = load_config(); set_global_seed(cfg["seed"])
    name, size = MODELS[args.model]
    if args.model == "retfound":                        # 同一结构，换成视网膜预训练权重
        from scripts.lib.fmencoder import FMEncoder
        net = FMEncoder(name, size, weights=cfg["lcnet"]["retfound_weights"]).net.cuda().eval()
    else:
        net = timm.create_model(name, pretrained=True, num_classes=0, img_size=size).cuda().eval()
    dc = net.pretrained_cfg
    mean = torch.tensor(dc["mean"], device="cuda").view(1, 3, 1, 1) * 255
    std = torch.tensor(dc["std"], device="cuda").view(1, 3, 1, 1) * 255
    man = pd.read_csv(ROOT / args.manifest, sep="\t", low_memory=False)
    man = man[man.included.astype(int) == 1]
    out = ROOT / cfg["paths"]["processed"] / "fm_embed" / args.model; out.mkdir(parents=True, exist_ok=True)
    files = []
    for role in args.roles:
        keys = man[man.role == role].image_key.to_numpy()
        dl = torch.utils.data.DataLoader(DS(keys, ROOT / cfg["paths"]["processed"] / "global1024", size), batch_size=32,
                                         num_workers=8)
        E = []
        for x in dl:
            x = (x.cuda().float() - mean) / std
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t = net.forward_features(x)
            E.append(torch.cat([t[:, 0], t[:, net.num_prefix_tokens:].mean(1)], 1).float().cpu().numpy())
        np.savez(out / f"{role}.npz", keys=keys, emb=np.concatenate(E).astype(np.float16))
        files.append(out / f"{role}.npz"); print(role, len(keys), flush=True)
    write_provenance(f"05h_fm_embed_{args.model}", [ROOT / args.manifest], files, seed=cfg["seed"],
                     extra={"model": name, "img_size": size, "roles": args.roles})


if __name__ == "__main__":
    main()
