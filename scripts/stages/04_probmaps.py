"""用冻结的分割器对全部纳入图像做整图推断，保存最大池化到 prob_map_size 的四通道概率图（uint8）与视野掩膜。
--ckpt/--subdir：换用另一分割器（如 segmenter_neg.pt → probmaps_neg）；--shard i/n：只处理第 i 份，便于多卡并行。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from scripts.lib.segnet import build_segmenter, predict_pooled  # noqa: E402


class CanonSet(torch.utils.data.Dataset):
    def __init__(self, keys, proc):
        self.keys, self.proc = list(keys), proc

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        return cv2.imread(str(self.proc / "canon" / f"{self.keys[i]}.jpg"))[:, :, ::-1].copy(), i


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="segmenter.pt"); ap.add_argument("--subdir", default="probmaps")
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    si, sn = map(int, args.shard.split("/"))
    cfg = load_config()
    set_global_seed(cfg["seed"])
    torch.set_num_threads(8)
    proc = ROOT / cfg["paths"]["processed"]
    outdir = proc / args.subdir; outdir.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t")
    keys = man.image_key[man.included == 1].tolist()
    todo = [k for k in keys[si::sn] if not (outdir / f"{k}.npz").exists()]
    device = torch.device("cuda")
    model = build_segmenter(cfg["segmenter"]["encoder"], 4, pretrained=False).to(device)
    model.load_state_dict(torch.load(ROOT / cfg["paths"]["checkpoints"] / args.ckpt, map_location=device, weights_only=True))
    model.eval()
    size = cfg["segmenter"]["prob_map_size"]
    dl = torch.utils.data.DataLoader(CanonSet(todo, proc), batch_size=None, num_workers=12)
    for n, (img, i) in enumerate(dl):
        k = todo[i]
        q = predict_pooled(model, img.numpy(), device, size)
        fov = cv2.resize(cv2.imread(str(proc / "fov" / f"{k}.png"), 0), (size, size), interpolation=cv2.INTER_NEAREST) > 0
        q = q * fov[None]
        np.savez_compressed(outdir / f"{k}.npz", logit=q, fov=fov.astype(np.uint8))
        if n % 1000 == 0:
            print(n, len(todo), flush=True)
    if sn > 1 and si != 0:
        return
    missing = [k for k in keys if not (outdir / f"{k}.npz").exists()]
    if missing:                      # 分片运行时只在所有分片完成后写出记录
        print(f"{len(missing)} maps still missing (other shards running); provenance not written", flush=True)
        return
    stage = "04_probmaps" if args.subdir == "probmaps" else f"04_{args.subdir}"
    (ROOT / "results" / stage).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_key": keys}).to_csv(ROOT / "results" / stage / "keys.tsv", sep="\t", index=False)
    write_provenance(stage, [ROOT / cfg["paths"]["checkpoints"] / args.ckpt],
                     [ROOT / "results" / stage / "keys.tsv"], seed=cfg["seed"])


if __name__ == "__main__":
    main()
