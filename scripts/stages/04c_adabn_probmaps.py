"""按相机适配分割器的归一化统计（AdaBN，Li 等 2016）：对一个评价队列的全部照片（不用任何标签），
重新估计分割器所有 BatchNorm 层的均值与方差（累积平均），权重不变；再用适配后的分割器生成病灶概率图，
写到 data/processed/probmaps_adabn/。分割器原本的统计量来自以病灶为中心的训练小块，与整张照片不同，所以开发队列本身
也用同样方式适配：DDR 训练区整张照片的统计量用于 DDR 开发区与测试区，校准在由此得到的开发区概率图上重新拟合（04b --subdir）。
用法：python scripts/stages/04c_adabn_probmaps.py --stats-from messidor2 --roles messidor2
      python scripts/stages/04c_adabn_probmaps.py --stats-from ddr_train --roles seg_dev ddr_test"""
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
from scripts.lib.segnet import build_segmenter, gpu_norm, predict_pooled  # noqa: E402


class CanonSet(torch.utils.data.Dataset):
    def __init__(self, keys, proc):
        self.keys, self.proc = list(keys), proc

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        return cv2.imread(str(self.proc / "canon" / f"{self.keys[i]}.jpg"))[:, :, ::-1].copy(), i


@torch.no_grad()
def adapt_bn(model, loader, device):
    """BatchNorm 层切到训练模式、动量置空（累积平均），前向一遍目标照片（含水平翻转），其余层保持推断模式。"""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats(); m.momentum = None; m.train()
    for img, _ in loader:
        x = gpu_norm(torch.from_numpy(np.ascontiguousarray(img.numpy().transpose(2, 0, 1)))[None].to(device))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x); model(x.flip(-1))
    for m in bns:
        m.eval()
    return len(bns)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stats-from", required=True); ap.add_argument("--roles", nargs="+", required=True)
    args = ap.parse_args()
    cfg = load_config(); set_global_seed(cfg["seed"]); torch.set_num_threads(8)
    proc = ROOT / cfg["paths"]["processed"]
    outdir = proc / "probmaps_adabn"; outdir.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    man = man[man.included.astype(int) == 1]
    pick = lambda r: (man[man.mask_role == "seg_dev"] if r == "seg_dev" else man[man.role == r]).image_key.tolist()
    src = pick(args.stats_from)
    keys = [k for r in args.roles for k in pick(r)]
    device = torch.device("cuda")
    model = build_segmenter(cfg["segmenter"]["encoder"], 4, pretrained=False).to(device)
    model.load_state_dict(torch.load(ROOT / cfg["paths"]["checkpoints"] / "segmenter.pt", map_location=device))
    model.eval()
    mk = lambda ks: torch.utils.data.DataLoader(CanonSet(ks, proc), batch_size=None, num_workers=8)
    n_bn = adapt_bn(model, mk(src), device)
    size = cfg["segmenter"]["prob_map_size"]
    for img, i in mk(keys):
        k = keys[i]
        q = predict_pooled(model, img.numpy(), device, size)
        fov = cv2.resize(cv2.imread(str(proc / "fov" / f"{k}.png"), 0), (size, size), interpolation=cv2.INTER_NEAREST) > 0
        np.savez_compressed(outdir / f"{k}.npz", logit=q * fov[None], fov=fov.astype(np.uint8))
    out = ROOT / "results/04c_adabn_probmaps"; out.mkdir(parents=True, exist_ok=True)
    tag = "_".join(args.roles)
    pd.DataFrame({"image_key": keys, "stats_from": args.stats_from}).to_csv(out / f"keys_{tag}.tsv", sep="\t", index=False)
    print(tag, len(keys), "photographs; statistics from", args.stats_from, len(src), ";", n_bn, "BatchNorm layers adapted", flush=True)
    write_provenance("04c_adabn_probmaps", [ROOT / cfg["paths"]["checkpoints"] / "segmenter.pt"],
                     [out / f"keys_{tag}.tsv"], seed=cfg["seed"])


if __name__ == "__main__":
    main()
