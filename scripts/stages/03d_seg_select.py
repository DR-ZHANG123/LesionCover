"""在 DDR 开发区比较不同分割器：除掩膜像素 AUPR（训练日志）外，用照片级指标衡量病灶图能否区分轻度 DR 与无 DR——
轻度 DR 按定义只有微动脉瘤，因此微动脉瘤图的峰值证据应在 1 级照片高、0 级照片低（无阈值的 AUC），
健康照片上的误报会直接压低该 AUC。只用开发区照片。
用法：python scripts/stages/03d_seg_select.py --ckpts segmenter.pt segmenter_neg.pt ...
输出 results/03d_seg_select/summary_<分割器名>.tsv（每个分割器一行）与 per_photo_<分割器名>.tsv.gz。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from scripts.lib.segnet import build_segmenter, gpu_norm  # noqa: E402

TOP = 10


class CanonSet(torch.utils.data.Dataset):
    def __init__(self, keys, proc):
        self.keys, self.proc = list(keys), proc

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        k = self.keys[i]
        img = cv2.imread(str(self.proc / "canon" / f"{k}.jpg"))[:, :, ::-1].copy()
        return img, cv2.imread(str(self.proc / "fov" / f"{k}.png"), 0), i


@torch.no_grad()
def ma_peaks(model, img, fov, device, size):
    """微动脉瘤概率图（翻转平均，最大池化到 size）上的局部峰值，降序返回。"""
    x = gpu_norm(torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].to(device))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = (torch.sigmoid(model(x).float()) + torch.sigmoid(model(x.flip(-1)).float()).flip(-1)) / 2
    p = F.max_pool2d(p[:, :1], p.shape[-1] // size)[0, 0]
    fv = torch.from_numpy(cv2.resize(fov, (size, size), interpolation=cv2.INTER_NEAREST) > 0).to(device)
    p = p * fv
    peak = (p == F.max_pool2d(p[None, None], 9, 1, 4)[0, 0]) & (p > 0.05)
    return torch.sort(p[peak], descending=True).values.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpts", nargs="+", required=True)
    args = ap.parse_args()
    cfg = load_config(); set_global_seed(cfg["seed"]); torch.set_num_threads(8)
    proc, device, size = ROOT / cfg["paths"]["processed"], torch.device("cuda"), cfg["segmenter"]["prob_map_size"]
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    dv = man[(man.included.astype(int) == 1) & (man.role == "ddr_dev") & man.grade.astype(str).isin(["0", "1"])]
    dv = dv.reset_index(drop=True); y = dv.grade.astype(int).to_numpy()
    out = ROOT / "results/03d_seg_select"; out.mkdir(parents=True, exist_ok=True)
    rows, per = [], []
    for ck in args.ckpts:
        model = build_segmenter(cfg["segmenter"]["encoder"], 4, pretrained=False).to(device)
        model.load_state_dict(torch.load(ROOT / cfg["paths"]["checkpoints"] / ck, map_location=device, weights_only=True))
        model.eval()
        top, n05 = np.zeros(len(dv)), np.zeros(len(dv))
        dl = torch.utils.data.DataLoader(CanonSet(dv.image_key, proc), batch_size=None, num_workers=8)
        for img, fov, i in dl:
            pk = ma_peaks(model, img.numpy(), fov.numpy(), device, size)
            top[i], n05[i] = pk[:TOP].sum(), (pk > 0.5).sum()
        per.append(pd.DataFrame({"ckpt": ck, "image_key": dv.image_key, "grade": y, "ma_top_sum": top, "ma_peaks_05": n05}))
        rows.append({"ckpt": ck, "n_none": int((y == 0).sum()), "n_mild": int((y == 1).sum()),
                     "auc_mild_vs_none_top": roc_auc_score(y, top), "auc_mild_vs_none_count": roc_auc_score(y, n05),
                     "median_peaks_none": float(np.median(n05[y == 0])), "median_peaks_mild": float(np.median(n05[y == 1]))})
        print(rows[-1], flush=True)
    tag = "_".join(Path(c).stem for c in args.ckpts)
    s = pd.DataFrame(rows); s.to_csv(out / f"summary_{tag}.tsv", sep="\t", index=False)
    pd.concat(per).to_csv(out / f"per_photo_{tag}.tsv.gz", sep="\t", index=False)
    write_provenance("03d_seg_select", [ROOT / cfg["paths"]["checkpoints"] / c for c in args.ckpts], [out / f"summary_{tag}.tsv"],
                     seed=cfg["seed"])


if __name__ == "__main__":
    main()
