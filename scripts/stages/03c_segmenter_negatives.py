"""用健康照片作阴性监督微调病灶分割器：DR 0 级按国际分级定义没有任何病灶，因此 DDR 分级训练区与 APTOS 2019（多相机）
的 0 级照片可作为“全阴性”掩膜。阴性补丁以原分割器在这些照片上预测出的病灶分量为中心（困难负样本），与带掩膜的训练补丁
按 1:1 混合；从 03_segmenter 的权重出发，按开发区掩膜的平均 AUPR 选轮次。只用训练区照片；不使用任何外部评价队列。
输出 data/checkpoints/segmenter_neg.pt 与 results/03c_segmenter_negatives/。
--name X 加 --roles/--hard-frac/--lr/--epochs 覆盖配置：输出 segmenter_neg_X.pt 与 results/03c_segmenter_negatives_X/（开发区比较用）。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.calib import calibrated_map, load_calibration  # noqa: E402
from scripts.lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from scripts.lib.segeval import evaluate, load_set  # noqa: E402
from scripts.lib.segnet import PatchSet, build_segmenter, dice_bce, gpu_norm  # noqa: E402

CFG = load_config()
CAL = load_calibration()


def spot_centres(key: str) -> tuple[str, np.ndarray]:
    """原分割器在该照片上超过校准阈值的像素（canon 坐标，每 7 个取 1 稀疏化）；无则返回空。"""
    p, _, tau = calibrated_map(key, CFG, CAL)
    yx = np.argwhere((p > tau[:, None, None]).any(0))[::7]
    return key, (yx * (CFG["preprocess"]["canon_size"] // p.shape[-1])).astype(np.int32)


class NegSet(torch.utils.data.Dataset):
    """每个样本 = 一张 0 级照片上的 per 个补丁，hard_frac 比例以预测病灶像素为中心，其余在视野内随机；掩膜全零。"""

    def __init__(self, keys, centres, patch, per, hard_frac, seed):
        self.keys, self.centres, self.p, self.per, self.hf, self.seed = keys, centres, patch, per, hard_frac, seed
        self.proc = ROOT / CFG["paths"]["processed"]
        self.epoch = 0

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, self.epoch, i, 7))
        k = self.keys[i]
        img = cv2.imread(str(self.proc / "canon" / f"{k}.jpg"))[:, :, ::-1]
        S, p, c = img.shape[0], self.p, self.centres[k]
        fov = None
        xs = []
        for _ in range(self.per):
            if len(c) and rng.random() < self.hf:
                cy, cx = c[rng.integers(len(c))]
            else:
                if fov is None:
                    fov = np.argwhere(cv2.imread(str(self.proc / "fov" / f"{k}.png"), 0) > 0)[::97]
                cy, cx = fov[rng.integers(len(fov))]
            y0 = int(np.clip(cy - p // 2 + rng.integers(-p // 4, p // 4 + 1), 0, S - p))
            x0 = int(np.clip(cx - p // 2 + rng.integers(-p // 4, p // 4 + 1), 0, S - p))
            a = img[y0:y0 + p, x0:x0 + p]
            if rng.random() < 0.5:
                a = a[:, ::-1]
            if rng.random() < 0.5:
                a = a[::-1]
            a = np.rot90(a, int(rng.integers(4)))
            a = cv2.convertScaleAbs(np.ascontiguousarray(a), alpha=rng.uniform(0.85, 1.15), beta=rng.uniform(-15, 15))
            xs.append(torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))))
        return torch.stack(xs), torch.zeros(self.per, 4, p, p, dtype=torch.uint8)


def loader(ds, bs, seed):
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=12, drop_last=True,
                                       persistent_workers=True, generator=torch.Generator().manual_seed(seed))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", default=""); ap.add_argument("--roles", nargs="*")
    ap.add_argument("--hard-frac", type=float); ap.add_argument("--lr", type=float); ap.add_argument("--epochs", type=int)
    ap.add_argument("--neg-weight", type=float)
    args = ap.parse_args()
    sc, nc = CFG["segmenter"], dict(CFG["segmenter_neg"])
    for k, v in (("negative_roles", args.roles), ("hard_frac", args.hard_frac), ("lr", args.lr), ("epochs", args.epochs),
                 ("neg_weight", args.neg_weight)):
        if v is not None:            # 0 也是合法的设定
            nc[k] = v
    sfx = f"_{args.name}" if args.name else ""
    set_global_seed(CFG["seed"])
    device = torch.device("cuda")
    out = ROOT / f"results/03c_segmenter_negatives{sfx}"; out.mkdir(parents=True, exist_ok=True)
    ck = ROOT / CFG["paths"]["checkpoints"]
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    man = man[man.included.astype(int) == 1]
    tr, dv = man[man.mask_role == "seg_train"], man[man.mask_role == "seg_dev"]
    # 阴性照片：分级训练区（DDR、APTOS）的 0 级照片，去掉带病灶掩膜的照片
    neg = man[man.role.isin(nc["negative_roles"]) & (man.grade.astype(str) == "0") & (man.mask_role == "")]
    keys = neg.image_key.tolist()
    with Pool(48) as pool:
        centres = dict(pool.map(spot_centres, keys, chunksize=16))
    n_spot = {r: int(sum(len(centres[k]) > 0 for k in g.image_key)) for r, g in neg.groupby("role")}
    print("negatives", neg.role.value_counts().to_dict(), "with predicted spots", n_spot, flush=True)

    tr_imgs, tr_masks, tr_fms = load_set(tr, CFG)
    dv_imgs, dv_masks, dv_fms = load_set(dv, CFG)
    per = sc["patches_per_image"]
    pos_ds = PatchSet(tr_imgs, tr_masks, tr_fms, sc["patch"], per, sc["lesion_patch_frac"], CFG["seed"])
    neg_ds = NegSet(keys, centres, sc["patch"], per, nc["hard_frac"], CFG["seed"])
    bs = max(1, sc["batch_size"] // per)
    pos_dl, neg_dl = loader(pos_ds, bs, CFG["seed"]), loader(neg_ds, bs, CFG["seed"] + 1)

    tm = [m for m in tr_masks]
    pos = np.array([np.mean([m[c].mean() for m in tm]) for c in range(4)])
    pw = torch.tensor(np.clip(np.sqrt((1 - pos) / pos) / 4, 1, 20), dtype=torch.float32, device=device)
    model = build_segmenter(sc["encoder"], 4, pretrained=False).to(device)
    model.load_state_dict(torch.load(ck / "segmenter.pt", map_location=device, weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=nc["lr"], weight_decay=sc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=nc["epochs"] * len(pos_dl))
    rec = {"epoch": -1, **{f"dev_aupr_{k}": v for k, v in evaluate(model, dv_imgs, dv_masks, dv_fms, device,
                                                                   sc["prob_map_size"]).items()}}
    print(json.dumps(rec), flush=True)
    best, log, t0 = -1.0, [rec], time.time()
    for ep in range(nc["epochs"]):
        model.train(); pos_ds.epoch = neg_ds.epoch = ep; tot = 0.0
        for (xp, yp), (xn, yn) in zip(pos_dl, neg_dl):
            x = gpu_norm(torch.cat([xp.flatten(0, 1), xn.flatten(0, 1)]).to(device, non_blocking=True))
            y = torch.cat([yp.flatten(0, 1), yn.flatten(0, 1)]).to(device, non_blocking=True).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x).float()
            if nc.get("neg_weight") is None:     # 阴性补丁与病灶补丁合并计算 Dice + BCE
                loss = dice_bce(logit, y, pw)
            else:                                # 病灶补丁照常；阴性补丁只用加权 BCE（全零目标），不进入 Dice
                n = yp.shape[0] * yp.shape[1]
                loss = dice_bce(logit[:n], y[:n], pw) + nc["neg_weight"] * F.binary_cross_entropy_with_logits(logit[n:], y[n:])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        rec = {"epoch": ep, "loss": tot / len(pos_dl), "minutes": (time.time() - t0) / 60}
        if (ep + 1) % nc["eval_every"] == 0 or ep == nc["epochs"] - 1:
            rec.update({f"dev_aupr_{k}": v for k, v in evaluate(model, dv_imgs, dv_masks, dv_fms, device,
                                                                 sc["prob_map_size"]).items()})
            if rec["dev_aupr_mean"] > best:
                best = rec["dev_aupr_mean"]
                torch.save(model.state_dict(), ck / f"segmenter_neg{sfx}.pt")
        log.append(rec); print(json.dumps(rec), flush=True)
    pd.DataFrame(log).to_csv(out / "train_log.tsv", sep="\t", index=False)
    (out / "summary.json").write_text(json.dumps({"best_dev_aupr_mean": best, "n_negative_photos": len(keys),
                                                  "negatives_by_role": neg.role.value_counts().to_dict(),
                                                  "with_predicted_spots": n_spot}, indent=2))
    write_provenance(f"03c_segmenter_negatives{sfx}", [ROOT / "results/02_manifest/images.tsv", ck / "segmenter.pt"],
                     [out / "train_log.tsv"], seed=CFG["seed"], extra=nc)


if __name__ == "__main__":
    main()
