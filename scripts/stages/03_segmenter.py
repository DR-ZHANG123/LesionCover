"""训练共同病灶分割器（只用 DDR 分级训练区的配对掩膜），按开发区掩膜的平均 AUPR 选轮次。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from scripts.lib.segeval import evaluate, load_set  # noqa: E402
from scripts.lib.segnet import PatchSet, build_segmenter, dice_bce, gpu_norm  # noqa: E402


def main():
    cfg = load_config()
    sc = cfg["segmenter"]
    set_global_seed(cfg["seed"])
    device = torch.device("cuda")
    out = ROOT / "results/03_segmenter"; out.mkdir(parents=True, exist_ok=True)
    ck = ROOT / cfg["paths"]["checkpoints"]; ck.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t")
    tr = man[(man.mask_role == "seg_train") & (man.included == 1)]
    dv = man[(man.mask_role == "seg_dev") & (man.included == 1)]
    tr_imgs, tr_masks, tr_fms = load_set(tr, cfg)
    dv_imgs, dv_masks, dv_fms = load_set(dv, cfg)
    print(f"train {len(tr_imgs)} dev {len(dv_imgs)}", flush=True)

    ds = PatchSet(tr_imgs, tr_masks, tr_fms, sc["patch"], sc["patches_per_image"], sc["lesion_patch_frac"], cfg["seed"])
    dl = torch.utils.data.DataLoader(ds, batch_size=max(1, sc["batch_size"] // sc["patches_per_image"]), shuffle=True,
                                     num_workers=16, drop_last=True, persistent_workers=True,
                                     generator=torch.Generator().manual_seed(cfg["seed"]))
    pos = np.array([np.mean([m[c].mean() for m in tr_masks]) for c in range(4)])
    pw = torch.tensor(np.clip(np.sqrt((1 - pos) / pos) / 4, 1, 20), dtype=torch.float32, device=device)
    model = build_segmenter(sc["encoder"], 4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=sc["lr"], weight_decay=sc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sc["epochs"] * len(dl))
    best, log, t0 = -1.0, [], time.time()
    for ep in range(sc["epochs"]):
        model.train(); ds.epoch = ep; tot = 0.0
        for x, y in dl:
            x = gpu_norm(x.flatten(0, 1).to(device, non_blocking=True)); y = y.flatten(0, 1).to(device, non_blocking=True).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x)
            loss = dice_bce(logit.float(), y, pw)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        rec = {"epoch": ep, "loss": tot / len(dl), "minutes": (time.time() - t0) / 60}
        if (ep + 1) % 20 == 0 or ep == sc["epochs"] - 1:
            rec.update({f"dev_aupr_{k}": v for k, v in evaluate(model, dv_imgs, dv_masks, dv_fms, device,
                                                                 sc["prob_map_size"]).items()})
            if rec["dev_aupr_mean"] > best:
                best = rec["dev_aupr_mean"]
                torch.save(model.state_dict(), ck / "segmenter.pt")
        log.append(rec); print(json.dumps(rec), flush=True)
    pd.DataFrame(log).to_csv(out / "train_log.tsv", sep="\t", index=False)
    (out / "summary.json").write_text(json.dumps({"best_dev_aupr_mean": best, "n_train": len(tr_imgs),
                                                  "n_dev": len(dv_imgs), "pos_weight": pw.tolist()}, indent=2))
    write_provenance("03_segmenter", [ROOT / "results/02_manifest/images.tsv"], [out / "train_log.tsv"], seed=cfg["seed"])


if __name__ == "__main__":
    main()
