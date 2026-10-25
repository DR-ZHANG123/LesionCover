"""训练并评价一个分级配置（方法 × 种子）。

用法：python scripts/stages/07_classify.py --method C --seed 11 [--k 4]
方法：G512 / G1024（仅全局）；U R T N D C Cmass S（全局 + K 个裁块，裁块来自 05_selection）。
开发区 QWK 最优的轮次用于预测 DDR 测试集与 DeepDRiD；输出逐图概率表。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.clsnet import Grader, quadratic_kappa  # noqa: E402
from scripts.lib.gradedata import hard_negative_weights, load_crops, make_loader  # noqa: E402
from scripts.lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from scripts.lib.trainutil import Ema, ordinal_loss, param_groups  # noqa: E402
from scripts.lib.variants import encoder_name, tokens  # noqa: E402


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P, A = np.zeros((len(loader.dataset), 5), np.float32), None
    for g, c, _, idx in loader:
        g = g.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True) if c.numel() else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit, a = model(g, c)
        P[idx.numpy()] = torch.softmax(logit.float(), 1).cpu().numpy()
        if a is not None:
            if A is None:
                A = np.zeros((len(loader.dataset), a.shape[1]), np.float32)
            A[idx.numpy()] = a.float().cpu().numpy()
    return P, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--variant", default="", help="训练设置，由 scripts/lib/variants.py 中的记号拼接（如 aux、auxmc、auxmcsmall、auxmcbaseema）")
    ap.add_argument("--recipe", default="", help="config 中 recipes 下的训练超参组合（只在 DDR 开发区上比较）")
    args = ap.parse_args()
    cfg = load_config()
    torch.set_num_threads(8)
    cc = dict(cfg["classifier"])
    if args.recipe:
        cc.update(cfg["recipes"][args.recipe])
    tk = tokens(args.variant)
    aux_w = cc["crop_aux_weight"] if "aux" in tk else 0.0
    # variant "auxart"：辅助头 + 跨相机伪影增强；"art"：仅增强（用于全局图基线）
    g_drop = cc["global_drop"] if "gdrop" in tk else 0.0
    set_global_seed(args.seed)
    device = torch.device("cuda")
    tag = f"{args.method}_k{args.k}_s{args.seed}" if not args.method.startswith("G") else f"{args.method}_s{args.seed}"
    if args.variant:
        tag += f"_{args.variant}"
    if args.recipe:
        tag += f"_{args.recipe}"
    out = ROOT / "results/07_classify" / tag
    out.mkdir(parents=True, exist_ok=True)
    ck = ROOT / cfg["paths"]["checkpoints"] / "classify"
    ck.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t")
    man = man[man.included == 1].copy()
    man["canon_path"] = [str(ROOT / cfg["paths"]["processed"] / "canon" / f"{k}.jpg") for k in man.image_key]
    # variant 含 mc：训练集加入 APTOS 2019（多相机）；模型选择仍只用 DDR 开发区
    train_roles = ["ddr_train", "aptos_train"] if "mc" in tk else ["ddr_train"]
    tr, dv = man[man.role.isin(train_roles)].reset_index(drop=True), man[man.role == "ddr_dev"].reset_index(drop=True)
    tr["grade"], dv["grade"] = tr.grade.astype(int), dv.grade.astype(int)
    gsize = cfg["preprocess"]["global_size_large"] if args.method == "G1024" else cfg["preprocess"]["global_size"]

    tr_ds, tr_dl = make_loader(tr, load_crops(args.method, args.k, tr, var=args.variant), cfg, gsize, True, args.seed, cc["batch_size"],
                               artifact_aug="art" in tk,
                               weights=hard_negative_weights(tr, cfg) if "hn" in tk else None)
    _, dv_dl = make_loader(dv, load_crops(args.method, args.k, dv, var=args.variant), cfg, gsize, False, args.seed, 2 * cc["batch_size"])

    backbone = encoder_name(args.variant, cfg, ROOT)
    if "rf" in tk:                               # RETFound 编码器及其官方微调设置（逐层学习率衰减）
        cc.update(cc["rf_recipe"])
    elif "dino" in tk:                           # DINOv2 ViT 的微调设置
        cc.update(cc["dino_recipe"])
    model = Grader(backbone, 5, cc["attn_dim"], not args.method.startswith("G"), crop_head=aux_w > 0,
                   drop_path=cc.get("drop_path", 0.0))
    model = model.to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(param_groups(model, cc), weight_decay=cc["weight_decay"])
    ema = Ema(model, cc["ema_decay"]) if "ema" in tk else None   # 含 ema：用滑动平均权重做轮次选择与预测
    ord_w = cc["ord_weight"] if "ord" in tk else 0.0
    freq = np.bincount(tr.grade, minlength=5).astype(np.float64)
    w = 1 / np.sqrt(freq); w = w / (w * freq).sum() * freq.sum()
    cw = torch.tensor(w, dtype=torch.float32, device=device)
    steps = cc["epochs"] * len(tr_dl); warm = cc["warmup_epochs"] * len(tr_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))

    best, log = -1.0, []
    t0 = time.time()
    for ep in range(cc["epochs"]):
        model.train(); tr_ds.epoch = ep
        tot, n = 0.0, 0
        for g, c, y, _ in tr_dl:
            g = g.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True) if c.numel() else None
            y = y.to(device)
            gm = (torch.rand(len(y), device=device) >= g_drop).float() if (g_drop > 0 and c is not None) else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit, _, aux = model(g, c, global_mask=gm, return_aux=True) if c is not None else (*model(g, c), None)
            ls = cc.get("label_smoothing", 0.0)
            loss = F.cross_entropy(logit.float(), y, weight=cw, label_smoothing=ls)
            if aux is not None:
                loss = loss + aux_w * F.cross_entropy(aux.float(), y, weight=cw, label_smoothing=ls)
            if ord_w > 0:
                loss = loss + ord_w * (ordinal_loss(logit, y) + (aux_w * ordinal_loss(aux, y) if aux is not None else 0.0))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            if ema is not None:
                ema.update(model)
            tot += loss.item() * len(y); n += len(y)
        P, _ = predict(ema.model if ema is not None else model, dv_dl, device)
        qwk = quadratic_kappa(dv.grade, P.argmax(1))
        log.append({"epoch": ep, "train_loss": tot / n, "dev_qwk": qwk, "minutes": (time.time() - t0) / 60})
        print(json.dumps(log[-1]), flush=True)
        if qwk > best:
            best = qwk
            torch.save((ema.model if ema is not None else model).state_dict(), ck / f"{tag}.pt")
    pd.DataFrame(log).to_csv(out / "train_log.tsv", sep="\t", index=False)

    model.load_state_dict(torch.load(ck / f"{tag}.pt", map_location=device))
    rows = []
    for role in ["ddr_dev", "ddr_test", "deepdrid", "idrid"]:
        sub = man[man.role == role].reset_index(drop=True)
        _, dl = make_loader(sub, load_crops(args.method, args.k, sub, var=args.variant), cfg, gsize, False, args.seed, 2 * cc["batch_size"])
        P, A = predict(model, dl, device)
        d = pd.DataFrame(P, columns=[f"p{i}" for i in range(5)])
        d.insert(0, "image_key", sub.image_key); d.insert(1, "role", role)
        if A is not None:
            for j in range(A.shape[1]):
                d[f"attn{j}"] = A[:, j]
        rows.append(d)
    pred = pd.concat(rows)
    pred.to_csv(out / "predictions.tsv.gz", sep="\t", index=False, float_format="%.5f")
    write_provenance(f"07_classify/{tag}", [ROOT / "results/02_manifest/images.tsv"], [out / "predictions.tsv.gz"],
                     seed=args.seed, extra={"best_dev_qwk": best, "train_minutes": (time.time() - t0) / 60})


if __name__ == "__main__":
    import os
    import traceback
    try:
        main()
    except BaseException:
        # 出错（如显存溢出）时立即退出，避免常驻的数据加载进程让失败的作业挂起并占住显存
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
