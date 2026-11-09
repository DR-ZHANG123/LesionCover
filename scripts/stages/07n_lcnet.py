"""训练 LesionCover 网络（scripts/lib/lcnet.py）并预测。

用法：python scripts/stages/07n_lcnet.py --seed 11 [--tag base] [--predict ddr_dev ddr_test ...]
只用 DDR 训练区训练、DDR 开发区选轮次；--predict 缺省只预测开发区，外部队列在设置定好后统一预测。
--lesion-aux/--global-aux/--no-lesion：消融（辅助头权重、去掉病灶 token 只留全局 token）。"""
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
from scripts.lib.clsnet import quadratic_kappa  # noqa: E402
from scripts.lib.lcdata import make_lc_loader  # noqa: E402
from scripts.lib.lcnet import LesionCoverNet  # noqa: E402
from scripts.lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from scripts.lib.trainutil import ordinal_loss  # noqa: E402


def _flip(g, tok, cnt, axis: int, P: float):
    """测试时翻转：axis 0 水平、1 垂直。图像、token 坐标（概率图坐标系，边长 P）与象限计数同步翻转。"""
    g = g.flip(-1 if axis == 0 else -2)
    tok = tok.clone(); tok[..., 3 + axis] = P - tok[..., 3 + axis]
    perm = [1, 0, 3, 2] if axis == 0 else [2, 3, 0, 1]                     # 象限 q = (x≥½) + 2(y≥½)
    cnt = cnt.view(cnt.shape[0], -1, 4)[..., perm].reshape(cnt.shape)
    return g, tok, cnt


@torch.no_grad()
def predict(model, loader, device, tta: bool = False):
    model.eval()
    P = np.zeros((len(loader.dataset), 5), np.float32)
    for g, tok, val, cnt, _, idx in loader:
        g, tok, val, cnt = (t.to(device, non_blocking=True) for t in (g, tok, val, cnt))
        views = [(g, tok, cnt)]
        if tta:
            views += [_flip(g, tok, cnt, a, model.P) for a in (0, 1)]
        pr = 0
        for gg, tt, cc in views:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pr = pr + torch.softmax(model(gg, tt, val, cc).float(), 1)
        P[idx.numpy()] = (pr / len(views)).cpu().numpy()
    return P


def split(man, role):
    d = man[man.role == role].reset_index(drop=True)
    return d, d.grade.fillna(-1).astype(int).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--predict", nargs="*", default=["ddr_dev"])
    ap.add_argument("--lesion-aux", type=float, default=None)
    ap.add_argument("--global-aux", type=float, default=None)
    ap.add_argument("--no-lesion", action="store_true")
    ap.add_argument("--hyper", type=int, default=None, help="病灶超图卷积层数（缺省读 config）")
    ap.add_argument("--ord", type=float, default=None, help="有序损失权重")
    ap.add_argument("--cons", type=float, default=None, help="病灶分支与融合输出的一致性损失权重")
    ap.add_argument("--text", action="store_true", help="加知识引导分类头（05f_text_prototypes 的分级文字原型）")
    ap.add_argument("--rule", action="store_true", help="分级规则模块（逐类逐象限软阈值，rulehead.py）")
    ap.add_argument("--rule-learn", action="store_true",
                    help="规则阈值自由学习：中性初始化 lcnet.rule_learn_init、单独学习率 rule_lr 且不衰减，并加只看规则特征的单调有序分级损失")
    ap.add_argument("--gate", action="store_true", help="按病灶负荷的融合/全局门控")
    ap.add_argument("--proto", type=float, default=None, help="病灶原型聚合，值为原型对比损失权重（protoagg.py）")
    ap.add_argument("--fm", default=None, help="基础模型编码器（timm 名称，如 vit_large_patch14_dinov2.lvd142m）")
    ap.add_argument("--fm-tokens", default=None, help="双编码器融合：基础模型（timm 名称）提供语义 token，ConvNeXt 保持为主编码器")
    ap.add_argument("--fm-rank", type=int, default=None, help="基础模型 LoRA 秩（0 = 完全冻结）")
    ap.add_argument("--fm-size", type=int, default=None)
    ap.add_argument("--grad-ckpt", action="store_true", help="ConvNeXt 梯度检查点（省显存）")
    ap.add_argument("--fm-weights", default=None, help="基础模型本地权重（如 RETFound）")
    ap.add_argument("--ablate", default="", help="逗号分隔：app,pos,attr,count（见 lcnet.py）")
    ap.add_argument("--predict-only", action="store_true")
    ap.add_argument("--tta", action="store_true", help="测试时水平/垂直翻转平均（只用于 --predict-only）")
    args = ap.parse_args()
    cfg = load_config(); cfg["_proc"] = ROOT / cfg["paths"]["processed"]
    lc = dict(cfg["lcnet"])
    if args.lesion_aux is not None:
        lc["lesion_aux_weight"] = args.lesion_aux
    if args.global_aux is not None:
        lc["global_aux_weight"] = args.global_aux
    lc["ablate"] = [a for a in args.ablate.split(",") if a]
    for k, v in (("hyper_layers", args.hyper), ("ord_weight", args.ord), ("cons_weight", args.cons)):
        if v is not None:
            lc[k] = v
    lc["rule"], lc["gate"] = args.rule or args.rule_learn, args.gate
    if args.rule_learn:
        lc["rule_init"] = lc["rule_learn_init"]
    else:
        lc["rule_aux_weight"] = 0.0
    if args.fm or args.fm_tokens:
        lc["fm"], lc["fm_tokens"] = args.fm, args.fm_tokens
        lc["fm_weights"] = lc["retfound_weights"] if args.fm_weights == "retfound" else args.fm_weights
    lc["grad_ckpt"] = args.grad_ckpt
    for k, v in (("fm_rank", args.fm_rank), ("fm_size", args.fm_size)):
        if v is not None:
            lc[k] = v
    lc["proto"], lc["proto_weight"] = args.proto is not None, args.proto or 0.0
    if args.text:
        lc["text_proto"] = str(ROOT / "results/05f_text_prototypes/prototypes.npy")
    torch.set_num_threads(8)
    set_global_seed(args.seed)
    device = torch.device("cuda")
    tag = f"LC_{args.tag}_s{args.seed}"
    out = ROOT / "results/07n_lcnet" / tag; out.mkdir(parents=True, exist_ok=True)
    ck = ROOT / cfg["paths"]["checkpoints"] / "lcnet"; ck.mkdir(parents=True, exist_ok=True)
    tdir = cfg["_proc"] / "lesion_tokens"

    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", low_memory=False)
    mm = ROOT / "results/01b_mmrdr/images.tsv"           # 第五外部队列 MMRDR（只预测，不参与训练与选择）
    if mm.exists():
        man = pd.concat([man, pd.read_csv(mm, sep="\t")[["image_key", "role", "grade", "included"]]])
    man = man[man.included.astype(int) == 1]
    tr, ytr = split(man, "ddr_train"); dv, ydv = split(man, "ddr_dev")
    model = LesionCoverNet(cfg["classifier"]["backbone"], lc, prob_size=cfg["segmenter"]["prob_map_size"],
                           drop_path=lc.get("drop_path", 0.0)).to(device).to(memory_format=torch.channels_last)

    def loader(keys, y, train, bs):
        _, dl = make_lc_loader(keys, y, cfg, train, args.seed, bs, tdir)
        if args.no_lesion:                     # 消融：病灶 token 全部设为无效（计数 token 也置零）
            dl.dataset.M = 0
        return dl

    t0, best = time.time(), -1.0
    if not args.predict_only:
        tr_ds, tr_dl = make_lc_loader(tr.image_key, ytr, cfg, True, args.seed, lc["batch_size"], tdir)
        dv_dl = loader(dv.image_key, ydv, False, 2 * lc["batch_size"])
        enc = [p for p in model.enc.parameters() if p.requires_grad]; ids = {id(p) for p in model.enc.parameters()}
        rp = model.rule.rule_params() if (args.rule_learn and model.rule is not None) else []
        ids |= {id(p) for p in rp}
        head = [p for p in model.parameters() if id(p) not in ids]
        groups = [{"params": enc, "lr": lc["head_lr"] if lc.get("fm") else lc["lr"]}, {"params": head, "lr": lc["head_lr"]}]
        if rp:
            groups.append({"params": rp, "lr": lc["rule_lr"], "weight_decay": 0.0})
        opt = torch.optim.AdamW(groups, weight_decay=lc["weight_decay"])
        freq = np.bincount(ytr, minlength=5).astype(np.float64)
        w = 1 / np.sqrt(freq); w = w / (w * freq).sum() * freq.sum()
        cw = torch.tensor(w, dtype=torch.float32, device=device)
        steps = lc["epochs"] * len(tr_dl); warm = lc["warmup_epochs"] * len(tr_dl)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
        log = []
        for ep in range(lc["epochs"]):
            model.train(); tr_ds.epoch = ep
            tot, n = 0.0, 0
            for g, tok, val, cnt, y, _ in tr_dl:
                g, tok, val, cnt, y = (t.to(device, non_blocking=True) for t in (g, tok, val, cnt, y))
                if args.no_lesion:
                    val = torch.zeros_like(val); cnt = torch.zeros_like(cnt)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit, ag, al = model(g, tok, val, cnt, return_aux=True)
                loss = F.cross_entropy(logit.float(), y, weight=cw)
                loss = loss + lc["global_aux_weight"] * F.cross_entropy(ag.float(), y, weight=cw)
                if not args.no_lesion:
                    loss = loss + lc["lesion_aux_weight"] * F.cross_entropy(al.float(), y, weight=cw)
                if lc["proto"] and model.proto_loss is not None:
                    loss = loss + lc["proto_weight"] * model.proto_loss
                if lc.get("rule_aux_weight", 0) > 0 and model.rule.score is not None:   # 规则有序分级：累积二分类
                    tgt = (y[:, None] > torch.arange(4, device=device)[None]).float()
                    loss = loss + lc["rule_aux_weight"] * F.binary_cross_entropy_with_logits(model.rule.score.float(), tgt)
                if lc.get("ord_weight", 0) > 0:          # 有序损失：期望等级与参考等级的平方误差
                    loss = loss + lc["ord_weight"] * ordinal_loss(logit, y)
                if lc.get("cons_weight", 0) > 0 and not args.no_lesion:
                    # 一致性：融合输出（停梯度）作为目标，约束只看病灶证据的分支与之一致（KL），使病灶通路承担分级证据
                    loss = loss + lc["cons_weight"] * F.kl_div(F.log_softmax(al.float(), 1),
                                                               F.softmax(logit.float().detach(), 1), reduction="batchmean")
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); sched.step()
                tot += loss.item() * len(y); n += len(y)
            P = predict(model, dv_dl, device)
            qwk = quadratic_kappa(ydv, P.argmax(1))
            mild = ydv == 1
            log.append({"epoch": ep, "train_loss": tot / n, "dev_qwk": qwk,
                        "dev_mild_as_none": float((P[mild].argmax(1) == 0).mean()), "minutes": (time.time() - t0) / 60})
            print(json.dumps(log[-1]), flush=True)
            if qwk > best:
                best = qwk
                torch.save(model.state_dict(), ck / f"{tag}.pt")
        pd.DataFrame(log).to_csv(out / "train_log.tsv", sep="\t", index=False)

    model.load_state_dict(torch.load(ck / f"{tag}.pt", map_location=device))
    rows = []
    for role in args.predict:
        sub, y = split(man, role)
        P = predict(model, loader(sub.image_key, np.maximum(y, 0), False, 2 * lc["batch_size"]), device, args.tta)
        d = pd.DataFrame(P, columns=[f"p{i}" for i in range(5)])
        d.insert(0, "image_key", sub.image_key.to_numpy()); d.insert(1, "role", role)
        rows.append(d)
    f = out / (("predictions" if set(args.predict) != {"ddr_dev"} else "predictions_dev") + ("_tta" if args.tta else "")
               + ".tsv.gz")
    pd.concat(rows).to_csv(f, sep="\t", index=False, float_format="%.5f")
    stage = f"07n_lcnet/{tag}" + ("/predict" if args.predict_only else "") + ("_tta" if args.tta else "")   # 只预测时不覆盖训练记录
    write_provenance(stage, [ROOT / "results/02_manifest/images.tsv"], [f], seed=args.seed,
                     extra={"best_dev_qwk": best, "train_minutes": (time.time() - t0) / 60, "lcnet": lc,
                            "no_lesion": args.no_lesion})


if __name__ == "__main__":
    import os
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
