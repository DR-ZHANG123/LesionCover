"""已发表的 DR 分级方法（GDRNet、CABNet）用其官方代码在同一 DDR 训练区重新训练，并在同一组队列上预测。

代码：GDRBench 官方仓库（config published_methods.code_dir，固定 commit），直接调用其 algorithms.py 中的算法类、
网络、损失与 FundusAug；超参取官方单源域设置（configs/datasets/GDRBench_ESDG.yaml：ResNet-50、224 输入、SGD lr 1e-3、
100 轮、batch 16、第 50 轮学习率 ×0.1）。与官方唯一的不同：
  1. 输入为本研究的规范化照片（与所有模型相同），FundusAug 所需视野掩膜由灰度阈值得到；
  2. DahLoss 的类别先验用 DDR 训练区的等级频数（官方字典中没有 DDR 单源域条目）；
  3. 轮次选择与本研究其余模型一致：DDR 开发区 κ 最优（沿用官方“前 30% 轮次不选”的约束）。
外部队列只预测一次，不参与任何选择。
用法：python scripts/stages/07p_published_methods.py --algorithm GDRNet --seed 11
输出 results/07_classify/<algorithm>_s<seed>/{predictions.tsv.gz, train_log.tsv, PROVENANCE.json}。"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

PRED_ROLES = ["ddr_dev", "ddr_test", "deepdrid", "idrid", "messidor2", "eyepacs", "mmrdr"]


class Fundus(Dataset):
    def __init__(self, paths, labels, img_tf, mask_tf=None, thr=10):
        self.p, self.y, self.tf, self.mtf, self.thr = list(paths), np.asarray(labels, int), img_tf, mask_tf, thr

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        im = Image.open(self.p[i]).convert("RGB")
        x = self.tf(im)
        if self.mtf is None:
            return x, int(self.y[i]), i
        m = Image.fromarray(((np.asarray(im.convert("L")) > self.thr) * 255).astype(np.uint8))
        return x, self.mtf(m), int(self.y[i]), 0, i


def import_official(code_dir: Path, size: int = 224):
    """size ≠ 224 时按比例改写官方 data_manager 中写死的输入边长（裁剪 re_size 与缩放 size = re_size × 256/224，取 8 的倍数以满足 FundusAug 的整数中心），
    其余增强与网络不变；改写后的模块在导入 algorithms 之前注册，使 GDRNet 的 post-FundusAug 使用同一边长。"""
    sys.path.insert(0, str(code_dir))
    if size != 224:
        import importlib.util
        import types
        import dataset  # noqa: F401,E402
        src = (code_dir / "dataset/data_manager.py").read_text()
        n = src.count("re_size = 224")
        src = src.replace("size = 256", f"size = {8 * round(size * 256 / 224 / 8)}").replace("re_size = 224", f"re_size = {size}")
        assert n == 3, n
        mod = types.ModuleType("dataset.data_manager"); mod.__file__ = str(code_dir / "dataset/data_manager.py")
        mod.__package__ = "dataset"
        sys.modules["dataset.data_manager"] = mod
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    import algorithms  # noqa: E402
    from configs.defaults import _C  # noqa: E402
    from dataset import data_manager  # noqa: E402
    from modeling.losses import DahLoss  # noqa: E402
    from utils.misc import get_scheduler  # noqa: E402
    return algorithms, _C, data_manager, DahLoss, get_scheduler


def manifest() -> pd.DataFrame:
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", low_memory=False)
    mm = pd.read_csv(ROOT / "results/01b_mmrdr/images.tsv", sep="\t", low_memory=False)
    man = pd.concat([man[["image_key", "role", "grade", "included"]], mm[["image_key", "role", "grade", "included"]]])
    return man[man.included.astype(int) == 1].reset_index(drop=True)


@torch.no_grad()
def predict(alg, dl, n):
    alg.eval()
    P = np.zeros((n, 5), np.float32)
    for x, _, idx in dl:
        out = alg.predict(x.cuda(non_blocking=True))
        out = out.view(len(idx), -1)
        P[idx.numpy()] = torch.softmax(out.float(), 1).cpu().numpy()
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--size", type=int, default=None, help="输入边长（缺省取 config published_methods.size，官方为 224）")
    args = ap.parse_args()
    cfg = load_config(); pm = cfg["published_methods"]
    code = ROOT / pm["code_dir"]
    size = args.size or pm.get("size", 224)
    algorithms, _C, dm, DahLoss, get_scheduler = import_official(code, size)
    set_global_seed(args.seed); torch.set_num_threads(8)
    tag = f"{args.algorithm}_s{args.seed}" + ("" if size == 224 else f"_{size}")
    out = ROOT / "results/07_classify" / tag; out.mkdir(parents=True, exist_ok=True)

    man = manifest()
    canon = ROOT / cfg["paths"]["processed"] / "canon"
    man["path"] = [str(canon / f"{k}.jpg") for k in man.image_key]
    tr = man[man.role == "ddr_train"].reset_index(drop=True)
    ytr = tr.grade.astype(int).to_numpy()

    c = _C.clone(); c.set_new_allowed(True)
    c.merge_from_file(str(code / f"configs/datasets/GDRBench_{pm['setting']}.yaml"))
    c.ALGORITHM, c.BACKBONE, c.DG_MODE = args.algorithm, pm["backbone"], pm["setting"]
    c.DATASET.NUM_CLASSES, c.DATASET.SOURCE_DOMAINS, c.DATASET.TARGET_DOMAINS = 5, ["DDR"], []
    c.VAL_EPOCH = 1

    counts = torch.tensor(np.bincount(ytr, minlength=5), dtype=torch.float32)

    class DDRDahLoss(DahLoss):          # 单源域 DDR：域先验为 1，类别先验取 DDR 训练区频数（其余与官方相同）
        def get_domain_label_prob(self):
            return torch.ones(1).cuda(), (counts / counts.sum()).cuda()
    algorithms.DahLoss = DDRDahLoss

    if args.algorithm == "GDRNet":
        tr_tf, te_tf, mask_tf = dm.get_pre_FundusAug(c)
    else:
        tr_tf, te_tf, mask_tf = dm.get_transform(c)
    nw = pm["num_workers"]
    g = torch.Generator(); g.manual_seed(args.seed)
    tr_dl = DataLoader(Fundus(tr.path, ytr, tr_tf, mask_tf, pm["fov_threshold"]), batch_size=c.BATCH_SIZE, shuffle=True,
                       num_workers=nw, drop_last=True, pin_memory=True, generator=g, persistent_workers=True)

    def eval_loader(d):
        return DataLoader(Fundus(d.path, d.grade.astype(int), te_tf), batch_size=64, shuffle=False, num_workers=nw)
    dv = man[man.role == "ddr_dev"].reset_index(drop=True); ydv = dv.grade.astype(int).to_numpy()
    dv_dl = eval_loader(dv)

    alg = algorithms.get_algorithm_class(args.algorithm)(5, c).cuda()
    sched = get_scheduler(alg.optimizer, c.EPOCHS)
    best, best_ep, log, t0 = -1.0, -1, [], time.time()
    ck = ROOT / cfg["paths"]["checkpoints"] / "published"; ck.mkdir(parents=True, exist_ok=True)
    ckf = ck / f"{tag}.pt"
    for i in range(c.EPOCHS):
        ep = i + 1; alg.train(); tot = []
        for x, m, y, d, _ in tr_dl:
            r = alg.update([x.cuda(non_blocking=True), m.cuda(non_blocking=True), y.cuda().long(), d.cuda().long()])
            tot.append(float(r["loss"]))
        alg.update_epoch(ep); sched.step()
        pdv = predict(alg, dv_dl, len(dv))
        k = cohen_kappa_score(ydv, pdv.argmax(1), weights="quadratic")
        log.append({"epoch": ep, "loss": np.mean(tot), "dev_qwk": k, "dev_acc": np.mean(pdv.argmax(1) == ydv),
                    "minutes": (time.time() - t0) / 60})
        print(log[-1], flush=True)
        if k > best and ep > c.EPOCHS * 0.3:
            best, best_ep = k, ep
            torch.save(alg.state_dict(), ckf)
        pd.DataFrame(log).to_csv(out / "train_log.tsv", sep="\t", index=False)

    alg.load_state_dict(torch.load(ckf, map_location="cuda"))
    rows = []
    for role in PRED_ROLES:
        d = man[man.role == role].reset_index(drop=True)
        P = predict(alg, eval_loader(d), len(d))
        rows.append(pd.DataFrame({"image_key": d.image_key, "role": role, **{f"p{j}": P[:, j] for j in range(5)}}))
        print(role, len(d), flush=True)
    pd.concat(rows).to_csv(out / "predictions.tsv.gz", sep="\t", index=False, float_format="%.5f")
    write_provenance(f"07_classify/{tag}", [ROOT / "results/02_manifest/images.tsv", ROOT / "results/01b_mmrdr/images.tsv"],
                     [out / "predictions.tsv.gz", out / "train_log.tsv"], seed=args.seed,
                     extra={"algorithm": args.algorithm, "code_commit": pm["code_commit"], "setting": pm["setting"], "input_size": size,
                            "best_epoch": best_ep, "best_dev_qwk": best})


if __name__ == "__main__":
    main()
