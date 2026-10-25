"""微动脉瘤遮挡实验：固定已训练的分级器（病灶覆盖裁块、辅助头主设置，三个种子），只改动裁块像素，比较
原样、遮挡（Telea 修补裁块内全部预测微动脉瘤分量的外扩掩膜）与对照（同样形状的掩膜随机平移到同一裁块内视野中、
不碰任何预测病灶的位置再修补，排除修补本身的影响）三种输入；全局 512 图不变。对象为 DDR 测试区全部照片；
带参考掩膜的照片另做一组以参考标注微动脉瘤为遮挡对象的版本。每种条件取三个模型的概率平均。
按参考等级（0、1、≥2）与裁块内预测微动脉瘤数（分量中心落在裁块内，同 08e；0、1–4、≥5）汇总判为无 DR 的比例、
P(DR)=1−p0 的变化及遮挡与对照之差（照片 bootstrap 95% 区间）。
输出 results/08k_occlusion/{per_photo.tsv.gz, summary.tsv, examples.png}。"""
from __future__ import annotations

import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.calib import calibrated_map, load_calibration  # noqa: E402
from scripts.lib.clsnet import Grader, backbone_for, to_u8  # noqa: E402
from scripts.lib.gradedata import load_crops  # noqa: E402
from scripts.lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from scripts.lib.selection import extract_components  # noqa: E402

cv2.setNumThreads(1)
CFG = load_config()
CAL = load_calibration()
OC = CFG["occlusion"]
PROC = ROOT / CFG["paths"]["processed"]
S, C = CFG["preprocess"]["canon_size"], CFG["selection"]["crop"]
P = [f"p{i}" for i in range(5)]
CONDS = ["intact", "ma", "ctrl", "ma_ref", "ctrl_ref"]
KER = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * OC["dilate_px"] + 1,) * 2)


def split_components(lab: np.ndarray):
    """连通域标记图 → 每个分量的 (xs, ys)。"""
    ys, xs = np.nonzero(lab)
    ids = lab[ys, xs]
    o = np.argsort(ids, kind="stable")
    ys, xs, ids = ys[o], xs[o], ids[o]
    cuts = np.flatnonzero(np.diff(ids)) + 1
    return list(zip(np.split(xs, cuts), np.split(ys, cuts)))


def predicted_components(key: str):
    """校准概率图上各类分量 → canon 像素坐标；返回 (微动脉瘤分量列表, 微动脉瘤中心 (canon), 全部预测病灶掩膜)。"""
    prob, _, tau = calibrated_map(key, CFG, CAL)
    comps = extract_components(prob, tau, CFG["selection"]["comp_min_pixels"])
    sc = S // prob.shape[-1]
    dx, dy = np.meshgrid(np.arange(sc), np.arange(sc))
    les = np.zeros((S, S), np.uint8)
    ma, cen = [], []
    for k, xx, yy, ce in zip(comps.cls, comps.px, comps.py, comps.centre):
        xs = (xx[:, None] * sc + dx.ravel()[None]).ravel(); ys = (yy[:, None] * sc + dy.ravel()[None]).ravel()
        les[ys, xs] = 1
        if k == 0:
            ma.append((xs, ys)); cen.append(ce * sc)
    return ma, np.asarray(cen, np.float32).reshape(-1, 2), les


def patches_in_crop(comps, x0, y0):
    """与裁块相交的分量 → [(外扩后的局部小掩膜 bool, 左上角 y, x)]（只取裁块内的部分）。"""
    d, out = OC["dilate_px"], []
    for xs, ys in comps:
        m = (xs >= x0) & (xs < x0 + C) & (ys >= y0) & (ys < y0 + C)
        if not m.any():
            continue
        lx, ly = xs[m] - x0, ys[m] - y0
        bx, by = lx.min() - d, ly.min() - d
        p = np.zeros((ly.max() - ly.min() + 1 + 2 * d, lx.max() - lx.min() + 1 + 2 * d), np.uint8)
        p[ly - by, lx - bx] = 1
        out.append((cv2.dilate(p, KER) > 0, by, bx))
    return out


def paint(mask: np.ndarray, p: np.ndarray, y: int, x: int) -> None:
    """把小掩膜 p 画到 mask 的 (y, x) 处（裁到边界内）。"""
    h, w = p.shape
    y0, x0, y1, x1 = max(y, 0), max(x, 0), min(y + h, C), min(x + w, C)
    if y1 > y0 and x1 > x0:
        mask[y0:y1, x0:x1] |= p[y0 - y:y1 - y, x0 - x:x1 - x]


def inpaint(tile: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return tile
    return cv2.inpaint(np.ascontiguousarray(tile), mask.astype(np.uint8), OC["inpaint_radius"], cv2.INPAINT_TELEA)


def occlude_and_control(tile, patches, free, rng):
    """遮挡：修补全部斑块。对照：每个斑块平移到 free（视野内、不碰病灶、互不重叠）的随机位置后修补。"""
    m = np.zeros((C, C), bool)
    for p, y, x in patches:
        paint(m, p, y, x)
    occ = inpaint(tile, m)
    ctrl, free, fail = np.zeros((C, C), bool), free.copy(), 0
    for p, _, _ in patches:
        h, w = p.shape
        ok = False
        if h < C and w < C:
            for _ in range(OC["ctrl_tries"]):
                y, x = int(rng.integers(0, C - h + 1)), int(rng.integers(0, C - w + 1))
                if not (p & ~free[y:y + h, x:x + w]).any():
                    ok = True
                    break
        if not ok:
            fail += 1
            continue
        ctrl[y:y + h, x:x + w] |= p
        free[y:y + h, x:x + w] &= ~p
    return occ, inpaint(tile, ctrl), fail


class OcclusionSet(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, crops: np.ndarray):
        self.df, self.crops = df.reset_index(drop=True), crops

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        key = r.image_key
        rng = np.random.default_rng((CFG["seed"], zlib.crc32(key.encode())))
        img = cv2.imread(str(PROC / "canon" / f"{key}.jpg"))[:, :, ::-1]
        g = cv2.imread(str(PROC / f"global{CFG['preprocess']['global_size']}" / f"{key}.jpg"))[:, :, ::-1]
        fov = cv2.imread(str(PROC / "fov" / f"{key}.png"), 0) > 0
        ma, cen, les = predicted_components(key)
        les_d = cv2.dilate(les, KER) > 0
        mp = PROC / "canon_masks" / f"{r.seg_key}.npy" if r.seg_key else None
        ref = mp is not None and mp.exists()
        if ref:
            rm = np.load(mp)
            ref_ma = split_components(cv2.connectedComponents(rm[0].astype(np.uint8), connectivity=8)[1])
            les_ref = cv2.dilate(rm.any(0).astype(np.uint8), KER) > 0
        tiles = {c: [] for c in CONDS}
        meta = np.zeros(6, np.int32)          # n_in, n_occ, ctrl_fail, has_ref, n_occ_ref, ctrl_fail_ref
        inside = np.zeros(len(cen), bool)
        for x0, y0 in self.crops[i]:
            x0, y0 = int(np.clip(x0, 0, S - C)), int(np.clip(y0, 0, S - C))
            t = np.ascontiguousarray(img[y0:y0 + C, x0:x0 + C])
            inside |= (cen[:, 0] >= x0) & (cen[:, 0] < x0 + C) & (cen[:, 1] >= y0) & (cen[:, 1] < y0 + C)
            free = fov[y0:y0 + C, x0:x0 + C] & ~les_d[y0:y0 + C, x0:x0 + C]
            pt = patches_in_crop(ma, x0, y0)
            occ, ctl, f = occlude_and_control(t, pt, free, rng)
            tiles["intact"].append(t); tiles["ma"].append(occ); tiles["ctrl"].append(ctl)
            meta[1] += len(pt); meta[2] += f
            if ref:
                pr = patches_in_crop(ref_ma, x0, y0)
                occ, ctl, f = occlude_and_control(t, pr, free & ~les_ref[y0:y0 + C, x0:x0 + C], rng)
                meta[4] += len(pr); meta[5] += f
            else:
                occ = ctl = t
            tiles["ma_ref"].append(occ); tiles["ctrl_ref"].append(ctl)
        meta[0], meta[3] = int(inside.sum()), int(ref)
        x = torch.stack([torch.stack([to_u8(t) for t in tiles[c]]) for c in CONDS])
        return to_u8(g), x, torch.from_numpy(meta), i


@torch.no_grad()
def predict(df, crops, dev):
    models = []
    for sd in CFG["seeds"]:
        tag = f"{OC['rule']}_k4_s{sd}_{OC['variant']}"
        m = Grader(backbone_for(tag, CFG, ROOT), 5, CFG["classifier"]["attn_dim"], True, crop_head="aux" in OC["variant"])
        m.load_state_dict(torch.load(ROOT / CFG["paths"]["checkpoints"] / "classify" / f"{tag}.pt", map_location="cpu",
                                     weights_only=True))
        models.append(m.to(dev).eval())
    ds = OcclusionSet(df, crops)
    dl = torch.utils.data.DataLoader(ds, batch_size=OC["batch_size"], num_workers=16, pin_memory=True)
    prob = np.zeros((len(ds), len(CONDS), 5), np.float32)
    meta = np.zeros((len(ds), 6), np.int32)
    for n, (g, x, mt, idx) in enumerate(dl):
        g, x = g.to(dev, non_blocking=True), x.to(dev, non_blocking=True)
        for ci in range(len(CONDS)):
            acc = 0
            for m in models:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit, _ = m(g, x[:, ci])
                acc = acc + torch.softmax(logit.float(), 1)
            prob[idx.numpy(), ci] = (acc / len(models)).cpu().numpy()
        meta[idx.numpy()] = mt.numpy()
        if n % 50 == 0:
            print(n * OC["batch_size"], len(ds), flush=True)
    return prob, meta


def boot(vals: dict, B: int, rng) -> dict:
    """vals：名称 → 每张照片的数值；返回各名称的 (均值, 下限, 上限, 双侧 P)。"""
    n = len(next(iter(vals.values())))
    idx = rng.integers(n, size=(B, n))
    out = {}
    for k, v in vals.items():
        d = v[idx].mean(1)
        out[k] = (float(v.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
                  float(min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / B))))
    return out


def summarize(pp: pd.DataFrame, B: int, rng) -> pd.DataFrame:
    rows = []
    for target, sfx, n_col in (("predicted", "", "n_ma_in_crops"), ("annotated", "_ref", "n_ma_in_crops")):
        d0 = pp if target == "predicted" else pp[pp.has_ref == 1]
        for gname, gm in (("no DR", d0.grade == 0), ("mild DR", d0.grade == 1), ("referable DR", d0.grade >= 2)):
            for bname, lo, hi in (("all", 0, 10**6), ("0", 0, 0), ("1-4", 1, 4), ("5+", 5, 10**6)):
                d = d0[gm & d0[n_col].between(lo, hi)]
                if len(d) < 2:
                    continue
                pdr = {c: 1 - d[f"p0_{c}"].to_numpy() for c in ("intact", "ma" + sfx, "ctrl" + sfx)}
                none = {c: (d[[f"{q}_{c}" for q in P]].to_numpy().argmax(1) == 0).astype(float) for c in pdr}
                i, m, c = "intact", "ma" + sfx, "ctrl" + sfx
                st = boot({"d_pdr_occ": pdr[m] - pdr[i], "d_pdr_ctrl": pdr[c] - pdr[i], "d_pdr_occ_vs_ctrl": pdr[m] - pdr[c],
                           "d_none_occ_vs_ctrl": none[m] - none[c], "none_intact": none[i], "none_occ": none[m],
                           "none_ctrl": none[c]}, B, rng)
                row = {"occluded": target, "grade_group": gname, "ma_in_crops": bname, "n": len(d),
                       "none_intact": none[i].mean(), "none_occ": none[m].mean(), "none_ctrl": none[c].mean(),
                       "pdr_intact": pdr[i].mean()}
                for k, (v, l, h, p) in st.items():
                    row.update({k: v, f"{k}_lo": l, f"{k}_hi": h, f"{k}_p": p})
                rows.append(row)
    return pd.DataFrame(rows)


def examples(df, crops, pp, out: Path) -> None:
    """视觉核对：裁块内预测微动脉瘤最多的三张轻度照片，各取微动脉瘤最多的裁块（原样 | 遮挡 | 对照）。"""
    ds = OcclusionSet(df, crops)
    top = pp[pp.grade == 1].nlargest(3, "n_ma_in_crops").index
    rows = []
    for i in top:
        _, x, _, _ = ds[int(i)]
        k = int(np.argmax([(x[0, j] != x[1, j]).sum().item() for j in range(x.shape[1])]))
        rows.append(np.concatenate([x[ci, k].numpy().transpose(1, 2, 0) for ci in range(3)], 1))
    cv2.imwrite(str(out / "examples.png"), np.concatenate(rows, 0)[:, :, ::-1])


def main():
    set_global_seed(CFG["seed"])
    t0 = time.time()
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    df = man[(man.role == OC["role"]) & (man.included.astype(int) == 1)].copy().reset_index(drop=True)
    df["grade"] = df.grade.astype(int)
    crops = load_crops(OC["rule"], 4, df)
    prob, meta = predict(df, crops, torch.device("cuda"))
    pp = df[["image_key", "grade"]].copy()
    for j, c in enumerate(["n_ma_in_crops", "n_occluded", "n_ctrl_failed", "has_ref", "n_ref_occluded", "n_ref_ctrl_failed"]):
        pp[c] = meta[:, j]
    for ci, c in enumerate(CONDS):
        for q in range(5):
            pp[f"p{q}_{c}"] = prob[:, ci, q]
    pp.loc[pp.has_ref == 0, [f"p{q}_{c}" for c in ("ma_ref", "ctrl_ref") for q in range(5)]] = np.nan
    out = ROOT / "results/08k_occlusion"; out.mkdir(parents=True, exist_ok=True)
    pp.to_csv(out / "per_photo.tsv.gz", sep="\t", index=False, float_format="%.5f")
    summ = summarize(pp, CFG["stats"]["n_boot"], np.random.default_rng(CFG["seed"]))
    summ.to_csv(out / "summary.tsv", sep="\t", index=False)
    examples(df, crops, pp, out)
    print(summ[["occluded", "grade_group", "ma_in_crops", "n", "none_intact", "none_occ", "none_ctrl", "d_pdr_occ",
                "d_pdr_occ_lo", "d_pdr_occ_hi", "d_pdr_ctrl", "d_pdr_occ_vs_ctrl", "d_pdr_occ_vs_ctrl_lo",
                "d_pdr_occ_vs_ctrl_hi"]].round(3).to_string())
    print(f"minutes {(time.time() - t0) / 60:.1f}; MA occluded {int(pp.n_occluded.sum())}, control placements failed "
          f"{int(pp.n_ctrl_failed.sum())}; annotated occluded {int(pp.n_ref_occluded.sum())}, failed "
          f"{int(pp.n_ref_ctrl_failed.sum())}")
    ck = [ROOT / CFG["paths"]["checkpoints"] / "classify" / f"{OC['rule']}_k4_s{s}_{OC['variant']}.pt" for s in CFG["seeds"]]
    write_provenance("08k_occlusion", [ROOT / "results/02_manifest/images.tsv", ROOT / "results/05_selection/crops.tsv.gz", *ck],
                     [out / "per_photo.tsv.gz", out / "summary.tsv"], seed=CFG["seed"])


if __name__ == "__main__":
    main()
