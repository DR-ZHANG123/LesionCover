"""相机复现斑点的描述（供 Fig. 5 与正文）：
(1) 各队列预测病灶分量复现率的分位数、超过阈值的比例、有复现分量的照片数，以及复现位置（任一类型复现率 ≥ 阈值的格）
占视野的比例（含 DDR 开发与训练区；来自 05d）；
(1b) 偶然重合的参照（无标签）：每张照片的全部分量整体随机平移（每轴 ±shift 格，20 次），破坏固定位置但保留病灶的
大致空间分布，重算复现率；给出各队列平移后最大复现率与超过阈值比例的均值（病灶越密，偶然重合越高）；
(2) Messidor-2 每张照片被标为复现的分量数，按相机设置（原始图像宽度）与参考等级汇总（照片 bootstrap 95% 区间）；
(3) 示例：主设置三个模型集成下，原输入判为 DR、去掉复现斑点后判为无 DR 的参考 0 级 Messidor-2 照片中，原病灶覆盖
裁块内复现微动脉瘤分量最多（且该裁块不含其他类型的复现分量）的一张，保存该裁块修补前后的图像与复现分量中心。
输出 results/08o_camera_spots/{recurrence_quantiles.tsv, chance_recurrence.tsv, spots_by_camera.tsv, example_before.png, example_after.png,
example_spots.tsv}。"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.gradedata import load_crops  # noqa: E402
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402

h7 = importlib.import_module("scripts.stages.07h_predict_camera")
CFG = load_config()
CP = CFG["camera_prior"]
S, C = CFG["preprocess"]["canon_size"], CFG["selection"]["crop"]
OUT = ROOT / "results/08o_camera_spots"
P = [f"lp{i}" for i in range(5)]


def hot_area(role: str) -> float:
    """任一类型复现率 ≥ 阈值的格占视野内切圆的比例。"""
    z = np.load(ROOT / "results/05d_camera_prior/recurrence_maps.npz")
    n = CFG["segmenter"]["prob_map_size"]
    hot = np.zeros((n, n), bool)
    for t in CFG["segmenter"]["classes"]:
        hot |= z[f"{role}_{t}"] >= CP["threshold"]
    yy, xx = np.mgrid[:n, :n]
    fov = (xx - (n - 1) / 2) ** 2 + (yy - (n - 1) / 2) ** 2 <= (n / 2) ** 2
    return float((hot & fov).sum() / fov.sum())


def quantiles(comp: pd.DataFrame, summ: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for role, d in comp.groupby("role"):
        v = d.recurrence.to_numpy()
        ma = d[d.lesion == "MA"].recurrence.to_numpy()
        rows.append({"role": role, "components": len(v), "median": np.median(v), "q99": np.quantile(v, .99),
                     "q999": np.quantile(v, .999), "max": v.max(), "flagged_share": (v >= CP["threshold"]).mean(),
                     "flagged_share_ma": (ma >= CP["threshold"]).mean(), "max_ma": ma.max(),
                     "photos": d.image_key.nunique(), "photos_flagged": d[d.flagged == 1].image_key.nunique(),
                     "hot_area": hot_area(role)})
    for role, d in summ[summ.role.isin(CP["reference_roles"])].groupby("role"):
        rows.append({"role": role, "components": int(d.components.sum()), "max": d.max_recurrence.max(),
                     "flagged_share": d.flagged_components.sum() / d.components.sum()})
    return pd.DataFrame(rows)


def recurrence_kdtree(comp: pd.DataFrame, cen: np.ndarray, r: int) -> np.ndarray:
    """每个分量：同队列其他照片中同类分量中心（取整）落在切比雪夫距离 ≤ r 内的照片比例（与 05d 的计数图等价）。"""
    out = np.zeros(len(comp), np.float32)
    photo = comp.image_key.astype("category").cat.codes.to_numpy()
    n = comp.image_key.nunique() if "n_photos" not in comp.attrs else comp.attrs["n_photos"]
    for t in comp.lesion.unique():
        idx = np.flatnonzero(comp.lesion.to_numpy() == t)
        xy = np.floor(cen[idx]).astype(np.int32)
        tree = cKDTree(xy)
        for k, nb in enumerate(tree.query_ball_point(xy, r, p=np.inf)):
            ph = np.unique(photo[idx[nb]])
            out[idx[k]] = (len(ph) - 1) / max(n - 1, 1)
    return out


def chance_recurrence(comp: pd.DataFrame, man: pd.DataFrame, reps: int, shift: int, rng) -> pd.DataFrame:
    size, r, thr = CFG["segmenter"]["prob_map_size"], CP["radius"], CP["threshold"]
    rows = []
    for role, d in comp.groupby("role"):
        d = d.reset_index(drop=True)
        d.attrs["n_photos"] = int((man.role == role).sum())
        cen = d[["x", "y"]].to_numpy(np.float64)
        obs = d.recurrence.to_numpy()                       # 05d 的计数图结果；此处重算只作核对（中心存为 0.1 格精度，取整偶有差 1 格）
        agree = float(np.mean(np.abs(recurrence_kdtree(d, cen, r) - obs) < 1e-3))
        codes = d.image_key.astype("category").cat.codes.to_numpy()
        mx, share = [], []
        for _ in range(reps):
            off = rng.integers(-shift, shift + 1, size=(codes.max() + 1, 2))
            sh = np.clip(cen + off[codes], 0, size - 1)
            v = recurrence_kdtree(d, sh, r)
            mx.append(v.max()); share.append((v >= thr).mean())
        rows.append({"role": role, "observed_max": float(obs.max()), "observed_share": float((obs >= thr).mean()),
                     "chance_max": float(np.mean(mx)), "chance_max_hi": float(np.max(mx)),
                     "chance_share": float(np.mean(share)), "components_per_photo": len(d) / d.attrs["n_photos"],
                     "reps": reps, "shift_cells": shift, "agreement_with_05d": agree})
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def spots_by_camera(comp, man, B, rng) -> pd.DataFrame:
    m = man[man.role == "messidor2"][["image_key", "grade", "orig_w", "orig_h"]].copy()
    f = comp[comp.role == "messidor2"].groupby("image_key").flagged.sum()
    m["flagged"] = m.image_key.map(f).fillna(0).to_numpy()
    m["grade_group"] = np.select([m.grade == 0, m.grade == 1], ["No DR", "Mild DR"], "Referable DR")
    rows = []
    for (w, h, g), d in m.groupby(["orig_w", "orig_h", "grade_group"]):
        v = d.flagged.to_numpy()
        bs = v[rng.integers(len(v), size=(B, len(v)))].mean(1)
        rows.append({"camera_setting": f"{w} × {h}", "orig_w": w, "grade_group": g, "n": len(v), "mean_flagged": v.mean(),
                     "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5), "share_any": (v > 0).mean()})
    return pd.DataFrame(rows)


def ensemble_grade(src: Path, tag: str, keys: pd.Series) -> np.ndarray:
    lst = []
    for sd in CFG["seeds"]:
        d = pd.read_csv(src / f"{tag.format(sd)}.tsv.gz", sep="\t").set_index("image_key").loc[keys]
        lst.append(np.exp(d[P].to_numpy(np.float64)))
    return np.mean(lst, 0).argmax(1)


def example(comp, man):
    m = man[(man.role == "messidor2") & (man.grade == 0)].reset_index(drop=True)
    before = ensemble_grade(ROOT / "results/07g_referable_scores", "C_k4_s{}_aux", m.image_key)
    after = ensemble_grade(ROOT / "results/07h_predict_camera", "C_k4_s{}_aux", m.image_key)
    m = m[(before >= 1) & (after == 0)].reset_index(drop=True)
    crops = load_crops("C", 4, m)
    fl = comp[(comp.role == "messidor2") & (comp.flagged == 1)]
    sc = S // CFG["segmenter"]["prob_map_size"]
    best = (-1, None)
    for i, key in enumerate(m.image_key):
        f = fl[fl.image_key == key]
        cen, ma = f[["x", "y"]].to_numpy() * sc, (f.lesion == "MA").to_numpy()
        for x0, y0 in crops[i]:
            ins = (cen[:, 0] >= x0) & (cen[:, 0] < x0 + C) & (cen[:, 1] >= y0) & (cen[:, 1] < y0 + C)
            n = int((ins & ma).sum())
            if n > best[0] and not (ins & ~ma).any():
                best = (n, (key, int(x0), int(y0), cen))
    n, (key, x0, y0, cen) = best
    img = cv2.imread(str(ROOT / CFG["paths"]["processed"] / "canon" / f"{key}.jpg"), cv2.IMREAD_COLOR)
    mask = h7.artefact_mask(key, fl[fl.image_key == key].component.to_numpy())
    fixed = cv2.inpaint(img, mask, CP["inpaint_radius"], cv2.INPAINT_TELEA)
    cv2.imwrite(str(OUT / "example_before.png"), img[y0:y0 + C, x0:x0 + C])
    cv2.imwrite(str(OUT / "example_after.png"), fixed[y0:y0 + C, x0:x0 + C])
    inside = cen[(cen[:, 0] >= x0) & (cen[:, 0] < x0 + C) & (cen[:, 1] >= y0) & (cen[:, 1] < y0 + C)]
    pd.DataFrame({"image_key": key, "crop_x0": x0, "crop_y0": y0, "x_in_crop": inside[:, 0] - x0,
                  "y_in_crop": inside[:, 1] - y0}).to_csv(OUT / "example_spots.tsv", sep="\t", index=False)
    print("example", key, "spots in crop", n, "candidates", len(m), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False, low_memory=False)
    man = man[man.included.astype(int) == 1].copy(); man["grade"] = man.grade.astype(int)
    comp = pd.read_csv(ROOT / "results/05d_camera_prior/components.tsv.gz", sep="\t")
    summ = pd.read_csv(ROOT / "results/05d_camera_prior/summary.tsv", sep="\t")
    q = quantiles(comp, summ); q.to_csv(OUT / "recurrence_quantiles.tsv", sep="\t", index=False)
    if not (OUT / "chance_recurrence.tsv").exists():
        ch = chance_recurrence(comp, man, CP["chance_reps"], CP["chance_shift"], np.random.default_rng(CFG["seed"]))
        ch.to_csv(OUT / "chance_recurrence.tsv", sep="\t", index=False)
    sb = spots_by_camera(comp, man, CFG["stats"]["n_boot"], np.random.default_rng(CFG["seed"]))
    sb.to_csv(OUT / "spots_by_camera.tsv", sep="\t", index=False)
    print(q.round(4).to_string()); print(sb.round(3).to_string())
    example(comp, man)
    write_provenance("08o_camera_spots", [ROOT / "results/05d_camera_prior/components.tsv.gz",
                                          ROOT / "results/05d_camera_prior/summary.tsv"],
                     [OUT / "recurrence_quantiles.tsv", OUT / "chance_recurrence.tsv", OUT / "spots_by_camera.tsv", OUT / "example_before.png",
                      OUT / "example_after.png", OUT / "example_spots.tsv"], seed=CFG["seed"])


if __name__ == "__main__":
    main()
