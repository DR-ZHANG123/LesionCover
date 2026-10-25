"""覆盖的上限来自检测还是选择：
(1) 以参考掩膜代替预测概率（理想检测）重新运行各选择规则，在同一评价标准下计算参考病灶覆盖；
(2) 参考病灶被预测分量触及的比例（病灶级检出），按病灶类型与面积。
掩膜图像为 DDR 开发区与测试区；理想检测只用作上界分析，不参与任何模型或参数选择。
输出 results/06b_detection_limit/{oracle_coverage.tsv, detection.tsv}。"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.refcov import CLASSES, crops_to_orig, ref_components_for, retained_fraction  # noqa: E402
from scripts.lib.repro import ROOT, load_config, write_provenance  # noqa: E402
from scripts.lib.calib import calibrated_map, load_calibration  # noqa: E402
from scripts.lib.selection import select_all  # noqa: E402

CFG = load_config()
RULES = ("T", "N", "D", "C")
CAL = load_calibration()


def oracle_prob(key: str, size: int) -> np.ndarray:
    m = np.load(ROOT / CFG["paths"]["processed"] / "canon_masks" / f"{key}.npy").astype(np.float32)
    return F.max_pool2d(torch.from_numpy(m)[None], m.shape[-1] // size)[0].numpy()


def one(args):
    row, nms, theta = args
    prob, fov, tau = calibrated_map(row["image_key"], CFG, CAL)
    comps = ref_components_for(row, CFG)
    cls = np.array([c[0] for c in comps], int)
    box = (row["box_x0"], row["box_y0"], row["box_side"])
    scale = CFG["preprocess"]["canon_size"] / prob.shape[-1]
    area = np.array([len(c[1]) for c in comps]) * (CFG["preprocess"]["canon_size"] / row["box_side"]) ** 2
    out_cov, out_det = [], []
    for src, p in (("predicted", prob), ("reference", oracle_prob(row["seg_key"], prob.shape[-1]) * fov[None])):
        r = select_all(p, fov, CFG, 4, np.random.default_rng(0), nms, theta, tau if src == "predicted" else 0.5,
                       methods=RULES)
        for m in RULES:
            rects = crops_to_orig(r[m] * scale, box, CFG["preprocess"]["canon_size"], CFG["selection"]["crop"])
            cov = retained_fraction(comps, rects).max(1) >= CFG["coverage_eval"]["retain_frac"] if comps else np.zeros(0, bool)
            out_cov += [(row["image_key"], row["mask_role"], src, m, CLASSES[c], float(a), int(v)) for c, a, v in zip(cls, area, cov)]
    # 病灶级检出：参考分量在 prob_map 坐标下的像素是否与该类预测分量（> 阈值）相交
    s_orig = prob.shape[-1] / row["box_side"]
    for (ci, xx, yy), a in zip(comps, area):
        px = np.clip(((xx - row["box_x0"]) * s_orig).astype(int), 0, prob.shape[-1] - 1)
        py = np.clip(((yy - row["box_y0"]) * s_orig).astype(int), 0, prob.shape[-1] - 1)
        hit = bool((prob[ci, py, px] > tau[ci]).any())
        out_det.append((row["image_key"], row["mask_role"], CLASSES[ci], float(a), int(hit)))
    return out_cov, out_det


def main():
    import json
    par = json.loads((ROOT / "results/05_selection/chosen_parameters.json").read_text())
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    man = man[(man.included.astype(int) == 1) & man.mask_role.isin(["seg_dev", "seg_test", "ext_seg"])]
    for c in ["box_x0", "box_y0", "box_side", "orig_h", "orig_w"]:
        man[c] = man[c].astype(int)
    with Pool(48) as pool:
        res = pool.map(one, [(r, par["nms_iou"], par["dpp_theta"]) for r in man.to_dict("records")], chunksize=1)
    cov = pd.DataFrame([x for a, _ in res for x in a], columns=["image_key", "mask_role", "input", "rule", "lesion", "area_canon", "covered"])
    det = pd.DataFrame([x for _, b in res for x in b], columns=["image_key", "mask_role", "lesion", "area_canon", "detected"])
    med = det.groupby(["mask_role", "lesion"]).area_canon.median().rename("med")
    det = det.join(med, on=["mask_role", "lesion"]); det["small"] = (det.area_canon <= det.med).astype(int)
    out = ROOT / "results/06b_detection_limit"; out.mkdir(parents=True, exist_ok=True)
    summ = cov.groupby(["mask_role", "input", "rule", "lesion"]).covered.mean().unstack("lesion")
    summ["macro"] = summ[CLASSES].mean(1)
    summ.reset_index().to_csv(out / "oracle_coverage.tsv", sep="\t", index=False)
    d = det.groupby(["mask_role", "lesion", "small"]).detected.mean().reset_index()
    d.to_csv(out / "detection.tsv", sep="\t", index=False)
    print(summ.round(3).to_string()); print(d.round(3).to_string())
    write_provenance("06b_detection_limit", [ROOT / "results/05_selection/chosen_parameters.json"],
                     [out / "oracle_coverage.tsv", out / "detection.tsv"])


if __name__ == "__main__":
    main()
