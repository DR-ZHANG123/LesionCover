"""病灶概率的逐类校准：p̂_c = σ(a_c·logit + b_c)（Platt 缩放，开发区掩膜拟合），分量阈值 τ_c 取开发区像素 F1 最大处。
所有选择规则读取同一校准后的概率图与阈值。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .repro import ROOT
from .segnet import decode_logit

CAL = ROOT / "results/04b_calibration/calibration.json"


def load_calibration(path: Path = CAL) -> dict:
    return json.loads(Path(path).read_text())


def calibrated_map(key: str, cfg: dict, cal: dict, subdir: str = "probmaps") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (p̂ (4,H,W) float32, 视野掩膜, 逐类阈值 τ)。视野外概率为 0。subdir：概率图目录（probmaps_adabn 为按相机适配的版本）。"""
    z = np.load(ROOT / cfg["paths"]["processed"] / subdir / f"{key}.npz")
    lg, fov = decode_logit(z["logit"]), z["fov"]
    a = np.array([cal[c]["a"] for c in cfg["segmenter"]["classes"]], np.float32)[:, None, None]
    b = np.array([cal[c]["b"] for c in cfg["segmenter"]["classes"]], np.float32)[:, None, None]
    p = 1 / (1 + np.exp(-(a * lg + b)))
    p *= fov[None]
    tau = np.array([cal[c]["tau"] for c in cfg["segmenter"]["classes"]], np.float32)
    return p.astype(np.float32), fov, tau
