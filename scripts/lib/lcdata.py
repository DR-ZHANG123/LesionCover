"""LesionCover 网络的数据：全局图 + 病灶 token。几何增强（翻转、90° 旋转）同时作用于图像与 token 坐标、象限计数。
工作进程每轮重建（persistent_workers=False），因此增强与 token 丢弃按（种子, 轮次, 照片）每轮重新抽取。"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .clsnet import to_u8

cv2.setNumThreads(1)


def _quad_perm(S: float, ops) -> np.ndarray:
    """象限 q=(x≥S/2)+2(y≥S/2) 在几何变换后的去向。"""
    perm = np.zeros(4, int)
    for q in range(4):
        x, y = (0.75 if q % 2 else 0.25) * S, (0.75 if q // 2 else 0.25) * S
        x, y = _apply(np.array([x]), np.array([y]), S, ops)
        perm[q] = int(x[0] >= S / 2) + 2 * int(y[0] >= S / 2)
    return perm


def _apply(x, y, S, ops):
    hf, vf, k = ops
    if hf:
        x = S - x
    if vf:
        y = S - y
    for _ in range(k):                         # np.rot90 一次：(x, y) → (y, S − x)
        x, y = y, S - x
    return x, y


class LesionTokenSet(torch.utils.data.Dataset):
    def __init__(self, keys, labels, cfg, train: bool, seed: int, token_dir, drop_dir=None):
        """drop_dir：可选，读取剔除相机斑点后的 token（同名 npz）。"""
        self.keys, self.labels = list(keys), np.asarray(labels)
        lc = cfg["lcnet"]
        self.gs, self.M, self.P = lc["global_size"], lc["max_tokens"], cfg["segmenter"]["prob_map_size"]
        self.n_les = len(cfg["segmenter"]["classes"])
        self.gdir = cfg["_proc"] / f"global{self.gs}"
        self.tdir = drop_dir or token_dir
        self.tdrop = lc["token_drop"] if train else 0.0
        self.train, self.seed, self.epoch = train, seed, 0

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, self.epoch, i)) if self.train else None
        g = cv2.imread(str(self.gdir / f"{self.keys[i]}.jpg"), cv2.IMREAD_COLOR)[:, :, ::-1]
        z = np.load(self.tdir / f"{self.keys[i]}.npz")
        tok, cnt = z["tokens"].copy(), z["counts"].copy()
        if self.train:
            g = cv2.convertScaleAbs(np.ascontiguousarray(g), alpha=rng.uniform(0.85, 1.15), beta=rng.uniform(-12, 12))
            ops = (rng.random() < 0.5, rng.random() < 0.5, int(rng.integers(4)))
            if ops[0]:
                g = g[:, ::-1]
            if ops[1]:
                g = g[::-1]
            g = np.rot90(g, ops[2])
            if len(tok):
                tok[:, 3], tok[:, 4] = _apply(tok[:, 3], tok[:, 4], self.P, ops)
                if ops[2] % 2:
                    tok[:, [5, 6]] = tok[:, [6, 5]]
                if self.tdrop > 0:
                    tok = tok[rng.random(len(tok)) >= self.tdrop]
            perm = _quad_perm(self.P, ops)
            c2 = cnt.reshape(self.n_les, 4); new = np.zeros_like(c2)
            new[:, perm] = c2
            cnt = new.ravel()
        if self.M == 0:                              # 消融（只留全局 token）：计数也置零
            cnt = np.zeros_like(cnt)
        out = np.zeros((self.M, tok.shape[1] if tok.ndim == 2 else 7), np.float32)
        valid = np.zeros(self.M, bool)
        n = min(len(tok), self.M)
        if n:
            out[:n] = tok[:n]; valid[:n] = True
        out[~valid, 5:7] = 1.0                       # 填充位：避免 log(0)
        return to_u8(np.ascontiguousarray(g)), torch.from_numpy(out), torch.from_numpy(valid), \
            torch.from_numpy(cnt.astype(np.float32)), int(self.labels[i]), i


def make_lc_loader(keys, labels, cfg, train, seed, bs, token_dir, drop_dir=None):
    ds = LesionTokenSet(keys, labels, cfg, train, seed, token_dir, drop_dir)
    gen = torch.Generator().manual_seed(seed)
    return ds, torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=train, num_workers=cfg["classifier"]["num_workers"],
                                           pin_memory=True, drop_last=train, persistent_workers=False, generator=gen)
