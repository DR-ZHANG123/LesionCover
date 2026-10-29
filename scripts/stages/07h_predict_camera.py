"""相机复现先验下的推理：已训练好的分级器（不再训练），输入换成去掉相机复现分量的照片——复现分量（05d）的外扩掩膜
在 canon 图上修补（Telea，同 08k），全局图由修补后的 canon 图按预处理相同方式缩放（INTER_AREA，JPEG 质量 95），
裁块取 05d 重新选择的坐标（无复现分量的照片与 05_selection 相同）。没有复现分量的照片输入与原来逐像素相同。
每个运行写 results/07h_predict_camera/<tag>.tsv.gz：image_key, role, lp0..lp4（全精度对数概率），c0..c4（裁块辅助头）。
用法：CUDA_VISIBLE_DEVICES=<gpu> python scripts/stages/07h_predict_camera.py --tags G512_s11 C_k4_s11_aux ...
敏感性分析：--prior 05d_camera_prior_t10 --out 07h_predict_camera_t10 --roles messidor2 idrid。"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.calib import calibrated_map, load_calibration  # noqa: E402
from scripts.lib.clsnet import Grader, backbone_for, to_u8  # noqa: E402
from scripts.lib.gradedata import load_crops  # noqa: E402
from scripts.lib.variants import global_size  # noqa: E402
from scripts.lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from scripts.lib.selection import extract_components  # noqa: E402

cv2.setNumThreads(1)
CFG = load_config()
CAL = load_calibration()
CP = CFG["camera_prior"]
PROC = ROOT / CFG["paths"]["processed"]
S, C = CFG["preprocess"]["canon_size"], CFG["selection"]["crop"]
KER = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CP["dilate_px"] + 1,) * 2)
JPEG = [cv2.IMWRITE_JPEG_QUALITY, CFG["preprocess"]["jpeg_quality"]]


def jpeg_roundtrip(img: np.ndarray) -> np.ndarray:
    """与预处理保存的文件一致：RGB → JPEG（质量同预处理）→ RGB。"""
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img[:, :, ::-1]), JPEG)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]


def artefact_mask(key: str, idx: np.ndarray) -> np.ndarray:
    """05d 标为复现的分量（extract_components 顺序下的序号 idx）→ canon 坐标外扩掩膜。"""
    prob, _, tau = calibrated_map(key, CFG, CAL)
    comps = extract_components(prob, tau, CFG["selection"]["comp_min_pixels"])
    sc = S // prob.shape[-1]
    m = np.zeros((S, S), np.uint8)
    for j in idx:
        for dy in range(sc):
            for dx in range(sc):
                m[comps.py[j] * sc + dy, comps.px[j] * sc + dx] = 1
    return cv2.dilate(m, KER)


class CameraSet(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, crops, flagged: dict, gsize: int):
        self.df, self.crops, self.flagged, self.gsize = df.reset_index(drop=True), crops, flagged, gsize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        key = self.df.image_key[i]
        g = cv2.imread(str(PROC / f"global{self.gsize}" / f"{key}.jpg"), cv2.IMREAD_COLOR)[:, :, ::-1]
        img = None
        if key in self.flagged:
            img = cv2.imread(str(PROC / "canon" / f"{key}.jpg"), cv2.IMREAD_COLOR)[:, :, ::-1]
            m = artefact_mask(key, self.flagged[key])
            img = cv2.inpaint(np.ascontiguousarray(img), m, CP["inpaint_radius"], cv2.INPAINT_TELEA)
            g = jpeg_roundtrip(cv2.resize(img, (self.gsize, self.gsize), interpolation=cv2.INTER_AREA))
            img = jpeg_roundtrip(img)
        if self.crops is None:
            return to_u8(g), torch.zeros(0, dtype=torch.uint8), i
        if img is None:
            img = cv2.imread(str(PROC / "canon" / f"{key}.jpg"), cv2.IMREAD_COLOR)[:, :, ::-1]
        tiles = []
        for x0, y0 in self.crops[i]:
            x0, y0 = int(np.clip(x0, 0, S - C)), int(np.clip(y0, 0, S - C))
            tiles.append(to_u8(img[y0:y0 + C, x0:x0 + C]))
        return to_u8(g), torch.stack(tiles), i


@torch.no_grad()
def run(model, loader, dev, aux: bool):
    n = len(loader.dataset)
    L = np.zeros((n, 5), np.float64); Q = np.full((n, 5), np.nan, np.float32)
    for g, c, idx in loader:
        g = g.to(dev, non_blocking=True)
        c = c.to(dev, non_blocking=True) if c.numel() else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if aux:
                logit, _, a = model(g, c, return_aux=True)
                Q[idx.numpy()] = torch.softmax(a.float(), 1).cpu().numpy()
            else:
                logit, _ = model(g, c)
        L[idx.numpy()] = torch.log_softmax(logit.double(), 1).cpu().numpy()
    return L, Q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--prior", default="05d_camera_prior"); ap.add_argument("--out", default="07h_predict_camera")
    ap.add_argument("--roles", nargs="+", default=CP["roles"])
    args = ap.parse_args()
    OUT = ROOT / "results" / args.out
    set_global_seed(CFG["seed"]); torch.set_num_threads(8)
    dev = torch.device("cuda")
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False, low_memory=False)
    man = man[(man.included.astype(int) == 1) & man.role.isin(args.roles)].copy()
    comp = pd.read_csv(ROOT / "results" / args.prior / "components.tsv.gz", sep="\t")
    fl = comp[comp.flagged == 1]
    flagged = {k: g.component.to_numpy() for k, g in fl.groupby("image_key")}
    crops_path = ROOT / "results" / args.prior / "crops.tsv.gz"
    OUT.mkdir(parents=True, exist_ok=True)
    for tag in args.tags:
        out = OUT / f"{tag}.tsv.gz"
        if out.exists():
            continue
        m = re.match(r"(?P<method>[A-Za-z0-9]+?)(?:_k(?P<k>\d+))?_s(?P<seed>\d+)(?:_(?P<var>[a-z0-9_]+))?$", tag)
        method, var = m["method"], (m["var"] or "")
        gsize = global_size(method, var, CFG)
        aux = "aux" in var
        model = Grader(backbone_for(tag, CFG, ROOT), 5, CFG["classifier"]["attn_dim"], not method.startswith("G"),
                       crop_head=aux).to(dev)
        model.load_state_dict(torch.load(ROOT / CFG["paths"]["checkpoints"] / "classify" / f"{tag}.pt", map_location=dev))
        model.eval()
        parts = []
        ref = pd.read_csv(ROOT / "results/07g_referable_scores" / f"{tag}.tsv.gz", sep="\t")
        for role in args.roles:
            sub = man[man.role == role].reset_index(drop=True)
            if not sub.image_key.isin(flagged).any():      # 无复现分量的队列：输入与原来相同，沿用 07g 的对数概率
                d = sub[["image_key", "role"]].merge(ref, on=["image_key", "role"], how="left")
                assert d.lp0.notna().all()
                for i in range(5):
                    d[f"c{i}"] = np.nan
                d["camera_flagged"] = 0
                parts.append(d)
                continue
            crops = load_crops(method, 4, sub, crops_path)
            dl = torch.utils.data.DataLoader(CameraSet(sub, crops, flagged, gsize), batch_size=32, num_workers=16,
                                             pin_memory=True)
            L, Q = run(model, dl, dev, aux)
            d = pd.DataFrame(np.c_[L, Q], columns=[f"lp{i}" for i in range(5)] + [f"c{i}" for i in range(5)])
            d.insert(0, "image_key", sub.image_key); d.insert(1, "role", role)
            d["camera_flagged"] = sub.image_key.isin(flagged).astype(int).to_numpy()
            parts.append(d)
            chk = d[d.camera_flagged == 0].merge(ref, on=["image_key", "role"], suffixes=("", "_ref"))
            a, b = chk[[f"lp{i}" for i in range(5)]].to_numpy(), chk[[f"lp{i}_ref" for i in range(5)]].to_numpy()
            print(tag, role, len(d), "flagged", int(d.camera_flagged.sum()), "unflagged grade agreement with original",
                  round(float((a.argmax(1) == b.argmax(1)).mean()), 4), "max |dP|",
                  round(float(np.abs(np.exp(a) - np.exp(b)).max()), 4), flush=True)
        pd.concat(parts).to_csv(out, sep="\t", index=False, float_format="%.8g")
        write_provenance(f"{args.out}/{tag}", [crops_path, ROOT / "results" / args.prior / "components.tsv.gz",
                                                         ROOT / "results/07g_referable_scores" / f"{tag}.tsv.gz"], [out],
                         seed=int(m["seed"]))
    if all((OUT / f"{t}.tsv.gz").exists() for t in args.tags):
        write_provenance(args.out, [crops_path], sorted(OUT.glob("*.tsv.gz")))


if __name__ == "__main__":
    main()
