"""完整推断成本：预处理、分割器、候选与选择（各方法）、分级网络前向；逐图耗时与峰值显存。
在 n_images 张 DeepDRiD 图像上测量（单 GPU、批大小 1，先预热）。输出 results/09_cost/cost.tsv。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.clsnet import Grader, to_u8  # noqa: E402
from scripts.lib.fundus import crop_square, foreground, fundus_box, to_canon  # noqa: E402
from scripts.lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from scripts.lib.calib import load_calibration  # noqa: E402
from scripts.lib.segnet import build_segmenter, decode_logit, predict_pooled  # noqa: E402
from scripts.lib.selection import select_all  # noqa: E402

N_IMAGES, WARMUP = 200, 10


def sync():
    torch.cuda.synchronize()


def main():
    cfg = load_config()
    torch.set_num_threads(8)
    set_global_seed(cfg["seed"])
    dev = torch.device("cuda")
    pc = cfg["preprocess"]
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    rows = man[(man.role == "deepdrid") & (man.included.astype(int) == 1)].head(N_IMAGES + WARMUP)
    seg = build_segmenter(cfg["segmenter"]["encoder"], 4, pretrained=False).to(dev).eval()
    seg.load_state_dict(torch.load(ROOT / cfg["paths"]["checkpoints"] / "segmenter.pt", map_location=dev))
    nets = {"global512": Grader(cfg["classifier"]["backbone"], 5, cfg["classifier"]["attn_dim"], False),
            "global1024": Grader(cfg["classifier"]["backbone"], 5, cfg["classifier"]["attn_dim"], False),
            "global512+4crops": Grader(cfg["classifier"]["backbone"], 5, cfg["classifier"]["attn_dim"], True)}
    nets = {k: v.to(dev).eval().to(memory_format=torch.channels_last) for k, v in nets.items()}
    cal = load_calibration()
    cls = cfg["segmenter"]["classes"]
    ca = np.array([cal[c]["a"] for c in cls], np.float32)[:, None, None]
    cb = np.array([cal[c]["b"] for c in cls], np.float32)[:, None, None]
    tau = np.array([cal[c]["tau"] for c in cls], np.float32)
    import json
    par = json.loads((ROOT / "results/05_selection/chosen_parameters.json").read_text())
    rec = []
    rng = np.random.default_rng(0)
    for n, r in enumerate(rows.itertuples()):
        t = {}
        t0 = time.perf_counter()
        img = cv2.imread(str(ROOT / r.path))[:, :, ::-1]
        fg = foreground(img, pc["fundus_threshold"])
        box = fundus_box(fg)
        canon = to_canon(img, box, pc["canon_size"])
        fov = cv2.resize(crop_square(fg, *box), (pc["canon_size"],) * 2, interpolation=cv2.INTER_NEAREST)
        canon[fov == 0] = 0
        t["preprocess"] = time.perf_counter() - t0
        sync(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
        size = cfg["segmenter"]["prob_map_size"]
        lg = decode_logit(predict_pooled(seg, canon, dev, size))
        sync(); t["segmenter"] = time.perf_counter() - t0
        t["segmenter_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        fv = cv2.resize(fov, (size, size), interpolation=cv2.INTER_NEAREST)
        prob = (1 / (1 + np.exp(-(ca * lg + cb)))) * fv[None]
        for m in ["T", "N", "D", "C"]:
            t0 = time.perf_counter()
            sel = select_all(prob, fv, cfg, 4, rng, par["nms_iou"], par["dpp_theta"], tau, methods=(m,))
            t[f"select_{m}"] = time.perf_counter() - t0
        crops = sel["C"] * (pc["canon_size"] // size)
        c = cfg["selection"]["crop"]
        for name, net in nets.items():
            gs = 1024 if name == "global1024" else 512
            g = to_u8(cv2.resize(canon, (gs, gs), interpolation=cv2.INTER_AREA))[None].to(dev)
            cr = torch.stack([to_u8(np.ascontiguousarray(canon[y:y + c, x:x + c])) for x, y in crops])[None].to(dev) \
                if net.use_crops else None
            sync(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                net(g, cr)
            sync(); t[f"grader_{name}"] = time.perf_counter() - t0
            t[f"grader_{name}_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        if n >= WARMUP:
            rec.append({"image_key": r.image_key, **t})
    df = pd.DataFrame(rec)
    out = ROOT / "results/09_cost"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cost_per_image.tsv", sep="\t", index=False)
    summ = df.drop(columns="image_key").agg(["median", "mean"]).T.reset_index().rename(columns={"index": "component"})
    summ.to_csv(out / "cost.tsv", sep="\t", index=False)
    print(summ.to_string())
    write_provenance("09_cost", [ROOT / cfg["paths"]["checkpoints"] / "segmenter.pt"], [out / "cost.tsv"],
                     seed=cfg["seed"], extra={"gpu": torch.cuda.get_device_name(0), "n_images": len(df)})


if __name__ == "__main__":
    main()
