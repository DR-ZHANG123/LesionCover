"""冻结分割器在开发区与测试区掩膜图像上的逐类像素 AUPR（视野内、最大池化到 prob_map_size）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from scripts.lib.segnet import build_segmenter  # noqa: E402
from scripts.lib.segeval import evaluate, load_set  # noqa: E402


def main():
    cfg = load_config()
    torch.set_num_threads(8)
    set_global_seed(cfg["seed"])
    dev = torch.device("cuda")
    model = build_segmenter(cfg["segmenter"]["encoder"], 4, pretrained=False).to(dev)
    model.load_state_dict(torch.load(ROOT / cfg["paths"]["checkpoints"] / "segmenter.pt", map_location=dev))
    man = pd.read_csv(ROOT / "results/02_manifest/images.tsv", sep="\t", keep_default_na=False)
    rows = []
    for role in ["seg_dev", "seg_test"]:
        sub = man[(man.mask_role == role) & (man.included.astype(int) == 1)]
        r = evaluate(model, *load_set(sub, cfg), dev, cfg["segmenter"]["prob_map_size"])
        rows.append({"set": role, "n_images": len(sub), **r})
    out = ROOT / "results/03b_seg_eval"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "seg_aupr.tsv", sep="\t", index=False)
    print(json.dumps(rows, indent=1))
    write_provenance("03b_seg_eval", [ROOT / cfg["paths"]["checkpoints"] / "segmenter.pt"], [out / "seg_aupr.tsv"],
                     seed=cfg["seed"])


if __name__ == "__main__":
    main()
