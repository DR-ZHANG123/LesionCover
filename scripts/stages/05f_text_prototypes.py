"""分级文字原型：用生物医学视觉—语言模型 BiomedCLIP 的文本编码器（冻结）编码国际临床 DR 分级标准各级的文字定义，
每级若干种表述取平均并 L2 归一化，得到 5 × 512 的类别原型，供 LesionCover 网络的知识引导分类头使用。
只编码一次，不训练文本编码器。输出 results/05f_text_prototypes/{prototypes.npy, prompts.tsv}。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

PROMPTS = {
    0: ["a colour fundus photograph of a healthy retina with no diabetic retinopathy",
        "a retinal photograph without microaneurysms, haemorrhages or exudates",
        "no apparent diabetic retinopathy on fundus photography"],
    1: ["mild non-proliferative diabetic retinopathy with microaneurysms only",
        "a fundus photograph showing a few small red dots, microaneurysms, and no other lesions",
        "early diabetic retinopathy with isolated microaneurysms"],
    2: ["moderate non-proliferative diabetic retinopathy with retinal haemorrhages, hard exudates or cotton wool spots",
        "a fundus photograph with more than just microaneurysms but less than severe non-proliferative retinopathy",
        "dot and blot haemorrhages and hard exudates in diabetic retinopathy"],
    3: ["severe non-proliferative diabetic retinopathy with many intraretinal haemorrhages in all four quadrants",
        "diabetic retinopathy with venous beading in two quadrants or intraretinal microvascular abnormalities",
        "severe diabetic retinopathy without neovascularization"],
    4: ["proliferative diabetic retinopathy with neovascularization",
        "a fundus photograph with new vessels on the disc or elsewhere, or vitreous haemorrhage",
        "proliferative diabetic retinopathy with preretinal haemorrhage"],
}
MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def main():
    import open_clip
    cfg = load_config(); set_global_seed(cfg["seed"])
    model, _ = open_clip.create_model_from_pretrained(MODEL)
    tok = open_clip.get_tokenizer(MODEL)
    model.eval()
    protos, rows = [], []
    with torch.no_grad():
        for g in range(5):
            z = model.encode_text(tok(PROMPTS[g], context_length=256)).float()
            z = torch.nn.functional.normalize(z, dim=-1).mean(0)
            protos.append(torch.nn.functional.normalize(z, dim=0).numpy())
            rows += [{"grade": g, "prompt": p} for p in PROMPTS[g]]
    P = np.stack(protos).astype(np.float32)
    out = ROOT / "results/05f_text_prototypes"; out.mkdir(parents=True, exist_ok=True)
    np.save(out / "prototypes.npy", P)
    pd.DataFrame(rows).to_csv(out / "prompts.tsv", sep="\t", index=False)
    print("cosine similarity between grade prototypes\n", np.round(P @ P.T, 3))
    write_provenance("05f_text_prototypes", [], [out / "prototypes.npy", out / "prompts.tsv"], seed=cfg["seed"],
                     extra={"text_encoder": MODEL})


if __name__ == "__main__":
    main()
