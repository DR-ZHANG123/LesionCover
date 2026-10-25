"""第四外部队列 EyePACS（Kaggle Diabetic Retinopathy Detection 2015 训练集，35,126 张照片、每位患者左右眼各一张、
五级 ICDR 分级）：从 HuggingFace 镜像 ctmedtech/EYEPACS 下载 EYEPACS.zip（照片已裁至眼底并缩放为 1,024 像素宽）与
trainLabels.csv，把照片解压到 data/raw/EyePACS/images/，分级写入 labels.csv（image, grade, patient_id, eye）。
zip、标签文件与每张照片的 SHA-256 记录在 results/00_download/eyepacs_files.tsv。
该队列只用于外部评价：全部模型与设置确定后预测一次，不参与任何训练、选参或选轮次。"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config  # noqa: E402

REPO = "ctmedtech/EYEPACS"
REVISION = "main"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    cfg = load_config()
    raw = ROOT / cfg["paths"]["raw_eyepacs"]
    hf = raw.parent / "EyePACS_hf"
    hf.mkdir(parents=True, exist_ok=True)
    for name in ("EYEPACS.zip", "trainLabels.csv"):
        if not (hf / name).exists():
            hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, local_dir=str(hf))
    files = [{"file": f"hf/{n}", "sha256": sha256_file(hf / n)} for n in ("EYEPACS.zip", "trainLabels.csv")]
    (raw / "images").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(hf / "EYEPACS.zip") as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.endswith(".jpeg"):
                continue
            data = z.read(info)
            name = Path(info.filename).name
            dst = raw / "images" / name
            if not dst.exists():
                dst.write_bytes(data)
            files.append({"file": f"images/{name}", "sha256": hashlib.sha256(data).hexdigest()})
    lab = pd.read_csv(hf / "trainLabels.csv").rename(columns={"level": "grade"})
    lab["patient_id"] = lab.image.str.split("_").str[0].astype(int)
    lab["eye"] = lab.image.str.split("_").str[1]
    have = {Path(f["file"]).stem for f in files if f["file"].startswith("images/")}
    lab = lab[lab.image.isin(have)].sort_values("image")
    assert lab.image.is_unique
    lab["image"] = lab.image + ".jpeg"
    lab.to_csv(raw / "labels.csv", index=False)
    out = ROOT / "results/00_download"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(files).to_csv(out / "eyepacs_files.tsv", sep="\t", index=False)
    print(len(lab), lab.patient_id.nunique(), lab.grade.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
