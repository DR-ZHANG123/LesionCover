"""Messidor-2 检查配对表（messidor-2.csv：每次检查左右眼两张照片，874 次检查）。官方随数据集发布于 ADCIS
（需登记）；这里取其公开副本（固定提交号），两个独立副本内容一致（仅换行符不同）。写入
data/raw/Messidor2/messidor2_pairs.csv（exam, image, eye），SHA-256 记录在 results/00_download/messidor2_pairs.tsv。
配对只用于按检查重抽样（同一检查的两眼同进同出），不改变任何照片或分级。"""
from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.lib.repro import ROOT, load_config  # noqa: E402

URLS = ["https://raw.githubusercontent.com/ok1zjf/RFM/b8c9322863b5e2d79df964e524d74ad53d1d1b7a/datasets/messidor2_pairs.csv",
        "https://raw.githubusercontent.com/THOUFIKUR/sih-matlab/10fd6ba4edf9ffd91fc8dcaf8b8fb86e7707f32d/data/messidor2/messidor-2.csv"]


def parse(raw: bytes) -> pd.DataFrame:
    t = pd.read_csv(io.BytesIO(raw), sep=";")
    t.columns = [c.strip() for c in t.columns]
    return t.apply(lambda s: s.str.strip())


def main():
    cfg = load_config()
    got = [urllib.request.urlopen(u, timeout=60).read() for u in URLS]
    tabs = [parse(g) for g in got]
    assert tabs[0].apply(lambda s: s.str.lower()).equals(tabs[1].apply(lambda s: s.str.lower())), "pairing copies differ"
    t = tabs[0]
    rows = [{"exam": i + 1, "image": r[side], "eye": side} for i, r in t.iterrows() for side in ("left", "right")]
    out = pd.DataFrame(rows)
    assert len(out) == 1748 and out.image.str.lower().is_unique
    dst = ROOT / cfg["paths"]["raw_messidor2"] / "messidor2_pairs.csv"
    out.to_csv(dst, index=False)
    rec = ROOT / "results/00_download"; rec.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"url": u, "sha256": hashlib.sha256(g).hexdigest()} for u, g in zip(URLS, got)]).to_csv(
        rec / "messidor2_pairs.tsv", sep="\t", index=False)
    print(len(t), "examinations")


if __name__ == "__main__":
    main()
