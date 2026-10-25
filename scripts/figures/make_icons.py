"""Fig. 1 中的单个示意图标：用图像生成模型生成（接口地址与密钥只从环境变量 IMAGE_API_BASE / IMAGE_API_KEY 读取），
再用 vtracer 转成矢量路径（SVG），供 Fig1_framework.py 以矢量形式绘入。生成结果不可重复，故原始 PNG、SVG 与提示词
都保存在 figures/source_data/icons/（icons.tsv），作图只读这些文件；已存在的图标不会重新生成。
用法：IMAGE_API_BASE=... IMAGE_API_KEY=... python scripts/figures/make_icons.py [--force name ...]"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import vtracer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "source_data" / "icons"
MODEL = "gpt-image-2"
STYLE = ("Flat scientific illustration icon in the style of BioRender, clean simple geometric shapes, thin dark outline, "
         "muted palette of navy (#2B4C7E), teal (#2A9D8F), warm red (#B2182B), sand (#E9C46A) and light grey, "
         "no text, no letters, no numbers, no shadow, no gradient, single object centred, isolated on a transparent background.")
ICONS = {
    "fundus_camera": "A retinal fundus camera, the ophthalmic imaging device with a lens, chin rest and base, seen from the side.",
    "segmenter": "A U-shaped encoder-decoder neural network drawn as stacked rectangular blocks descending and ascending, with skip connections.",
    "encoder": "A deep convolutional neural network drawn as a row of 3D blocks that become smaller from left to right.",
    "magnifier": "A magnifying glass over a small patch of retina with a few tiny red dots.",
    "grade_report": "A medical clipboard report with a five-step colour severity scale from green to red.",
    "hospital": "A small hospital building with a medical cross.",
    "globe": "A globe with four location pins in different continents.",
    "eraser": "An eraser removing a few tiny red dots from a light orange surface.",
    "stethoscope": "A stethoscope.",
}


def generate(name: str, base: str, key: str) -> dict:
    body = json.dumps({"model": MODEL, "prompt": f"{ICONS[name]} {STYLE}", "size": "1024x1024", "n": 1,
                       "background": "transparent", "output_format": "png"}).encode()
    req = urllib.request.Request(f"{base.rstrip('/')}/images/generations", data=body, method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    png = OUT / f"{name}.png"
    png.write_bytes(base64.b64decode(d["data"][0]["b64_json"]))
    vectorize(name)
    return {"icon": name, "model": MODEL, "prompt": f"{ICONS[name]} {STYLE}", "created_utc": dt.datetime.utcnow().isoformat(timespec="seconds")}


def vectorize(name: str) -> None:
    """透明背景先铺白再矢量化（vtracer 不处理透明通道）；作图时去掉铺满画布的白色底块。"""
    from PIL import Image
    im = Image.open(OUT / f"{name}.png").convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im)
    flat = OUT / f"{name}_white.png"
    bg.convert("RGB").save(flat)
    vtracer.convert_image_to_svg_py(str(flat), str(OUT / f"{name}.svg"), colormode="color",
                                    hierarchical="cutout", mode="spline", filter_speckle=6, color_precision=8,
                                    layer_difference=24, corner_threshold=60, length_threshold=4.0, splice_threshold=45,
                                    path_precision=3)
    flat.unlink()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", nargs="*", default=[]); args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base, key = os.environ.get("IMAGE_API_BASE"), os.environ.get("IMAGE_API_KEY")
    todo = [n for n in ICONS if n in args.force or not (OUT / f"{n}.png").exists()]
    if todo and not (base and key):
        sys.exit("IMAGE_API_BASE / IMAGE_API_KEY not set")
    log = OUT / "icons.tsv"
    old = pd.read_csv(log, sep="\t") if log.exists() else pd.DataFrame(columns=["icon", "model", "prompt", "created_utc"])
    with ThreadPoolExecutor(5) as ex:
        rows = list(ex.map(lambda n: generate(n, base, key), todo))
    for n in ICONS:                      # 已有 PNG 的图标只重新矢量化
        if n not in todo:
            vectorize(n)
    new = pd.concat([old[~old.icon.isin(todo)], pd.DataFrame(rows)]).sort_values("icon")
    new.to_csv(log, sep="\t", index=False)
    print("generated", todo)


if __name__ == "__main__":
    main()
