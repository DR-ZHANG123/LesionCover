"""把 figures/source_data/icons/<name>.svg（vtracer 输出的填色路径）以矢量 PathPatch 画进 matplotlib 坐标系。
去掉铺满画布的白色底块；保持图标宽高比，放在给定的正方形区域内（数据坐标，需 aspect=equal）。"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
from svgpath2mpl import parse_path

ICON_DIR = Path(__file__).resolve().parents[2] / "figures" / "source_data" / "icons"
NS = "{http://www.w3.org/2000/svg}"
_CACHE: dict[str, tuple] = {}


def _load(name: str):
    if name not in _CACHE:
        root = ET.parse(ICON_DIR / f"{name}.svg").getroot()
        W, H = float(root.get("width")), float(root.get("height"))
        items = []
        for el in root.iter(f"{NS}path"):
            try:
                p = parse_path(el.get("d"))
            except ValueError:                                 # 空路径
                continue
            if len(p.vertices) < 3:
                continue
            m = re.search(r"translate\(([-\d.]+)[ ,]+([-\d.]+)\)", el.get("transform", ""))
            v = p.vertices + (np.array([float(m.group(1)), float(m.group(2))]) if m else 0.0)
            fill = el.get("fill", "#000000")
            rgb = [int(fill[i:i + 2], 16) for i in (1, 3, 5)]
            touches = (v[:, 0].min() <= 1 or v[:, 1].min() <= 1 or v[:, 0].max() >= W - 1 or v[:, 1].max() >= H - 1)
            if min(rgb) > 225 and touches:                     # 与画布边缘相连的白色区域 = 背景
                continue
            items.append((v, p.codes, fill))
        allv = np.concatenate([v for v, _, _ in items])
        _CACHE[name] = (items, allv.min(0), allv.max(0))
    return _CACHE[name]


def draw_icon(ax, name: str, x: float, y: float, size: float, zorder: int = 3) -> None:
    """图标的可见部分等比缩放进左下角 (x, y)、边长 size 的正方形，居中。"""
    items, lo, hi = _load(name)
    span = hi - lo
    k = size / span.max()
    ox = x + (size - span[0] * k) / 2
    oy = y + (size - span[1] * k) / 2
    for v, codes, fill in items:
        xy = np.column_stack([ox + (v[:, 0] - lo[0]) * k, oy + (hi[1] - v[:, 1]) * k])
        ax.add_patch(PathPatch(MPath(xy, codes), facecolor=fill, edgecolor="none", lw=0, zorder=zorder))
