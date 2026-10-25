"""Fig. 1 的示意图标，全部为矢量：
- 相机、放大镜、病历、医院、地球、听诊器取自 Health Icons（healthicons.org，MIT 许可证；固定提交号的 SVG、许可证与
  SHA-256 见 figures/source_data/icons_healthicons/），按单色填充绘入；
- 网络（U 形分割器、编码器）与橡皮擦由本模块用几何图形直接绘制。
draw_icon(ax, name, x, y, size)：在左下角 (x, y)、边长 size 的正方形内绘制（数据坐标，需 aspect=equal）。"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.transforms import Affine2D
from svgpath2mpl import parse_path

ICON_DIR = Path(__file__).resolve().parents[2] / "figures" / "source_data" / "icons_healthicons"
NS = "{http://www.w3.org/2000/svg}"
NAVY, TEAL, RED, SAND, GREY = "#2B4C7E", "#2A9D8F", "#B2182B", "#E9C46A", "#9E9E9E"
HEALTH = {"fundus_camera": ("camera", NAVY), "magnifier": ("magnifying-glass", TEAL),
          "grade_report": ("medical-records", RED), "hospital": ("hospital", NAVY),
          "globe": ("globe", TEAL), "stethoscope": ("stethoscope", NAVY)}


def _evenodd(p: MPath) -> MPath:
    """matplotlib 按非零环绕规则填充；SVG 的 evenodd 路径先细分成多边形，再按嵌套深度交替设定方向，使洞保持为洞。"""
    s = 64.0
    polys = [q / s for q in p.to_polygons(transform=Affine2D().scale(s), closed_only=True) if len(q) >= 3]
    paths = [MPath(q) for q in polys]
    verts, codes = [], []
    for i, q in enumerate(polys):
        depth = sum(paths[j].contains_point(q[0]) for j in range(len(polys)) if j != i)
        area = 0.5 * np.sum(q[:-1, 0] * q[1:, 1] - q[1:, 0] * q[:-1, 1])
        if (area > 0) == (depth % 2 == 1):                       # 偶数层逆时针、奇数层顺时针
            q = q[::-1]
        verts.append(q)
        codes += [MPath.MOVETO] + [MPath.LINETO] * (len(q) - 2) + [MPath.CLOSEPOLY]
    return MPath(np.concatenate(verts), codes)


def _health(ax, fname: str, color: str, x: float, y: float, size: float, zorder: int) -> None:
    root = ET.parse(ICON_DIR / f"{fname}.svg").getroot()
    vb = [float(v) for v in root.get("viewBox").split()]
    k = size / max(vb[2], vb[3])
    for el in root.iter(f"{NS}path"):
        if el.get("fill") == "none" or not el.get("d"):
            continue
        p = parse_path(el.get("d"))
        if el.get("fill-rule") == "evenodd":
            p = _evenodd(p)
        xy = np.column_stack([x + (p.vertices[:, 0] - vb[0]) * k, y + size - (p.vertices[:, 1] - vb[1]) * k])
        ax.add_patch(PathPatch(MPath(xy, p.codes), facecolor=color, edgecolor="none", lw=0, zorder=zorder))


def _box3d(ax, x, y, w, h, d, color, zorder):
    """立方体块：正面矩形 + 顶面与侧面平行四边形（d 为纵深偏移）。"""
    ax.add_patch(Rectangle((x, y), w, h, fc=color, ec="white", lw=0.3, zorder=zorder))
    ax.add_patch(Polygon([(x, y + h), (x + d, y + h + d), (x + w + d, y + h + d), (x + w, y + h)], closed=True,
                         fc=color, ec="white", lw=0.3, alpha=0.75, zorder=zorder))
    ax.add_patch(Polygon([(x + w, y), (x + w + d, y + d), (x + w + d, y + h + d), (x + w, y + h)], closed=True,
                         fc=color, ec="white", lw=0.3, alpha=0.55, zorder=zorder))


def _encoder(ax, x, y, size, zorder):
    """卷积编码器：从左到右逐级变小的方块。"""
    cols = [NAVY, TEAL, RED, SAND, GREY]
    hs = [0.70, 0.56, 0.43, 0.32, 0.22]
    w, d, gap = size * 0.12, size * 0.06, size * 0.05
    cx = x + size * 0.03
    for c, h in zip(cols, hs):
        _box3d(ax, cx, y + size * (0.47 - h / 2), w, size * h, d, c, zorder)
        cx += w + gap


def _unet(ax, x, y, size, zorder):
    """U 形编码–解码网络：左侧逐级下行，右侧逐级上行，同级之间以细线表示跳连。"""
    lv = [(0.72, 0.34), (0.50, 0.27), (0.31, 0.20), (0.15, 0.14)]          # (中心纵向位置, 块高)
    w, d = size * 0.10, size * 0.04
    for i, (yy, h) in enumerate(lv):
        c = [NAVY, TEAL, RED, SAND][i]
        xl = x + size * (0.02 + 0.12 * i)
        _box3d(ax, xl, y + size * (yy - h / 2), w, size * h, d, c, zorder)
        if i < 3:
            xr = x + size * (0.92 - 0.12 * i) - w
            _box3d(ax, xr, y + size * (yy - h / 2), w, size * h, d, c, zorder)
            ax.plot([xl + w + d, xr], [y + size * yy] * 2, color=GREY, lw=0.5, ls=(0, (2, 1)), zorder=zorder - 1)


def _eraser(ax, x, y, size, zorder):
    """浅色视网膜片上的小红点，与一块斜置的橡皮擦。"""
    ax.add_patch(FancyBboxPatch((x + size * 0.04, y + size * 0.06), size * 0.82, size * 0.30,
                                boxstyle=f"round,pad=0,rounding_size={size * 0.1}", fc="#F6D9A8", ec=NAVY, lw=0.5,
                                zorder=zorder))
    for dx, dy in [(0.16, 0.22), (0.25, 0.14), (0.31, 0.27), (0.40, 0.18)]:
        ax.add_patch(Circle((x + size * dx, y + size * dy), size * 0.028, fc=RED, ec="none", zorder=zorder + 1))
    ang = np.deg2rad(40)
    L, Wd = size * 0.56, size * 0.24
    cx, cy = x + size * 0.62, y + size * 0.58
    u, v = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])
    c0 = np.array([cx, cy])
    for a, b, fc in ((-0.5, -0.1, "#E8EEF4"), (-0.1, 0.5, TEAL)):
        pts = [c0 + u * a * L - v * Wd / 2, c0 + u * b * L - v * Wd / 2, c0 + u * b * L + v * Wd / 2, c0 + u * a * L + v * Wd / 2]
        ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=NAVY, lw=0.5, zorder=zorder + 2))


def draw_icon(ax, name: str, x: float, y: float, size: float, zorder: int = 3) -> None:
    if name in HEALTH:
        f, c = HEALTH[name]
        _health(ax, f, c, x, y, size, zorder)
    elif name == "encoder":
        _encoder(ax, x, y, size, zorder)
    elif name == "segmenter":
        _unet(ax, x, y, size, zorder)
    elif name == "eraser":
        _eraser(ax, x, y, size, zorder)
    else:
        raise KeyError(name)
