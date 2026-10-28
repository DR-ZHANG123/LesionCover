"""作图公共样式（单栏 84 mm、1.5 栏 129 mm、双栏 174 mm，最大高 234 mm；Arial 7–8 pt 起；矢量 PDF + 600 dpi PNG）。

每个作图脚本只读 results/ 下的表，把面板输入写到 figures/source_data/<figure>_<panel>.tsv，再存 PDF/PNG。
子图字母由 align_panel_labels 统一按行列对齐；overlap_report 检查文字与文字、文字与数据点的重叠。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

MM = 1 / 25.4
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
W1, W15, W2, HMAX = 84 * MM, 129 * MM, 174 * MM, 234 * MM

# 方法颜色：本研究方法深红，几何去重/多样性对照蓝绿，其余灰阶与浅色；图中名称与正文一致
METHOD_COLORS = {"G512": "#9E9E9E", "G1024": "#616161", "U": "#BDBDBD", "R": "#D7B98E", "T": "#E0A458",
                 "N": "#3A7DBA", "D": "#2A9D8F", "C": "#B2182B", "Cmass": "#E57373", "S": "#8E6BB0"}
METHOD_LABELS = {"G512": "Global 512 only", "G1024": "Global 1024 only", "U": "Fixed quadrants",
                 "R": "Random crops", "T": "Heatmap ranking", "N": "Lesion score + NMS",
                 "D": "Lesion score + DPP", "C": "LesionCover", "Cmass": "LesionCover, area-weighted",
                 "S": "Lesion score top-K"}
LESION_LABELS = {"MA": "Microaneurysm", "HE": "Haemorrhage", "EX": "Hard exudate", "SE": "Soft exudate"}
LESION_COLORS = {"MA": "#D62728", "HE": "#8C2D04", "EX": "#E6AB02", "SE": "#1F78B4"}


def apply_style(base_fontsize: float = 8.0) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "legend.fontsize": base_fontsize - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.1,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": "#CCCCCC",
        "legend.borderpad": 0.4,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.dpi": 600,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def panel_label(ax, letter: str, dx: float = -0.14, dy: float = 1.06) -> None:
    """Register a panel letter; save_fig places all of a figure's letters together.

    Letters are positioned from the drawn extent of each panel rather than from
    offsets inside it, so that letters on one row share a baseline and letters in
    one column share a left edge, however large each panel is. `dx` and `dy` are
    accepted for compatibility and no longer used.
    """
    fig = ax.figure
    if not hasattr(fig, "_panel_labels"):
        fig._panel_labels = []
    fig._panel_labels.append((ax, letter))


def _spans_overlap(a0, a1, b0, b1) -> bool:
    inter = min(a1, b1) - max(a0, b0)
    return inter > 0.5 * min(a1 - a0, b1 - b0)


def align_panel_labels(fig, pad_pt: float = 4.0, col_tol: float = 0.07) -> None:
    """Place registered panel letters on shared rows and columns.

    Rows: panels whose axes overlap vertically by more than half of the shorter
    one form a row, and every letter of the row sits just above the highest drawn
    top in it (titles and labels included). Columns: letters whose panels start
    within `col_tol` of the figure width share the leftmost of their left edges.
    """
    items = getattr(fig, "_panel_labels", [])
    if not items:
        return
    col_tol = getattr(fig, "_col_tol", col_tol)         # 图像面板按宽高比缩小时可放宽
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    rec = []
    for ax, letter in items:
        tb = inv.transform_bbox(ax.get_tightbbox(r))
        pos = ax.get_position()
        rec.append({"ax": ax, "letter": letter, "x0": tb.x0, "top": tb.y1,
                    "y0": pos.y0, "y1": pos.y1, "px0": pos.x0, "px1": pos.x1})
    # rows
    rows = []
    for it in sorted(rec, key=lambda d: -d["y1"]):
        for row in rows:
            if any(_spans_overlap(it["y0"], it["y1"], o["y0"], o["y1"]) for o in row):
                row.append(it); break
        else:
            rows.append([it])
    pad = pad_pt / 72 / fig.get_figheight()
    for row in rows:
        y = max(o["top"] for o in row) + pad
        for o in row:
            o["y"] = y
    # columns: panels that overlap horizontally, or whose drawn left edges lie
    # within `col_tol` of each other, share one left edge (the leftmost)
    parent = list(range(len(rec)))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for i_ in range(len(rec)):
        for j_ in range(i_ + 1, len(rec)):
            a_, b_ = rec[i_], rec[j_]
            # overlap is measured against the wider panel, so a full-width panel
            # does not chain every column of the row above into one
            inter = min(a_["px1"], b_["px1"]) - max(a_["px0"], b_["px0"])
            wide = max(a_["px1"] - a_["px0"], b_["px1"] - b_["px0"])
            if inter > 0.9 * wide or abs(a_["x0"] - b_["x0"]) < col_tol:
                parent[find(i_)] = find(j_)
    groups = {}
    for k, o in enumerate(rec):
        groups.setdefault(find(k), []).append(o)
    for col in groups.values():
        x = min(o["x0"] for o in col)
        for o in col:
            o["x"] = x
    for o in rec:
        t = fig.text(o["x"], o["y"], o["letter"], fontsize=11, fontweight="bold",
                     ha="left", va="bottom")
        t._panel_letter = True
    fig._panel_labels = []


def save_fig(fig, root: Path, name: str) -> list[Path]:
    import os
    align_panel_labels(fig)
    if os.environ.get("FIG_OVERLAP_REPORT"):
        for msg in overlap_report(fig):
            print(f"OVERLAP {name}: {msg}")
    outdir = root / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = outdir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def save_source(df: pd.DataFrame, root: Path, figure: str, panel: str) -> Path:
    d = root / "figures" / "source_data"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{figure}_{panel}.tsv"
    df.to_csv(p, sep="\t", index=False, float_format="%.6g")
    return p


def overlap_report(fig, point_pad: float = 1.0) -> list[str]:
    """Text that collides with other text or sits on plotted points.

    Every visible text artist (labels, annotations, legends, tick labels) is
    measured after drawing. Two texts whose boxes intersect, or a text box that
    covers a scatter point, are reported. Heatmap cell values are exempt because
    they are meant to sit on their cells.
    """
    import numpy as np
    from matplotlib.collections import PathCollection
    from matplotlib.legend import Legend
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # tick labels exist for ticks outside the view as well; only the drawn ones count
    all_ticks, drawn = set(), set()
    for ax_ in fig.axes:
        for axis in (ax_.xaxis, ax_.yaxis):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                all_ticks |= {id(tk.label1), id(tk.label2)}
            for tk in axis._update_ticks():
                drawn |= {id(tk.label1), id(tk.label2)}
    texts = []
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if id(t) in all_ticks and id(t) not in drawn:
            continue
        if getattr(t, "_overlap_exempt", False):
            continue
        texts.append(t)
    # Text.get_window_extent on an Annotation measures the text alone; the
    # annotation's own method also spans its leader line
    boxes = [(t, mpl.text.Text.get_window_extent(t, r)) for t in texts]
    issues = []
    for i in range(len(boxes)):
        ti, bi = boxes[i]
        for j in range(i + 1, len(boxes)):
            tj, bj = boxes[j]
            if ti.get_rotation() or tj.get_rotation():
                continue        # axis-aligned boxes of rotated labels overlap falsely
            if bi.overlaps(bj) and bi.width > 0 and bj.width > 0:
                issues.append(f"text '{ti.get_text()[:30]}' overlaps '{tj.get_text()[:30]}'")
    legend_boxes = [lg.get_window_extent(r) for lg in fig.findobj(Legend)]
    for col in fig.findobj(PathCollection):
        if not col.get_visible() or col.axes is None or col.get_offsets().size == 0:
            continue
        pts = col.get_offset_transform().transform(np.asarray(col.get_offsets()))
        for t, b in boxes:
            if t.axes is not col.axes and t.figure is fig and t.axes is not None:
                continue
            inside = ((pts[:, 0] > b.x0 + point_pad) & (pts[:, 0] < b.x1 - point_pad) &
                      (pts[:, 1] > b.y0 + point_pad) & (pts[:, 1] < b.y1 - point_pad))
            if inside.any():
                issues.append(f"text '{t.get_text()[:30]}' covers {int(inside.sum())} point(s)")
        for b in legend_boxes:
            inside = ((pts[:, 0] > b.x0) & (pts[:, 0] < b.x1) &
                      (pts[:, 1] > b.y0) & (pts[:, 1] < b.y1))
            if inside.any() and col.get_label().startswith("_"):
                issues.append(f"legend covers {int(inside.sum())} point(s)")
    issues += _legend_data_overlaps(fig, r)
    return sorted(set(issues))


def _legend_data_overlaps(fig, r) -> list[str]:
    """Legends drawn over plotted data of their own panel: lines (including error-bar
    whiskers and reference lines), marker-only lines and bars."""
    import numpy as np
    from matplotlib.collections import LineCollection
    issues = []
    for ax_ in fig.axes:
        lg = ax_.get_legend()
        if lg is None or not lg.get_visible():
            continue
        b = lg.get_window_extent(r)
        n_line, n_bar = 0, 0
        for ln in ax_.lines:
            if not ln.get_visible() or len(ln.get_xdata()) == 0:
                continue
            path = ln.get_transform().transform_path(ln.get_path())
            if ln.get_linestyle() in ("None", "none", "", " "):          # 只画标记：看标记中心是否落在图例框内
                v = path.vertices
                n_line += int(((v[:, 0] > b.x0) & (v[:, 0] < b.x1) & (v[:, 1] > b.y0) & (v[:, 1] < b.y1)).any())
            elif path.intersects_bbox(b, filled=False):
                n_line += 1
        for col in ax_.collections:
            if isinstance(col, LineCollection) and col.get_visible():
                tr = col.get_transform()
                n_line += sum(tr.transform_path(p).intersects_bbox(b, filled=False) for p in col.get_paths())
        for p in ax_.patches:
            if p.get_visible() and p.get_height() > 0 and p.get_window_extent(r).overlaps(b):
                n_bar += 1
        if n_line:
            issues.append(f"legend of panel '{ax_.get_title()[:30]}' crosses {n_line} data line(s)")
        if n_bar:
            issues.append(f"legend of panel '{ax_.get_title()[:30]}' covers {n_bar} bar(s)")
    return issues


def place_labels(ax, xs, ys, labels, fontsize: float = 5.4, italic: bool = False,
                 color: str = "#222222", min_r: float = 9, max_r: float = 70,
                 avoid_axes=None) -> list:
    """Annotate points with labels that land in empty space.

    For each label, candidate positions on rings of growing radius around its
    point are scored against every plotted point in the axes, every label already
    placed, the legend and the axes frame; the first position touching nothing is
    taken, otherwise the one covering the fewest points. A thin leader line joins
    label and point. Axes limits are frozen first, so later autoscaling cannot
    move the points out from under their labels.
    """
    import numpy as np
    from matplotlib.collections import PathCollection
    from matplotlib.transforms import Bbox
    fig = ax.figure
    ax.set_xlim(ax.get_xlim()); ax.set_ylim(ax.get_ylim())
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts = []
    for col in ax.collections:
        if isinstance(col, PathCollection) and col.get_offsets().size:
            pts.append(col.get_offset_transform().transform(np.asarray(col.get_offsets())))
    for ln in ax.lines:
        xy = ln.get_xydata()
        if len(xy):
            pts.append(ax.transData.transform(xy))
    pts = np.vstack(pts) if pts else np.zeros((0, 2))
    frame = ax.get_window_extent(r)
    blocked = [lg.get_window_extent(r) for lg in ([ax.get_legend()] if ax.get_legend() else [])]
    blocked += [mpl.text.Text.get_window_extent(t, r) for t in ax.texts if t.get_text().strip()]
    for other in (avoid_axes or []):
        blocked.append(other.get_window_extent(r))
    px = fig.dpi / 72.0
    placed, out = [], []
    angles = np.deg2rad(np.arange(0, 360, 22.5))
    for x, y, lab in zip(xs, ys, labels):
        t = ax.text(0, 0, lab, fontsize=fontsize, fontstyle="italic" if italic else "normal")
        w, h = t.get_window_extent(r).width, t.get_window_extent(r).height
        t.remove()
        p0 = ax.transData.transform((x, y))
        best, best_cost = None, None
        for rad in np.arange(min_r, max_r + 1, 6):
            for a in angles:
                cx, cy = p0[0] + rad * px * np.cos(a), p0[1] + rad * px * np.sin(a)
                b = Bbox.from_bounds(cx - w / 2 - 1.5, cy - h / 2 - 1.5, w + 3, h + 3)
                if b.x0 < frame.x0 or b.x1 > frame.x1 or b.y0 < frame.y0 or b.y1 > frame.y1:
                    continue
                if any(b.overlaps(o) for o in placed + blocked):
                    continue
                cover = int(((pts[:, 0] > b.x0) & (pts[:, 0] < b.x1) &
                             (pts[:, 1] > b.y0) & (pts[:, 1] < b.y1)).sum())
                cost = cover * 1000 + rad
                if best_cost is None or cost < best_cost:
                    best, best_cost = (cx - p0[0], cy - p0[1], b), cost
            if best_cost is not None and best_cost < 1000:
                break
        if best is None:
            raise SystemExit(f"place_labels: no room for '{lab}'")
        dx, dy, b = best
        placed.append(b)
        out.append(ax.annotate(lab, (x, y), xytext=(dx / px, dy / px),
                               textcoords="offset points", ha="center", va="center",
                               fontsize=fontsize, color=color, zorder=7,
                               fontstyle="italic" if italic else "normal",
                               # an invisible plate around the text clips the leader
                               # line at its edge, so the line never runs through it
                               bbox=dict(boxstyle="square,pad=0.15", fc="none", ec="none"),
                               arrowprops=dict(arrowstyle="-", lw=0.35, color="#777777",
                                               shrinkA=0, shrinkB=1.5)))
    return out
