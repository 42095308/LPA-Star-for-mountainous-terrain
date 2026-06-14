"""按 Nature 风格重新拼版路径后处理对比图。"""

from __future__ import annotations

import shutil
import sys
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_revised_paper_assets import (
    bspline_or_polyline,
    load_scene_arrays,
    los_prune_path,
    material_icon_path,
    paste_icon_with_outline,
    path_distance_km,
    render_path_postprocessing_panel,
    shortest_path,
    terminal_index,
)


DPI = 600
ELEVATION_MAX_M = 2081.0
OUTPUT_STEM = "fig_3_3_path_postprocessing"
SCENARIO_CONFIG = "scenarios/huashan.json"
CACHE_DIR = PROJECT_ROOT / "intermediate_artifacts" / "figures" / "huashan" / "_fig_3_3_nature_cache"
INTERMEDIATE_FIGURE_DIR = PROJECT_ROOT / "intermediate_artifacts" / "figures" / "huashan"


def configure_matplotlib() -> None:
    """设置符合 Nature 图件要求的基础字体和导出参数。"""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 6,
            "axes.linewidth": 0.6,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
        }
    )


def reference_terrain_cmap() -> LinearSegmentedColormap:
    """复现三维地形面板使用的高程配色。"""
    return LinearSegmentedColormap.from_list(
        "huashan_reference_terrain",
        [
            (0.00, "#0B321F"),
            (0.16, "#1F5631"),
            (0.34, "#4D7041"),
            (0.50, "#8B8058"),
            (0.66, "#A38E63"),
            (0.80, "#8D8677"),
            (0.92, "#CFC8B8"),
            (1.00, "#F4F1EA"),
        ],
    )


def english_task_name(name: str) -> str:
    """把场景端点名称转换为图内英文标签。"""
    if name.startswith("虚拟配送站"):
        suffix = name.removeprefix("虚拟配送站")
        return f"Virtual depot {suffix}" if suffix else "Virtual depot"
    if name.startswith("铏氭嫙閰嶉€佺珯"):
        suffix = name.removeprefix("铏氭嫙閰嶉€佺珯")
        return f"Virtual depot {suffix}" if suffix else "Virtual depot"
    mapping = {
        "南峰": "South Peak",
        "北峰": "North Peak",
        "东峰": "East Peak",
        "西峰": "West Peak",
        "中峰": "Central Peak",
        "鍗楀嘲": "South Peak",
        "鍖楀嘲": "North Peak",
        "涓滃嘲": "East Peak",
        "瑗垮嘲": "West Peak",
        "涓嘲": "Central Peak",
    }
    return mapping.get(name, name)


def compute_union_bbox(images: list[np.ndarray], padding: int = 40) -> tuple[int, int, int, int]:
    """对两个三维面板使用同一裁剪框，保证空间比例一致。"""
    boxes: list[tuple[int, int, int, int]] = []
    for image in images:
        rgb = np.asarray(Image.fromarray(image).convert("RGB"))
        mask = np.any(rgb < 247, axis=2)
        if not np.any(mask):
            boxes.append((0, 0, rgb.shape[1], rgb.shape[0]))
            continue
        ys, xs = np.where(mask)
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    x0 = max(0, min(box[0] for box in boxes) - padding)
    y0 = max(0, min(box[1] for box in boxes) - padding)
    x1 = min(images[0].shape[1], max(box[2] for box in boxes) + padding)
    y1 = min(images[0].shape[0], max(box[3] for box in boxes) + padding)
    return x0, y0, x1, y1


def crop_panel(image: np.ndarray, bbox: tuple[int, int, int, int], start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> Image.Image:
    """裁剪面板并贴回起终点图标。"""
    panel = Image.fromarray(image).convert("RGBA")
    x0, y0, x1, y1 = bbox
    cropped = panel.crop((x0, y0, x1, y1))
    uav_icon = Image.open(material_icon_path(PROJECT_ROOT, "uav.png"))
    target_icon = Image.open(material_icon_path(PROJECT_ROOT, "target.png"))
    paste_icon_with_outline(cropped, uav_icon, (start_xy[0] - x0, start_xy[1] - y0), 58, outline_px=4)
    paste_icon_with_outline(cropped, target_icon, (goal_xy[0] - x0, goal_xy[1] - y0), 46, outline_px=3)
    return cropped.convert("RGB")


def draw_custom_legend(fig: plt.Figure) -> None:
    """绘制支持 UAV 位图图标的紧凑图例。"""
    legend_ax = fig.add_axes([0.155, 0.050, 0.690, 0.058])
    legend_ax.set_xlim(0.0, 1.0)
    legend_ax.set_ylim(0.0, 1.0)
    legend_ax.set_axis_off()

    entries = [
        ("line", 0.000, "#D55E00", "Original path"),
        ("dash", 0.205, "#6B6B6B", "LOS-pruned line"),
        ("line", 0.430, "#0072B2", "B-spline trajectory"),
    ]
    for kind, x, color, label in entries:
        linestyle = (0, (3, 2)) if kind == "dash" else "solid"
        line_width = 1.5 if kind == "dash" else 2.0
        legend_ax.plot([x, x + 0.050], [0.50, 0.50], color=color, lw=line_width, ls=linestyle, solid_capstyle="butt")
        legend_ax.text(x + 0.065, 0.50, label, va="center", ha="left", fontsize=5.5, color="black")

    uav_icon = Image.open(material_icon_path(PROJECT_ROOT, "uav.png")).convert("RGBA")
    uav_artist = OffsetImage(uav_icon, zoom=0.90, interpolation="nearest")
    legend_ax.add_artist(AnnotationBbox(uav_artist, (0.662, 0.50), frameon=False, xycoords=legend_ax.transAxes))
    legend_ax.text(0.700, 0.50, "Start node", va="center", ha="left", fontsize=5.5, color="black")

    legend_ax.scatter([0.845], [0.50], marker="*", s=36, color="#E12A1C", edgecolor="white", linewidth=0.25, zorder=3)
    legend_ax.text(0.880, 0.50, "Target node", va="center", ha="left", fontsize=5.5, color="black")


def render_cache(cache_dir: Path = CACHE_DIR) -> list[Path]:
    """使用 PyVista 渲染两个三维路径面板缓存。"""
    data = load_scene_arrays(PROJECT_ROOT, SCENARIO_CONFIG, max_surface_points=96)

    start_name = str(data.cfg.get("default_start", "虚拟配送站1"))
    goal_name = str(data.cfg.get("default_goal", "南峰"))
    raw_path = shortest_path(data, terminal_index(data, start_name), terminal_index(data, goal_name))
    pruned_path = los_prune_path(data, raw_path)
    curve = bspline_or_polyline(data, pruned_path)
    raw_len = path_distance_km(data.nodes[list(raw_path), :3])
    pruned_len = path_distance_km(data.nodes[list(pruned_path), :3])
    curve_len = path_distance_km(curve)

    panel_size = (2120, 1420)
    panel_raw, raw_start_xy, raw_goal_xy, terrain_cmap, elev_display_max = render_path_postprocessing_panel(
        data,
        raw_path,
        pruned_path,
        curve,
        smooth=False,
        window_size=panel_size,
    )
    panel_smooth, smooth_start_xy, smooth_goal_xy, _, _ = render_path_postprocessing_panel(
        data,
        raw_path,
        pruned_path,
        curve,
        smooth=True,
        window_size=panel_size,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_panel_path = cache_dir / "panel_raw.png"
    smooth_panel_path = cache_dir / "panel_smooth.png"
    metadata_path = cache_dir / "metadata.json"
    Image.fromarray(panel_raw).save(raw_panel_path)
    Image.fromarray(panel_smooth).save(smooth_panel_path)
    metadata = {
        "start_name": start_name,
        "goal_name": goal_name,
        "raw_start_xy": list(raw_start_xy),
        "raw_goal_xy": list(raw_goal_xy),
        "smooth_start_xy": list(smooth_start_xy),
        "smooth_goal_xy": list(smooth_goal_xy),
        "raw_node_count": len(raw_path),
        "pruned_node_count": len(pruned_path),
        "raw_len": raw_len,
        "pruned_len": pruned_len,
        "curve_len": curve_len,
        "elev_display_max": ELEVATION_MAX_M,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return [raw_panel_path, smooth_panel_path, metadata_path]


def load_cache(cache_dir: Path = CACHE_DIR) -> tuple[np.ndarray, np.ndarray, dict]:
    """读取三维面板缓存和路径指标。"""
    raw_panel_path = cache_dir / "panel_raw.png"
    smooth_panel_path = cache_dir / "panel_smooth.png"
    metadata_path = cache_dir / "metadata.json"
    missing = [path for path in (raw_panel_path, smooth_panel_path, metadata_path) if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"缺少面板缓存：{missing_text}")
    panel_raw = np.asarray(Image.open(raw_panel_path).convert("RGB"))
    panel_smooth = np.asarray(Image.open(smooth_panel_path).convert("RGB"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return panel_raw, panel_smooth, metadata


def build_optimized_figure(cache_dir: Path = CACHE_DIR) -> list[Path]:
    """生成优化后的 PNG、PDF、SVG 和 TIFF，并同步到场景中间图目录。"""
    configure_matplotlib()
    panel_raw, panel_smooth, metadata = load_cache(cache_dir)

    bbox = compute_union_bbox([panel_raw, panel_smooth])
    panels = [
        crop_panel(panel_raw, bbox, tuple(metadata["raw_start_xy"]), tuple(metadata["raw_goal_xy"])),
        crop_panel(panel_smooth, bbox, tuple(metadata["smooth_start_xy"]), tuple(metadata["smooth_goal_xy"])),
    ]

    fig = plt.figure(figsize=(6.35, 2.98), dpi=DPI)
    ax_a = fig.add_axes([0.045, 0.348, 0.462, 0.610])
    ax_b = fig.add_axes([0.493, 0.348, 0.462, 0.610])
    cax = fig.add_axes([0.390, 0.238, 0.220, 0.017])

    for ax, panel in zip((ax_a, ax_b), panels):
        ax.imshow(panel)
        ax.set_axis_off()

    fig.text(0.276, 0.318, "(a) Initial discrete path", ha="center", va="top", fontsize=6.4)
    fig.text(0.724, 0.318, "(b) LOS-pruned B-spline trajectory", ha="center", va="top", fontsize=6.4)

    scalar = mpl.cm.ScalarMappable(
        norm=Normalize(vmin=0.0, vmax=ELEVATION_MAX_M),
        cmap=reference_terrain_cmap(),
    )
    colorbar = fig.colorbar(scalar, cax=cax, orientation="horizontal")
    colorbar.set_ticks([0, 1000, 2081])
    colorbar.ax.set_title("Elevation (m)", fontsize=5.1, pad=1.6)
    colorbar.ax.tick_params(labelsize=4.7, width=0.4, length=1.5, pad=1)
    colorbar.outline.set_linewidth(0.4)

    task_text = (
        f"{english_task_name(str(metadata['start_name']))} to {english_task_name(str(metadata['goal_name']))}; "
        f"nodes {int(metadata['raw_node_count'])} to {int(metadata['pruned_node_count'])}; "
        f"length {float(metadata['raw_len']):.2f} to {float(metadata['pruned_len']):.2f} to {float(metadata['curve_len']):.2f} km"
    )
    fig.text(0.5, 0.180, task_text, ha="center", va="center", fontsize=5.8)

    draw_custom_legend(fig)

    out_dir = PROJECT_ROOT / "final_results" / "paper_revision" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{OUTPUT_STEM}.png"
    pdf_path = out_dir / f"{OUTPUT_STEM}.pdf"
    svg_path = out_dir / f"{OUTPUT_STEM}.svg"
    tiff_path = out_dir / f"{OUTPUT_STEM}.tiff"

    print(f"正在导出 SVG：{svg_path}", flush=True)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.01)
    print(f"正在导出 PDF：{pdf_path}", flush=True)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    print(f"正在导出 PNG：{png_path}", flush=True)
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"正在导出 TIFF：{tiff_path}", flush=True)
    with Image.open(png_path) as image:
        image.convert("RGB").save(tiff_path, dpi=(DPI, DPI))

    copied: list[Path] = []
    for path in (png_path, pdf_path, svg_path, tiff_path):
        target = INTERMEDIATE_FIGURE_DIR / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target)
    return [png_path, pdf_path, svg_path, tiff_path, *copied]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="优化 fig_3_3_path_postprocessing 的 Nature 风格拼版。")
    parser.add_argument("--render-cache", action="store_true", help="仅生成 PyVista 三维面板缓存。")
    parser.add_argument("--compose-cache", action="store_true", help="仅使用已缓存面板生成最终图。")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR, help="面板缓存目录。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.render_cache:
        paths = render_cache(args.cache_dir)
        print("已生成三维面板缓存：")
    else:
        if not args.compose_cache:
            print("未指定模式，默认使用已缓存面板生成最终图。")
        paths = build_optimized_figure(args.cache_dir)
        print("已生成 Nature 风格优化图：")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
