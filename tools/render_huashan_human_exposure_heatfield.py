"""生成华山三维地形上的游客人员暴露风险连续热力场。

该脚本只读取已有 DEM、经纬度网格、场景配置和 OSM 文件，使用 Python 代码
把步道、观景点、服务设施和道路等离散要素转换为连续风险代理场，再贴合
PyVista 三维地形表面渲染。输出为新文件，不覆盖原始地形图或既有结果。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from article_planner.scenario_config import load_scenario_config
from human_risk_osm import (
    LEVELS,
    apply_scene_risk_keywords,
    build_lonlat_tree,
    classify_level,
    dedup_lines,
    dedup_points,
    draw_line_mask,
    is_line_way,
    line_bbox_intersects,
    lonlat_to_rc,
    parse_osm,
    point_in_bbox,
)
from render_huashan_dem_pyvista import (
    build_ground_shadow,
    configure_reference_lights,
    read_resolution,
    style_bounds_actor,
)
from huashan_peak_annotations import draw_peak_annotations, peak_world_points, project_peak_points


DEFAULT_SCENE = PROJECT_ROOT / "scenarios" / "huashan.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "intermediate_artifacts" / "data" / "huashan"
DEFAULT_OUT = (
    PROJECT_ROOT
    / "final_results"
    / "paper_revision"
    / "figures"
    / "fig_2_1b_huashan_human_exposure_risk_python_heatfield.png"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "final_results"
    / "paper_revision"
    / "figures"
    / "fig_2_1b_huashan_human_exposure_risk_python_heatfield_summary.json"
)
DEFAULT_PDF_OUT = DEFAULT_OUT.with_suffix(".pdf")


TERRAIN_CMAP = LinearSegmentedColormap.from_list(
    "huashan_muted_reference_terrain",
    [
        (0.00, "#405A47"),
        (0.16, "#5B6D5D"),
        (0.34, "#777B68"),
        (0.52, "#969183"),
        (0.68, "#AEA79A"),
        (0.82, "#C2BFB7"),
        (0.93, "#DDDBD5"),
        (1.00, "#F5F3EF"),
    ],
)

RISK_COLOR_STOPS = [
    (0.00, "#F9D976", 0.00),
    (0.25, "#F9D976", 0.00),
    (0.45, "#F9D976", 0.30),
    (0.55, "#F39C34", 0.45),
    (0.80, "#C83232", 0.60),
    (1.00, "#C83232", 0.60),
]

RISK_CMAP = LinearSegmentedColormap.from_list(
    "human_exposure_risk_warm",
    [(value, color) for value, color, _ in RISK_COLOR_STOPS],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成华山三维人员暴露风险连续热力场。")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="场景配置 JSON。")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="华山中间数据目录。")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出 PNG 路径。")
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY, help="输出摘要 JSON 路径。")
    parser.add_argument("--stride", type=int, default=2, help="三维地形采样步长。")
    parser.add_argument("--vertical-exag", type=float, default=1.45, help="垂向夸张系数。")
    parser.add_argument("--smooth-sigma", type=float, default=0.45, help="地形显示平滑系数。")
    parser.add_argument("--risk-stride", type=int, default=2, help="风险热力层投影采样步长。")
    parser.add_argument("--width", type=int, default=3600, help="输出图像宽度。")
    parser.add_argument("--height", type=int, default=2200, help="输出图像高度。")
    parser.add_argument("--hide-grid", action="store_true", help="隐藏三维坐标网格。")
    parser.add_argument("--hide-elevation-colorbar", action="store_true", help="隐藏高程色标。")
    parser.add_argument("--hide-risk-legend", action="store_true", help="隐藏人员暴露风险图例。")
    parser.add_argument("--no-high-contour", action="store_true", help="隐藏高风险细轮廓线。")
    parser.add_argument("--hide-peak-labels", action="store_true", help="隐藏华山五峰单字母标注。")
    return parser.parse_args()


def resolve_osm_path(config: dict) -> Path:
    raw = Path(str(config.get("osm_file") or "data/raw/huashan/map.osm"))
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """优先使用 Arial，使图中文字符合 Nature 风格并保持清晰。"""
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/timesbd.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def extract_osm_features(config: dict, osm_path: Path, lon_grid: np.ndarray, lat_grid: np.ndarray) -> tuple[dict, dict]:
    """从 OSM 中提取分级点线要素，后续只用于计算连续风险场。"""
    apply_scene_risk_keywords(config)
    lon_min, lon_max = float(np.nanmin(lon_grid)), float(np.nanmax(lon_grid))
    lat_min, lat_max = float(np.nanmin(lat_grid)), float(np.nanmax(lat_grid))
    nodes, tagged_nodes, ways, _ = parse_osm(osm_path)

    lines = {level: [] for level in LEVELS}
    points = {level: [] for level in LEVELS}

    for node_id, tags in tagged_nodes:
        level = classify_level(tags)
        if level not in LEVELS or node_id not in nodes:
            continue
        lon, lat = nodes[node_id]
        if point_in_bbox(lon, lat, lon_min, lon_max, lat_min, lat_max):
            points[level].append((float(lon), float(lat)))

    for way in ways:
        level = classify_level(way.tags)
        if level not in LEVELS:
            continue
        coords = [nodes[ref] for ref in way.refs if ref in nodes]
        if not coords or not line_bbox_intersects(coords, lon_min, lon_max, lat_min, lat_max):
            continue
        if is_line_way(way.tags, level) and len(coords) >= 2:
            lines[level].append([(float(lon), float(lat)) for lon, lat in coords])
        else:
            in_crop = [
                (lon, lat)
                for lon, lat in coords
                if point_in_bbox(lon, lat, lon_min, lon_max, lat_min, lat_max)
            ]
            if in_crop:
                points[level].append(
                    (
                        float(np.mean([item[0] for item in in_crop])),
                        float(np.mean([item[1] for item in in_crop])),
                    )
                )

    return (
        {level: dedup_lines(lines[level]) for level in LEVELS},
        {level: dedup_points(points[level]) for level in LEVELS},
    )


def add_lonlat_points_to_mask(
    mask: np.ndarray,
    lonlat: Sequence[tuple[float, float]],
    tree,
    rows: int,
    cols: int,
) -> int:
    if not lonlat:
        return 0
    count = 0
    for r, c in lonlat_to_rc(lonlat, tree, rows, cols):
        mask[int(r), int(c)] = True
        count += 1
    return count


def add_scene_target_hotspots(mask: np.ndarray, config: dict, tree, rows: int, cols: int) -> int:
    """把场景配置中的主峰目标纳入高风险核心节点。"""
    targets = config.get("targets", {})
    if not isinstance(targets, dict):
        return 0
    lonlat = []
    for item in targets.values():
        if not isinstance(item, dict):
            continue
        if "lon" in item and "lat" in item:
            lonlat.append((float(item["lon"]), float(item["lat"])))
    return add_lonlat_points_to_mask(mask, lonlat, tree, rows, cols)


def gaussian_from_mask(mask: np.ndarray, peak: float, sigma_m: float, resolution_m: float) -> np.ndarray:
    """根据到要素的距离生成平滑衰减场。"""
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=float)
    distance_m = distance_transform_edt(~mask) * float(resolution_m)
    return float(peak) * np.exp(-(distance_m**2) / (2.0 * float(sigma_m) ** 2))


def build_risk_field(
    z: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    lines: dict,
    points: dict,
    config: dict,
    resolution_m: float,
) -> tuple[np.ndarray, dict]:
    """把 OSM 离散要素转换为低、中、高分明的连续人员暴露风险场。"""
    tree, rows, cols = build_lonlat_tree(lon_grid, lat_grid)
    line_masks = {level: np.zeros((rows, cols), dtype=bool) for level in LEVELS}
    point_masks = {level: np.zeros((rows, cols), dtype=bool) for level in LEVELS}
    line_density = np.zeros((rows, cols), dtype=float)

    for level in LEVELS:
        for line in lines[level]:
            rc_line = lonlat_to_rc(line, tree, rows, cols)
            if len(rc_line) < 2:
                continue
            single = np.zeros((rows, cols), dtype=bool)
            draw_line_mask(single, rc_line)
            line_masks[level] |= single
            line_density += single.astype(float)
        add_lonlat_points_to_mask(point_masks[level], points[level], tree, rows, cols)

    target_mask = np.zeros((rows, cols), dtype=bool)
    target_count = add_scene_target_hotspots(target_mask, config, tree, rows, cols)

    intersection_mask = np.zeros((rows, cols), dtype=bool)
    if np.max(line_density) > 1.0:
        intersection_seed = gaussian_filter(line_density, sigma=1.0, mode="nearest")
        nonzero = intersection_seed[intersection_seed > 0]
        if nonzero.size:
            intersection_mask |= intersection_seed >= float(np.percentile(nonzero, 99.82))

    activity_points = point_masks[1] | point_masks[2] | point_masks[3]
    clustered_points = np.zeros((rows, cols), dtype=bool)
    if np.any(activity_points):
        point_density = gaussian_filter(activity_points.astype(float), sigma=4.6, mode="nearest")
        nonzero = point_density[point_density > 0]
        if nonzero.size:
            clustered_points |= point_density >= float(np.percentile(nonzero, 99.35))

    core_mask = target_mask | point_masks[1] | point_masks[3] | intersection_mask | clustered_points

    main_trail_mask = line_masks[1] | line_masks[2]
    normal_trail_mask = line_masks[3]
    trail_mask = main_trail_mask | normal_trail_mask
    road_mask = line_masks[4]
    all_line_mask = trail_mask | road_mask

    broad_halo = np.maximum(
        gaussian_from_mask(all_line_mask, peak=0.22, sigma_m=150.0, resolution_m=resolution_m),
        gaussian_from_mask(road_mask, peak=0.14, sigma_m=120.0, resolution_m=resolution_m),
    )
    medium_corridor = np.maximum.reduce(
        [
            gaussian_from_mask(main_trail_mask, peak=0.70, sigma_m=80.0, resolution_m=resolution_m),
            gaussian_from_mask(normal_trail_mask, peak=0.45, sigma_m=50.0, resolution_m=resolution_m),
            gaussian_from_mask(road_mask, peak=0.55, sigma_m=90.0, resolution_m=resolution_m),
        ]
    )
    medium_corridor = np.minimum(medium_corridor, 0.72)
    medium_nodes = gaussian_from_mask(point_masks[2], peak=0.62, sigma_m=120.0, resolution_m=resolution_m)
    medium_nodes = np.minimum(medium_nodes, 0.70)

    hotspot_core = gaussian_from_mask(core_mask, peak=1.00, sigma_m=130.0, resolution_m=resolution_m)
    hotspot_halo = gaussian_from_mask(core_mask, peak=0.68, sigma_m=185.0, resolution_m=resolution_m)
    high_nodes = np.maximum(hotspot_core, hotspot_halo)

    risk = np.clip(0.92 * high_nodes + 0.82 * medium_corridor + 0.62 * medium_nodes + 0.55 * broad_halo, 0.0, 1.0)
    risk = gaussian_filter(risk, sigma=1.3, mode="nearest")
    risk = np.clip(risk, 0.0, 1.0)
    risk[risk < 0.25] = 0.0

    summary = {
        "osm_feature_counts": {
            f"L{level}": {"lines": len(lines[level]), "points": len(points[level])}
            for level in LEVELS
        },
        "target_hotspots": int(target_count),
        "core_hotspot_pixels": int(np.count_nonzero(core_mask)),
        "intersection_hotspot_pixels": int(np.count_nonzero(intersection_mask)),
        "clustered_point_hotspot_pixels": int(np.count_nonzero(clustered_points)),
        "risk_min": float(np.min(risk)),
        "risk_max": float(np.max(risk)),
        "risk_mean": float(np.mean(risk)),
        "risk_p95": float(np.percentile(risk, 95)),
        "risk_area_ge_0p35": float(np.mean(risk >= 0.35)),
        "risk_area_ge_0p70": float(np.mean(risk >= 0.70)),
        "risk_area_ge_0p75": float(np.mean(risk >= 0.75)),
        "risk_area_ge_0p80": float(np.mean(risk >= 0.80)),
    }
    return risk.astype(np.float32), summary


def build_surface_grid(
    z: np.ndarray,
    resolution_m: float,
    stride: int,
    vertical_exag: float,
    smooth_sigma: float,
    lift_m: float = 0.0,
) -> pv.StructuredGrid:
    """构建与地形一致的结构化曲面。"""
    stride = max(1, int(stride))
    z_clean = np.nan_to_num(z, nan=float(np.nanmedian(z)))
    z_smooth = gaussian_filter(z_clean, sigma=max(0.0, float(smooth_sigma)), mode="nearest")
    z_sample = z_smooth[::stride, ::stride]
    elev_sample = z_clean[::stride, ::stride]
    rows, cols = z_sample.shape
    base_elev = float(np.nanmin(z_clean))

    x = (np.arange(cols, dtype=float) - 0.5 * (cols - 1)) * resolution_m * stride
    y = (np.arange(rows - 1, -1, -1, dtype=float) - 0.5 * (rows - 1)) * resolution_m * stride
    xx, yy = np.meshgrid(x, y)
    zz = np.maximum(z_sample - base_elev + float(lift_m), 0.0) * vertical_exag

    grid = pv.StructuredGrid(xx, yy, zz)
    grid["Elevation"] = elev_sample.ravel(order="F")
    return grid


def interpolate_stops(value: float, stops: list[tuple[float, str, float]]) -> tuple[int, int, int, int]:
    value = float(np.clip(value, 0.0, 1.0))
    for i in range(len(stops) - 1):
        v0, c0, a0 = stops[i]
        v1, c1, a1 = stops[i + 1]
        if v0 <= value <= v1:
            t = 0.0 if v1 == v0 else (value - v0) / (v1 - v0)
            rgb0 = np.asarray(to_rgb(c0), dtype=float)
            rgb1 = np.asarray(to_rgb(c1), dtype=float)
            rgb = rgb0 * (1.0 - t) + rgb1 * t
            alpha = float(a0) * (1.0 - t) + float(a1) * t
            return (
                int(np.round(rgb[0] * 255)),
                int(np.round(rgb[1] * 255)),
                int(np.round(rgb[2] * 255)),
                int(np.round(alpha * 255)),
            )
    c = to_rgb(stops[-1][1])
    return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), int(stops[-1][2] * 255))


def risk_to_rgba(risk: np.ndarray) -> np.ndarray:
    """把风险值映射为带透明度的暖色 RGBA。"""
    flat = risk.ravel(order="F")
    rgba = np.zeros((flat.size, 4), dtype=np.uint8)
    for idx, value in enumerate(flat):
        rgba[idx] = interpolate_stops(float(value), RISK_COLOR_STOPS)
    rgba[flat <= 0.045, 3] = 0
    return rgba


def risk_to_opacity(risk: np.ndarray) -> np.ndarray:
    """显式控制风险曲面的逐点透明度。"""
    stops_x = np.asarray([value for value, _, _ in RISK_COLOR_STOPS], dtype=float)
    stops_a = np.asarray([alpha for _, _, alpha in RISK_COLOR_STOPS], dtype=float)
    alpha = np.interp(np.clip(risk, 0.0, 1.0).ravel(order="F"), stops_x, stops_a)
    alpha[risk.ravel(order="F") <= 0.045] = 0.0
    return np.clip(alpha, 0.0, 1.0)


def add_risk_overlay(
    plotter: pv.Plotter,
    z: np.ndarray,
    risk: np.ndarray,
    resolution_m: float,
    stride: int,
    vertical_exag: float,
    smooth_sigma: float,
    draw_contour: bool,
) -> pv.StructuredGrid:
    """构建贴合地形的风险曲面，后续投影到当前三维相机视角。"""
    overlay = build_surface_grid(
        z,
        resolution_m=resolution_m,
        stride=stride,
        vertical_exag=vertical_exag,
        smooth_sigma=smooth_sigma,
        lift_m=15.0,
    )
    risk_sample = np.clip(risk[:: max(1, int(stride)), :: max(1, int(stride))], 0.0, 1.0)
    overlay["Risk"] = risk_sample.ravel(order="F")
    return overlay


def project_world_points(plotter: pv.Plotter, points: np.ndarray, image_height: int) -> np.ndarray:
    """把三维世界坐标投影到截图像素坐标。"""
    renderer = plotter.renderer
    projected = np.empty((points.shape[0], 3), dtype=float)
    for idx, point in enumerate(points):
        renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
        renderer.WorldToDisplay()
        x, y, depth = renderer.GetDisplayPoint()
        projected[idx] = (float(x), float(image_height) - float(y), float(depth))
    return projected


def draw_projected_heatfield(image: Image.Image, plotter: pv.Plotter, risk_surface: pv.StructuredGrid) -> Image.Image:
    """按当前三维相机投影，把连续风险场半透明叠加到地形截图上。"""
    width, height = image.size
    risk = np.asarray(risk_surface["Risk"], dtype=float)
    dims = tuple(int(v) for v in risk_surface.dimensions)
    cols, rows = dims[0], dims[1]
    if cols <= 1 or rows <= 1:
        return image

    risk_grid = risk.reshape((cols, rows), order="F").T
    points_2d = project_world_points(plotter, risk_surface.points, height)
    x_grid = points_2d[:, 0].reshape((cols, rows), order="F").T
    y_grid = points_2d[:, 1].reshape((cols, rows), order="F").T
    d_grid = points_2d[:, 2].reshape((cols, rows), order="F").T

    cells: list[tuple[float, list[tuple[float, float]], tuple[int, int, int, int]]] = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            value = float(np.mean(risk_grid[r : r + 2, c : c + 2]))
            if value <= 0.045:
                continue
            polygon = [
                (float(x_grid[r, c]), float(y_grid[r, c])),
                (float(x_grid[r, c + 1]), float(y_grid[r, c + 1])),
                (float(x_grid[r + 1, c + 1]), float(y_grid[r + 1, c + 1])),
                (float(x_grid[r + 1, c]), float(y_grid[r + 1, c])),
            ]
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            if max(xs) < -20 or min(xs) > width + 20 or max(ys) < -20 or min(ys) > height + 20:
                continue
            rgba = interpolate_stops(value, RISK_COLOR_STOPS)
            depth = float(np.mean(d_grid[r : r + 2, c : c + 2]))
            cells.append((depth, polygon, rgba))

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for _, polygon, rgba in sorted(cells, key=lambda item: item[0], reverse=True):
        draw.polygon(polygon, fill=rgba)

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def soften_terrain_backdrop(image: Image.Image) -> Image.Image:
    """降低地形底图饱和度和阴影对比，让风险层成为视觉主体。"""
    image = ImageEnhance.Color(image).enhance(0.75)
    image = ImageEnhance.Brightness(image).enhance(1.05)
    image = ImageEnhance.Contrast(image).enhance(0.88)
    return image


def draw_projected_contours(
    image: Image.Image,
    plotter: pv.Plotter,
    risk_surface: pv.StructuredGrid,
    levels: Sequence[float] = (0.80, 0.92),
) -> Image.Image:
    """绘制高风险区细轮廓，增强关键热点形状边界。"""
    if float(np.nanmax(risk_surface["Risk"])) < min(levels):
        return image

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    height = image.size[1]
    styles = {
        0.80: ((150, 42, 35, 80), 2),
        0.92: ((120, 28, 28, 95), 2),
    }

    for level in levels:
        contour = risk_surface.contour(isosurfaces=[float(level)], scalars="Risk")
        if contour.n_points == 0 or contour.lines.size == 0:
            continue
        projected = project_world_points(plotter, contour.points, height)
        rgba, line_width = styles.get(float(level), ((132, 31, 26, 140), 2))
        lines = np.asarray(contour.lines, dtype=int)
        idx = 0
        while idx < len(lines):
            n = int(lines[idx])
            ids = lines[idx + 1 : idx + 1 + n]
            idx += n + 1
            if n < 2:
                continue
            polyline = [(float(projected[i, 0]), float(projected[i, 1])) for i in ids]
            draw.line(polyline, fill=rgba, width=line_width, joint="curve")

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def draw_vertical_gradient(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], stops: list[tuple[float, str, float]]) -> None:
    x0, y0, x1, y1 = box
    height = max(1, y1 - y0)
    for yy in range(height):
        value = 1.0 - yy / max(1, height - 1)
        r, g, b, _ = interpolate_stops(value, stops)
        draw.line((x0, y0 + yy, x1, y0 + yy), fill=(r, g, b), width=1)


def draw_risk_legend(image_path: Path) -> None:
    """绘制横向 Low、Medium、High 三等级风险图例。"""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    scale = width / 3600.0

    title_font = load_font(max(24, int(36 * scale)))
    label_font = load_font(max(22, int(32 * scale)))
    bar_w = int(400 * scale)
    bar_h = int(36 * scale)
    x0 = width - int(925 * scale)
    y0 = height - int(188 * scale)
    x1 = x0 + bar_w
    y1 = y0 + bar_h

    title = "Normalized human exposure risk"
    tb = draw.textbbox((0, 0), title, font=title_font)
    draw.text((x0 + bar_w / 2 - (tb[2] - tb[0]) / 2, y0 - int(46 * scale)), title, fill="black", font=title_font)
    for xx in range(bar_w):
        value = xx / max(1, bar_w - 1)
        r, g, b, _ = interpolate_stops(value, RISK_COLOR_STOPS)
        draw.line((x0 + xx, y0, x0 + xx, y1), fill=(r, g, b), width=1)
    draw.rectangle((x0, y0, x1, y1), outline="#E6D8C0", width=max(1, int(1.0 * scale)))

    labels = [("Low", x0), ("Medium", x0 + bar_w // 2), ("High", x1)]
    label_y = y1 + int(10 * scale)
    for text, x in labels:
        lb = draw.textbbox((0, 0), text, font=label_font)
        draw.text((x - (lb[2] - lb[0]) / 2, label_y), text, fill="black", font=label_font)

    image.save(image_path)


def draw_elevation_colorbar(image_path: Path, elev_max: float) -> None:
    """用地形配色绘制高程色标，避免 PyVista 色标缩放后文字过细。"""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, _ = image.size
    scale = width / 3600.0

    title_font = load_font(max(24, int(36 * scale)))
    label_font = load_font(max(20, int(30 * scale)))
    bar_w = int(50 * scale)
    bar_h = int(1030 * scale)
    x0 = width - int(315 * scale)
    y0 = int(300 * scale)
    x1 = x0 + bar_w
    y1 = y0 + bar_h

    draw.rectangle((x0 - int(18 * scale), y0 - int(74 * scale), x1 + int(130 * scale), y1 + int(20 * scale)), fill="white")
    for yy in range(bar_h):
        t = 1.0 - yy / max(1, bar_h - 1)
        rgb = tuple(int(round(c * 255)) for c in TERRAIN_CMAP(t)[:3])
        draw.line((x0, y0 + yy, x1, y0 + yy), fill=rgb, width=1)
    draw.rectangle((x0, y0, x1, y1), outline="#777777", width=max(1, int(1.1 * scale)))

    title = "Elevation (m)"
    tb = draw.textbbox((0, 0), title, font=title_font)
    draw.text((x0 + bar_w / 2 - (tb[2] - tb[0]) / 2, y0 - int(58 * scale)), title, fill="black", font=title_font)

    for value in np.linspace(0.0, float(elev_max), 5):
        pos = y1 - (value / max(float(elev_max), 1e-6)) * bar_h
        draw.line((x1, pos, x1 + int(12 * scale), pos), fill="black", width=max(1, int(1.4 * scale)))
        label = f"{value:.0f}"
        lb = draw.textbbox((0, 0), label, font=label_font)
        draw.text((x1 + int(18 * scale), pos - (lb[3] - lb[1]) / 2), label, fill="black", font=label_font)

    image.save(image_path)


def save_png_as_pdf(png_path: Path, pdf_path: Path, dpi: int = 600) -> Path:
    """把最终 PNG 同步保存为单页 PDF，便于论文排版使用。"""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(png_path) as image:
        image.convert("RGB").save(pdf_path, "PDF", resolution=int(dpi))
    return pdf_path


def render_scene(args: argparse.Namespace) -> Path:
    config = load_scenario_config(args.scene, workdir=PROJECT_ROOT)
    data_dir = args.data_dir
    dem_path = data_dir / "Z_crop.npy"
    meta_path = data_dir / "Z_crop_meta.json"
    geo_path = data_dir / "Z_crop_geo.npz"
    osm_path = resolve_osm_path(config)

    for required in (dem_path, meta_path, geo_path, osm_path):
        if not required.exists():
            raise FileNotFoundError(f"缺少必要输入文件：{required}")

    z = np.asarray(np.load(dem_path), dtype=float)
    geo = np.load(geo_path)
    lon_grid = np.asarray(geo["lon_grid"], dtype=float)
    lat_grid = np.asarray(geo["lat_grid"], dtype=float)
    resolution_m = read_resolution(meta_path)

    lines, points = extract_osm_features(config, osm_path, lon_grid, lat_grid)
    risk, risk_summary = build_risk_field(z, lon_grid, lat_grid, lines, points, config, resolution_m)

    terrain_grid = build_surface_grid(
        z,
        resolution_m=resolution_m,
        stride=args.stride,
        vertical_exag=args.vertical_exag,
        smooth_sigma=args.smooth_sigma,
        lift_m=0.0,
    )
    bounds = terrain_grid.bounds
    x_mid = 0.5 * (bounds[0] + bounds[1])
    y_mid = 0.5 * (bounds[2] + bounds[3])
    z_mid = 0.45 * bounds[5]
    span = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    elev_display_max = float(np.nanpercentile(z, 99.7))

    pv.global_theme.smooth_shading = True
    pv.global_theme.multi_samples = 8
    plotter = pv.Plotter(off_screen=True, window_size=(int(args.width), int(args.height)))
    plotter.enable_depth_peeling(number_of_peels=24, occlusion_ratio=0.0)
    plotter.set_background("white")
    plotter.add_mesh(
        build_ground_shadow(bounds, span),
        scalars="Shadow_rgba",
        rgb=True,
        opacity=1.0,
        show_scalar_bar=False,
        lighting=False,
        use_transparency=True,
    )
    plotter.add_mesh(
        terrain_grid,
        scalars="Elevation",
        cmap=TERRAIN_CMAP,
        clim=(0.0, elev_display_max),
        show_scalar_bar=False,
        smooth_shading=True,
        split_sharp_edges=False,
        ambient=0.26,
        diffuse=0.58,
        specular=0.03,
        specular_power=10,
        pbr=False,
    )
    risk_surface = add_risk_overlay(
        plotter,
        z=z,
        risk=risk,
        resolution_m=resolution_m,
        stride=max(1, int(args.risk_stride)),
        vertical_exag=args.vertical_exag,
        smooth_sigma=args.smooth_sigma,
        draw_contour=not args.no_high_contour,
    )

    configure_reference_lights(plotter, (x_mid, y_mid, z_mid), span, bounds[5])
    plotter.enable_anti_aliasing("ssaa")
    plotter.enable_ssao(radius=0.14, bias=0.024, kernel_size=128, blur=True)

    if not args.hide_grid:
        actor = plotter.show_bounds(
            bounds=(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]),
            axes_ranges=(bounds[0], bounds[1], bounds[2], bounds[3], 0.0, elev_display_max),
            show_xlabels=True,
            show_ylabels=True,
            show_zlabels=True,
            xtitle="East–west distance (m)",
            ytitle="North–south distance (m)",
            ztitle="Elevation (m)",
            font_family="arial",
            font_size=26,
            fmt="%.0f",
            n_xlabels=5,
            n_ylabels=5,
            n_zlabels=5,
            grid="back",
            location="outer",
            ticks="outside",
            all_edges=True,
            corner_factor=1.0,
            padding=0.0,
            use_3d_text=True,
            color="black",
        )
        style_bounds_actor(actor)

    camera_pos = (
        (x_mid - 1.34 * span, y_mid - 1.12 * span, bounds[5] + 1.10 * span),
        (x_mid + 0.01 * span, y_mid + 0.00 * span, 0.32 * bounds[5]),
        (0.0, 0.0, 1.0),
    )
    plotter.camera_position = camera_pos
    plotter.camera.view_angle = 32.0
    plotter.camera.clipping_range = (10.0, 6.5 * span)
    plotter.camera.zoom(0.98)
    plotter.camera.SetWindowCenter(0.0, -0.12)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    peak_labels = []
    if not args.hide_peak_labels:
        plotter.render()
        peak_labels = project_peak_points(
            plotter,
            peak_world_points(
                config_or_path=args.scene,
                geo_path=geo_path,
                z=z,
                resolution_m=resolution_m,
                vertical_exag=args.vertical_exag,
                lift_scene_m=150.0,
            ),
            (int(args.width), int(args.height)),
        )
    image = Image.fromarray(plotter.screenshot(transparent_background=False, return_img=True)).convert("RGB")
    image = soften_terrain_backdrop(image)
    image = draw_projected_heatfield(image, plotter, risk_surface)
    if not args.no_high_contour:
        image = draw_projected_contours(image, plotter, risk_surface)
    if peak_labels:
        image = draw_peak_annotations(image, peak_labels, reference_width=float(args.width))
    plotter.close()
    image.save(args.out)

    if not args.hide_elevation_colorbar:
        draw_elevation_colorbar(args.out, elev_display_max)
    if not args.hide_risk_legend:
        draw_risk_legend(args.out)
    pdf_out = save_png_as_pdf(args.out, args.out.with_suffix(".pdf"), dpi=600)

    summary = {
        "scene": str(config.get("scene_name") or "huashan"),
        "dem": str(dem_path),
        "geo": str(geo_path),
        "osm": str(osm_path),
        "output": str(args.out),
        "pdf_output": str(pdf_out),
        "rendering": {
            "width": int(args.width),
            "height": int(args.height),
            "stride": int(args.stride),
            "risk_stride": int(args.risk_stride),
            "vertical_exag": float(args.vertical_exag),
            "risk_color_stops": RISK_COLOR_STOPS,
            "high_contour": not args.no_high_contour,
        },
        "risk_summary": risk_summary,
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return args.out


def main() -> None:
    args = parse_args()
    out = render_scene(args)
    print(f"[完成] Python 生成的华山人员暴露风险图：{out}")
    print(f"[完成] 风险场生成摘要：{args.summary_json}")


if __name__ == "__main__":
    main()
