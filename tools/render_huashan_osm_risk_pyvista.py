"""在华山 PyVista 三维地形上叠加 OSM 游客暴露风险要素。

脚本读取已有 DEM、经纬度网格和本地 OSM 文件，将 L1-L4 游客暴露风险直接映射到
三维地形表面。线状风险以 Tube 显示，点状风险以三维球体显示，避免二维专题图和三维
地形割裂。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from article_planner.scenario_config import load_scenario_config
from human_risk_osm import (
    LEVELS,
    _norm,
    apply_scene_risk_keywords,
    build_lonlat_tree,
    classify_level,
    dedup_lines,
    dedup_points,
    is_line_way,
    line_bbox_intersects,
    parse_osm,
    point_in_bbox,
)
from render_huashan_dem_pyvista import (
    build_ground_shadow,
    build_structured_grid,
    configure_reference_lights,
    read_resolution,
    style_bounds_actor,
)
from huashan_peak_annotations import draw_peak_annotations, peak_world_points, project_peak_points


DEFAULT_SCENE = PROJECT_ROOT / "scenarios" / "huashan.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "intermediate_artifacts" / "data" / "huashan"
DEFAULT_OUT = (
    PROJECT_ROOT
    / "intermediate_artifacts"
    / "figures"
    / "huashan"
    / "fig_2_1b_huashan_human_risk_pyvista_3d.png"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "intermediate_artifacts"
    / "figures"
    / "huashan"
    / "fig_2_1b_huashan_human_risk_pyvista_3d_summary.json"
)


GRAY_TERRAIN_CMAP = LinearSegmentedColormap.from_list(
    "huashan_gray_risk_backdrop",
    [
        (0.00, "#2E2E2E"),
        (0.22, "#555555"),
        (0.45, "#777777"),
        (0.68, "#9B9B9B"),
        (0.86, "#BEBEBE"),
        (1.00, "#D8D8D8"),
    ],
)


RISK_STYLES = {
    1: {
        "label": "L1 High-risk trails",
        "color": "#D13B2F",
        "tube_radius": 80.0,
        "point_radius": 100.0,
        "z_offset_m": 120.0,
    },
    2: {
        "label": "L2 Peaks and scenic paths",
        "color": "#F0A51A",
        "tube_radius": 60.0,
        "point_radius": 80.0,
        "z_offset_m": 100.0,
    },
    3: {
        "label": "L3 Service hotspots",
        "color": "#008C95",
        "tube_radius": 60.0,
        "point_radius": 100.0,
        "z_offset_m": 120.0,
    },
    4: {
        "label": "L4 Access roads",
        "color": "#8E3DA8",
        "tube_radius": 80.0,
        "point_radius": 80.0,
        "z_offset_m": 100.0,
    },
}


@dataclass
class RiskFeatureSet:
    lines: Dict[int, List[List[Tuple[float, float]]]]
    points: Dict[int, List[Tuple[float, float]]]
    names: Dict[int, List[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在华山三维 DEM 上叠加 OSM L1-L4 游客暴露风险。")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="场景配置 JSON。")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="华山中间数据目录。")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出 PNG 路径。")
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY, help="输出风险要素摘要 JSON。")
    parser.add_argument("--stride", type=int, default=2, help="地形网格采样步长。")
    parser.add_argument("--vertical-exag", type=float, default=1.45, help="垂向夸张系数。")
    parser.add_argument("--smooth-sigma", type=float, default=0.45, help="地形显示平滑系数。")
    parser.add_argument("--line-sample-cells", type=float, default=4.0, help="风险线贴地采样间隔，单位为 DEM 栅格。")
    parser.add_argument("--width", type=int, default=3600, help="输出图像宽度。")
    parser.add_argument("--height", type=int, default=2200, help="输出图像高度。")
    parser.add_argument("--hide-grid", action="store_true", help="隐藏盒式坐标网格。")
    parser.add_argument("--hide-scalar-bar", action="store_true", help="隐藏高程色标。")
    parser.add_argument("--hide-legend", action="store_true", help="隐藏风险等级图例。")
    parser.add_argument("--hide-peak-labels", action="store_true", help="隐藏华山五峰单字母标注。")
    return parser.parse_args()


def resolve_osm_path(config: dict) -> Path:
    raw = Path(str(config.get("osm_file") or "data/raw/huashan/map.osm"))
    if raw.is_absolute():
        return raw
    return PROJECT_ROOT / raw


def extract_osm_risk_features(config: dict, osm_path: Path, lon_grid: np.ndarray, lat_grid: np.ndarray) -> RiskFeatureSet:
    """复用二维风险图的分类规则，从 OSM 提取 L1-L4 点线要素。"""
    apply_scene_risk_keywords(config)
    lon_min, lon_max = float(np.nanmin(lon_grid)), float(np.nanmax(lon_grid))
    lat_min, lat_max = float(np.nanmin(lat_grid)), float(np.nanmax(lat_grid))
    nodes, tagged_nodes, ways, _ = parse_osm(osm_path)

    lines: Dict[int, List[List[Tuple[float, float]]]] = {level: [] for level in LEVELS}
    points: Dict[int, List[Tuple[float, float]]] = {level: [] for level in LEVELS}
    names: Dict[int, List[str]] = {level: [] for level in LEVELS}

    for node_id, tags in tagged_nodes:
        level = classify_level(tags)
        if level not in LEVELS or node_id not in nodes:
            continue
        lon, lat = nodes[node_id]
        if not point_in_bbox(lon, lat, lon_min, lon_max, lat_min, lat_max):
            continue
        points[level].append((lon, lat))
        name = _norm(tags.get("name"))
        if name:
            names[level].append(name)

    for way in ways:
        level = classify_level(way.tags)
        if level not in LEVELS:
            continue
        coords = [nodes[ref] for ref in way.refs if ref in nodes]
        if not coords or not line_bbox_intersects(coords, lon_min, lon_max, lat_min, lat_max):
            continue

        name = _norm(way.tags.get("name"))
        if name:
            names[level].append(name)

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

    return RiskFeatureSet(
        lines={level: dedup_lines(lines[level]) for level in LEVELS},
        points={level: dedup_points(points[level]) for level in LEVELS},
        names={level: sorted(set(names[level])) for level in LEVELS},
    )


def rc_to_scene_point(
    row: int,
    col: int,
    z: np.ndarray,
    resolution_m: float,
    vertical_exag: float,
    z_min: float,
    z_offset_m: float,
) -> tuple[float, float, float, float]:
    """把 DEM 行列号转换为与三维地形一致的场景坐标。"""
    rows, cols = z.shape
    row = int(np.clip(row, 0, rows - 1))
    col = int(np.clip(col, 0, cols - 1))
    elev = float(z[row, col])
    x = (float(col) - 0.5 * float(cols - 1)) * resolution_m
    y = (float(rows - 1 - row) - 0.5 * float(rows - 1)) * resolution_m
    z_scene = (elev - z_min + z_offset_m) * vertical_exag
    return x, y, z_scene, elev


def lonlat_to_scene_points(
    lonlat: Sequence[Tuple[float, float]],
    tree,
    rows: int,
    cols: int,
    z: np.ndarray,
    resolution_m: float,
    vertical_exag: float,
    z_min: float,
    z_offset_m: float,
) -> np.ndarray:
    if not lonlat:
        return np.empty((0, 3), dtype=float)
    query = np.asarray(lonlat, dtype=float)
    _, idx = tree.query(query, k=1)
    idx = np.asarray(idx, dtype=int)
    rr = idx // cols
    cc = idx % cols
    pts = [
        rc_to_scene_point(int(r), int(c), z, resolution_m, vertical_exag, z_min, z_offset_m)[:3]
        for r, c in zip(rr, cc)
    ]
    return np.asarray(pts, dtype=float)


def resample_line_by_dem_cells(
    line: Sequence[Tuple[float, float]],
    tree,
    rows: int,
    cols: int,
    z: np.ndarray,
    resolution_m: float,
    vertical_exag: float,
    z_min: float,
    z_offset_m: float,
    sample_cells: float,
) -> np.ndarray:
    """将 OSM 折线按 DEM 栅格重采样，使 Tube 沿地形起伏而不是悬成直线。"""
    if len(line) < 2:
        return np.empty((0, 3), dtype=float)

    query = np.asarray(line, dtype=float)
    _, idx = tree.query(query, k=1)
    idx = np.asarray(idx, dtype=int)
    rc = np.column_stack([idx // cols, idx % cols]).astype(float)

    sampled: list[tuple[int, int]] = []
    step_cells = max(1.0, float(sample_cells))
    for i in range(len(rc) - 1):
        r0, c0 = rc[i]
        r1, c1 = rc[i + 1]
        dist = max(abs(r1 - r0), abs(c1 - c0))
        count = max(2, int(np.ceil(dist / step_cells)) + 1)
        rr = np.linspace(r0, r1, count)
        cc = np.linspace(c0, c1, count)
        for r, c in zip(rr, cc):
            sampled.append((int(np.rint(r)), int(np.rint(c))))

    deduped: list[tuple[int, int]] = []
    for r, c in sampled:
        item = (int(np.clip(r, 0, rows - 1)), int(np.clip(c, 0, cols - 1)))
        if not deduped or item != deduped[-1]:
            deduped.append(item)

    if len(deduped) < 2:
        return np.empty((0, 3), dtype=float)

    pts = [
        rc_to_scene_point(r, c, z, resolution_m, vertical_exag, z_min, z_offset_m)[:3]
        for r, c in deduped
    ]
    return np.asarray(pts, dtype=float)


def build_polyline_mesh(lines: Sequence[np.ndarray]) -> pv.PolyData | None:
    """把多条三维折线合并为一个 PolyData，供 Tube 滤波使用。"""
    valid = [line for line in lines if line.shape[0] >= 2]
    if not valid:
        return None

    points: list[np.ndarray] = []
    cells: list[int] = []
    offset = 0
    for line in valid:
        n = int(line.shape[0])
        points.append(line)
        cells.extend([n, *range(offset, offset + n)])
        offset += n

    mesh = pv.PolyData(np.vstack(points))
    mesh.lines = np.asarray(cells, dtype=np.int64)
    return mesh


def add_risk_overlays(
    plotter: pv.Plotter,
    features: RiskFeatureSet,
    z: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    resolution_m: float,
    vertical_exag: float,
    line_sample_cells: float,
) -> dict:
    """添加 L1-L4 风险 Tube 和点球标，并返回摘要信息。"""
    tree, rows, cols = build_lonlat_tree(lon_grid, lat_grid)
    z_min = float(np.nanmin(z))
    summary: dict = {"levels": {}, "point_records": {}}

    for level in LEVELS:
        style = RISK_STYLES[level]
        line_points = [
            resample_line_by_dem_cells(
                line,
                tree,
                rows,
                cols,
                z,
                resolution_m,
                vertical_exag,
                z_min,
                float(style["z_offset_m"]),
                line_sample_cells,
            )
            for line in features.lines[level]
        ]
        line_mesh = build_polyline_mesh(line_points)
        rendered_line_count = 0
        if line_mesh is not None:
            tube = line_mesh.tube(radius=float(style["tube_radius"]), n_sides=22, capping=True)
            plotter.add_mesh(
                tube,
                color=str(style["color"]),
                show_scalar_bar=False,
                smooth_shading=True,
                lighting=False,
                ambient=1.0,
                diffuse=0.0,
                specular=0.0,
                emissive=True,
                label=str(style["label"]),
            )
            rendered_line_count = len([line for line in line_points if line.shape[0] >= 2])

        pts = lonlat_to_scene_points(
            features.points[level],
            tree,
            rows,
            cols,
            z,
            resolution_m,
            vertical_exag,
            z_min,
            float(style["z_offset_m"]) + 18.0,
        )
        if pts.size:
            point_cloud = pv.PolyData(pts)
            sphere = pv.SphereSource(
                radius=float(style["point_radius"]),
                theta_resolution=22,
                phi_resolution=14,
            ).output
            glyphs = point_cloud.glyph(geom=sphere, scale=False, orient=False)
            plotter.add_mesh(
                glyphs,
                color=str(style["color"]),
                show_scalar_bar=False,
                smooth_shading=True,
                lighting=False,
                ambient=1.0,
                diffuse=0.0,
                specular=0.0,
                emissive=True,
            )

        summary["levels"][f"L{level}"] = {
            "lines_in_osm": len(features.lines[level]),
            "lines_rendered": rendered_line_count,
            "points_in_osm": len(features.points[level]),
            "points_rendered": int(pts.shape[0]) if pts.size else 0,
            "named_features": features.names[level],
            "style": {
                "color": style["color"],
                "tube_radius_m": style["tube_radius"],
                "point_radius_m": style["point_radius"],
                "z_offset_m": style["z_offset_m"],
            },
        }

    return summary


def load_legend_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """优先使用 Times New Roman，保证图例和坐标轴风格一致。"""
    candidates = [
        Path("C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/timesbd.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_academic_legend(image_path: Path) -> None:
    """绘制黑字白底边框图例，避免 PyVista 内置图例把文字染成类别颜色。"""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, _ = image.size
    scale = width / 3600.0

    font = load_legend_font(max(34, int(52 * scale)))
    margin = int(86 * scale)
    pad_x = int(28 * scale)
    pad_y = int(24 * scale)
    row_h = int(66 * scale)
    swatch_w = int(56 * scale)
    swatch_h = int(30 * scale)
    gap = int(20 * scale)

    entries = [(str(RISK_STYLES[level]["label"]), str(RISK_STYLES[level]["color"])) for level in LEVELS]
    text_width = 0
    text_height = 0
    for label, _ in entries:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = max(text_width, bbox[2] - bbox[0])
        text_height = max(text_height, bbox[3] - bbox[1])

    panel_w = pad_x * 2 + swatch_w + gap + text_width
    panel_h = pad_y * 2 + row_h * len(entries)
    x0 = margin + int(64 * scale)
    y0 = margin
    x1 = x0 + panel_w
    y1 = y0 + panel_h
    draw.rectangle((x0, y0, x1, y1), fill="white", outline="#222222", width=max(2, int(2 * scale)))

    for idx, (label, color) in enumerate(entries):
        y = y0 + pad_y + idx * row_h + (row_h - swatch_h) // 2
        sx0 = x0 + pad_x
        sy0 = y
        sx1 = sx0 + swatch_w
        sy1 = sy0 + swatch_h
        draw.rectangle((sx0, sy0, sx1, sy1), fill=color, outline=color)
        text_y = y0 + pad_y + idx * row_h + (row_h - text_height) // 2 - int(3 * scale)
        draw.text((sx1 + gap, text_y), label, fill="black", font=font)

    image.save(image_path)


def draw_elevation_colorbar(image_path: Path, elev_max: float) -> None:
    """在 PNG 输出上叠加清晰的高程色标，避免三维渲染缩放后文字过细。"""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, _ = image.size
    scale = width / 3600.0

    title_font = load_legend_font(max(28, int(42 * scale)))
    label_font = load_legend_font(max(24, int(36 * scale)))
    bar_w = int(50 * scale)
    bar_h = int(1420 * scale)
    x0 = width - int(315 * scale)
    y0 = int(245 * scale)
    x1 = x0 + bar_w
    y1 = y0 + bar_h
    pad = int(18 * scale)
    label_gap = int(18 * scale)

    panel = (x0 - pad, y0 - int(78 * scale), x1 + int(118 * scale), y1 + pad)
    draw.rectangle(panel, fill="white")

    for yy in range(bar_h):
        t = 1.0 - yy / max(1, bar_h - 1)
        gray = int(round(46 + (216 - 46) * t))
        draw.line((x0, y0 + yy, x1, y0 + yy), fill=(gray, gray, gray), width=1)
    draw.rectangle((x0, y0, x1, y1), outline="#555555", width=max(1, int(1.2 * scale)))

    title = "Elevation (m)"
    tb = draw.textbbox((0, 0), title, font=title_font)
    draw.text((x0 + bar_w / 2 - (tb[2] - tb[0]) / 2, y0 - int(58 * scale)), title, fill="black", font=title_font)

    ticks = np.linspace(0.0, float(elev_max), 7)
    for value in ticks:
        pos = y1 - (value / max(float(elev_max), 1e-6)) * bar_h
        draw.line((x1, pos, x1 + int(12 * scale), pos), fill="black", width=max(2, int(2 * scale)))
        label = f"{value:.0f}"
        lb = draw.textbbox((0, 0), label, font=label_font)
        draw.text((x1 + label_gap, pos - (lb[3] - lb[1]) / 2), label, fill="black", font=label_font)

    image.save(image_path)


def render_scene(args: argparse.Namespace) -> Path:
    config = load_scenario_config(args.scene, workdir=PROJECT_ROOT)
    data_dir = args.data_dir
    dem_path = data_dir / "Z_crop.npy"
    meta_path = data_dir / "Z_crop_meta.json"
    geo_path = data_dir / "Z_crop_geo.npz"
    osm_path = resolve_osm_path(config)
    if not dem_path.exists():
        raise FileNotFoundError(f"缺少 DEM 缓存：{dem_path}")
    if not geo_path.exists():
        raise FileNotFoundError(f"缺少经纬度网格：{geo_path}")
    if not osm_path.exists():
        raise FileNotFoundError(f"缺少 OSM 文件：{osm_path}")

    z = np.asarray(np.load(dem_path), dtype=float)
    geo = np.load(geo_path)
    lon_grid = np.asarray(geo["lon_grid"], dtype=float)
    lat_grid = np.asarray(geo["lat_grid"], dtype=float)
    resolution_m = read_resolution(meta_path)
    features = extract_osm_risk_features(config, osm_path, lon_grid, lat_grid)

    grid = build_structured_grid(z, resolution_m, args.stride, args.vertical_exag, args.smooth_sigma)
    bounds = grid.bounds
    x_mid = 0.5 * (bounds[0] + bounds[1])
    y_mid = 0.5 * (bounds[2] + bounds[3])
    z_mid = 0.45 * bounds[5]
    span = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    elev_display_max = float(np.nanpercentile(z, 99.7))

    pv.global_theme.smooth_shading = True
    pv.global_theme.multi_samples = 8
    plotter = pv.Plotter(off_screen=True, window_size=(int(args.width), int(args.height)))
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
        grid,
        scalars="Elevation",
        cmap=GRAY_TERRAIN_CMAP,
        clim=(0.0, elev_display_max),
        show_scalar_bar=False,
        scalar_bar_args={
            "title": "Elevation (m)",
            "color": "black",
            "title_font_size": 22,
            "label_font_size": 18,
            "font_family": "times",
            "vertical": True,
            "position_x": 0.80,
            "position_y": 0.15,
            "width": 0.030,
            "height": 0.70,
            "fmt": "%.0f",
            "outline": True,
            "n_labels": 7,
        },
        smooth_shading=True,
        split_sharp_edges=False,
        ambient=0.12,
        diffuse=0.62,
        specular=0.02,
        specular_power=10,
        pbr=False,
    )

    overlay_summary = add_risk_overlays(
        plotter,
        features,
        z,
        lon_grid,
        lat_grid,
        resolution_m,
        args.vertical_exag,
        args.line_sample_cells,
    )

    configure_reference_lights(plotter, (x_mid, y_mid, z_mid), span, bounds[5])
    plotter.enable_anti_aliasing("ssaa")
    plotter.enable_ssao(radius=0.22, bias=0.012, kernel_size=256, blur=True)

    if not args.hide_grid:
        actor = plotter.show_bounds(
            bounds=(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]),
            axes_ranges=(bounds[0], bounds[1], bounds[2], bounds[3], 0.0, elev_display_max),
            show_xlabels=True,
            show_ylabels=True,
            show_zlabels=True,
            xtitle="East-West (m)",
            ytitle="North-South (m)",
            ztitle="Elevation (m)",
            font_family="times",
            font_size=28,
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
    plotter.camera.zoom(0.90)

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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(plotter.screenshot(transparent_background=False, return_img=True)).convert("RGB")
    plotter.close()
    if peak_labels:
        image = draw_peak_annotations(image, peak_labels, reference_width=float(args.width))
    image.save(args.out)
    if not args.hide_scalar_bar:
        draw_elevation_colorbar(args.out, elev_display_max)
    if not args.hide_legend:
        draw_academic_legend(args.out)

    summary = {
        "scene": str(config.get("scene_name") or "huashan"),
        "osm_file": str(osm_path),
        "dem": str(dem_path),
        "output": str(args.out),
        "vertical_exag": float(args.vertical_exag),
        "line_sample_cells": float(args.line_sample_cells),
        "elevation_colorbar_max_m": elev_display_max,
        **overlay_summary,
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return args.out


def main() -> None:
    args = parse_args()
    out = render_scene(args)
    print(f"[完成] 华山 OSM 游客暴露风险三维叠加图：{out}")
    print(f"[完成] 风险要素摘要：{args.summary_json}")


if __name__ == "__main__":
    main()
