"""生成本轮论文结构调整所需的图表和正文片段。

脚本只读取已有的场景中间数据与正式实验 CSV，不重新运行 benchmark。
环境与方法构建图同步写入 intermediate_artifacts，第四章正式结果图表写入 final_results。
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LightSource, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from article_planner.scenario_config import load_scenario_config, resolve_resolution_m, scenario_output_dir
from human_risk_osm import apply_scene_risk_keywords, classify_level, is_line_way, parse_osm

try:
    from tools.huashan_peak_annotations import (
        MAP_OFFSETS,
        draw_peak_annotations,
        map_peak_screen_points,
        peak_world_points,
        project_peak_points,
    )
except ModuleNotFoundError:
    from huashan_peak_annotations import (
        MAP_OFFSETS,
        draw_peak_annotations,
        map_peak_screen_points,
        peak_world_points,
        project_peak_points,
    )


SCENE_CONFIGS = {
    "huashan": "scenarios/huashan.json",
    "huangshan": "scenarios/huangshan.json",
    "emeishan": "scenarios/emeishan.json",
}
SCENE_NAMES_CN = {
    "huashan": "华山",
    "huangshan": "黄山",
    "emeishan": "峨眉山",
}
METHOD_ORDER = ["M-P", "M-A", "M-F", "M-R", "M-V"]
METHOD_TO_BASELINE = {
    "M-P": "B4_Proposed_LPA_Layered",
    "M-A": "B2_GlobalAstar_Layered",
    "M-F": "B3_LPA_SingleLayer",
    "M-R": "B5_RegularLayered_LPA",
    "M-V": "B1_Voxel_Dijkstra",
}
BASELINE_TO_METHOD = {v: k for k, v in METHOD_TO_BASELINE.items()}
BASELINE_TO_METHOD["B6_RegularLayered_LPA"] = "M-R"
METHOD_COLORS = {
    "M-P": "#D55E00",
    "M-A": "#0072B2",
    "M-F": "#6A51A3",
    "M-R": "#009E73",
    "M-V": "#8C510A",
}
LAYER_COLORS = ["#2B7BBA", "#2CA25F", "#E69F00"]
LAYER_NAMES_CN = ["任务端点层", "区域支路层", "骨干通行层"]


@dataclass
class SceneArrays:
    """单个场景的中间结果数组。"""

    name: str
    cfg: dict
    data_dir: Path
    figure_dir: Path
    resolution_m: float
    z: np.ndarray
    risk_human: np.ndarray
    risk_l1: np.ndarray | None
    risk_l2: np.ndarray | None
    risk_l3: np.ndarray | None
    risk_l4: np.ndarray | None
    risk_comm: np.ndarray
    comm_summary: dict
    floor: np.ndarray
    ceiling: np.ndarray
    layer_mid: np.ndarray
    layer_allowed: np.ndarray
    nodes: np.ndarray
    edges: np.ndarray
    terminal_status: dict
    tasks: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成论文图表结构调整后的正式资产。")
    parser.add_argument("--workdir", type=str, default=".", help="项目根目录。")
    parser.add_argument("--scenario-config", type=str, default="scenarios/huashan.json", help="用于第二章和第三章示例图的场景配置。")
    parser.add_argument("--out-dir", type=str, default="final_results/paper_revision", help="修订图表和正文片段输出目录。")
    parser.add_argument("--dpi", type=int, default=600, help="PNG 输出分辨率。")
    parser.add_argument("--max-surface-points", type=int, default=96, help="三维曲面单边最大采样点数。")
    return parser.parse_args()


def configure_matplotlib() -> None:
    """设置接近论文模板的基础绘图风格。"""
    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.8,
        }
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path, required: bool = True) -> list[dict]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"缺少输入文件：{path}")
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def to_float(value: object, default: float = float("nan")) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def to_int(value: object, default: int = 0) -> int:
    val = to_float(value)
    return int(round(val)) if math.isfinite(val) else default


def choose_stride(rows: int, cols: int, max_surface_points: int) -> int:
    return max(1, int(math.ceil(max(rows, cols) / max(24, max_surface_points))))


def xy_grids(rows: int, cols: int, resolution_m: float, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(0, cols, stride, dtype=float) * resolution_m / 1000.0
    y = np.arange(rows - 1, -1, -stride, dtype=float) * resolution_m / 1000.0
    return np.meshgrid(x, y)


def map_extent(arr: np.ndarray, resolution_m: float) -> list[float]:
    rows, cols = arr.shape
    return [0.0, cols * resolution_m / 1000.0, 0.0, rows * resolution_m / 1000.0]


def save_figure_pair(fig: plt.Figure, out_dir: Path, basename: str, dpi: int) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{basename}.pdf"
    png_path = out_dir / f"{basename}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def load_scene_arrays(root: Path, scenario_config: str, max_surface_points: int) -> SceneArrays:
    cfg = load_scenario_config(scenario_config, root)
    scene_name = str(cfg.get("scene_name", "default"))
    data_dir = scenario_output_dir(cfg, root)
    figure_dir = root / "intermediate_artifacts" / "figures" / scene_name
    figure_dir.mkdir(parents=True, exist_ok=True)
    required = [
        "Z_crop.npy",
        "risk_human.npy",
        "risk_comm.npy",
        "floor.npy",
        "ceiling.npy",
        "layer_mid.npy",
        "layer_allowed.npy",
        "graph_nodes.npy",
        "graph_edges.npy",
        "graph_terminal_status.json",
    ]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{scene_name} 缺少中间结果文件：{', '.join(missing)}")

    task_path = data_dir / "generated_tasks.json"
    tasks: list[dict] = []
    if task_path.exists():
        tasks = [dict(item) for item in read_json(task_path).get("tasks", [])]

    def optional_array(name: str) -> np.ndarray | None:
        path = data_dir / name
        if not path.exists():
            return None
        return np.asarray(np.load(path), dtype=float)

    comm_summary_path = data_dir / "communication_summary.json"

    return SceneArrays(
        name=scene_name,
        cfg=cfg,
        data_dir=data_dir,
        figure_dir=figure_dir,
        resolution_m=resolve_resolution_m(cfg, data_dir),
        z=np.asarray(np.load(data_dir / "Z_crop.npy"), dtype=float),
        risk_human=np.asarray(np.load(data_dir / "risk_human.npy"), dtype=float),
        risk_l1=optional_array("risk_l1.npy"),
        risk_l2=optional_array("risk_l2.npy"),
        risk_l3=optional_array("risk_l3.npy"),
        risk_l4=optional_array("risk_l4.npy"),
        risk_comm=np.asarray(np.load(data_dir / "risk_comm.npy"), dtype=float),
        comm_summary=read_json(comm_summary_path) if comm_summary_path.exists() else {},
        floor=np.asarray(np.load(data_dir / "floor.npy"), dtype=float),
        ceiling=np.asarray(np.load(data_dir / "ceiling.npy"), dtype=float),
        layer_mid=np.asarray(np.load(data_dir / "layer_mid.npy"), dtype=float),
        layer_allowed=np.asarray(np.load(data_dir / "layer_allowed.npy"), dtype=bool),
        nodes=np.asarray(np.load(data_dir / "graph_nodes.npy"), dtype=float),
        edges=np.asarray(np.load(data_dir / "graph_edges.npy"), dtype=int),
        terminal_status=read_json(data_dir / "graph_terminal_status.json"),
        tasks=tasks,
    )


def style_map_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left")
    ax.set_xlabel("东西向距离，km")
    ax.set_ylabel("南北向距离，km")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def style_3d_axis(ax, data: SceneArrays, title: str, elev: float = 34.0, azim: float = -128.0) -> None:
    rows, cols = data.z.shape
    ax.set_xlim(0.0, cols * data.resolution_m / 1000.0)
    ax.set_ylim(0.0, rows * data.resolution_m / 1000.0)
    ax.set_xlabel("东西向，km", labelpad=7)
    ax.set_ylabel("南北向，km", labelpad=7)
    ax.set_zlabel("高程，m", labelpad=7)
    ax.set_title(title, loc="left", pad=10)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.0, 1.0, 0.42))
    ax.xaxis.pane.set_alpha(0.02)
    ax.yaxis.pane.set_alpha(0.02)
    ax.zaxis.pane.set_alpha(0.02)
    ax.grid(True, alpha=0.22)


def style_publication_terrain_axis(ax, data: SceneArrays, title: str, elev: float = 34.0, azim: float = -128.0) -> None:
    """为论文地形渲染图设置极简三维坐标轴。"""
    rows, cols = data.z.shape
    x_max = cols * data.resolution_m / 1000.0
    y_max = rows * data.resolution_m / 1000.0
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel("东西向距离，km", labelpad=5)
    ax.set_ylabel("南北向距离，km", labelpad=5)
    ax.set_zlabel("高程，m", labelpad=5)
    ax.set_title("")
    ax.text2D(0.03, 0.95, title, transform=ax.transAxes, fontsize=10.8, fontweight="semibold")
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.0, 1.0, 0.34), zoom=1.36)
    ax.set_xticks(np.arange(0.0, x_max + 0.1, 2.0))
    ax.set_yticks(np.arange(0.0, y_max + 0.1, 2.0))
    ax.tick_params(axis="both", which="major", pad=1, labelsize=7.8, colors="#1E1E1E")
    ax.tick_params(axis="z", which="major", pad=1, labelsize=7.8, colors="#1E1E1E")
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 0.0)
        axis._axinfo["grid"]["linewidth"] = 0.0
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.18
        try:
            axis.line.set_color("#1A1A1A")
            axis.line.set_linewidth(0.72)
        except Exception:
            pass


def style_path_3d_axis(ax, data: SceneArrays, title: str, path_nodes: Sequence[int]) -> None:
    """为路径后处理图设置紧凑三维视角，只显示路径邻域。"""
    pts = data.nodes[list(path_nodes), :3]
    min_xy = np.min(pts[:, :2], axis=0)
    max_xy = np.max(pts[:, :2], axis=0)
    center = 0.5 * (min_xy + max_xy)
    span = max(float(np.max(max_xy - min_xy)), 1.6)
    half = 0.5 * span + 0.45
    ax.set_xlim(max(0.0, center[0] - half), min(data.z.shape[1] * data.resolution_m / 1000.0, center[0] + half))
    ax.set_ylim(max(0.0, center[1] - half), min(data.z.shape[0] * data.resolution_m / 1000.0, center[1] + half))
    z_min = max(float(np.min(data.z)), float(np.min(pts[:, 2])) - 260.0)
    z_max = float(np.max(pts[:, 2])) + 220.0
    ax.set_zlim(z_min, z_max)
    ax.set_title(title, loc="left", pad=2, fontsize=9.0)
    ax.view_init(elev=30, azim=-124)
    ax.set_box_aspect((1.25, 1.0, 0.36))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)


def plot_surface(
    ax,
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    *,
    antialiased: bool = False,
    shade: bool = False,
    **kwargs,
):
    surface = ax.plot_surface(xx, yy, zz, linewidth=0, antialiased=antialiased, shade=shade, **kwargs)
    try:
        surface.set_rasterized(True)
    except Exception:
        pass
    return surface


TERRAIN_REALISTIC_CMAP = LinearSegmentedColormap.from_list(
    "huashan_realistic_terrain",
    [
        (0.00, "#1D4E3A"),
        (0.18, "#3F7D3A"),
        (0.38, "#8A8F4B"),
        (0.58, "#B49A62"),
        (0.76, "#8B8173"),
        (0.90, "#C4BDB1"),
        (1.00, "#F2F0E8"),
    ],
)


def shaded_terrain_rgb(z: np.ndarray, resolution_m: float, vert_exag: float = 1.75) -> np.ndarray:
    """生成带光照和自然地貌色带的地形贴图。"""
    z_clean = np.nan_to_num(z, nan=float(np.nanmedian(z)))
    z_relief = gaussian_filter(z_clean, sigma=0.65, mode="nearest")
    light = LightSource(azdeg=315, altdeg=40)
    rgb = light.shade(
        z_relief,
        cmap=TERRAIN_REALISTIC_CMAP,
        vert_exag=vert_exag,
        dx=resolution_m,
        dy=resolution_m,
        blend_mode="soft",
        fraction=1.22,
    )
    fill = LightSource(azdeg=120, altdeg=20).hillshade(
        z_relief,
        vert_exag=1.10,
        dx=resolution_m,
        dy=resolution_m,
    )
    rgb[..., :3] = np.clip(rgb[..., :3] * (0.95 + 0.08 * fill[..., None]), 0.0, 1.0)
    rgb[..., 3] = 1.0
    return rgb


def hillshade_gray(z: np.ndarray, resolution_m: float, vert_exag: float = 1.25) -> np.ndarray:
    """生成二维风险图使用的灰度地形阴影底图。"""
    light = LightSource(azdeg=315, altdeg=46)
    z_clean = np.nan_to_num(z, nan=float(np.nanmedian(z)))
    shade = light.hillshade(z_clean, vert_exag=vert_exag, dx=resolution_m, dy=resolution_m)
    return np.clip(0.20 + 0.78 * shade, 0.0, 1.0)


def map_xy_mesh(data: SceneArrays) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = data.z.shape
    x = np.linspace(0.0, cols * data.resolution_m / 1000.0, cols)
    y = np.linspace(0.0, rows * data.resolution_m / 1000.0, rows)
    return np.meshgrid(x, y)


def outline_risk_level(
    ax: plt.Axes,
    data: SceneArrays,
    risk: np.ndarray | None,
    level: float,
    color: str,
    label: str,
    linewidth: float = 1.25,
) -> Line2D | None:
    if risk is None or float(np.nanmax(risk)) < level:
        return None
    xx, yy = map_xy_mesh(data)
    ax.contour(xx, yy, np.flipud(risk), levels=[level], colors=[color], linewidths=linewidth, alpha=0.96)
    return Line2D([0], [0], color=color, lw=2.0, label=label)


def annotate_sources(ax: plt.Axes, data: SceneArrays, max_range_km: float) -> list[Line2D]:
    """在通信图上绘制通信源和理论覆盖半径。"""
    sources = list(data.comm_summary.get("sources", []))
    handles: dict[str, Line2D] = {}
    rows, cols = data.z.shape
    x_min, x_max = 0.0, cols * data.resolution_m / 1000.0
    y_min, y_max = 0.0, rows * data.resolution_m / 1000.0
    theta = np.linspace(0.0, 2.0 * math.pi, 361)
    for item in sources:
        x = float(item.get("x_km", np.nan))
        y = float(item.get("y_km", np.nan))
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        source_type = str(item.get("source", ""))
        is_depot = source_type == "virtual_depot"
        color = "#C62828" if is_depot else "#1565C0"
        marker = "s" if is_depot else "^"
        label = "Delivery Station (Main BS)" if is_depot else "Edge Relay Node"
        ax.scatter(
            [x],
            [y],
            s=34,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=0.65,
            zorder=8,
        )
        arc_x = x + max_range_km * np.cos(theta)
        arc_y = y + max_range_km * np.sin(theta)
        inside = (arc_x >= x_min) & (arc_x <= x_max) & (arc_y >= y_min) & (arc_y <= y_max)
        ax.plot(
            np.where(inside, arc_x, np.nan),
            np.where(inside, arc_y, np.nan),
            color=color,
            lw=0.78,
            ls=(0, (4, 3)),
            alpha=0.34,
            zorder=3,
            clip_on=True,
        )
        handles.setdefault(
            label,
            Line2D([0], [0], marker=marker, color="none", markerfacecolor=color, markeredgecolor="white", markersize=7, label=label),
        )
    return list(handles.values())


OSM_RISK_STYLES = {
    1: {"dark": "#C62828", "light": "#FFCDD2", "label": "L1 危险路段", "marker": "X"},
    2: {"dark": "#EF6C00", "light": "#FFE0B2", "label": "L2 主游线和峰顶", "marker": "^"},
    3: {"dark": "#D9A300", "light": "#FFF3B0", "label": "L3 服务和停留点", "marker": "s"},
    4: {"dark": "#2E7D32", "light": "#C8E6C9", "label": "L4 低风险道路", "marker": "o"},
}


def geo_bounds(data: SceneArrays) -> tuple[float, float, float, float]:
    geo_path = data.data_dir / "Z_crop_geo.npz"
    if geo_path.exists():
        geo = np.load(geo_path)
        lon_grid = np.asarray(geo["lon_grid"], dtype=float)
        lat_grid = np.asarray(geo["lat_grid"], dtype=float)
        return (
            float(np.nanmin(lon_grid)),
            float(np.nanmax(lon_grid)),
            float(np.nanmin(lat_grid)),
            float(np.nanmax(lat_grid)),
        )
    summary_path = data.data_dir / "osm_feature_summary.json"
    if summary_path.exists():
        bbox = read_json(summary_path).get("bbox_wgs84", {})
        return (
            float(bbox["lon_min"]),
            float(bbox["lon_max"]),
            float(bbox["lat_min"]),
            float(bbox["lat_max"]),
        )
    raise FileNotFoundError("缺少用于OSM坐标转换的经纬度边界文件。")


def lonlat_to_scene_xy(data: SceneArrays, lon: float, lat: float, bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    lon_min, lon_max, lat_min, lat_max = bounds
    rows, cols = data.z.shape
    width_km = cols * data.resolution_m / 1000.0
    height_km = rows * data.resolution_m / 1000.0
    x = (lon - lon_min) / max(lon_max - lon_min, 1e-12) * width_km
    y = (lat - lat_min) / max(lat_max - lat_min, 1e-12) * height_km
    return float(x), float(y)


def load_osm_risk_features(data: SceneArrays) -> tuple[dict[int, list[list[tuple[float, float]]]], dict[int, list[tuple[float, float]]]]:
    """读取真实OSM线要素，避免用缓冲栅格轮廓造成粗线效果。"""
    osm_file = data.cfg.get("osm_file")
    if not osm_file:
        return {lv: [] for lv in OSM_RISK_STYLES}, {lv: [] for lv in OSM_RISK_STYLES}
    osm_path = Path(str(osm_file))
    if not osm_path.is_absolute():
        osm_path = PROJECT_ROOT / osm_path
    if not osm_path.exists():
        return {lv: [] for lv in OSM_RISK_STYLES}, {lv: [] for lv in OSM_RISK_STYLES}

    apply_scene_risk_keywords(data.cfg)
    nodes, tagged_nodes, ways, _ = parse_osm(osm_path)
    bounds = geo_bounds(data)
    lon_min, lon_max, lat_min, lat_max = bounds
    lines: dict[int, list[list[tuple[float, float]]]] = {lv: [] for lv in OSM_RISK_STYLES}
    points: dict[int, list[tuple[float, float]]] = {lv: [] for lv in OSM_RISK_STYLES}

    def in_bounds(lon: float, lat: float) -> bool:
        return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max

    for node_id, tags in tagged_nodes:
        lv = classify_level(tags)
        if lv not in OSM_RISK_STYLES or node_id not in nodes:
            continue
        lon, lat = nodes[node_id]
        if in_bounds(lon, lat):
            points[lv].append(lonlat_to_scene_xy(data, lon, lat, bounds))

    for way in ways:
        lv = classify_level(way.tags)
        if lv not in OSM_RISK_STYLES:
            continue
        coords = [nodes[ref] for ref in way.refs if ref in nodes]
        if not coords:
            continue
        if not any(in_bounds(lon, lat) for lon, lat in coords):
            continue
        if is_line_way(way.tags, lv) and len(coords) >= 2:
            lines[lv].append([lonlat_to_scene_xy(data, lon, lat, bounds) for lon, lat in coords])
        else:
            lon = float(np.mean([item[0] for item in coords]))
            lat = float(np.mean([item[1] for item in coords]))
            if in_bounds(lon, lat):
                points[lv].append(lonlat_to_scene_xy(data, lon, lat, bounds))
    return lines, points


def plot_osm_risk_features(
    ax: plt.Axes,
    lines: dict[int, list[list[tuple[float, float]]]],
    points: dict[int, list[tuple[float, float]]],
) -> list[Line2D]:
    handles: list[Line2D] = []
    for lv, style in OSM_RISK_STYLES.items():
        for line in lines.get(lv, []):
            if len(line) < 2:
                continue
            xs = [xy[0] for xy in line]
            ys = [xy[1] for xy in line]
            ax.plot(xs, ys, color=style["light"], lw=0.74, alpha=0.42, solid_capstyle="round", zorder=7 + lv, clip_on=True)
            ax.plot(xs, ys, color=style["dark"], lw=0.34, alpha=0.96, solid_capstyle="round", zorder=8 + lv, clip_on=True)
        if points.get(lv):
            arr = np.asarray(points[lv], dtype=float)
            ax.scatter(
                arr[:, 0],
                arr[:, 1],
                s=10,
                marker=style["marker"],
                facecolors=style["light"],
                edgecolors=style["dark"],
                linewidths=0.30,
                alpha=0.90,
                zorder=12 + lv,
            )
        handles.append(Line2D([0], [0], color=style["dark"], lw=0.65, label=style["label"]))
    return handles


def build_fig_2_1a(data: SceneArrays, out_dir: Path, dpi: int, max_surface_points: int) -> list[Path]:
    """生成单幅图2.1a，华山场景三维地形渲染。"""
    stride = choose_stride(*data.z.shape, max(max_surface_points, 420))
    xx, yy = xy_grids(*data.z.shape, data.resolution_m, stride=stride)
    z_display = gaussian_filter(np.nan_to_num(data.z, nan=float(np.nanmedian(data.z))), sigma=0.42, mode="nearest")
    z_plot = z_display[::stride, ::stride]
    terrain_rgb = shaded_terrain_rgb(data.z, data.resolution_m)
    z_norm = Normalize(vmin=float(np.nanpercentile(data.z, 1.0)), vmax=float(np.nanpercentile(data.z, 99.5)))

    with plt.rc_context({"font.family": ["Times New Roman", "SimSun", "Microsoft YaHei", "DejaVu Serif"]}):
        fig = plt.figure(figsize=(6.20, 4.35))
        ax = fig.add_axes([-0.10, -0.05, 0.86, 1.03], projection="3d")
        plot_surface(
            ax,
            xx,
            yy,
            z_plot,
            facecolors=terrain_rgb[::stride, ::stride],
            alpha=1.0,
            antialiased=True,
        )
        z_base = float(np.nanmin(data.z)) - 130.0
        ax.contour(
            xx,
            yy,
            z_plot,
            levels=np.arange(450.0, 2200.0, 150.0),
            zdir="z",
            offset=z_base,
            colors="#625D57",
            linewidths=0.28,
            alpha=0.30,
        )
        ax.set_zlim(z_base, float(np.nanmax(data.z)) + 130.0)
        style_publication_terrain_axis(ax, data, "华山场景三维地形渲染", elev=36, azim=-132)
        ax.set_zticks([500, 1000, 1500, 2000])
        cax = fig.add_axes([0.82, 0.24, 0.023, 0.48])
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=z_norm, cmap=TERRAIN_REALISTIC_CMAP),
            cax=cax,
        )
        colorbar.set_label("高程，m", labelpad=5, fontsize=8.4)
        colorbar.ax.tick_params(labelsize=7.5, width=0.45, length=2.4, pad=1)
        colorbar.outline.set_linewidth(0.45)

        paths = list(save_figure_pair(fig, out_dir, "fig_2_1a_huashan_terrain_rendering", dpi))
    copy_paths = copy_asset_pair(paths, data.figure_dir)
    return paths + copy_paths


def build_fig_2_1b(data: SceneArrays, out_dir: Path, dpi: int, max_surface_points: int) -> list[Path]:
    """生成单幅图2.1b，华山场景人员暴露风险要素。"""
    extent = map_extent(data.z, data.resolution_m)
    relief = hillshade_gray(data.z, data.resolution_m)
    relief_display = np.flipud(relief)

    fig, ax = plt.subplots(figsize=(6.15, 5.55), constrained_layout=True)
    ax.imshow(relief_display, extent=extent, origin="lower", cmap="gray", vmin=0, vmax=1)
    osm_lines, osm_points = load_osm_risk_features(data)
    risk_handles = plot_osm_risk_features(ax, osm_lines, osm_points)
    style_map_axis(ax, "华山场景人员暴露风险要素")
    ax.set_aspect("equal", adjustable="box")
    if risk_handles:
        ax.legend(handles=risk_handles, loc="upper right", frameon=True, framealpha=0.82, fontsize=7.4, title="OSM风险要素")

    paths = list(save_figure_pair(fig, out_dir, "fig_2_1b_huashan_human_exposure_risk", dpi))
    copy_paths = copy_asset_pair(paths, data.figure_dir)
    return paths + copy_paths


def build_fig_2_1c(data: SceneArrays, out_dir: Path, dpi: int, max_surface_points: int) -> list[Path]:
    """生成单幅图2.1c，华山场景区域支路层通信视距可达性。"""
    from PIL import Image, ImageChops, ImageDraw, ImageFont
    import contourpy

    branch_comm = data.risk_comm[1] if data.risk_comm.ndim == 3 else data.risk_comm
    comm_params = dict(data.comm_summary.get("params", {}))
    risk_threshold = float(comm_params.get("risk_threshold", 0.55))
    max_range_km = float(comm_params.get("max_range_km", data.cfg.get("communication", {}).get("max_range_km", 5.0)))
    reachability = np.clip(1.0 - branch_comm, 0.0, 1.0)

    width = int(round(6.15 * dpi))
    height = int(round(5.55 * dpi))
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    pt = dpi / 72.0
    rows, cols = branch_comm.shape
    x_min, x_max = 0.0, cols * data.resolution_m / 1000.0
    y_min, y_max = 0.0, rows * data.resolution_m / 1000.0
    y_axis_max = y_max + max(0.35, 0.035 * (y_max - y_min))

    def load_font(size_pt: float, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        font_candidates = [
            Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/Arial.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
        ]
        for font_path in font_candidates:
            if font_path.exists():
                return ImageFont.truetype(str(font_path), int(round(size_pt * pt)))
        return ImageFont.load_default()

    tick_font = load_font(7.2)
    label_font = load_font(8.0)
    legend_font = load_font(5.9)
    colorbar_font = load_font(7.0)
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    nearest = getattr(Image, "Resampling", Image).NEAREST

    left = int(round(0.70 * dpi))
    top_margin = int(round(0.18 * dpi))
    bottom_margin = int(round(0.58 * dpi))
    colorbar_width = int(round(0.075 * dpi))
    colorbar_gap = int(round(0.23 * dpi))
    right_margin = int(round(0.42 * dpi))
    available_w = width - left - colorbar_gap - colorbar_width - right_margin
    available_h = height - top_margin - bottom_margin
    aspect = (x_max - x_min) / max(y_axis_max - y_min, 1e-9)
    plot_w = min(available_w, int(round(available_h * aspect)))
    plot_h = min(available_h, int(round(plot_w / max(aspect, 1e-9))))
    plot_left = left
    plot_top = top_margin + max(0, (available_h - plot_h) // 2)
    plot_right = plot_left + plot_w
    plot_bottom = plot_top + plot_h

    def color_ramp(values: np.ndarray) -> np.ndarray:
        stops = np.array(
            [
                [255, 255, 217],
                [237, 248, 177],
                [199, 233, 180],
                [127, 205, 187],
                [65, 182, 196],
                [29, 145, 192],
                [34, 94, 168],
                [12, 44, 132],
            ],
            dtype=float,
        )
        pos = np.linspace(0.0, 1.0, len(stops))
        clipped = np.clip(values, 0.0, 1.0)
        channels = [np.interp(clipped, pos, stops[:, channel]) for channel in range(3)]
        return np.stack(channels, axis=-1).astype(np.uint8)

    def text_size(text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def draw_rotated_text(text: str, xy: tuple[int, int], font: ImageFont.ImageFont, angle: int) -> None:
        tw, th = text_size(text, font)
        text_layer = Image.new("RGBA", (tw + 20, th + 20), (255, 255, 255, 0))
        text_draw = ImageDraw.Draw(text_layer)
        text_draw.text((10, 10), text, font=font, fill=(0, 0, 0, 255))
        rotated = text_layer.rotate(angle, expand=True)
        canvas.paste(rotated, xy, rotated)

    def world_to_pixel(x: float, y: float) -> tuple[int, int]:
        px = plot_left + int(round((x - x_min) / max(x_max - x_min, 1e-9) * plot_w))
        py = plot_bottom - int(round((y - y_min) / max(y_axis_max - y_min, 1e-9) * plot_h))
        return px, py

    terrain_top = world_to_pixel(x_min, y_max)[1]
    terrain_h = plot_bottom - terrain_top
    relief = hillshade_gray(data.z, data.resolution_m)
    relief_img = Image.fromarray(np.clip(relief * 255.0, 0, 255).astype(np.uint8), mode="L")
    relief_img = relief_img.resize((plot_w, terrain_h), resampling).convert("RGBA")

    rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
    rgba[..., :3] = color_ramp(reachability)
    rgba[..., 3] = np.where((branch_comm <= risk_threshold) & np.isfinite(reachability), int(0.88 * 255), 0).astype(np.uint8)
    reachability_img = Image.fromarray(rgba, mode="RGBA").resize((plot_w, terrain_h), resampling)
    relief_img.alpha_composite(reachability_img)

    nlos_alpha = ((branch_comm > risk_threshold) & np.isfinite(branch_comm)).astype(np.uint8) * 86
    nlos_img = np.zeros((rows, cols, 4), dtype=np.uint8)
    nlos_img[..., :3] = np.array([229, 88, 18], dtype=np.uint8)
    nlos_img[..., 3] = nlos_alpha
    nlos_overlay = Image.fromarray(nlos_img, mode="RGBA").resize((plot_w, terrain_h), nearest)
    relief_img.alpha_composite(nlos_overlay)

    nlos_mask = Image.fromarray(
        ((branch_comm > risk_threshold) & np.isfinite(branch_comm)).astype(np.uint8) * 255,
        mode="L",
    ).resize((plot_w, terrain_h), nearest)
    hatch = Image.new("RGBA", (plot_w, terrain_h), (255, 255, 255, 0))
    hatch_draw = ImageDraw.Draw(hatch)
    hatch_spacing = max(14, int(round(0.055 * dpi)))
    hatch_width = max(1, int(round(0.22 * pt)))
    for start in range(-terrain_h, plot_w + terrain_h, hatch_spacing):
        hatch_draw.line(
            [(start, terrain_h), (start + terrain_h, 0)],
            fill=(174, 42, 12, 145),
            width=hatch_width,
        )
    hatch.putalpha(ImageChops.multiply(hatch.getchannel("A"), nlos_mask))
    relief_img.alpha_composite(hatch)
    canvas.paste(relief_img.convert("RGB"), (plot_left, terrain_top))

    overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # 用 contourpy 只提取等值线坐标，避免调用 Matplotlib 的绘图后端。
    x_coords = np.linspace(x_min, x_max, cols)
    y_coords = np.linspace(y_min, y_max, rows)
    comm_for_contour = np.flipud(gaussian_filter(branch_comm, sigma=0.85))
    contour_generator = contourpy.contour_generator(x=x_coords, y=y_coords, z=comm_for_contour)
    for segment in contour_generator.lines(risk_threshold):
        if len(segment) < 2:
            continue
        pixels = [world_to_pixel(float(x), float(y)) for x, y in segment]
        overlay_draw.line(pixels, fill=(0, 95, 115, 245), width=max(2, int(round(0.75 * pt))), joint="curve")

    sources = list(data.comm_summary.get("sources", []))
    coverage_width = max(2, int(round(0.48 * pt)))
    marker_radius = max(8, int(round(3.6 * pt)))
    for item in sources:
        x = float(item.get("x_km", np.nan))
        y = float(item.get("y_km", np.nan))
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        source_type = str(item.get("source", ""))
        is_depot = source_type == "virtual_depot"
        color = (198, 40, 40, 255) if is_depot else (21, 101, 192, 255)
        coverage_color = (132, 18, 18, 180) if is_depot else (8, 56, 130, 180)
        theta = np.linspace(0.0, 2.0 * math.pi, 97)
        circle_points = [(x + max_range_km * math.cos(t), y + max_range_km * math.sin(t)) for t in theta]
        for start in range(0, len(circle_points) - 1, 4):
            sub = circle_points[start : start + 3]
            clipped = [(cx, cy) for cx, cy in sub if x_min <= cx <= x_max and y_min <= cy <= y_max]
            if len(clipped) >= 2:
                overlay_draw.line([world_to_pixel(cx, cy) for cx, cy in clipped], fill=coverage_color, width=coverage_width)
        px, py = world_to_pixel(x, y)
        if is_depot:
            overlay_draw.rectangle(
                [px - marker_radius, py - marker_radius, px + marker_radius, py + marker_radius],
                fill=color,
                outline=(255, 255, 255, 255),
                width=max(2, int(round(0.25 * pt))),
            )
        else:
            points = [
                (px, py - marker_radius - 2),
                (px - marker_radius - 2, py + marker_radius),
                (px + marker_radius + 2, py + marker_radius),
            ]
            overlay_draw.polygon(points, fill=color)
            overlay_draw.line(points + [points[0]], fill=(255, 255, 255, 255), width=max(2, int(round(0.25 * pt))))

    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    axis_width = max(2, int(round(0.24 * pt)))
    tick_len = max(7, int(round(0.95 * pt)))
    draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline=(0, 0, 0), width=axis_width)

    def format_tick(value: float) -> str:
        if abs(value - round(value)) < 0.05:
            return f"{int(round(value))}"
        return f"{value:.1f}"

    for value in np.linspace(x_min, x_max, 6):
        px, _ = world_to_pixel(float(value), y_min)
        draw.line([(px, plot_bottom), (px, plot_bottom + tick_len)], fill=(0, 0, 0), width=axis_width)
        label = format_tick(float(value))
        tw, th = text_size(label, tick_font)
        draw.text((px - tw // 2, plot_bottom + tick_len + int(0.12 * dpi)), label, font=tick_font, fill=(0, 0, 0))
    for value in np.linspace(y_min, y_max, 6):
        _, py = world_to_pixel(x_min, float(value))
        draw.line([(plot_left - tick_len, py), (plot_left, py)], fill=(0, 0, 0), width=axis_width)
        label = format_tick(float(value))
        tw, th = text_size(label, tick_font)
        draw.text((plot_left - tick_len - int(0.10 * dpi) - tw, py - th // 2), label, font=tick_font, fill=(0, 0, 0))

    xlabel = "East-West Distance (km)"
    ylabel = "North-South Distance (km)"
    tw, th = text_size(xlabel, label_font)
    draw.text((plot_left + (plot_w - tw) // 2, height - int(0.22 * dpi) - th), xlabel, font=label_font, fill=(0, 0, 0))
    y_label_w, y_label_h = text_size(ylabel, label_font)
    draw_rotated_text(ylabel, (int(0.10 * dpi), plot_top + (plot_h - y_label_w) // 2), label_font, 90)

    cbar_left = plot_right + colorbar_gap
    cbar_top = plot_top + int(round(0.07 * plot_h))
    cbar_h = plot_h - int(round(0.14 * plot_h))
    gradient_values = np.linspace(1.0, 0.0, cbar_h)[:, None]
    colorbar_arr = np.repeat(color_ramp(gradient_values), colorbar_width, axis=1)
    colorbar_img = Image.fromarray(colorbar_arr.astype(np.uint8), mode="RGB")
    canvas.paste(colorbar_img, (cbar_left, cbar_top))
    draw.rectangle([cbar_left, cbar_top, cbar_left + colorbar_width, cbar_top + cbar_h], outline=(0, 0, 0), width=axis_width)
    for value in np.linspace(0.0, 1.0, 6):
        y_pos = cbar_top + int(round((1.0 - value) * cbar_h))
        draw.line([(cbar_left + colorbar_width, y_pos), (cbar_left + colorbar_width + tick_len, y_pos)], fill=(0, 0, 0), width=axis_width)
        label = f"{value:.1f}"
        draw.text((cbar_left + colorbar_width + tick_len + int(0.08 * dpi), y_pos - text_size(label, colorbar_font)[1] // 2), label, font=colorbar_font, fill=(0, 0, 0))
    colorbar_label = "LOS Reachability"
    cb_label_w, cb_label_h = text_size(colorbar_label, label_font)
    draw_rotated_text(
        colorbar_label,
        (cbar_left - int(round(0.22 * dpi)), cbar_top + (cbar_h - cb_label_w) // 2),
        label_font,
        90,
    )

    legend_entries = [
        ("line", (0, 95, 115), "LOS Boundary"),
        ("hatched_zone", (229, 88, 18), "NLOS High-Risk Zone"),
        ("coverage_line", (132, 18, 18), "Main BS Coverage Radius"),
        ("coverage_line", (8, 56, 130), "Relay Node Coverage Radius"),
        ("square", (198, 40, 40), "Delivery Station (Main BS)"),
        ("triangle", (21, 101, 192), "Edge Relay Node"),
    ]
    swatch_w = int(round(0.18 * dpi))
    row_gap = int(round(0.075 * dpi))
    legend_pad_x = int(round(0.10 * dpi))
    legend_pad_y = int(round(0.075 * dpi))
    max_text_w = max(text_size(label, legend_font)[0] for _, _, label in legend_entries)
    row_h = max(text_size("Ag", legend_font)[1], int(round(0.11 * dpi)))
    legend_w = legend_pad_x * 2 + swatch_w + int(round(0.08 * dpi)) + max_text_w
    legend_h = legend_pad_y * 2 + len(legend_entries) * row_h + (len(legend_entries) - 1) * row_gap
    legend_x = plot_left + int(round(0.07 * dpi))
    legend_y = plot_top + int(round(0.07 * dpi))
    draw.rectangle([legend_x, legend_y, legend_x + legend_w, legend_y + legend_h], fill=(255, 255, 255), outline=(0, 0, 0), width=axis_width)
    for idx, (kind, color, label) in enumerate(legend_entries):
        row_y = legend_y + legend_pad_y + idx * (row_h + row_gap) + row_h // 2
        swatch_x = legend_x + legend_pad_x
        if kind == "line":
            draw.line([(swatch_x, row_y), (swatch_x + swatch_w, row_y)], fill=color, width=max(3, int(round(0.55 * pt))))
        elif kind == "coverage_line":
            dash_y = row_y
            dash_len = max(4, swatch_w // 4)
            gap = max(3, swatch_w // 9)
            dash_x = swatch_x
            while dash_x < swatch_x + swatch_w:
                draw.line(
                    [(dash_x, dash_y), (min(dash_x + dash_len, swatch_x + swatch_w), dash_y)],
                    fill=color,
                    width=max(2, int(round(0.42 * pt))),
                )
                dash_x += dash_len + gap
        elif kind == "hatched_zone":
            zone_box = [swatch_x, row_y - row_h // 3, swatch_x + swatch_w, row_y + row_h // 3]
            draw.rectangle(zone_box, fill=color)
            hatch_gap = max(5, swatch_w // 5)
            for hatch_x in range(swatch_x - row_h, swatch_x + swatch_w, hatch_gap):
                draw.line(
                    [(hatch_x, zone_box[3]), (hatch_x + row_h, zone_box[1])],
                    fill=(130, 30, 9),
                    width=max(1, int(round(0.16 * pt))),
                )
        elif kind == "square":
            size = row_h // 2
            draw.rectangle([swatch_x + swatch_w // 2 - size, row_y - size, swatch_x + swatch_w // 2 + size, row_y + size], fill=color)
        else:
            size = row_h // 2
            draw.polygon(
                [
                    (swatch_x + swatch_w // 2, row_y - size),
                    (swatch_x + swatch_w // 2 - size, row_y + size),
                    (swatch_x + swatch_w // 2 + size, row_y + size),
                ],
                fill=color,
            )
        draw.text((swatch_x + swatch_w + int(round(0.08 * dpi)), row_y - text_size(label, legend_font)[1] // 2), label, font=legend_font, fill=(0, 0, 0))

    bounds_wgs84 = geo_bounds(data)

    def lonlat_to_pixel(lon: float, lat: float) -> tuple[int, int]:
        x_km, y_km = lonlat_to_scene_xy(data, lon, lat, bounds_wgs84)
        return world_to_pixel(x_km, y_km)

    peak_labels = map_peak_screen_points(data.cfg, lonlat_to_pixel)
    canvas = draw_peak_annotations(canvas, peak_labels, offsets=MAP_OFFSETS, reference_width=float(width), font_size_px=int(round(56 * width / 3600.0)))

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "fig_2_1c_huashan_communication_reachability.pdf"
    png_path = out_dir / "fig_2_1c_huashan_communication_reachability.png"
    canvas.save(png_path, dpi=(dpi, dpi))
    canvas.convert("RGB").save(pdf_path, "PDF", resolution=dpi)
    paths = [pdf_path, png_path]
    copy_paths = copy_asset_pair(paths, data.figure_dir)
    return paths + copy_paths


def build_fig_3_2(data: SceneArrays, out_dir: Path, dpi: int, max_surface_points: int) -> list[Path]:
    """生成图3.2，自适应安全飞行走廊与三层飞行中面。"""
    rows, cols = data.z.shape
    row = rows // 2
    x = np.arange(cols, dtype=float) * data.resolution_m / 1000.0
    thickness = data.ceiling - data.floor
    extent = map_extent(data.z, data.resolution_m)
    stride = choose_stride(rows, cols, max_surface_points)
    xx, yy = xy_grids(rows, cols, data.resolution_m, stride=stride)

    fig = plt.figure(figsize=(13.0, 4.05), constrained_layout=True)
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.fill_between(x, data.floor[row], data.ceiling[row], color="#B0BEC5", alpha=0.34, label="安全走廊")
    ax0.plot(x, data.z[row], color="#5D4037", lw=1.4, label="地形")
    ax0.plot(x, data.floor[row], color="#546E7A", lw=1.3, label="下边界")
    ax0.plot(x, data.ceiling[row], color="#455A64", lw=1.3, label="上边界")
    for lid, color in enumerate(LAYER_COLORS):
        ax0.plot(x, data.layer_mid[lid, row], color=color, lw=1.25, label=LAYER_NAMES_CN[lid])
    ax0.set_title("a  典型剖面", loc="left")
    ax0.set_xlabel("东西向距离，km")
    ax0.set_ylabel("高程，m")
    ax0.grid(True, alpha=0.25)
    ax0.legend(frameon=False, ncol=2, fontsize=7.7, loc="upper left")

    ax1 = fig.add_subplot(1, 3, 2)
    im = ax1.imshow(np.flipud(thickness), extent=extent, origin="lower", cmap="viridis")
    style_map_axis(ax1, "b  走廊厚度平面图")
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cb.set_label("厚度，m")

    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    plot_surface(ax2, xx, yy, data.z[::stride, ::stride], cmap="terrain", alpha=0.32)
    plot_surface(ax2, xx, yy, data.floor[::stride, ::stride], color="#78909C", alpha=0.17)
    plot_surface(ax2, xx, yy, data.ceiling[::stride, ::stride], color="#CFD8DC", alpha=0.20)
    for lid, color in enumerate(LAYER_COLORS):
        layer = np.ma.masked_where(~data.layer_allowed[lid, ::stride, ::stride], data.layer_mid[lid, ::stride, ::stride])
        plot_surface(ax2, xx, yy, layer, color=color, alpha=0.31)
    style_3d_axis(ax2, data, "c  走廊包络与三层中面", elev=31, azim=-126)
    handles = [Line2D([0], [0], color=color, lw=2.2, label=name) for color, name in zip(LAYER_COLORS, LAYER_NAMES_CN)]
    ax2.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.98))

    fig.suptitle("图3.2  自适应安全飞行走廊与三层飞行中面", y=1.03, fontsize=12)
    paths = list(save_figure_pair(fig, out_dir, "fig_3_2_safe_corridor_layers", dpi))
    copy_paths = copy_asset_pair(paths, data.figure_dir)
    return paths + copy_paths


def node_layers(nodes: np.ndarray) -> np.ndarray:
    return np.rint(nodes[:, 3]).astype(int)


def edge_lengths(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    lengths = np.zeros(len(edges), dtype=float)
    for idx, edge in enumerate(edges):
        u, v = int(edge[0]), int(edge[1])
        diff = nodes[v, :3] - nodes[u, :3]
        diff[:2] *= 1000.0
        lengths[idx] = float(np.linalg.norm(diff))
    return lengths


def build_adjacency(nodes: np.ndarray, edges: np.ndarray) -> list[list[tuple[int, int]]]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(len(nodes))]
    for eid, edge in enumerate(edges):
        u, v = int(edge[0]), int(edge[1])
        adjacency[u].append((v, eid))
        adjacency[v].append((u, eid))
    return adjacency


def shortest_path(data: SceneArrays, start_idx: int, goal_idx: int) -> list[int]:
    """按边长求一条原始图搜索折线路径，用于后处理示意。"""
    lengths = edge_lengths(data.nodes, data.edges)
    adjacency = build_adjacency(data.nodes, data.edges)
    dist = np.full(len(data.nodes), np.inf)
    prev = np.full(len(data.nodes), -1, dtype=int)
    dist[start_idx] = 0.0
    heap: list[tuple[float, int]] = [(0.0, int(start_idx))]
    while heap:
        cur_dist, u = heapq.heappop(heap)
        if cur_dist > dist[u] + 1e-12:
            continue
        if u == int(goal_idx):
            break
        for v, eid in adjacency[u]:
            cand = cur_dist + float(lengths[eid])
            if cand + 1e-12 < dist[v]:
                dist[v] = cand
                prev[v] = u
                heapq.heappush(heap, (cand, int(v)))
    if not np.isfinite(dist[goal_idx]):
        raise RuntimeError("示例起终点在当前图中不连通。")
    path = []
    cur = int(goal_idx)
    while cur >= 0:
        path.append(cur)
        if cur == int(start_idx):
            break
        cur = int(prev[cur])
    return list(reversed(path))


def km_to_rc(data: SceneArrays, x_km: float, y_km: float) -> tuple[int, int]:
    rows, cols = data.z.shape
    col = int(np.clip(round(x_km * 1000.0 / data.resolution_m), 0, cols - 1))
    row = int(np.clip(round((rows - 1) - y_km * 1000.0 / data.resolution_m), 0, rows - 1))
    return row, col


def segment_is_safe(data: SceneArrays, p0: np.ndarray, p1: np.ndarray, sample_count: int = 28) -> bool:
    """检查直连线段是否仍处在安全飞行走廊内。"""
    for t in np.linspace(0.0, 1.0, sample_count):
        p = p0 + (p1 - p0) * float(t)
        row, col = km_to_rc(data, float(p[0]), float(p[1]))
        z = float(p[2])
        if z + 1e-9 < float(data.floor[row, col]) or z - 1e-9 > float(data.ceiling[row, col]):
            return False
    return True


def los_prune_path(data: SceneArrays, raw_path: Sequence[int]) -> list[int]:
    """贪心执行 LOS 直连剪枝。"""
    if len(raw_path) <= 2:
        return list(raw_path)
    out = [int(raw_path[0])]
    i = 0
    coords = data.nodes[:, :3]
    while i < len(raw_path) - 1:
        chosen = i + 1
        for j in range(len(raw_path) - 1, i, -1):
            if segment_is_safe(data, coords[int(raw_path[i])], coords[int(raw_path[j])]):
                chosen = j
                break
        out.append(int(raw_path[chosen]))
        i = chosen
    return out


def curve_is_safe(data: SceneArrays, curve: np.ndarray) -> bool:
    for p in curve:
        row, col = km_to_rc(data, float(p[0]), float(p[1]))
        z = float(p[2])
        if z + 1e-9 < float(data.floor[row, col]) or z - 1e-9 > float(data.ceiling[row, col]):
            return False
    return True


def bspline_or_polyline(data: SceneArrays, pruned_path: Sequence[int], n_points: int = 180) -> np.ndarray:
    """生成 B 样条轨迹；若越出走廊则退回剪枝折线插值。"""
    pts = np.asarray(data.nodes[list(pruned_path), :3], dtype=float)
    if len(pts) >= 4:
        try:
            degree = min(3, len(pts) - 1)
            tck, _ = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], s=0.0, k=degree)
            u_new = np.linspace(0.0, 1.0, n_points)
            curve = np.asarray(splev(u_new, tck), dtype=float).T
            if curve_is_safe(data, curve):
                return curve
        except Exception:
            pass
    samples: list[np.ndarray] = []
    for a, b in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0.0, 1.0, max(2, n_points // max(1, len(pts) - 1)), endpoint=False):
            samples.append(a + (b - a) * float(t))
    samples.append(pts[-1])
    return np.asarray(samples, dtype=float)


def path_distance_km(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    diffs[:, :2] *= 1000.0
    return float(np.sum(np.linalg.norm(diffs, axis=1)) / 1000.0)


def terminal_index(data: SceneArrays, name: str, layer: int = 0) -> int:
    terminals = data.terminal_status.get("terminals", {})
    if name not in terminals:
        raise KeyError(f"终端不存在：{name}")
    indices = terminals[name].get("indices", [])
    if len(indices) <= layer:
        raise IndexError(f"终端 {name} 缺少第 {layer} 层锚点。")
    return int(indices[layer])


def plot_path_context_edges(ax, data: SceneArrays, path_nodes: Sequence[int], alpha: float = 0.18) -> None:
    """只渲染路径邻域中的少量边，避免三层全图过密。"""
    layers = node_layers(data.nodes)
    path_xy = data.nodes[list(path_nodes), :2]
    min_xy = np.min(path_xy, axis=0) - 0.55
    max_xy = np.max(path_xy, axis=0) + 0.55
    selected = []
    for eid, edge in enumerate(data.edges):
        u, v = int(edge[0]), int(edge[1])
        mid = 0.5 * (data.nodes[u, :2] + data.nodes[v, :2])
        if np.all(mid >= min_xy) and np.all(mid <= max_xy):
            selected.append(eid)
    if len(selected) > 420:
        rng = np.random.default_rng(20260516)
        selected = sorted(rng.choice(selected, size=420, replace=False).astype(int).tolist())
    for eid in selected:
        u, v = int(data.edges[eid, 0]), int(data.edges[eid, 1])
        lid = int(max(layers[u], layers[v]))
        ax.plot(
            [data.nodes[u, 0], data.nodes[v, 0]],
            [data.nodes[u, 1], data.nodes[v, 1]],
            [data.nodes[u, 2], data.nodes[v, 2]],
            color=LAYER_COLORS[min(lid, 2)],
            lw=0.55,
            alpha=alpha,
        )


def draw_uav_icon(ax, point: np.ndarray, size_km: float = 0.16) -> None:
    """用线段绘制简化 UAV 图标。"""
    x, y, z = map(float, point)
    wing = size_km
    tail = size_km * 0.62
    ax.plot([x - wing, x, x + wing], [y - tail, y + tail, y - tail], [z, z + 8.0, z], color="#222222", lw=1.8)
    ax.plot([x, x], [y + tail, y - tail * 1.25], [z + 8.0, z - 2.0], color="#222222", lw=1.5)


def plot_one_path_panel(ax, data: SceneArrays, raw_path: Sequence[int], pruned_path: Sequence[int], curve: np.ndarray, smooth: bool) -> None:
    stride = choose_stride(*data.z.shape, 72)
    xx, yy = xy_grids(*data.z.shape, data.resolution_m, stride=stride)
    plot_surface(ax, xx, yy, data.z[::stride, ::stride], cmap="Greys", alpha=0.16)
    plot_path_context_edges(ax, data, raw_path)
    raw_pts = data.nodes[list(raw_path), :3]
    pruned_pts = data.nodes[list(pruned_path), :3]
    if smooth:
        ax.plot(pruned_pts[:, 0], pruned_pts[:, 1], pruned_pts[:, 2], color="#6B6B6B", lw=1.1, alpha=0.55, linestyle="--")
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color="#0072B2", lw=2.7)
    else:
        ax.plot(raw_pts[:, 0], raw_pts[:, 1], raw_pts[:, 2], color="#D55E00", lw=2.45)
    start = raw_pts[0]
    goal = raw_pts[-1]
    draw_uav_icon(ax, start)
    ax.scatter([goal[0]], [goal[1]], [goal[2]], marker="*", s=100, color="#CC79A7", edgecolor="white", linewidth=0.6)
    style_path_3d_axis(ax, data, "b  LOS剪枝与B样条平滑后" if smooth else "a  图搜索原始折线路径", raw_path)


def material_icon_path(root: Path, filename: str) -> Path:
    """优先读取用户指定的 raw/material 图标，不存在时回退到 data/material。"""
    candidates = [
        root / "raw" / "material" / filename,
        root / "data" / "material" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"缺少图标素材：{filename}")


def transform_path_points_for_pyvista(points: np.ndarray, data: SceneArrays, vertical_exag: float, lift_m: float = 72.0) -> np.ndarray:
    """把图搜索节点坐标转换到 PyVista DEM 的居中米制坐标系。"""
    rows, cols = data.z.shape
    base_elev = float(np.nanmin(data.z))
    xy_center = np.array(
        [
            0.5 * (cols - 1) * data.resolution_m,
            0.5 * (rows - 1) * data.resolution_m,
        ],
        dtype=float,
    )
    out = np.asarray(points, dtype=float).copy()
    out[:, 0] = out[:, 0] * 1000.0 - xy_center[0]
    out[:, 1] = out[:, 1] * 1000.0 - xy_center[1]
    out[:, 2] = np.maximum(out[:, 2] - base_elev, 0.0) * vertical_exag + lift_m
    return out


def line_polydata_from_segments(segments: Sequence[np.ndarray]):
    """把若干线段转换为 PyVista PolyData。"""
    import pyvista as pv

    points: list[np.ndarray] = []
    cells: list[int] = []
    for segment in segments:
        arr = np.asarray(segment, dtype=float)
        if len(arr) < 2:
            continue
        start = len(points)
        points.extend(arr)
        cells.extend([len(arr), *range(start, start + len(arr))])
    if not points:
        return pv.PolyData()
    poly = pv.PolyData(np.asarray(points, dtype=float))
    poly.lines = np.asarray(cells, dtype=np.int64)
    return poly


def polyline_from_points(points: np.ndarray):
    """把连续折线点转换为 PyVista PolyData。"""
    return line_polydata_from_segments([np.asarray(points, dtype=float)])


def sphere_mesh_from_points(points: np.ndarray, radius: float = 58.0, theta_steps: int = 18, phi_steps: int = 10):
    """用三角面片手工构造小球，避免 Windows 下 pv.Sphere 的原生崩溃。"""
    import pyvista as pv

    centers = np.asarray(points, dtype=float)
    if len(centers) == 0:
        return pv.PolyData()

    theta = np.linspace(0.0, 2.0 * np.pi, theta_steps, endpoint=False)
    phi = np.linspace(0.0, np.pi, phi_steps + 1)
    vertices: list[np.ndarray] = []
    faces: list[int] = []

    for center in centers:
        start = len(vertices)
        vertices.append(center + np.array([0.0, 0.0, radius], dtype=float))
        for phi_value in phi[1:-1]:
            sin_phi = float(np.sin(phi_value))
            cos_phi = float(np.cos(phi_value))
            for theta_value in theta:
                vertices.append(
                    center
                    + np.array(
                        [
                            radius * sin_phi * np.cos(theta_value),
                            radius * sin_phi * np.sin(theta_value),
                            radius * cos_phi,
                        ],
                        dtype=float,
                    )
                )
        south = len(vertices)
        vertices.append(center + np.array([0.0, 0.0, -radius], dtype=float))

        ring_count = phi_steps - 1
        first_ring = start + 1
        for j in range(theta_steps):
            faces.extend([3, start, first_ring + j, first_ring + (j + 1) % theta_steps])
        for ring_index in range(ring_count - 1):
            ring_a = first_ring + ring_index * theta_steps
            ring_b = ring_a + theta_steps
            for j in range(theta_steps):
                faces.extend(
                    [
                        4,
                        ring_a + j,
                        ring_a + (j + 1) % theta_steps,
                        ring_b + (j + 1) % theta_steps,
                        ring_b + j,
                    ]
                )
        last_ring = first_ring + (ring_count - 1) * theta_steps
        for j in range(theta_steps):
            faces.extend([3, south, last_ring + (j + 1) % theta_steps, last_ring + j])

    return pv.PolyData(np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64))


def dashed_segments_from_points(points: np.ndarray, dash_m: float = 210.0, gap_m: float = 130.0) -> list[np.ndarray]:
    """将连续折线拆成虚线段，供 PyVista 渲染 LOS 剪枝折线。"""
    segments: list[np.ndarray] = []
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return segments
    for p0, p1 in zip(pts[:-1], pts[1:]):
        vec = p1 - p0
        length = float(np.linalg.norm(vec))
        if length <= 1e-9:
            continue
        direction = vec / length
        cursor = 0.0
        while cursor < length:
            end = min(cursor + dash_m, length)
            segments.append(np.vstack([p0 + direction * cursor, p0 + direction * end]))
            cursor += dash_m + gap_m
    return segments


def paste_icon_with_outline(canvas, icon, center_xy: tuple[float, float], size_px: int, outline_px: int = 5) -> None:
    """在最终拼图上粘贴带白色描边的 PNG 图标。"""
    from PIL import Image, ImageChops, ImageFilter

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    icon_rgba = icon.convert("RGBA").resize((size_px, size_px), resampling)
    alpha = icon_rgba.getchannel("A")
    outline_alpha = alpha.filter(ImageFilter.MaxFilter(outline_px * 2 + 1))
    outline_alpha = ImageChops.subtract(outline_alpha, alpha)
    outlined = Image.new("RGBA", (size_px, size_px), (255, 255, 255, 0))
    outlined.putalpha(outline_alpha)
    outlined.alpha_composite(icon_rgba)
    x = int(round(center_xy[0] - 0.5 * size_px))
    y = int(round(center_xy[1] - 0.5 * size_px))
    canvas.alpha_composite(outlined, (x, y))


def draw_reference_colorbar(draw, canvas, cmap, x: int, y: int, width: int, height: int, elev_max: float, font) -> None:
    """绘制与 PyVista 地形一致的全局高程色标。"""
    from PIL import Image

    values = np.linspace(1.0, 0.0, height)[:, None]
    rgba = np.asarray(cmap(values), dtype=float)
    rgb = np.repeat(np.round(rgba[:, :, :3] * 255).astype(np.uint8), width, axis=1)
    canvas.paste(Image.fromarray(rgb, mode="RGB"), (x, y))
    draw.rectangle([x, y, x + width, y + height], outline=(0, 0, 0), width=1)
    ticks = np.linspace(0.0, elev_max, 7)
    for tick in ticks:
        ty = y + int(round((1.0 - tick / max(elev_max, 1e-9)) * height))
        draw.line([(x + width, ty), (x + width + 12, ty)], fill=(0, 0, 0), width=2)
        label = f"{int(round(tick))}"
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((x + width + 22, ty - (bbox[3] - bbox[1]) // 2), label, fill=(0, 0, 0), font=font)
    title = "Elevation (m)"
    bbox = draw.textbbox((0, 0), title, font=font)
    draw.text((x - 8, y - (bbox[3] - bbox[1]) - 18), title, fill=(0, 0, 0), font=font)


def render_path_postprocessing_panel(
    data: SceneArrays,
    raw_path: Sequence[int],
    pruned_path: Sequence[int],
    curve: np.ndarray,
    *,
    smooth: bool,
    window_size: tuple[int, int],
):
    """渲染单个路径后处理三维面板，并返回端点投影坐标。"""
    import pyvista as pv
    from render_huashan_dem_pyvista import (
        REFERENCE_TERRAIN_CMAP,
        build_ground_shadow,
        build_structured_grid,
        configure_reference_lights,
        style_bounds_actor,
    )

    vertical_exag = 1.45
    grid = build_structured_grid(data.z, data.resolution_m, stride=2, vertical_exag=vertical_exag, smooth_sigma=0.45)
    bounds = grid.bounds
    x_mid = 0.5 * (bounds[0] + bounds[1])
    y_mid = 0.5 * (bounds[2] + bounds[3])
    z_mid = 0.45 * bounds[5]
    span = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    elev_display_max = 2081.0

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
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
        cmap=REFERENCE_TERRAIN_CMAP,
        clim=(0.0, elev_display_max),
        show_scalar_bar=False,
        smooth_shading=True,
        split_sharp_edges=False,
        ambient=0.20,
        diffuse=0.84,
        specular=0.10,
        specular_power=18,
        pbr=False,
    )

    raw_pts = data.nodes[list(raw_path), :3]
    pruned_pts = data.nodes[list(pruned_path), :3]
    raw_vtk = transform_path_points_for_pyvista(raw_pts, data, vertical_exag)
    pruned_vtk = transform_path_points_for_pyvista(pruned_pts, data, vertical_exag)
    curve_vtk = transform_path_points_for_pyvista(np.asarray(curve, dtype=float), data, vertical_exag)

    layers = node_layers(data.nodes)
    path_xy = raw_pts[:, :2]
    min_xy = np.min(path_xy, axis=0) - 0.75
    max_xy = np.max(path_xy, axis=0) + 0.75
    grouped_segments: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: []}
    for edge in data.edges:
        u, v = int(edge[0]), int(edge[1])
        mid = 0.5 * (data.nodes[u, :2] + data.nodes[v, :2])
        if not (np.all(mid >= min_xy) and np.all(mid <= max_xy)):
            continue
        lid = int(min(max(layers[u], layers[v]), 2))
        segment = transform_path_points_for_pyvista(data.nodes[[u, v], :3], data, vertical_exag, lift_m=38.0)
        grouped_segments[lid].append(segment)

    rng = np.random.default_rng(20260516)
    for lid, segments in grouped_segments.items():
        if len(segments) > 210:
            selected = rng.choice(len(segments), size=210, replace=False)
            segments = [segments[int(i)] for i in sorted(selected)]
        poly = line_polydata_from_segments(segments)
        if poly.n_points:
            plotter.add_mesh(
                poly,
                color=LAYER_COLORS[min(lid, 2)],
                line_width=1.2,
                opacity=0.30,
                render_lines_as_tubes=True,
                lighting=False,
            )

    if smooth:
        dashed = line_polydata_from_segments(dashed_segments_from_points(pruned_vtk))
        if dashed.n_points:
            plotter.add_mesh(
                dashed.tube(radius=22.0, n_sides=10),
                color="#6B6B6B",
                opacity=0.78,
                smooth_shading=True,
                lighting=False,
            )
        plotter.add_mesh(
            polyline_from_points(curve_vtk).tube(radius=42.0, n_sides=18),
            color="#0072B2",
            smooth_shading=True,
            ambient=0.80,
            diffuse=0.40,
            specular=0.18,
        )
    else:
        plotter.add_mesh(
            polyline_from_points(raw_vtk).tube(radius=42.0, n_sides=18),
            color="#D55E00",
            smooth_shading=True,
            ambient=0.80,
            diffuse=0.40,
            specular=0.18,
        )
        turning_points = raw_vtk[1:-1]
        if len(turning_points):
            plotter.add_mesh(
                sphere_mesh_from_points(turning_points, radius=58.0, theta_steps=18, phi_steps=10),
                color="#F28E1C",
                smooth_shading=True,
                ambient=0.82,
                diffuse=0.46,
                specular=0.16,
                specular_power=18,
            )

    configure_reference_lights(plotter, (x_mid, y_mid, z_mid), span, bounds[5])
    plotter.enable_anti_aliasing("ssaa")
    try:
        plotter.enable_ssao(radius=0.22, bias=0.012, kernel_size=256, blur=True)
    except Exception:
        pass

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
        font_size=21,
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
    plotter.camera.zoom(0.82)
    plotter.render()

    def project(point: np.ndarray) -> tuple[float, float]:
        renderer = plotter.renderer
        renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
        renderer.WorldToDisplay()
        display = renderer.GetDisplayPoint()
        return float(display[0]), float(window_size[1] - display[1])

    start_xy = project(raw_vtk[0] + np.array([0.0, 0.0, 105.0]))
    goal_xy = project(raw_vtk[-1] + np.array([0.0, 0.0, 105.0]))
    peak_labels = project_peak_points(
        plotter,
        peak_world_points(
            config_or_path=data.cfg,
            geo_path=data.data_dir / "Z_crop_geo.npz",
            z=data.z,
            resolution_m=data.resolution_m,
            vertical_exag=vertical_exag,
            lift_scene_m=140.0,
        ),
        window_size,
    )
    image = plotter.screenshot(transparent_background=False, return_img=True)
    plotter.close()
    image = np.asarray(draw_peak_annotations(image, peak_labels, reference_width=float(window_size[0])), dtype=np.uint8)
    return image, start_xy, goal_xy, REFERENCE_TERRAIN_CMAP, elev_display_max


def build_fig_3_3(data: SceneArrays, out_dir: Path, dpi: int) -> list[Path]:
    """生成图3.3，路径后处理前后对比。"""
    from PIL import Image, ImageDraw, ImageFont

    start_name = str(data.cfg.get("default_start", "虚拟配送站1"))
    goal_name = str(data.cfg.get("default_goal", "南峰"))
    raw_path = shortest_path(data, terminal_index(data, start_name), terminal_index(data, goal_name))
    pruned_path = los_prune_path(data, raw_path)
    curve = bspline_or_polyline(data, pruned_path)
    raw_len = path_distance_km(data.nodes[list(raw_path), :3])
    pruned_len = path_distance_km(data.nodes[list(pruned_path), :3])
    curve_len = path_distance_km(curve)

    panel_size = (2050, 1460)
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

    final_w, final_h = 4800, 2400
    canvas = Image.new("RGBA", (final_w, final_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        candidates = [
            Path("C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf"),
            Path("C:/Windows/Fonts/timesbi.ttf" if bold else "C:/Windows/Fonts/timesi.ttf"),
            Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
        ]
        for path in candidates:
            if path.exists():
                return ImageFont.truetype(str(path), size)
        return ImageFont.load_default()

    title_font = load_font(54, bold=True)
    note_font = load_font(39)
    legend_font = load_font(36)
    colorbar_font = load_font(32)

    def crop_resize_panel(image: np.ndarray, start_xy: tuple[float, float], goal_xy: tuple[float, float]):
        panel_img = Image.fromarray(image).convert("RGBA")
        rgb = np.asarray(panel_img.convert("RGB"))
        mask = np.any(rgb < 246, axis=2)
        if np.any(mask):
            ys, xs = np.where(mask)
            pad = 48
            x0 = max(0, int(xs.min()) - pad)
            y0 = max(0, int(ys.min()) - pad)
            x1 = min(panel_img.width, int(xs.max()) + pad)
            y1 = min(panel_img.height, int(ys.max()) + pad)
        else:
            x0, y0, x1, y1 = 0, 0, panel_img.width, panel_img.height
        cropped = panel_img.crop((x0, y0, x1, y1))
        display_size = (2090, 1460)
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        sx = display_size[0] / max(1, x1 - x0)
        sy = display_size[1] / max(1, y1 - y0)
        start_new = ((start_xy[0] - x0) * sx, (start_xy[1] - y0) * sy)
        goal_new = ((goal_xy[0] - x0) * sx, (goal_xy[1] - y0) * sy)
        return cropped.resize(display_size, resampling), start_new, goal_new

    left_x, right_x = 95, 2325
    panel_y = 185
    panel_img_raw, raw_start_xy, raw_goal_xy = crop_resize_panel(panel_raw, raw_start_xy, raw_goal_xy)
    panel_img_smooth, smooth_start_xy, smooth_goal_xy = crop_resize_panel(panel_smooth, smooth_start_xy, smooth_goal_xy)
    canvas.alpha_composite(panel_img_raw, (left_x, panel_y))
    canvas.alpha_composite(panel_img_smooth, (right_x, panel_y))

    uav_icon = Image.open(material_icon_path(PROJECT_ROOT, "uav.png"))
    target_icon = Image.open(material_icon_path(PROJECT_ROOT, "target.png"))
    icon_size = 64
    paste_icon_with_outline(canvas, uav_icon, (left_x + raw_start_xy[0], panel_y + raw_start_xy[1]), icon_size)
    paste_icon_with_outline(canvas, target_icon, (left_x + raw_goal_xy[0], panel_y + raw_goal_xy[1]), icon_size)
    paste_icon_with_outline(canvas, uav_icon, (right_x + smooth_start_xy[0], panel_y + smooth_start_xy[1]), icon_size)
    paste_icon_with_outline(canvas, target_icon, (right_x + smooth_goal_xy[0], panel_y + smooth_goal_xy[1]), icon_size)

    draw.text((70, 44), "(a) Initial Discrete Path (LPA*)", fill=(0, 0, 0), font=title_font)
    draw.text((2365, 44), "(b) Smoothed Trajectory (LOS Pruning & B-spline)", fill=(0, 0, 0), font=title_font)

    colorbar_x = 4510
    colorbar_y = 195
    colorbar_h = 1500
    colorbar_w = 70
    draw_reference_colorbar(draw, canvas, terrain_cmap, colorbar_x, colorbar_y, colorbar_w, colorbar_h, elev_display_max, colorbar_font)

    def english_task_name(name: str) -> str:
        if name.startswith("虚拟配送站"):
            suffix = name.removeprefix("虚拟配送站")
            return f"Virtual Depot {suffix}" if suffix else "Virtual Depot"
        mapping = {
            "南峰": "South Peak",
            "北峰": "North Peak",
            "东峰": "East Peak",
            "西峰": "West Peak",
            "中峰": "Central Peak",
        }
        return mapping.get(name, name)

    note_lines = [
        f"Example Task: {english_task_name(start_name)} to {english_task_name(goal_name)}",
        f"Nodes {len(raw_path)} → {len(pruned_path)}; Length {raw_len:.2f} km → {pruned_len:.2f} km → {curve_len:.2f} km",
    ]
    legend_items = [
        ("line", "#D55E00", "Original Path"),
        ("dash", "#6B6B6B", "LOS Pruned Line"),
        ("line", "#0072B2", "B-spline Trajectory"),
        ("icon_uav", None, "Start Node (UAV)"),
        ("icon_target", None, "Target Node"),
    ]
    legend_rows = [legend_items[:3], legend_items[3:]]
    symbol_widths = {
        "line": 128,
        "dash": 128,
        "icon_uav": 70,
        "icon_target": 70,
    }
    label_gap = 32
    item_gap = 68

    def text_width(text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> int:
        bbox = draw.textbbox((0, 0), text, font=font)
        return int(bbox[2] - bbox[0])

    def legend_item_width(item: tuple[str, str | None, str]) -> int:
        kind, _, label = item
        return symbol_widths[kind] + label_gap + text_width(label, legend_font)

    row_item_widths = [[legend_item_width(item) for item in row] for row in legend_rows]
    row_widths = [
        sum(widths) + item_gap * (len(widths) - 1)
        for widths in row_item_widths
    ]
    note_widths = [text_width(line, note_font) for line in note_lines]
    legend_box_width = max(max(note_widths), max(row_widths)) + 190
    legend_box_height = 405
    legend_left = int(round((final_w - legend_box_width) / 2))
    legend_top = 1940
    legend_box = (legend_left, legend_top, legend_left + int(legend_box_width), legend_top + legend_box_height)
    draw.rectangle(legend_box, fill=(255, 255, 255), outline=(0, 0, 0), width=3)

    note_y = legend_box[1] + 28
    for line, width in zip(note_lines, note_widths):
        note_x = legend_box[0] + (legend_box[2] - legend_box[0] - width) // 2
        draw.text((note_x, note_y), line, fill=(0, 0, 0), font=note_font)
        note_y += 48

    for row, widths, row_width, legend_y in zip(legend_rows, row_item_widths, row_widths, [legend_box[1] + 170, legend_box[1] + 275]):
        x = legend_box[0] + (legend_box[2] - legend_box[0] - row_width) // 2
        for (kind, color, label), item_width in zip(row, widths):
            symbol_start = int(round(x))
            symbol_mid_y = legend_y + 24
            if kind == "line":
                draw.line(
                    [(symbol_start, symbol_mid_y), (symbol_start + symbol_widths[kind], symbol_mid_y)],
                    fill=color,
                    width=16,
                )
            elif kind == "dash":
                dash_x = symbol_start
                symbol_end = symbol_start + symbol_widths[kind]
                while dash_x < symbol_end:
                    draw.line(
                        [(dash_x, symbol_mid_y), (min(dash_x + 34, symbol_end), symbol_mid_y)],
                        fill=color,
                        width=10,
                    )
                    dash_x += 52
            elif kind == "icon_uav":
                paste_icon_with_outline(canvas, uav_icon, (symbol_start + symbol_widths[kind] / 2, symbol_mid_y), 54, outline_px=4)
            else:
                paste_icon_with_outline(canvas, target_icon, (symbol_start + symbol_widths[kind] / 2, symbol_mid_y), 54, outline_px=4)
            label_x = symbol_start + symbol_widths[kind] + label_gap
            draw.text((label_x, legend_y - 2), label, fill=(0, 0, 0), font=legend_font)
            x += item_width + item_gap

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig_3_3_path_postprocessing.png"
    pdf_path = out_dir / "fig_3_3_path_postprocessing.pdf"
    canvas.convert("RGB").save(png_path, dpi=(dpi, dpi))
    canvas.convert("RGB").save(pdf_path, "PDF", resolution=dpi)
    paths = [pdf_path, png_path]
    copy_paths = copy_asset_pair(paths, data.figure_dir)
    summary = {
        "start": start_name,
        "goal": goal_name,
        "raw_node_count": len(raw_path),
        "los_node_count": len(pruned_path),
        "raw_length_km": raw_len,
        "los_length_km": pruned_len,
        "curve_length_km": curve_len,
    }
    (out_dir / "fig_3_3_path_postprocessing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths + copy_paths


def copy_asset_pair(paths: Sequence[Path], target_dir: Path) -> list[Path]:
    """把正式资产同步到场景中间图目录，便于按当前目录结构归档。"""
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in paths:
        dst = target_dir / src.name
        dst.write_bytes(src.read_bytes())
        copied.append(dst)
    return copied


def normalize_baseline(row: dict) -> str:
    baseline = str(row.get("baseline", row.get("baseline_id", ""))).strip()
    return BASELINE_TO_METHOD.get(baseline, str(row.get("method", row.get("code", baseline))).strip())


def load_single_rows(root: Path) -> list[dict]:
    rows = read_csv_rows(root / "final_results" / "_summaries" / "E1_E2_three_mountain_single_final.csv")
    out = []
    for row in rows:
        method = BASELINE_TO_METHOD.get(str(row.get("baseline", "")).strip())
        scene = str(row.get("scene_name", "")).strip()
        if method in METHOD_ORDER and scene in SCENE_CONFIGS:
            item = dict(row)
            item["method"] = method
            item["scene"] = scene
            out.append(item)
    return out


def value_from(row: dict, *fields: str) -> float:
    for field in fields:
        val = to_float(row.get(field))
        if math.isfinite(val):
            return val
    return float("nan")


def performance_scores(rows: Sequence[dict], metric_specs: Sequence[tuple[str, str, bool]]) -> np.ndarray:
    """按场景归一化后求各方法平均表现分，分值越高越好。"""
    score_sum = np.zeros((len(METHOD_ORDER), len(metric_specs)), dtype=float)
    score_count = np.zeros_like(score_sum)
    for scene in sorted({r["scene"] for r in rows}):
        scene_rows = [r for r in rows if r["scene"] == scene]
        by_method = {r["method"]: r for r in scene_rows}
        for col, (_label, field, higher_better) in enumerate(metric_specs):
            vals = np.asarray([value_from(by_method.get(m, {}), field) for m in METHOD_ORDER], dtype=float)
            finite = np.isfinite(vals)
            if not finite.any():
                continue
            lo = float(np.nanmin(vals[finite]))
            hi = float(np.nanmax(vals[finite]))
            span = max(hi - lo, 1e-12)
            for row_idx, val in enumerate(vals):
                if not math.isfinite(val):
                    continue
                score = (val - lo) / span
                if not higher_better:
                    score = 1.0 - score
                score_sum[row_idx, col] += score
                score_count[row_idx, col] += 1.0
    return np.divide(score_sum, score_count, out=np.full_like(score_sum, np.nan), where=score_count > 0)


def build_fig_4_1(root: Path, out_dir: Path, dpi: int) -> list[Path]:
    """生成图4.1，主方法与 baseline 综合表现热图。"""
    rows = load_single_rows(root)
    metrics = [
        ("成功率", "success_rate", True),
        ("重规划时间", "mean_replan_ms", False),
        ("路径代价", "mean_path_cost", False),
        ("路径长度", "mean_length_km", False),
        ("风险暴露", "mean_risk_exposure_integral", False),
        ("通信覆盖", "mean_comm_coverage_ratio", True),
    ]
    scores = performance_scores(rows, metrics)
    fig, ax = plt.subplots(figsize=(7.6, 3.85))
    im = ax.imshow(scores, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m[0] for m in metrics], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels(METHOD_ORDER)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            if math.isfinite(scores[i, j]):
                ax.text(j, i, f"{scores[i, j]:.2f}", ha="center", va="center", fontsize=8.2, color="#1B1B1B")
    cb = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.025)
    cb.set_label("跨场景归一化表现，越高越好")
    ax.set_title("图4.1  主方法与baseline的综合表现", loc="left")
    ax.tick_params(length=0)
    return list(save_figure_pair(fig, out_dir, "fig_4_1_overall_performance_heatmap", dpi))


def load_experiment_rows(root: Path, filename: str) -> list[dict]:
    rows: list[dict] = []
    for scene in SCENE_CONFIGS:
        path = root / "final_results" / scene / "E3_E4_matrix_final" / filename
        for row in read_csv_rows(path):
            item = dict(row)
            item["scene"] = scene
            rows.append(item)
    return rows


def grouped_mean(rows: Sequence[dict], x_field: str, fields: Sequence[str]) -> dict[int, dict[str, float]]:
    groups: dict[int, list[dict]] = {}
    for row in rows:
        groups.setdefault(to_int(row.get(x_field)), []).append(row)
    out: dict[int, dict[str, float]] = {}
    for x, items in groups.items():
        out[x] = {}
        for field in fields:
            vals = [to_float(r.get(field)) for r in items]
            vals = [v for v in vals if math.isfinite(v)]
            out[x][field] = float(np.mean(vals)) if vals else float("nan")
    return out


def build_fig_4_2(root: Path, out_dir: Path, dpi: int) -> list[Path]:
    """生成图4.2，连续事件重规划分析。"""
    rows = load_experiment_rows(root, "experiment_B.csv")
    fields = [
        "b4_mean_cumulative_ms",
        "b4_p50_cumulative_ms",
        "b4_p95_cumulative_ms",
        "b2_mean_cumulative_ms",
        "b2_p50_cumulative_ms",
        "b2_p95_cumulative_ms",
        "b4_mean_cumulative_expanded",
        "b4_p50_cumulative_expanded",
        "b4_p95_cumulative_expanded",
        "b2_mean_cumulative_expanded",
        "b2_p50_cumulative_expanded",
        "b2_p95_cumulative_expanded",
    ]
    grouped = grouped_mean(rows, "k_events", fields)
    x = np.asarray(sorted(grouped), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
    for method, prefix in [("M-P", "b4"), ("M-A", "b2")]:
        color = METHOD_COLORS[method]
        mean = np.asarray([grouped[int(k)][f"{prefix}_mean_cumulative_ms"] for k in x])
        p50 = np.asarray([grouped[int(k)][f"{prefix}_p50_cumulative_ms"] for k in x])
        p95 = np.asarray([grouped[int(k)][f"{prefix}_p95_cumulative_ms"] for k in x])
        axes[0].plot(x, mean, marker="o", color=color, label=method)
        axes[0].fill_between(x, p50, p95, color=color, alpha=0.15, linewidth=0)
        expanded = np.asarray([grouped[int(k)][f"{prefix}_mean_cumulative_expanded"] for k in x])
        exp_p50 = np.asarray([grouped[int(k)][f"{prefix}_p50_cumulative_expanded"] for k in x])
        exp_p95 = np.asarray([grouped[int(k)][f"{prefix}_p95_cumulative_expanded"] for k in x])
        axes[1].plot(x, expanded, marker="s", color=color, label=method)
        axes[1].fill_between(x, exp_p50, exp_p95, color=color, alpha=0.14, linewidth=0)
    axes[0].set_title("a  累计重规划时间")
    axes[0].set_xlabel("连续事件数 K")
    axes[0].set_ylabel("时间，ms")
    axes[1].set_title("b  累计扩展节点")
    axes[1].set_xlabel("连续事件数 K")
    axes[1].set_ylabel("扩展节点数")
    for ax in axes:
        ax.grid(True, alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle("图4.2  连续事件重规划分析", y=1.04, fontsize=12)
    return list(save_figure_pair(fig, out_dir, "fig_4_2_consecutive_event_replanning", dpi))


def build_fig_4_3(root: Path, out_dir: Path, dpi: int) -> list[Path]:
    """生成图4.3，图规模敏感性分析。"""
    rows = load_experiment_rows(root, "experiment_C.csv")
    fields = [
        "graph_nodes",
        "graph_edges",
        "b4_success_rate",
        "b2_success_rate",
        "b2_over_b4_time_ratio",
        "b2_over_b4_time_ratio_mean_of_means",
    ]
    grouped: dict[str, dict[str, float]] = {}
    for scale in ["small", "medium", "large"]:
        items = [r for r in rows if str(r.get("scale")) == scale]
        if not items:
            continue
        grouped[scale] = {}
        for field in fields:
            vals = [to_float(r.get(field)) for r in items]
            vals = [v for v in vals if math.isfinite(v)]
            grouped[scale][field] = float(np.mean(vals)) if vals else float("nan")
    ordered = [scale for scale in ["small", "medium", "large"] if scale in grouped]
    x = np.arange(len(ordered), dtype=float)
    labels = ordered
    nodes = np.asarray([grouped[scale]["graph_nodes"] for scale in ordered], dtype=float)
    edges = np.asarray([grouped[scale]["graph_edges"] for scale in ordered], dtype=float)
    b4_success = np.asarray([grouped[scale]["b4_success_rate"] * 100.0 for scale in ordered], dtype=float)
    b2_success = np.asarray([grouped[scale]["b2_success_rate"] * 100.0 for scale in ordered], dtype=float)
    ratio = np.asarray([grouped[scale]["b2_over_b4_time_ratio_mean_of_means"] for scale in ordered], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.85), constrained_layout=True)
    ax0 = axes[0]
    ax0.bar(x - 0.17, nodes, width=0.32, color="#80B1D3", label="节点数")
    ax0.bar(x + 0.17, edges, width=0.32, color="#FDB462", label="边数")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylabel("图规模")
    ax0.set_title("a  节点数与边数")
    ax0.legend(frameon=False)

    ax1 = axes[1]
    ax1.plot(x, b4_success, marker="o", color=METHOD_COLORS["M-P"], label="M-P成功率")
    ax1.plot(x, b2_success, marker="s", color=METHOD_COLORS["M-A"], label="M-A成功率")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("成功率，%")
    ax1b = ax1.twinx()
    ax1b.axhline(1.0, color="#555555", linestyle="--", lw=1.0, alpha=0.7)
    ax1b.plot(x, ratio, marker="D", color="#4D4D4D", label="MA与MP时间比")
    ax1b.set_ylabel("时间比")
    ax1.set_title("b  成功率与时间比")
    lines, line_labels = ax1.get_legend_handles_labels()
    lines2, line_labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines + lines2, line_labels + line_labels2, frameon=False, loc="upper left")
    for ax in [ax0, ax1]:
        ax.grid(True, axis="y", alpha=0.26)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax1b.spines["top"].set_visible(False)
    fig.suptitle("图4.3  图规模敏感性分析", y=1.04, fontsize=12)
    return list(save_figure_pair(fig, out_dir, "fig_4_3_graph_scale_sensitivity", dpi))


def load_structural_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for scene in SCENE_CONFIGS:
        path = root / "final_results" / scene / "E1_E2_single_final" / "benchmark_structural_ablation.csv"
        for row in read_csv_rows(path):
            item = dict(row)
            item["scene"] = scene
            method = str(item.get("code") or item.get("method_id") or "").strip()
            if method in METHOD_ORDER:
                item["method"] = method
                rows.append(item)
    return rows


def relative_ablation_matrix(rows: Sequence[dict]) -> tuple[np.ndarray, list[str]]:
    metrics = [
        ("成功率", "success_rate", True),
        ("重规划时间", "mean_replan_ms", False),
        ("扩展节点", "mean_expanded", False),
        ("路径代价", "mean_path_cost", False),
        ("风险暴露", "mean_risk_exposure_integral", False),
        ("通信覆盖", "mean_comm_coverage_ratio", True),
    ]
    changes = np.zeros((len(METHOD_ORDER), len(metrics)), dtype=float)
    counts = np.zeros_like(changes)
    for scene in sorted({r["scene"] for r in rows}):
        by_method = {r["method"]: r for r in rows if r["scene"] == scene}
        base = by_method.get("M-P")
        if base is None:
            continue
        for col, (_label, field, higher_better) in enumerate(metrics):
            base_val = value_from(base, field)
            if not math.isfinite(base_val) or abs(base_val) < 1e-12:
                continue
            for row_idx, method in enumerate(METHOD_ORDER):
                val = value_from(by_method.get(method, {}), field)
                if not math.isfinite(val):
                    continue
                rel = (val - base_val) / abs(base_val)
                if not higher_better:
                    rel = -rel
                changes[row_idx, col] += rel * 100.0
                counts[row_idx, col] += 1.0
    matrix = np.divide(changes, counts, out=np.full_like(changes, np.nan), where=counts > 0)
    return matrix, [m[0] for m in metrics]


def build_fig_4_4(root: Path, out_dir: Path, dpi: int) -> list[Path]:
    """生成图4.4，消融实验影响热图。"""
    rows = load_structural_rows(root)
    matrix, labels = relative_ablation_matrix(rows)
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    finite = matrix[np.isfinite(matrix)]
    vmax = max(20.0, float(np.nanpercentile(np.abs(finite), 85)) if finite.size else 50.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(matrix, cmap="RdBu", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels(METHOD_ORDER)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if math.isfinite(val):
                ax.text(j, i, f"{val:+.0f}%", ha="center", va="center", fontsize=8.0, color="#111111")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cb.set_label("相对MP变化，正值表示指标改善")
    ax.set_title("图4.4  消融实验关键影响", loc="left")
    ax.tick_params(length=0)
    return list(save_figure_pair(fig, out_dir, "fig_4_4_ablation_impact_heatmap", dpi))


def scene_table_rows(root: Path) -> list[list[str]]:
    table = [["场景", "场景范围", "高程范围", "任务数量", "节点数", "边数"]]
    for scene, cfg_path in SCENE_CONFIGS.items():
        data = load_scene_arrays(root, cfg_path, 64)
        crop_km = float(data.cfg.get("crop", {}).get("crop_size_m", data.z.shape[0] * data.resolution_m)) / 1000.0
        table.append(
            [
                SCENE_NAMES_CN[scene],
                f"{crop_km:.0f} km × {crop_km:.0f} km",
                f"{float(np.nanmin(data.z)):.0f} 到 {float(np.nanmax(data.z)):.0f} m",
                str(len(data.tasks) or int(data.cfg.get("task_generation", {}).get("pair_count", 0))),
                str(len(data.nodes)),
                str(len(data.edges)),
            ]
        )
    return table


def parameter_table_rows(root: Path) -> list[list[str]]:
    cfg = load_scenario_config("scenarios/huashan.json", root)
    terrain = cfg.get("terrain_sampling", {})
    corridor = cfg.get("adaptive_corridor", {})
    comm = cfg.get("communication", {})
    return [
        ["类别", "参数", "取值"],
        ["UAV", "巡航速度", "15 m/s"],
        ["UAV", "参考功率", "500 W"],
        ["UAV", "质量", "5 kg"],
        ["UAV", "最大爬升角", "30°"],
        ["边连接", "同层最大连接距离", "250 m"],
        ["边连接", "跨层最大水平距离", "250 m"],
        ["边连接", "碰撞检测安全净空", "30 m"],
        ["安全走廊", "基础下边界偏移", f"{to_float(corridor.get('base_floor_offset_m')):.0f} m"],
        ["安全走廊", "基础上边界偏移", f"{to_float(corridor.get('base_ceiling_offset_m')):.0f} m"],
        ["安全走廊", "最小走廊厚度", f"{to_float(corridor.get('min_thickness_m')):.0f} m"],
        ["安全走廊", "三层相对位置", "0.22、0.52、0.82"],
        ["节点采样", "支路层节点预算", str(int(to_float(terrain.get("branch_node_budget"))))],
        ["节点采样", "骨干层节点预算", str(int(to_float(terrain.get("backbone_node_budget"))))],
        ["风险权重", "地形、人员、通信", "0.45、0.35、0.20"],
        ["代价权重", "时间、能耗、风险", "由 benchmark 统一设置"],
        ["区域事件", "事件半径", "0.8 km"],
        ["区域事件", "事件强度", "1.0"],
        ["通信", "最大视距范围", f"{to_float(comm.get('max_range_km')):.1f} km"],
        ["重复次数", "single benchmark", "30 次"],
        ["重复次数", "matrix关键组合", "约30 次"],
    ]


def main_result_table_rows(root: Path) -> list[list[str]]:
    rows = load_single_rows(root)
    table = [["场景", "方法", "成功率", "重规划时间 ms", "路径代价", "长度 km", "风险暴露", "通信覆盖率"]]
    for scene in SCENE_CONFIGS:
        by_method = {r["method"]: r for r in rows if r["scene"] == scene}
        for method in METHOD_ORDER:
            r = by_method.get(method)
            if not r:
                continue
            table.append(
                [
                    SCENE_NAMES_CN[scene],
                    method,
                    f"{value_from(r, 'success_rate'):.2f}",
                    f"{value_from(r, 'mean_replan_ms'):.2f}",
                    f"{value_from(r, 'mean_path_cost'):.3f}",
                    f"{value_from(r, 'mean_length_km'):.3f}",
                    f"{value_from(r, 'mean_risk_exposure_integral'):.3f}",
                    f"{value_from(r, 'mean_comm_coverage_ratio'):.3f}",
                ]
            )
    return table


def summary_table_rows(root: Path) -> list[list[str]]:
    exp_b = load_experiment_rows(root, "experiment_B.csv")
    exp_c = load_experiment_rows(root, "experiment_C.csv")
    exp_d = load_experiment_rows(root, "experiment_D.csv")
    structural = load_structural_rows(root)
    row_b = [r for r in exp_b if str(r.get("scene")) == "huashan" and to_int(r.get("k_events")) == 10][0]
    row_c = [r for r in exp_c if str(r.get("scene")) == "emeishan" and str(r.get("scale")) == "large"][0]
    row_d = [r for r in exp_d if str(r.get("scene")) == "huashan" and to_int(r.get("intensity_index")) == 4][0]
    ablation_mf = [r for r in structural if r["scene"] == "huashan" and r["method"] == "M-F"][0]
    ablation_mr = [r for r in structural if r["scene"] == "huashan" and r["method"] == "M-R"][0]
    return [
        ["实验项", "关键设置", "MP结果", "MA或对照结果", "主要含义"],
        [
            "连续事件",
            "华山、K=10",
            f"累计时间 {to_float(row_b.get('b4_mean_cumulative_ms')):.2f} ms，扩展 {to_float(row_b.get('b4_mean_cumulative_expanded')):.1f}",
            f"累计时间 {to_float(row_b.get('b2_mean_cumulative_ms')):.2f} ms，扩展 {to_float(row_b.get('b2_mean_cumulative_expanded')):.1f}",
            f"MA与MP时间比 {to_float(row_b.get('b2_over_b4_time_ratio_mean_of_means')):.2f}",
        ],
        [
            "图规模敏感性",
            "峨眉山、large、K=5",
            f"成功率 {to_float(row_c.get('b4_success_rate')):.2f}",
            f"MA成功率 {to_float(row_c.get('b2_success_rate')):.2f}",
            f"节点 {to_int(row_c.get('graph_nodes'))}，边 {to_int(row_c.get('graph_edges'))}",
        ],
        [
            "工作量机制",
            "华山、事件强度4",
            f"事件均值扩展 {to_float(row_d.get('b4_mean_event_expanded')):.1f}",
            f"MA事件均值扩展 {to_float(row_d.get('b2_mean_event_expanded')):.1f}",
            f"扩展节点减少 {100.0 * to_float(row_d.get('expanded_reduction')):.1f}%",
        ],
        [
            "消融实验",
            "华山、移除三层结构",
            f"MP风险 {value_from([r for r in structural if r['scene'] == 'huashan' and r['method'] == 'M-P'][0], 'mean_risk_exposure_integral'):.3f}",
            f"MF风险 {value_from(ablation_mf, 'mean_risk_exposure_integral'):.3f}",
            "单层图降低长度但提高风险暴露",
        ],
        [
            "消融实验",
            "华山、移除地形感知分层",
            f"MP代价 {value_from([r for r in structural if r['scene'] == 'huashan' and r['method'] == 'M-P'][0], 'mean_path_cost'):.3f}",
            f"MR代价 {value_from(ablation_mr, 'mean_path_cost'):.3f}",
            "规则分层路径质量不稳定",
        ],
    ]


def markdown_table(rows: Sequence[Sequence[str]]) -> str:
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, sep, *body])


def write_table_files(out_dir: Path, basename: str, title: str, rows: Sequence[Sequence[str]], note: str = "") -> Path:
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    md_path = table_dir / f"{basename}.md"
    csv_path = table_dir / f"{basename}.csv"
    md = f"{title}\n\n{markdown_table(rows)}"
    if note:
        md = f"{md}\n\n表注：{note}"
    md_path.write_text(md + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return md_path


def build_tables(root: Path, out_dir: Path) -> list[Path]:
    paths = [
        write_table_files(out_dir, "table_4_1_test_instances_and_graph_scale", "表4.1  测试实例与图规模", scene_table_rows(root)),
        write_table_files(out_dir, "table_4_2_parameter_settings", "表4.2  参数设置", parameter_table_rows(root)),
        write_table_files(
            out_dir,
            "table_4_3_three_scene_main_results",
            "表4.3  三场景主要对比结果",
            main_result_table_rows(root),
            "MP为本文完整方法，MA、MF、MR、MV分别表示全局A*重算、单层图LPA*、规则三层图LPA*和体素全局搜索。成功率越高越好，重规划时间、路径代价、长度和风险暴露越低越好，通信覆盖率越高越好。表中数值为重复实验均值。",
        ),
        write_table_files(
            out_dir,
            "table_4_4_key_sensitivity_and_ablation_summary",
            "表4.4  连续事件、图规模敏感性和消融实验关键汇总",
            summary_table_rows(root),
            "该表只保留支撑机制解释的关键组合，完整分布与趋势由图4.2至图4.4展示。",
        ),
    ]
    combined = "\n\n".join(path.read_text(encoding="utf-8").strip() for path in paths)
    combined_path = out_dir / "tables" / "chapter4_compact_tables.md"
    combined_path.write_text(combined + "\n", encoding="utf-8")
    paths.append(combined_path)
    return paths


def write_revised_text(out_dir: Path) -> Path:
    text = """# 论文章节结构调整正文片段

## 2.1 山地物流任务与环境输入

本文以真实山地物流任务为研究对象，将地形高程、人员暴露风险、通信视距可达性和任务端点共同作为规划环境输入。环境输入图不再采用三联图集中展示，而是拆分为三个可独立排版的单图。图2.1a给出华山场景三维地形渲染，用于说明研究区高程起伏和山脊谷地结构。图2.1b给出人员暴露风险要素，由OSM空间要素生成，用细线轮廓和点标记标示L1至L4分级OSM风险要素，不再叠加连续暴露风险色标或热力层，以避免风险来源类别和连续风险值混淆。图2.1c给出区域支路层通信视距可达性，用于表达DEM遮挡、通信源和可通信边界之间的空间关系。这三幅图均用于说明问题场景和环境输入，不作为实验对比结果。

## 3.2 DEM驱动的自适应安全飞行走廊补充说明

图3.2展示自适应安全飞行走廊与三层飞行中面的构建结果。典型剖面说明走廊下边界、上边界和三层中面随地形变化而变化，走廊厚度平面图反映复杂地形与端点邻域对可飞空间的局部调节，三维图进一步给出走廊包络和三层中面的空间关系。该图属于方法构建结果，应放在第三章安全走廊建模或三层网络构建附近，而不放入第四章实验结果。

## 3.3 路径后处理说明

图搜索得到的路径由三层航线图上的离散节点序列组成，能够直接用于统计路径代价、路径长度、风险暴露和通信覆盖等指标。为便于轨迹连续化表达和论文可视化，本文在离散路径之后加入轻量后处理。后处理先执行LOS直连剪枝，在安全飞行走廊约束满足的前提下删除冗余中间节点，再对剪枝后的折线进行B样条平滑，使轨迹形态更接近连续飞行表达。若B样条采样点越出安全走廊，则回退到LOS剪枝折线，必要时回退到原始图路径。需要强调的是，第四章实验表格中的路径代价、路径长度、风险暴露和通信覆盖仍基于离散图路径统计，LOS剪枝和B样条平滑只用于轨迹连续化表达和可视化输出，不改变实验评价口径。

## 4 实验研究重排说明

第四章应集中呈现实验对比结果，原图4.1的地形、人员暴露风险和通信环境内容前移并拆分为图2.1a至图2.1c，原图4.2的安全走廊和三层中面内容前移为图3.2。第四章表格压缩为四张，分别对应测试实例与图规模、参数设置、三场景主要对比结果，以及连续事件、图规模敏感性和消融实验关键汇总。baseline方法和评价指标不再单独占表，而是在实验设置段落与表4.3表注中说明。

图4.1使用归一化热图展示主方法与baseline在时间、代价、长度、风险和通信覆盖上的综合差异。图4.2使用折线图和误差带展示连续事件数量增加时MP与MA的累计重规划时间和累计扩展节点变化。图4.3联合展示节点数、边数、成功率和MA与MP时间比，用于解释图规模变化对重规划收益的影响。图4.4使用相对MP变化热图展示消融实验中移除增量机制、三层结构、地形感知分层和换用体素搜索后的关键影响。
"""
    path = out_dir / "revised_chapter_text.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_manifest(out_dir: Path, paths: Sequence[Path]) -> Path:
    manifest = {
        "说明": "本清单由 tools/build_revised_paper_assets.py 生成，所有实验图表均基于已有结果文件，不重跑实验。",
        "生成文件": [str(path) for path in paths],
    }
    path = out_dir / "asset_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    root = Path(args.workdir).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_out = out_dir / "figures"
    data = load_scene_arrays(root, args.scenario_config, args.max_surface_points)

    produced: list[Path] = []
    produced += build_fig_2_1a(data, figure_out, args.dpi, args.max_surface_points)
    produced += build_fig_2_1b(data, figure_out, args.dpi, args.max_surface_points)
    produced += build_fig_2_1c(data, figure_out, args.dpi, args.max_surface_points)
    produced += build_fig_3_2(data, figure_out, args.dpi, args.max_surface_points)
    produced += build_fig_3_3(data, figure_out, args.dpi)
    produced += build_fig_4_1(root, figure_out, args.dpi)
    produced += build_fig_4_2(root, figure_out, args.dpi)
    produced += build_fig_4_3(root, figure_out, args.dpi)
    produced += build_fig_4_4(root, figure_out, args.dpi)
    produced += build_tables(root, out_dir)
    produced.append(write_revised_text(out_dir))
    produced.append(write_manifest(out_dir, produced))

    print("[完成] 论文结构调整资产已生成：")
    for path in produced:
        print(f"  {path}")


if __name__ == "__main__":
    main()
