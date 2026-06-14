"""将图 3.2 从三联图重构为双联论文图。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_revised_paper_assets import (
    LAYER_COLORS,
    SceneArrays,
    dashed_segments_from_points,
    load_scene_arrays,
    line_polydata_from_segments,
    node_layers,
    transform_path_points_for_pyvista,
)


DPI = 600
OUTPUT_STEM = "fig_3_2_safe_corridor_layers"
PROFILE_OUTPUT_STEM = "fig_3_2_corridor_profile_flight_mid_surfaces"
GRAPH_OUTPUT_STEM = "fig_3_2_layered_route_graph_transparent_terrain"
SCENARIO_CONFIG = "scenarios/huashan.json"
CACHE_DIR = PROJECT_ROOT / "intermediate_artifacts" / "figures" / "huashan" / "_fig_3_2_dual_panel_cache"
INTERMEDIATE_FIGURE_DIR = PROJECT_ROOT / "intermediate_artifacts" / "figures" / "huashan"

LAYER_LABELS = ["Terminal layer", "Regional route layer", "Backbone layer"]
TERRAIN_COLOR = "#6A4A3C"
FLOOR_COLOR = "#52636E"
CEILING_COLOR = "#394B55"
CORRIDOR_COLOR = "#9EABB3"


def configure_matplotlib() -> None:
    """设置论文图的基础字体与导出参数。"""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.55,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.3,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def crop_nonwhite(image: Image.Image, padding: int = 14) -> Image.Image:
    """裁剪白色边缘，保留少量缓冲空白。"""
    rgb = np.asarray(image.convert("RGB"))
    mask = np.any(rgb < 246, axis=2)
    if not np.any(mask):
        return image.convert("RGB")
    ys, xs = np.where(mask)
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(rgb.shape[1], int(xs.max()) + padding)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(rgb.shape[0], int(ys.max()) + padding)
    return image.crop((x0, y0, x1, y1)).convert("RGB")


def sample_nearest_edges(edge_ids: np.ndarray, edge_mid_xy: np.ndarray, focus_xy: np.ndarray, max_count: int) -> np.ndarray:
    """从候选边中选取距离局部焦点最近的少量边。"""
    if len(edge_ids) <= max_count:
        return np.asarray(edge_ids, dtype=int)
    order = np.argsort(np.linalg.norm(edge_mid_xy[edge_ids] - focus_xy.reshape(1, 2), axis=1))
    return np.asarray(edge_ids[order[:max_count]], dtype=int)


def choose_focus_terminal(data: SceneArrays) -> tuple[str, np.ndarray, list[int]]:
    """选择一个同时包含端点柱、同层边和跨层边的典型局部区域。"""
    layers = node_layers(data.nodes)
    edges = np.asarray(data.edges, dtype=int)
    edge_mid_xy = 0.5 * (data.nodes[edges[:, 0], :2] + data.nodes[edges[:, 1], :2])
    graph_center = np.mean(data.nodes[:, :2], axis=0)
    best: tuple[float, str, np.ndarray, list[int]] | None = None
    for name, meta in (data.terminal_status.get("terminals") or {}).items():
        indices = [int(v) for v in meta.get("indices", [])]
        if len(indices) < 3:
            continue
        focus_xy = np.mean(data.nodes[indices, :2], axis=0)
        local = np.linalg.norm(edge_mid_xy - focus_xy.reshape(1, 2), axis=1) <= 1.25
        same_count = int(np.sum((edges[:, 2] == 0) & local))
        climb_count = int(np.sum((edges[:, 2] == 2) & local))
        vertical_count = int(np.sum((edges[:, 2] == 1) & local))
        center_penalty = float(np.linalg.norm(focus_xy - graph_center))
        score = same_count * 0.010 + climb_count * 0.025 + vertical_count * 0.50 - center_penalty * 0.45
        if best is None or score > best[0]:
            best = (score, str(name), focus_xy, indices)
    if best is None:
        return "local mechanism", np.mean(data.nodes[:, :2], axis=0), []
    return best[1], best[2], best[3]


def select_mechanism_edges(data: SceneArrays, focus_xy: np.ndarray, focus_indices: list[int]) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """只保留局部机制解释所需的节点和三类代表性边。"""
    layers = node_layers(data.nodes)
    edges = np.asarray(data.edges, dtype=int)
    u = edges[:, 0]
    v = edges[:, 1]
    edge_mid_xy = 0.5 * (data.nodes[u, :2] + data.nodes[v, :2])
    local = np.linalg.norm(edge_mid_xy - focus_xy.reshape(1, 2), axis=1) <= 1.08
    edge_types = edges[:, 2]

    same_by_layer: dict[int, np.ndarray] = {}
    for lid, max_count in enumerate((9, 16, 18)):
        candidates = np.where((edge_types == 0) & (layers[u] == lid) & (layers[v] == lid) & local)[0]
        same_by_layer[lid] = sample_nearest_edges(candidates, edge_mid_xy, focus_xy, max_count)

    focus_set = set(int(v) for v in focus_indices)
    vertical_candidates = np.where(
        (edge_types == 1)
        & (
            np.isin(u, list(focus_set))
            | np.isin(v, list(focus_set))
            | local
        )
    )[0]
    vertical_edges = sample_nearest_edges(vertical_candidates, edge_mid_xy, focus_xy, 6)

    climb_candidates = np.where((edge_types == 2) & local)[0]
    climb_edges = sample_nearest_edges(climb_candidates, edge_mid_xy, focus_xy, 18)

    selected = np.concatenate([*same_by_layer.values(), vertical_edges, climb_edges])
    selected_nodes = set(int(v) for v in focus_indices)
    for eid in selected:
        selected_nodes.add(int(edges[int(eid), 0]))
        selected_nodes.add(int(edges[int(eid), 1]))
    node_ids = np.asarray(sorted(selected_nodes), dtype=int)
    return same_by_layer, vertical_edges, climb_edges, node_ids


def polydata_from_edge_ids(data: SceneArrays, points_vtk: np.ndarray, edge_ids: np.ndarray):
    """将一组边转换为 PyVista 折线对象。"""
    edges = np.asarray(data.edges, dtype=int)
    segments = [points_vtk[[int(edges[int(eid), 0]), int(edges[int(eid), 1])], :] for eid in edge_ids]
    return line_polydata_from_segments(segments)


def dashed_polydata_from_edge_ids(data: SceneArrays, points_vtk: np.ndarray, edge_ids: np.ndarray):
    """将跨层斜向边转换为分段虚线对象。"""
    edges = np.asarray(data.edges, dtype=int)
    segments: list[np.ndarray] = []
    for eid in edge_ids:
        u, v = int(edges[int(eid), 0]), int(edges[int(eid), 1])
        segments.extend(dashed_segments_from_points(points_vtk[[u, v], :], dash_m=92.0, gap_m=58.0))
    return line_polydata_from_segments(segments)


def render_graph_scene(cache_dir: Path = CACHE_DIR) -> list[Path]:
    """使用 PyVista 渲染透明华山三维地形及三层节点边连接。"""
    import pyvista as pv
    from render_huashan_dem_pyvista import (
        REFERENCE_TERRAIN_CMAP,
        build_structured_grid,
        configure_reference_lights,
    )

    data = load_scene_arrays(PROJECT_ROOT, SCENARIO_CONFIG, max_surface_points=96)
    vertical_exag = 1.25
    grid = build_structured_grid(data.z, data.resolution_m, stride=3, vertical_exag=vertical_exag, smooth_sigma=0.42)
    bounds = grid.bounds
    x_mid = 0.5 * (bounds[0] + bounds[1])
    y_mid = 0.5 * (bounds[2] + bounds[3])
    span = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    elev_display_max = 2081.0

    pv.global_theme.smooth_shading = True
    pv.global_theme.multi_samples = 8
    plotter = pv.Plotter(off_screen=True, window_size=(2100, 1480))
    plotter.set_background("white")
    plotter.add_mesh(
        grid,
        scalars="Elevation",
        cmap=REFERENCE_TERRAIN_CMAP,
        clim=(0.0, elev_display_max),
        opacity=0.24,
        show_scalar_bar=False,
        smooth_shading=True,
        ambient=0.55,
        diffuse=0.52,
        specular=0.04,
        specular_power=15,
    )

    points_vtk = transform_path_points_for_pyvista(data.nodes[:, :3], data, vertical_exag, lift_m=96.0)
    layers = node_layers(data.nodes)
    focus_name, focus_xy, focus_indices = choose_focus_terminal(data)
    same_by_layer, vertical_edges, climb_edges, node_ids = select_mechanism_edges(data, focus_xy, focus_indices)

    for lid, color in enumerate(LAYER_COLORS):
        poly = polydata_from_edge_ids(data, points_vtk, same_by_layer[lid])
        if poly.n_points:
            plotter.add_mesh(
                poly,
                color=color,
                line_width=4.6,
                opacity=0.94,
                render_lines_as_tubes=False,
                lighting=False,
            )
    if len(vertical_edges):
        vertical_poly = polydata_from_edge_ids(data, points_vtk, vertical_edges)
        plotter.add_mesh(
            vertical_poly,
            color="#242424",
            line_width=7.0,
            opacity=0.96,
            render_lines_as_tubes=False,
            lighting=False,
        )
    if len(climb_edges):
        climb_poly = dashed_polydata_from_edge_ids(data, points_vtk, climb_edges)
        plotter.add_mesh(
            climb_poly,
            color="#5C6670",
            line_width=6.0,
            opacity=0.86,
            render_lines_as_tubes=False,
            lighting=False,
        )

    for lid, color in enumerate(LAYER_COLORS):
        layer_points = points_vtk[node_ids[layers[node_ids] == lid]]
        if len(layer_points):
            plotter.add_points(
                layer_points,
                color=color,
                point_size=24.0 if lid == 0 else 20.0,
                opacity=0.96,
                render_points_as_spheres=True,
            )

    if len(node_ids):
        focus_vtk = np.mean(points_vtk[node_ids], axis=0)
    else:
        focus_z = float(np.mean(data.nodes[focus_indices, 2])) if focus_indices else float(np.nanmedian(data.z))
        focus_vtk = transform_path_points_for_pyvista(
            np.asarray([[float(focus_xy[0]), float(focus_xy[1]), focus_z]], dtype=float),
            data,
            vertical_exag,
            lift_m=96.0,
        )[0]
    local_span = 980.0
    configure_reference_lights(plotter, (float(focus_vtk[0]), float(focus_vtk[1]), float(focus_vtk[2])), local_span, bounds[5])
    plotter.enable_anti_aliasing("ssaa")
    try:
        plotter.enable_ssao(radius=0.16, bias=0.012, kernel_size=192, blur=True)
    except Exception:
        pass

    plotter.camera_position = (
        (
            float(focus_vtk[0] - 1.55 * local_span),
            float(focus_vtk[1] - 1.18 * local_span),
            float(focus_vtk[2] + 1.38 * local_span),
        ),
        (
            float(focus_vtk[0] + 0.08 * local_span),
            float(focus_vtk[1] + 0.02 * local_span),
            float(focus_vtk[2] + 0.04 * local_span),
        ),
        (0.0, 0.0, 1.0),
    )
    plotter.camera.view_angle = 28.0
    plotter.camera.clipping_range = (10.0, 5.5 * span)
    plotter.camera.zoom(1.95)
    plotter.render()

    cache_dir.mkdir(parents=True, exist_ok=True)
    scene_path = cache_dir / "transparent_graph_scene.png"
    metadata_path = cache_dir / "metadata.json"
    image = Image.fromarray(plotter.screenshot(transparent_background=False, return_img=True)).convert("RGB")
    plotter.close()
    crop_nonwhite(image, padding=18).save(scene_path)
    metadata_path.write_text(
        json.dumps(
            {
                "node_count": int(len(data.nodes)),
                "edge_count": int(len(data.edges)),
                "shown_node_count": int(len(node_ids)),
                "shown_same_layer_edge_count": int(sum(len(v) for v in same_by_layer.values())),
                "shown_vertical_edge_count": int(len(vertical_edges)),
                "shown_cross_layer_edge_count": int(len(climb_edges)),
                "focus_terminal": focus_name,
                "elevation_max_m": elev_display_max,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return [scene_path, metadata_path]


def draw_profile(ax: plt.Axes, data: SceneArrays) -> None:
    """绘制安全走廊及三层飞行中面剖面。"""
    rows, cols = data.z.shape
    row = rows // 2
    x = np.arange(cols, dtype=float) * data.resolution_m / 1000.0
    ax.fill_between(x, data.floor[row], data.ceiling[row], color=CORRIDOR_COLOR, alpha=0.50, lw=0)
    ax.plot(x, data.z[row], color=TERRAIN_COLOR, lw=1.05)
    ax.plot(x, data.floor[row], color=FLOOR_COLOR, lw=0.95)
    ax.plot(x, data.ceiling[row], color=CEILING_COLOR, lw=0.95)
    for lid, color in enumerate(LAYER_COLORS):
        ax.plot(x, data.layer_mid[lid, row], color=color, lw=1.18)
    ax.set_xlim(float(x.min()), float(x.max()))
    ymin = float(np.nanmin(data.floor[row])) - 55.0
    ymax = float(np.nanmax(data.ceiling[row])) + 65.0
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("East-West distance (km)", labelpad=1.0)
    ax.set_ylabel("Elevation (m)", labelpad=2.0)
    ax.grid(True, color="#BDBDBD", alpha=0.28, lw=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figure_handles() -> list:
    """构建左右面板共用图例。"""
    handles: list = [
        Patch(facecolor=CORRIDOR_COLOR, edgecolor="none", alpha=0.50, label="Safe corridor"),
        Line2D([0], [0], color=TERRAIN_COLOR, lw=1.25, label="Terrain"),
        Line2D([0], [0], color=FLOOR_COLOR, lw=1.1, label="Lower boundary"),
        Line2D([0], [0], color=CEILING_COLOR, lw=1.1, label="Upper boundary"),
    ]
    handles.extend(Line2D([0], [0], color=color, lw=1.7, marker="o", markersize=3.0, label=label) for color, label in zip(LAYER_COLORS, LAYER_LABELS))
    return handles


def profile_legend_handles() -> list:
    """构建剖面单图的独立图例，图例样式与剖面线型保持一致。"""
    handles: list = [
        Patch(facecolor=CORRIDOR_COLOR, edgecolor="none", alpha=0.50, label="Safe corridor"),
        Line2D([0], [0], color=TERRAIN_COLOR, lw=1.25, label="Terrain"),
        Line2D([0], [0], color=FLOOR_COLOR, lw=1.1, label="Lower boundary"),
        Line2D([0], [0], color=CEILING_COLOR, lw=1.1, label="Upper boundary"),
    ]
    handles.extend(Line2D([0], [0], color=color, lw=1.7, label=label) for color, label in zip(LAYER_COLORS, LAYER_LABELS))
    return handles


def graph_layer_handles() -> list:
    """构建三层航线网络单图的层级颜色图例。"""
    return [
        Line2D([0], [0], color=color, lw=1.7, marker="o", markersize=3.0, label=label)
        for color, label in zip(LAYER_COLORS, LAYER_LABELS)
    ]


def edge_type_handles() -> list:
    """构建右图内部的边类型图例。"""
    return [
        Line2D([0], [0], color="#333333", lw=1.2, linestyle="-", label="Intra-layer edge"),
        Line2D([0], [0], color="#333333", lw=1.9, linestyle="-", label="Terminal vertical access"),
        Line2D([0], [0], color="#333333", lw=1.4, linestyle=(0, (3, 2)), alpha=0.80, label="Cross-layer transition"),
    ]


def export_figure(fig: plt.Figure, out_dir: Path, output_stem: str, pad_inches: float = 0.015) -> list[Path]:
    """导出 PNG、PDF、SVG 和 TIFF 四种论文图格式。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{output_stem}.png"
    pdf_path = out_dir / f"{output_stem}.pdf"
    svg_path = out_dir / f"{output_stem}.svg"
    tiff_path = out_dir / f"{output_stem}.tiff"

    fig.savefig(svg_path, bbox_inches="tight", pad_inches=pad_inches)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=pad_inches)
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    with Image.open(png_path) as image:
        image.save(tiff_path, dpi=(DPI, DPI), compression="raw")
    return [png_path, pdf_path, svg_path, tiff_path]


def copy_to_intermediate(paths: list[Path]) -> list[Path]:
    """将正式结果目录中的图同步到中间图目录。"""
    INTERMEDIATE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for path in paths:
        target = INTERMEDIATE_FIGURE_DIR / path.name
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def compose_split_figures(data: SceneArrays, scene_image: Image.Image, out_dir: Path) -> list[Path]:
    """在不影响原双联图的前提下，额外导出两张互不关联的专利用单图。"""
    outputs: list[Path] = []

    profile_fig = plt.figure(figsize=(3.58, 2.82), dpi=DPI)
    profile_ax = profile_fig.add_axes([0.135, 0.245, 0.820, 0.700])
    draw_profile(profile_ax, data)
    profile_fig.legend(
        handles=profile_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.52, 0.010),
        ncol=4,
        frameon=False,
        columnspacing=0.78,
        handlelength=1.45,
        handletextpad=0.34,
        borderaxespad=0.0,
        fontsize=5.3,
    )
    profile_paths = export_figure(profile_fig, out_dir, PROFILE_OUTPUT_STEM)
    outputs.extend(profile_paths)
    outputs.extend(copy_to_intermediate(profile_paths))

    graph_fig = plt.figure(figsize=(3.90, 2.76), dpi=DPI)
    graph_ax = graph_fig.add_axes([0.010, 0.145, 0.980, 0.840])
    graph_ax.imshow(scene_image)
    graph_ax.set_axis_off()
    edge_legend = graph_ax.legend(
        handles=edge_type_handles(),
        loc="upper right",
        bbox_to_anchor=(0.988, 0.988),
        frameon=True,
        fancybox=False,
        framealpha=0.86,
        facecolor="white",
        edgecolor="#D2D2D2",
        fontsize=5.1,
        handlelength=1.58,
        borderpad=0.26,
        labelspacing=0.25,
    )
    edge_legend.get_frame().set_linewidth(0.45)
    graph_fig.legend(
        handles=graph_layer_handles(),
        loc="lower center",
        bbox_to_anchor=(0.50, 0.018),
        ncol=3,
        frameon=False,
        columnspacing=0.92,
        handlelength=1.55,
        handletextpad=0.36,
        borderaxespad=0.0,
        fontsize=5.7,
    )
    graph_paths = export_figure(graph_fig, out_dir, GRAPH_OUTPUT_STEM)
    outputs.extend(graph_paths)
    outputs.extend(copy_to_intermediate(graph_paths))
    return outputs


def compose_figure(cache_dir: Path = CACHE_DIR) -> list[Path]:
    """拼版双联图并导出 PNG、PDF、SVG 和 TIFF。"""
    configure_matplotlib()
    data = load_scene_arrays(PROJECT_ROOT, SCENARIO_CONFIG, max_surface_points=96)
    scene_path = cache_dir / "transparent_graph_scene.png"
    if not scene_path.exists():
        raise FileNotFoundError(f"缺少三维场景缓存：{scene_path}")
    scene_image = Image.open(scene_path).convert("RGB")

    fig = plt.figure(figsize=(7.15, 3.18), dpi=DPI)
    ax_profile = fig.add_axes([0.070, 0.255, 0.385, 0.655])
    ax_scene = fig.add_axes([0.506, 0.245, 0.430, 0.675])
    draw_profile(ax_profile, data)
    ax_scene.imshow(scene_image)
    ax_scene.set_axis_off()
    edge_legend = ax_scene.legend(
        handles=edge_type_handles(),
        loc="upper right",
        bbox_to_anchor=(0.988, 0.988),
        frameon=True,
        fancybox=False,
        framealpha=0.86,
        facecolor="white",
        edgecolor="#D2D2D2",
        fontsize=4.9,
        handlelength=1.58,
        borderpad=0.26,
        labelspacing=0.25,
    )
    edge_legend.get_frame().set_linewidth(0.45)

    fig.text(0.262, 0.142, "(a) Corridor profile and flight mid surfaces", ha="center", va="top", fontsize=7.0)
    fig.text(0.721, 0.142, "(b) Layered route graph on transparent terrain", ha="center", va="top", fontsize=7.0)
    fig.legend(
        handles=figure_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=4,
        frameon=False,
        columnspacing=1.05,
        handlelength=1.8,
        handletextpad=0.45,
        borderaxespad=0.0,
        fontsize=6.2,
    )

    out_dir = PROJECT_ROOT / "final_results" / "paper_revision" / "figures"
    dual_paths = export_figure(fig, out_dir, OUTPUT_STEM)
    split_paths = compose_split_figures(data, scene_image, out_dir)
    return [*dual_paths, *copy_to_intermediate(dual_paths), *split_paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="优化图 3.2 为双联安全走廊与三层航路图。")
    parser.add_argument("--render-cache", action="store_true", help="只渲染 PyVista 三维场景缓存。")
    parser.add_argument("--compose-cache", action="store_true", help="只用现有缓存拼版并导出最终图。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.render_cache:
        paths = render_graph_scene()
    elif args.compose_cache:
        paths = compose_figure()
    else:
        render_graph_scene()
        paths = compose_figure()
    print("生成图 3.2 双联图：")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
