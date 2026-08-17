"""生成方法流程图元素小图标。

本脚本面向山地无人机物流论文的方法流程图，生成可独立使用的透明 PNG、
可编辑 SVG、总览 PNG 和中文清单。图标覆盖地形输入、风险场、飞行走廊、
三层航线网络、图抽象压缩、事件影响检测、LOS 剪枝、B-spline 平滑和连续轨迹。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
from scipy.interpolate import splev, splprep

try:
    import networkx as nx
except ImportError:
    nx = None


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "intermediate_artifacts" / "figures" / "method_flow_element_icons"
SVG_DIR = OUT_DIR / "svg"
PNG_DIR = OUT_DIR / "png"
CONTACT_SHEET = OUT_DIR / "method_flow_element_icons_contact_sheet.png"
MANIFEST = OUT_DIR / "method_flow_element_icons_manifest.md"

ICON_DPI = 256

INK = "#263746"
MUTED = "#657487"
GRID = "#C7D2DF"
BLUE = "#2B74B8"
BLUE_LIGHT = "#DCECF8"
CYAN = "#53A6B8"
GREEN = "#4C8B58"
GREEN_LIGHT = "#DFEEDA"
ORANGE = "#E49335"
YELLOW = "#F2C94C"
RED = "#D34E45"
RED_LIGHT = "#F6D7D3"
PURPLE = "#7B6BB3"
PAPER = "#F7FAFC"
WHITE = "#FFFFFF"


def choose_font_family() -> list[str]:
    """优先加载可显示中文的本机字体。"""
    font_paths = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for font_path in font_paths:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            return [font_name, "Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
    return ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": choose_font_family(),
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.7,
        "figure.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
    }
)


@dataclass(frozen=True)
class IconSpec:
    file_stem: str
    title: str
    method: str
    description: str
    builder: Callable[[], plt.Figure]


def ensure_dirs() -> None:
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)


def new_icon_axis() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(2.0, 2.0), dpi=ICON_DPI, facecolor="none")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def new_icon_3d_axis() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(2.0, 2.0), dpi=ICON_DPI, facecolor="none")
    ax = fig.add_axes((0, 0, 1, 1), projection="3d")
    ax.set_axis_off()
    ax.view_init(elev=32, azim=-55)
    ax.set_box_aspect((1, 1, 0.55))
    return fig, ax


def gaussian_field(n: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-2.8, 2.8, n)
    y = np.linspace(-2.8, 2.8, n)
    xx, yy = np.meshgrid(x, y)
    z = (
        1.10 * np.exp(-((xx + 1.05) ** 2 + (yy - 0.25) ** 2) / 1.20)
        + 0.82 * np.exp(-((xx - 1.15) ** 2 + (yy + 0.90) ** 2) / 0.82)
        + 0.52 * np.exp(-((xx - 0.10) ** 2 + (yy - 1.25) ** 2) / 1.65)
        + 0.11 * np.sin(2.2 * xx) * np.cos(1.6 * yy)
    )
    z = (z - z.min()) / (z.max() - z.min())
    return xx, yy, z


def draw_panel_background(ax: plt.Axes, color: str = PAPER) -> None:
    panel = Polygon(
        np.array([[0.08, 0.12], [0.92, 0.12], [0.92, 0.88], [0.08, 0.88]]),
        closed=True,
        facecolor=color,
        edgecolor="none",
        alpha=0.95,
        zorder=-10,
    )
    ax.add_patch(panel)


def draw_plane(
    ax: plt.Axes,
    center: tuple[float, float],
    width: float,
    height: float,
    color: str,
    edge: str = GRID,
    alpha: float = 0.72,
) -> np.ndarray:
    cx, cy = center
    pts = np.array(
        [
            [cx - width * 0.42, cy - height * 0.08],
            [cx + width * 0.18, cy + height * 0.28],
            [cx + width * 0.48, cy + height * 0.08],
            [cx - width * 0.14, cy - height * 0.30],
        ]
    )
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=edge, linewidth=1.1, alpha=alpha))
    for t in np.linspace(0.25, 0.75, 3):
        p1 = pts[0] * (1 - t) + pts[1] * t
        p2 = pts[3] * (1 - t) + pts[2] * t
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=edge, lw=0.42, alpha=0.65)
        q1 = pts[0] * (1 - t) + pts[3] * t
        q2 = pts[1] * (1 - t) + pts[2] * t
        ax.plot([q1[0], q2[0]], [q1[1], q2[1]], color=edge, lw=0.42, alpha=0.65)
    return pts


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = INK, lw: float = 1.5) -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=lw, color=color)
    ax.add_patch(arrow)


def add_lightning(ax: plt.Axes, offset: tuple[float, float], scale: float, color: str = YELLOW, edge: str = ORANGE) -> None:
    ox, oy = offset
    pts = np.array(
        [
            [0.46, 0.88],
            [0.35, 0.58],
            [0.49, 0.60],
            [0.40, 0.30],
            [0.66, 0.66],
            [0.51, 0.64],
            [0.60, 0.88],
        ]
    )
    pts = (pts - np.array([0.50, 0.60])) * scale + np.array([ox, oy])
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=edge, linewidth=1.2, joinstyle="round"))


def graph_from_edges(edges: list[tuple[str, str]], weighted: bool = False) -> list[tuple[str, str]]:
    if nx is None:
        return edges
    graph = nx.Graph()
    for index, (u, v) in enumerate(edges):
        if weighted:
            graph.add_edge(u, v, weight=1.0 + index * 0.17)
        else:
            graph.add_edge(u, v)
    return list(graph.edges())


def draw_network(
    ax: plt.Axes,
    positions: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    node_colors: dict[str, str] | None = None,
    edge_colors: dict[tuple[str, str], str] | None = None,
    edge_widths: dict[tuple[str, str], float] | None = None,
    default_node: str = CYAN,
    default_edge: str = MUTED,
    default_width: float = 1.1,
    node_size: float = 0.022,
    alpha: float = 1.0,
) -> None:
    node_colors = node_colors or {}
    edge_colors = edge_colors or {}
    edge_widths = edge_widths or {}
    for u, v in graph_from_edges(edges):
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        key = (u, v) if (u, v) in edge_colors else (v, u)
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=edge_colors.get(key, default_edge),
            lw=edge_widths.get(key, default_width),
            alpha=alpha,
            solid_capstyle="round",
            zorder=1,
        )
    for name, (x, y) in positions.items():
        ax.add_patch(
            Circle(
                (x, y),
                node_size,
                facecolor=node_colors.get(name, default_node),
                edgecolor=WHITE,
                linewidth=0.9,
                zorder=3,
            )
        )


def draw_dem_terrain() -> plt.Figure:
    fig, ax = new_icon_axis()
    xx, yy, z = gaussian_field()
    extent = (0.12, 0.88, 0.18, 0.82)
    x = np.linspace(extent[0], extent[1], z.shape[1])
    y = np.linspace(extent[2], extent[3], z.shape[0])
    ax.contourf(x, y, z, levels=26, cmap="terrain", alpha=0.96, antialiased=True)
    ax.contour(
        x,
        y,
        z,
        levels=np.linspace(0.16, 0.90, 8),
        colors=WHITE,
        linewidths=0.55,
        alpha=0.72,
    )
    ax.contour(
        x,
        y,
        z,
        levels=[0.72],
        colors=INK,
        linewidths=1.1,
        alpha=0.35,
    )
    ax.plot([0.12, 0.88, 0.88, 0.12, 0.12], [0.18, 0.18, 0.82, 0.82, 0.18], color=INK, lw=1.0, alpha=0.34)
    ax.plot([0.15, 0.30, 0.45, 0.62, 0.83], [0.26, 0.52, 0.38, 0.68, 0.49], color="#7E8B66", lw=1.4, alpha=0.70)
    return fig


def draw_risk_heatmap() -> plt.Figure:
    fig, ax = new_icon_axis()
    xx, yy, terrain = gaussian_field()
    risk = np.clip(0.78 * terrain + 0.34 * np.exp(-((xx - 0.7) ** 2 + (yy + 0.55) ** 2) / 0.36), 0, 1)
    extent = (0.10, 0.90, 0.14, 0.84)
    x = np.linspace(extent[0], extent[1], risk.shape[1])
    y = np.linspace(extent[2], extent[3], risk.shape[0])
    ax.contourf(x, y, risk, levels=28, cmap="magma", alpha=0.96, antialiased=True)
    ax.contour(
        x,
        y,
        risk,
        levels=[0.28, 0.48, 0.68, 0.82],
        colors=[BLUE_LIGHT, WHITE, YELLOW, RED_LIGHT],
        linewidths=[0.55, 0.7, 0.9, 1.1],
        alpha=0.86,
    )
    hot_spots = np.array([[0.33, 0.62], [0.61, 0.43], [0.73, 0.68]])
    ax.scatter(hot_spots[:, 0], hot_spots[:, 1], s=[34, 46, 28], color=RED, edgecolor=WHITE, linewidth=0.7, zorder=4)
    ax.add_patch(Circle((0.61, 0.43), 0.13, facecolor="none", edgecolor=RED_LIGHT, linewidth=1.6, alpha=0.9))
    return fig


def draw_risk_surface_3d() -> plt.Figure:
    fig, ax = new_icon_3d_axis()
    xx, yy, z = gaussian_field(80)
    z = 0.58 * z
    ax.plot_surface(xx, yy, z, cmap="magma", linewidth=0, antialiased=True, alpha=0.95, shade=True)
    ax.contourf(xx, yy, z, zdir="z", offset=-0.10, levels=16, cmap="magma", alpha=0.32)
    ax.plot_wireframe(xx[::8, ::8], yy[::8, ::8], z[::8, ::8], color=WHITE, linewidth=0.18, alpha=0.32)
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.8, 2.8)
    ax.set_zlim(-0.10, 0.66)
    return fig


def draw_flight_mid_surfaces() -> plt.Figure:
    fig, ax = new_icon_axis()
    x = np.linspace(0.13, 0.87, 160)
    colors = [BLUE, ORANGE, GREEN]
    fills = ["#CFE3F5", "#F7D9AE", "#D9EBCF"]
    for index, base in enumerate([0.65, 0.48, 0.31]):
        center = base + 0.035 * np.sin(2.5 * np.pi * (x - 0.12) + index * 0.42)
        tilt = 0.045 * (x - 0.50)
        upper = center + tilt + 0.045
        lower = center + tilt - 0.045
        ax.fill_between(x, lower, upper, color=fills[index], edgecolor=colors[index], linewidth=1.1, alpha=0.88, zorder=3 - index)
        ax.plot(x, center + tilt, color=colors[index], lw=1.6, alpha=0.96)
        ax.plot([x[0], x[-1]], [lower[0], lower[-1]], color=MUTED, lw=0.45, alpha=0.38)
    ax.plot([0.18, 0.80], [0.76, 0.48], color=INK, lw=0.8, alpha=0.35, linestyle=(0, (2, 2)))
    ax.plot([0.18, 0.80], [0.21, 0.48], color=INK, lw=0.8, alpha=0.35, linestyle=(0, (2, 2)))
    return fig


def draw_contours_2d() -> plt.Figure:
    fig, ax = new_icon_axis()
    xx, yy, z = gaussian_field()
    x = np.linspace(0.12, 0.88, z.shape[1])
    y = np.linspace(0.15, 0.85, z.shape[0])
    levels = np.linspace(0.18, 0.90, 8)
    ax.contourf(x, y, z, levels=levels, cmap="Greens", alpha=0.18)
    contour_colors = [GREEN, CYAN, BLUE, PURPLE, ORANGE, RED, INK]
    ax.contour(x, y, z, levels=levels[:-1], colors=contour_colors, linewidths=1.0, alpha=0.92)
    ax.plot([0.16, 0.84, 0.84, 0.16, 0.16], [0.19, 0.19, 0.81, 0.81, 0.19], color=INK, lw=0.9, alpha=0.22)
    ax.scatter([0.31, 0.69], [0.60, 0.36], s=22, color=[ORANGE, BLUE], edgecolor=WHITE, linewidth=0.6, zorder=4)
    return fig


def draw_three_layer_airway_network() -> plt.Figure:
    fig, ax = new_icon_axis()
    draw_plane(ax, (0.48, 0.72), 0.78, 0.30, BLUE_LIGHT, "#9CBFDA", 0.62)
    draw_plane(ax, (0.52, 0.51), 0.78, 0.30, GREEN_LIGHT, "#A9CFA4", 0.62)
    draw_plane(ax, (0.48, 0.30), 0.78, 0.30, "#F8E0B7", "#D5B27A", 0.64)

    positions = {
        "t1": (0.22, 0.24),
        "t2": (0.42, 0.31),
        "t3": (0.66, 0.27),
        "t4": (0.80, 0.36),
        "r1": (0.25, 0.48),
        "r2": (0.48, 0.55),
        "r3": (0.69, 0.49),
        "r4": (0.80, 0.58),
        "b1": (0.24, 0.69),
        "b2": (0.48, 0.76),
        "b3": (0.72, 0.70),
        "b4": (0.82, 0.77),
    }
    edges = [
        ("t1", "t2"),
        ("t2", "t3"),
        ("t3", "t4"),
        ("r1", "r2"),
        ("r2", "r3"),
        ("r3", "r4"),
        ("b1", "b2"),
        ("b2", "b3"),
        ("b3", "b4"),
        ("t2", "r2"),
        ("t4", "r4"),
        ("r1", "b1"),
        ("r3", "b3"),
        ("t1", "r1"),
        ("r4", "b4"),
    ]
    node_colors = {name: ORANGE for name in positions if name.startswith("t")}
    node_colors.update({name: GREEN for name in positions if name.startswith("r")})
    node_colors.update({name: BLUE for name in positions if name.startswith("b")})
    draw_network(ax, positions, edges, node_colors=node_colors, default_edge=INK, default_width=1.15, node_size=0.024, alpha=0.86)
    return fig


def draw_graph_abstraction_compression() -> plt.Figure:
    fig, ax = new_icon_axis()
    left_pos = {
        "a": (0.12, 0.33),
        "b": (0.18, 0.62),
        "c": (0.29, 0.45),
        "d": (0.34, 0.70),
        "e": (0.39, 0.25),
        "f": (0.49, 0.56),
        "g": (0.52, 0.34),
    }
    dense_edges = [
        ("a", "b"),
        ("a", "c"),
        ("a", "e"),
        ("b", "c"),
        ("b", "d"),
        ("c", "d"),
        ("c", "e"),
        ("c", "f"),
        ("d", "f"),
        ("e", "f"),
        ("e", "g"),
        ("f", "g"),
        ("b", "f"),
    ]
    draw_network(
        ax,
        left_pos,
        dense_edges,
        default_node=MUTED,
        default_edge=GRID,
        default_width=0.8,
        node_size=0.017,
        alpha=0.76,
    )

    right_pos = {
        "a": (0.68, 0.31),
        "b": (0.70, 0.62),
        "d": (0.84, 0.69),
        "g": (0.88, 0.34),
        "f": (0.79, 0.50),
    }
    compressed_edges = [("a", "b"), ("b", "d"), ("d", "f"), ("f", "g"), ("a", "g"), ("b", "f")]
    if nx is not None:
        graph = nx.Graph()
        graph.add_weighted_edges_from(
            [
                ("a", "b", 1.0),
                ("b", "d", 1.1),
                ("d", "f", 0.9),
                ("f", "g", 1.2),
                ("a", "g", 1.7),
                ("b", "f", 1.4),
            ]
        )
        tree = list(nx.minimum_spanning_tree(graph).edges())
        compressed_edges = tree + [("a", "g")]
    draw_network(
        ax,
        right_pos,
        compressed_edges,
        default_node=BLUE,
        default_edge=BLUE,
        default_width=1.35,
        node_size=0.022,
        alpha=0.96,
    )
    add_arrow(ax, (0.55, 0.50), (0.64, 0.50), BLUE, 1.8)
    ax.add_patch(Circle((0.79, 0.50), 0.11, facecolor=BLUE_LIGHT, edgecolor=BLUE, linewidth=0.9, alpha=0.38))
    return fig


def draw_affected_edge_detection() -> plt.Figure:
    fig, ax = new_icon_axis()
    positions = {
        "n1": (0.18, 0.32),
        "n2": (0.25, 0.64),
        "n3": (0.45, 0.48),
        "n4": (0.58, 0.72),
        "n5": (0.72, 0.55),
        "n6": (0.80, 0.28),
        "n7": (0.48, 0.24),
    }
    edges = [
        ("n1", "n2"),
        ("n1", "n3"),
        ("n1", "n7"),
        ("n2", "n3"),
        ("n3", "n4"),
        ("n3", "n5"),
        ("n3", "n7"),
        ("n4", "n5"),
        ("n5", "n6"),
        ("n6", "n7"),
    ]
    affected = {("n3", "n4"), ("n3", "n5"), ("n3", "n7")}
    edge_colors = {edge: ORANGE for edge in affected}
    edge_widths = {edge: 2.8 for edge in affected}
    draw_network(
        ax,
        positions,
        edges,
        default_node=CYAN,
        default_edge=MUTED,
        default_width=1.0,
        edge_colors=edge_colors,
        edge_widths=edge_widths,
        node_size=0.024,
        alpha=0.92,
    )
    ax.add_patch(Circle((0.46, 0.49), 0.19, facecolor=RED_LIGHT, edgecolor=RED, linewidth=1.2, alpha=0.42, zorder=0))
    add_lightning(ax, (0.43, 0.55), 0.55)
    return fig


def draw_los_pruning() -> plt.Figure:
    fig, ax = new_icon_axis()
    path_points = np.array(
        [
            [0.10, 0.28],
            [0.23, 0.58],
            [0.39, 0.40],
            [0.55, 0.69],
            [0.72, 0.53],
            [0.88, 0.76],
        ]
    )
    pruned = path_points[[0, 2, 4, 5]]
    ax.add_patch(Circle((0.52, 0.50), 0.11, facecolor="#E7ECE9", edgecolor="#9BAAA0", linewidth=0.8, alpha=0.85))
    ax.plot(path_points[:, 0], path_points[:, 1], color=MUTED, lw=1.2, linestyle=(0, (3, 3)), alpha=0.82)
    ax.scatter(path_points[:, 0], path_points[:, 1], s=22, color=WHITE, edgecolor=MUTED, linewidth=1.0, zorder=3)
    ax.plot(pruned[:, 0], pruned[:, 1], color=BLUE, lw=2.6, solid_capstyle="round", zorder=4)
    ax.scatter(pruned[:, 0], pruned[:, 1], s=32, color=BLUE, edgecolor=WHITE, linewidth=0.8, zorder=5)
    for point in path_points[[1, 3]]:
        ax.plot([point[0] - 0.035, point[0] + 0.035], [point[1] - 0.035, point[1] + 0.035], color=RED, lw=2.1, zorder=6)
        ax.plot([point[0] - 0.035, point[0] + 0.035], [point[1] + 0.035, point[1] - 0.035], color=RED, lw=2.1, zorder=6)
    add_arrow(ax, (0.13, 0.22), (0.84, 0.72), CYAN, 1.0)
    return fig


def draw_bspline_smoothing() -> plt.Figure:
    fig, ax = new_icon_axis()
    controls = np.array(
        [
            [0.10, 0.30],
            [0.22, 0.55],
            [0.35, 0.42],
            [0.50, 0.70],
            [0.68, 0.50],
            [0.88, 0.76],
        ]
    )
    tck, _ = splprep([controls[:, 0], controls[:, 1]], s=0.018, k=3)
    uu = np.linspace(0, 1, 160)
    sx, sy = splev(uu, tck)
    ax.plot(controls[:, 0], controls[:, 1], color=MUTED, lw=1.1, linestyle=(0, (3, 3)), alpha=0.76)
    ax.scatter(controls[:, 0], controls[:, 1], s=22, color=ORANGE, edgecolor=WHITE, linewidth=0.7, zorder=4)
    ax.plot(sx, sy, color=BLUE, lw=3.0, solid_capstyle="round", zorder=5)
    ax.fill_between(sx, np.array(sy) - 0.035, np.array(sy) + 0.035, color=BLUE_LIGHT, alpha=0.36, zorder=2)
    add_arrow(ax, (0.23, 0.23), (0.76, 0.30), GREEN, 1.1)
    return fig


def draw_continuous_trajectory() -> plt.Figure:
    fig, ax = new_icon_axis()
    controls = np.array(
        [
            [0.10, 0.24],
            [0.25, 0.36],
            [0.37, 0.61],
            [0.56, 0.48],
            [0.70, 0.72],
            [0.90, 0.82],
        ]
    )
    tck, _ = splprep([controls[:, 0], controls[:, 1]], s=0.030, k=3)
    uu = np.linspace(0, 1, 180)
    sx, sy = splev(uu, tck)
    sx = np.asarray(sx)
    sy = np.asarray(sy)
    lower = sy - 0.055
    upper = sy + 0.055
    ax.fill_between(sx, lower, upper, color=GREEN_LIGHT, alpha=0.44, zorder=1)
    points = np.column_stack([sx, sy]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    collection = LineCollection(segments, cmap="viridis", linewidth=3.2, capstyle="round", zorder=5)
    collection.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(collection)
    ax.scatter([sx[0], sx[-1]], [sy[0], sy[-1]], s=[44, 52], color=[GREEN, RED], edgecolor=WHITE, linewidth=1.0, zorder=6)
    direction = np.array([sx[-1] - sx[-8], sy[-1] - sy[-8]])
    direction = direction / np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    tip = np.array([sx[-1], sy[-1]])
    tail = tip - direction * 0.07
    wing = normal * 0.035
    ax.add_patch(Polygon(np.vstack([tip, tail + wing, tail - wing]), closed=True, facecolor=INK, edgecolor=WHITE, linewidth=0.7, zorder=7))
    ax.plot(sx[::18], upper[::18], color=GREEN, lw=0.6, alpha=0.48)
    ax.plot(sx[::18], lower[::18], color=GREEN, lw=0.6, alpha=0.48)
    return fig


ICONS = [
    IconSpec("dem_terrain", "DEM 地形", "Python 生成，矢量简化", "以合成高程场表达山地 DEM 输入", draw_dem_terrain),
    IconSpec("risk_heatmap", "风险热力图", "Python", "以二维热场表达规划层风险字段", draw_risk_heatmap),
    IconSpec("risk_surface_3d", "三维风险曲面", "Python", "以三维曲面表达风险积分场的空间起伏", draw_risk_surface_3d),
    IconSpec("flight_mid_surfaces", "flight mid-surfaces", "Python 曲线面", "以三条中面表达自适应飞行走廊层", draw_flight_mid_surfaces),
    IconSpec("contours_2d", "2D contours", "Python", "以等值线表达走廊投影与高度分区", draw_contours_2d),
    IconSpec("three_layer_airway_network", "三层航线网络", "Python，networkx 语义", "以终端、区域和骨干三层节点表达航线网络", draw_three_layer_airway_network),
    IconSpec("graph_abstraction_compression", "图抽象压缩示意", "Python，networkx 语义", "以稠密图到压缩图表达图规模削减", draw_graph_abstraction_compression),
    IconSpec("affected_edge_detection", "受影响边检测", "Python，networkx 语义", "以局部事件半径标出受扰边集", draw_affected_edge_detection),
    IconSpec("los_pruning", "LOS pruning", "Python，networkx 语义", "以可视直连删除冗余离散节点", draw_los_pruning),
    IconSpec("bspline_smoothing", "B-spline smoothing", "Python，scipy", "以控制点和样条曲线表达轨迹平滑", draw_bspline_smoothing),
    IconSpec("continuous_trajectory", "continuous trajectory", "Python", "以渐变连续曲线表达最终飞行轨迹", draw_continuous_trajectory),
]


def save_icon(spec: IconSpec) -> dict[str, str]:
    fig = spec.builder()
    svg_path = SVG_DIR / f"{spec.file_stem}.svg"
    png_path = PNG_DIR / f"{spec.file_stem}.png"
    fig.savefig(svg_path, format="svg", transparent=True)
    fig.savefig(png_path, format="png", dpi=ICON_DPI, transparent=True)
    plt.close(fig)
    return {
        "file_stem": spec.file_stem,
        "title": spec.title,
        "method": spec.method,
        "description": spec.description,
        "svg": str(svg_path),
        "png": str(png_path),
    }


def make_contact_sheet(rows: list[dict[str, str]]) -> None:
    cols = 4
    sheet_rows = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(sheet_rows, cols, figsize=(cols * 2.45, sheet_rows * 2.45), dpi=180)
    axes_array = np.atleast_1d(axes).ravel()
    for ax in axes_array:
        ax.axis("off")
    for ax, row in zip(axes_array, rows, strict=False):
        image = plt.imread(row["png"])
        ax.imshow(image)
        ax.text(0.03, -0.08, row["title"], transform=ax.transAxes, ha="left", va="top", fontsize=9, color=INK)
        ax.text(0.03, -0.18, row["file_stem"], transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=MUTED)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.08, wspace=0.18, hspace=0.36)
    fig.savefig(CONTACT_SHEET, dpi=180, facecolor=WHITE, transparent=False)
    plt.close(fig)


def write_manifest(rows: list[dict[str, str]]) -> None:
    lines = [
        "# 方法流程图元素图标清单",
        "",
        "本目录由 `tools/generate_method_flow_element_icons.py` 生成。图标用于搭建方法流程图中的语义元素，整体遵循地形输入、风险场构建、飞行走廊生成、航线网络建模、事件驱动局部更新和连续轨迹后处理的证据链。",
        "",
        f"图标总览：`{CONTACT_SHEET.name}`",
        "",
        "输出格式：每个元素同时包含可缩放 SVG 和透明 PNG。SVG 中的曲线、节点、等值线和面片保持为图形路径，PNG 为 512 像素方形画布，适合直接插入 Word、PPT 和图形编辑器。",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        svg_rel = Path(row["svg"]).relative_to(OUT_DIR)
        png_rel = Path(row["png"]).relative_to(OUT_DIR)
        lines.extend(
            [
                f"{index}. {row['title']}",
                f"   文件名：`{row['file_stem']}`",
                f"   推荐方式：{row['method']}",
                f"   图标含义：{row['description']}",
                f"   SVG：`{svg_rel}`",
                f"   PNG：`{png_rel}`",
                "",
            ]
        )
    MANIFEST.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    rows = [save_icon(spec) for spec in ICONS]
    make_contact_sheet(rows)
    write_manifest(rows)
    print(f"输出目录：{OUT_DIR}")
    print(f"图标数量：{len(rows)}")
    print(f"总览图：{CONTACT_SHEET}")
    print(f"清单：{MANIFEST}")


if __name__ == "__main__":
    main()
