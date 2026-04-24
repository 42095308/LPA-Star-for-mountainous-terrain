"""
基于既有 DEM、可飞走廊和分层图结果绘制论文主图。

输出四个子图：
1. DEM 山体与安全走廊
2. 三层自适应飞行曲面
3. 三层航路拓扑与终端锚点柱
4. 三层网络上的示例路径
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from article_planner.scenario_config import (
    load_scenario_config,
    resolve_resolution_m,
    scenario_output_dir,
)


matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

LAYER_LABELS = [
    "Terminal Approach Layer",
    "Regional Branch Layer",
    "Backbone Layer",
]
LAYER_SHORT_LABELS = ["Terminal", "Branch", "Backbone"]
LAYER_COLORS = ["#2196F3", "#4CAF50", "#FF8F00"]
CORRIDOR_FLOOR_COLOR = "#5E7A87"
CORRIDOR_CEILING_COLOR = "#B0BEC5"
EXPLODE_OFFSETS_M = np.array([0.0, 80.0, 160.0], dtype=float)
FIGURE_NAME = "three_layer_framework"
RNG_SEED = 20260424


@dataclass
class FigureData:
    """主图绘制所需的全部数据。"""

    scene_name: str
    output_dir: Path
    figure_dir: Path
    resolution_m: float
    z: np.ndarray
    floor: np.ndarray
    ceiling: np.ndarray
    layer_mid: np.ndarray
    layer_allowed: np.ndarray
    nodes: np.ndarray
    edges: np.ndarray
    terminal_status: dict
    tasks: list[dict]
    rows: int
    cols: int
    node_layers: np.ndarray
    x_full_km: np.ndarray
    y_full_km: np.ndarray
    x_plot_km: np.ndarray
    y_plot_km: np.ndarray
    stride: int
    edge_lengths_m: np.ndarray
    adjacency: list[list[tuple[int, int]]]


@dataclass
class ExampleRoute:
    """示例路径及其说明信息。"""

    task_id: str
    start_name: str
    goal_name: str
    path: list[int]
    layer_sequence: list[int]
    distance_km: float


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="绘制 Terrain-aware three-layer airway network construction 主图。")
    parser.add_argument("--scenario-config", type=str, default="", help="场景配置 JSON 路径。")
    parser.add_argument("--workdir", type=str, default=".", help="项目工作目录。")
    parser.add_argument("--task-id", type=str, default="", help="可选：指定用于子图 D 的任务 ID。")
    parser.add_argument("--dpi", type=int, default=320, help="PNG 导出分辨率。")
    parser.add_argument("--max-surface-points", type=int, default=92, help="单边最多绘制多少个曲面采样点。")
    return parser.parse_args()


def choose_stride(rows: int, cols: int, max_surface_points: int) -> int:
    """根据 DEM 尺寸自动选择曲面降采样步长。"""
    max_dim = max(int(rows), int(cols))
    target = max(24, int(max_surface_points))
    return max(1, int(math.ceil(max_dim / target)))


def load_json(path: Path) -> dict:
    """读取 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_pair(u: int, v: int) -> tuple[int, int]:
    """把无向边端点标准化为有序二元组。"""
    return (int(u), int(v)) if int(u) <= int(v) else (int(v), int(u))


def km_to_rc(x_km: float, y_km: float, rows: int, cols: int, resolution_m: float) -> tuple[int, int]:
    """km 坐标转换为栅格行列号。"""
    c = int(np.clip(round(x_km * 1000.0 / resolution_m), 0, cols - 1))
    r = int(np.clip(round((rows - 1) - y_km * 1000.0 / resolution_m), 0, rows - 1))
    return r, c


def build_xy_grids(rows: int, cols: int, resolution_m: float, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构建完整和降采样后的 km 坐标网格。"""
    x_full = np.arange(cols, dtype=float) * resolution_m / 1000.0
    y_full = np.arange(rows - 1, -1, -1, dtype=float) * resolution_m / 1000.0
    x_plot = x_full[::stride]
    y_plot = y_full[::stride]
    xx_plot, yy_plot = np.meshgrid(x_plot, y_plot)
    return x_full, y_full, xx_plot, yy_plot


def build_graph_structures(nodes: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    """基于节点和边构建边长数组与邻接表。"""
    edge_lengths = np.zeros(len(edges), dtype=float)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(len(nodes))]
    for eid, edge in enumerate(edges):
        u, v = int(edge[0]), int(edge[1])
        dx = float(nodes[v, 0] - nodes[u, 0]) * 1000.0
        dy = float(nodes[v, 1] - nodes[u, 1]) * 1000.0
        dz = float(nodes[v, 2] - nodes[u, 2])
        length_m = math.sqrt(dx * dx + dy * dy + dz * dz)
        edge_lengths[eid] = length_m
        adjacency[u].append((v, eid))
        adjacency[v].append((u, eid))
    return edge_lengths, adjacency


def load_figure_data(args: argparse.Namespace) -> FigureData:
    """读取主图绘制所需的现有结果文件。"""
    root = Path(args.workdir).resolve()
    scene_cfg = load_scenario_config(args.scenario_config or None, root)
    scene_name = str(scene_cfg.get("scene_name", "default"))
    output_dir = scenario_output_dir(scene_cfg, root)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        "Z_crop.npy",
        "floor.npy",
        "ceiling.npy",
        "layer_mid.npy",
        "layer_allowed.npy",
        "graph_nodes.npy",
        "graph_edges.npy",
        "graph_terminal_status.json",
    ]
    for filename in required_files:
        path = output_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"缺少 {path}，请先完成对应场景的建图流程。")

    z = np.asarray(np.load(output_dir / "Z_crop.npy"), dtype=float)
    floor = np.asarray(np.load(output_dir / "floor.npy"), dtype=float)
    ceiling = np.asarray(np.load(output_dir / "ceiling.npy"), dtype=float)
    layer_mid = np.asarray(np.load(output_dir / "layer_mid.npy"), dtype=float)
    layer_allowed = np.asarray(np.load(output_dir / "layer_allowed.npy"), dtype=bool)
    nodes = np.asarray(np.load(output_dir / "graph_nodes.npy"), dtype=float)
    edges = np.asarray(np.load(output_dir / "graph_edges.npy"), dtype=int)
    terminal_status = load_json(output_dir / "graph_terminal_status.json")

    tasks: list[dict] = []
    task_path = output_dir / "generated_tasks.json"
    if task_path.exists():
        task_payload = load_json(task_path)
        tasks = [dict(item) for item in task_payload.get("tasks", [])]

    rows, cols = z.shape
    resolution_m = resolve_resolution_m(scene_cfg, output_dir)
    stride = choose_stride(rows, cols, args.max_surface_points)
    x_full_km, y_full_km, x_plot_km, y_plot_km = build_xy_grids(rows, cols, resolution_m, stride)
    node_layers = np.rint(nodes[:, 3]).astype(int)
    edge_lengths_m, adjacency = build_graph_structures(nodes, edges)

    return FigureData(
        scene_name=scene_name,
        output_dir=output_dir,
        figure_dir=figure_dir,
        resolution_m=resolution_m,
        z=z,
        floor=floor,
        ceiling=ceiling,
        layer_mid=layer_mid,
        layer_allowed=layer_allowed,
        nodes=nodes,
        edges=edges,
        terminal_status=terminal_status,
        tasks=tasks,
        rows=rows,
        cols=cols,
        node_layers=node_layers,
        x_full_km=x_full_km,
        y_full_km=y_full_km,
        x_plot_km=x_plot_km,
        y_plot_km=y_plot_km,
        stride=stride,
        edge_lengths_m=edge_lengths_m,
        adjacency=adjacency,
    )


def choose_example_task(data: FigureData, task_id: str = "") -> tuple[dict, str, str]:
    """选择用于子图 D 的示例任务。"""
    terminals = data.terminal_status.get("terminals", {})
    reachable_tasks = []
    for task in data.tasks:
        if task_id and str(task.get("task_id", "")) != str(task_id):
            continue
        start_name = str(task.get("depot", ""))
        goal_name = str(task.get("target", ""))
        if start_name not in terminals or goal_name not in terminals:
            continue
        if not terminals[start_name].get("reachable", False):
            continue
        if not terminals[goal_name].get("reachable", False):
            continue
        reachable_tasks.append((task, start_name, goal_name))

    if reachable_tasks:
        reachable_tasks.sort(
            key=lambda item: (
                float(item[0].get("distance_km", 0.0)),
                float(item[0].get("elevation_gain_m", 0.0)),
            ),
            reverse=True,
        )
        return reachable_tasks[0]

    if task_id:
        raise ValueError(f"未找到 task_id={task_id} 的可达任务。")

    depots = []
    targets = []
    for name, meta in terminals.items():
        indices = meta.get("indices", [])
        if len(indices) < 3 or not meta.get("reachable", False):
            continue
        if str(meta.get("source")) == "virtual_depot":
            depots.append((name, meta))
        else:
            targets.append((name, meta))

    if not depots or not targets:
        raise RuntimeError("缺少可用于示例路径的配送站或目标点。")

    best = None
    for depot_name, depot_meta in depots:
        for target_name, target_meta in targets:
            s_idx = int(depot_meta["indices"][0])
            g_idx = int(target_meta["indices"][0])
            distance_km = float(np.linalg.norm(data.nodes[s_idx, :2] - data.nodes[g_idx, :2]))
            task = {
                "task_id": "AUTO",
                "depot": depot_name,
                "target": target_name,
                "distance_km": distance_km,
                "elevation_gain_m": float(data.nodes[g_idx, 2] - data.nodes[s_idx, 2]),
            }
            candidate = (distance_km, task, depot_name, target_name)
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None:
        raise RuntimeError("自动选择示例路径失败。")
    return best[1], best[2], best[3]


def solve_semantic_example_path(data: FigureData, start_idx: int, goal_idx: int) -> list[int]:
    """
    求一条具有层级语义的示例路径。

    状态机约束：
    1. 起点阶段只能在末端/支路层接入；
    2. 必须通过斜向爬升边进入骨干层；
    3. 必须再通过斜向爬升边下降回支路层；
    4. 最终通过末端层完成进近。
    """

    def layer_of(node_idx: int) -> int:
        return int(data.node_layers[int(node_idx)])

    start_state = (int(start_idx), 0)
    goal_state = (int(goal_idx), 2)
    dist: dict[tuple[int, int], float] = {start_state: 0.0}
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start_state: None}
    heap: list[tuple[float, tuple[int, int]]] = [(0.0, start_state)]

    while heap:
        cur_dist, state = heapq.heappop(heap)
        if not math.isclose(cur_dist, dist.get(state, float("inf")), rel_tol=1e-12, abs_tol=1e-12):
            continue
        if state == goal_state:
            break

        u, phase = state
        lu = layer_of(u)
        for v, eid in data.adjacency[u]:
            edge_type = int(data.edges[eid, 2])
            lv = layer_of(v)
            next_phase: int | None = None

            if phase == 0:
                if edge_type == 1 and {lu, lv} == {0, 1}:
                    next_phase = 0
                elif edge_type == 0 and lu in {0, 1} and lv in {0, 1}:
                    next_phase = 0
                elif edge_type == 2 and lu == 1 and lv == 2:
                    next_phase = 1
            elif phase == 1:
                if edge_type == 0 and lu in {1, 2} and lv in {1, 2}:
                    next_phase = 1
                elif edge_type == 2 and lu == 1 and lv == 2:
                    next_phase = 1
                elif edge_type == 2 and lu == 2 and lv == 1:
                    next_phase = 2
            else:
                if edge_type == 1 and {lu, lv} == {0, 1}:
                    next_phase = 2
                elif edge_type == 0 and lu in {0, 1} and lv in {0, 1}:
                    next_phase = 2

            if next_phase is None:
                continue

            new_dist = cur_dist + float(data.edge_lengths_m[eid])
            next_state = (int(v), int(next_phase))
            if new_dist + 1e-12 < dist.get(next_state, float("inf")):
                dist[next_state] = new_dist
                prev[next_state] = state
                heapq.heappush(heap, (new_dist, next_state))

    if goal_state in dist:
        states: list[tuple[int, int]] = []
        cur_state: tuple[int, int] | None = goal_state
        while cur_state is not None:
            states.append(cur_state)
            cur_state = prev[cur_state]
        states.reverse()
        return [int(node_idx) for node_idx, _phase in states]

    # 如果严格语义路径不可达，则退化为普通最短路，保证脚本仍能出图。
    dist_nodes = np.full(len(data.nodes), float("inf"), dtype=float)
    prev_nodes = np.full(len(data.nodes), -1, dtype=int)
    dist_nodes[start_idx] = 0.0
    heap_nodes: list[tuple[float, int]] = [(0.0, int(start_idx))]
    while heap_nodes:
        cur_dist, u = heapq.heappop(heap_nodes)
        if not math.isclose(cur_dist, float(dist_nodes[u]), rel_tol=1e-12, abs_tol=1e-12):
            continue
        if u == int(goal_idx):
            break
        for v, eid in data.adjacency[u]:
            new_dist = cur_dist + float(data.edge_lengths_m[eid])
            if new_dist + 1e-12 < float(dist_nodes[v]):
                dist_nodes[v] = new_dist
                prev_nodes[v] = int(u)
                heapq.heappush(heap_nodes, (new_dist, int(v)))

    if not np.isfinite(dist_nodes[goal_idx]):
        raise RuntimeError("示例路径求解失败：起点与终点在当前图中不连通。")

    path = []
    cur = int(goal_idx)
    while cur >= 0:
        path.append(cur)
        if cur == int(start_idx):
            break
        cur = int(prev_nodes[cur])
    path.reverse()
    return path


def build_example_route(data: FigureData, task_id: str = "") -> ExampleRoute:
    """构建子图 D 所需的示例路径信息。"""
    task, start_name, goal_name = choose_example_task(data, task_id=task_id)
    terminals = data.terminal_status.get("terminals", {})
    start_idx = int(terminals[start_name]["indices"][0])
    goal_idx = int(terminals[goal_name]["indices"][0])
    path = solve_semantic_example_path(data, start_idx, goal_idx)
    layer_sequence = []
    for node_idx in path:
        layer_id = int(data.node_layers[node_idx])
        if not layer_sequence or layer_sequence[-1] != layer_id:
            layer_sequence.append(layer_id)

    route_distance_km = 0.0
    for idx in range(len(path) - 1):
        u = path[idx]
        v = path[idx + 1]
        route_distance_km += float(
            np.linalg.norm(
                [
                    (data.nodes[v, 0] - data.nodes[u, 0]) * 1000.0,
                    (data.nodes[v, 1] - data.nodes[u, 1]) * 1000.0,
                    data.nodes[v, 2] - data.nodes[u, 2],
                ]
            )
            / 1000.0
        )

    return ExampleRoute(
        task_id=str(task.get("task_id", "AUTO")),
        start_name=start_name,
        goal_name=goal_name,
        path=path,
        layer_sequence=layer_sequence,
        distance_km=route_distance_km,
    )


def set_surface_rasterized(surface) -> None:
    """让 3D 曲面在 PDF 中按栅格输出，避免矢量文件过大。"""
    try:
        surface.set_rasterized(True)
    except Exception:
        pass


def add_surface(
    ax,
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    **kwargs,
):
    """统一包装 3D 曲面绘制。"""
    surface = ax.plot_surface(xx, yy, zz, linewidth=0.0, antialiased=False, shade=False, **kwargs)
    set_surface_rasterized(surface)
    return surface


def prepare_surface(data: FigureData, array_2d: np.ndarray, offset_m: float = 0.0, mask: np.ndarray | None = None) -> np.ndarray:
    """把二维栅格切成绘图分辨率并叠加可视化偏移。"""
    sampled = np.asarray(array_2d[:: data.stride, :: data.stride], dtype=float) + float(offset_m)
    if mask is not None:
        sampled = np.ma.masked_where(~mask[:: data.stride, :: data.stride], sampled)
    return sampled


def style_3d_axes(
    ax,
    data: FigureData,
    z_min: float,
    z_max: float,
    elev: float,
    azim: float,
) -> None:
    """统一设置 3D 视角与坐标轴风格。"""
    ax.set_xlim(float(data.x_full_km.min()), float(data.x_full_km.max()))
    ax.set_ylim(float(data.y_full_km.min()), float(data.y_full_km.max()))
    ax.set_zlim(float(z_min), float(z_max))
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.0, 1.0, 0.45))
    ax.set_xlabel("East-West (km)", labelpad=8)
    ax.set_ylabel("South-North (km)", labelpad=8)
    ax.set_zlabel("Altitude (m)", labelpad=8)
    ax.tick_params(labelsize=8)
    ax.xaxis.pane.set_alpha(0.02)
    ax.yaxis.pane.set_alpha(0.02)
    ax.zaxis.pane.set_alpha(0.02)
    ax.grid(True, alpha=0.25)


def sample_indices(indices: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """固定随机种子采样索引，避免图面过密。"""
    if ratio >= 1.0 or len(indices) == 0:
        return indices
    count = max(1, int(round(len(indices) * ratio)))
    if count >= len(indices):
        return indices
    chosen = rng.choice(indices, size=count, replace=False)
    return np.sort(chosen.astype(int))


def edge_pair_set_from_terminal_status(terminal_status: dict) -> set[tuple[int, int]]:
    """提取所有终端接入边。"""
    pairs: set[tuple[int, int]] = set()
    for meta in (terminal_status.get("terminals") or {}).values():
        for edge in meta.get("connect_edges", []):
            if len(edge) >= 2:
                pairs.add(normalize_pair(int(edge[0]), int(edge[1])))
    return pairs


def plot_terrain_and_corridor(ax, data: FigureData) -> None:
    """子图 A：DEM 山体与安全走廊。"""
    z_surface = prepare_surface(data, data.z)
    floor_surface = prepare_surface(data, data.floor)
    ceiling_surface = prepare_surface(data, data.ceiling)

    add_surface(
        ax,
        data.x_plot_km,
        data.y_plot_km,
        z_surface,
        cmap="terrain",
        alpha=0.96,
    )
    add_surface(
        ax,
        data.x_plot_km,
        data.y_plot_km,
        floor_surface,
        color=CORRIDOR_FLOOR_COLOR,
        alpha=0.15,
    )
    add_surface(
        ax,
        data.x_plot_km,
        data.y_plot_km,
        ceiling_surface,
        color=CORRIDOR_CEILING_COLOR,
        alpha=0.15,
    )

    sample_points = [
        (int(data.rows * 0.32), int(data.cols * 0.28)),
        (int(data.rows * 0.60), int(data.cols * 0.72)),
    ]
    for row, col in sample_points:
        x_km = col * data.resolution_m / 1000.0
        y_km = (data.rows - 1 - row) * data.resolution_m / 1000.0
        ax.plot(
            [x_km, x_km],
            [y_km, y_km],
            [float(data.floor[row, col]), float(data.ceiling[row, col])],
            color="#455A64",
            lw=1.2,
            alpha=0.9,
            linestyle="--",
        )

    row = int(data.rows * 0.52)
    y_cross = (data.rows - 1 - row) * data.resolution_m / 1000.0
    ax.plot(
        data.x_full_km,
        np.full_like(data.x_full_km, y_cross),
        data.ceiling[row, :],
        color="#607D8B",
        lw=1.4,
        alpha=0.75,
    )

    ax.text(
        float(data.x_full_km[int(data.cols * 0.18)]),
        float(data.y_full_km[int(data.rows * 0.18)]),
        float(data.ceiling[int(data.rows * 0.18), int(data.cols * 0.18)]) + 10.0,
        "Ceiling",
        color="#546E7A",
        fontsize=10,
    )
    ax.text(
        float(data.x_full_km[int(data.cols * 0.12)]),
        float(data.y_full_km[int(data.rows * 0.22)]),
        float(data.floor[int(data.rows * 0.22), int(data.cols * 0.12)]) - 8.0,
        "Floor",
        color="#455A64",
        fontsize=10,
    )
    ax.set_title("(a) DEM terrain and adaptive flyable corridor", pad=12, fontsize=13, loc="left")


def plot_layer_surfaces(ax, data: FigureData) -> None:
    """子图 B：三层自适应飞行曲面。"""
    terrain_surface = prepare_surface(data, data.z)
    add_surface(
        ax,
        data.x_plot_km,
        data.y_plot_km,
        terrain_surface,
        cmap="terrain",
        alpha=0.38,
    )

    for lid, color in enumerate(LAYER_COLORS):
        surface = prepare_surface(
            data,
            data.layer_mid[lid],
            offset_m=float(EXPLODE_OFFSETS_M[lid]),
            mask=data.layer_allowed[lid],
        )
        add_surface(
            ax,
            data.x_plot_km,
            data.y_plot_km,
            surface,
            color=color,
            alpha=0.34,
        )

    ax.text2D(
        0.03,
        0.06,
        "Visual offsets: +0 / +80 / +160 m",
        transform=ax.transAxes,
        fontsize=10,
        color="#37474F",
    )
    ax.set_title("(b) Three adaptive flight-layer surfaces", pad=12, fontsize=13, loc="left")


def plot_edge_subset(
    ax,
    data: FigureData,
    edge_indices: Iterable[int],
    z_vis: np.ndarray,
    color: str,
    linewidth: float,
    alpha: float,
    linestyle: str = "-",
) -> None:
    """绘制一组边。"""
    for eid in edge_indices:
        u, v = int(data.edges[eid, 0]), int(data.edges[eid, 1])
        ax.plot(
            [data.nodes[u, 0], data.nodes[v, 0]],
            [data.nodes[u, 1], data.nodes[v, 1]],
            [z_vis[u], z_vis[v]],
            color=color,
            lw=linewidth,
            alpha=alpha,
            linestyle=linestyle,
        )


def plot_topology(ax, data: FigureData, rng: np.random.Generator) -> None:
    """子图 C：三层航路拓扑、终端锚点柱与接入关系。"""
    terrain_surface = prepare_surface(data, data.z)
    add_surface(
        ax,
        data.x_plot_km,
        data.y_plot_km,
        terrain_surface,
        cmap="terrain",
        alpha=0.18,
    )

    z_vis = data.nodes[:, 2] + EXPLODE_OFFSETS_M[data.node_layers]
    access_pairs = edge_pair_set_from_terminal_status(data.terminal_status)
    edge_types = data.edges[:, 2].astype(int)
    u_idx = data.edges[:, 0].astype(int)
    v_idx = data.edges[:, 1].astype(int)
    u_layer = data.node_layers[u_idx]
    v_layer = data.node_layers[v_idx]

    access_mask = np.array(
        [normalize_pair(int(u_idx[k]), int(v_idx[k])) in access_pairs for k in range(len(data.edges))],
        dtype=bool,
    )
    vertical_edges = np.where(edge_types == 1)[0]
    climb_edges = np.where(edge_types == 2)[0]
    branch_edges = np.where(
        (edge_types == 0)
        & (u_layer == 1)
        & (v_layer == 1)
        & (~access_mask)
    )[0]
    backbone_edges = np.where(
        (edge_types == 0)
        & (u_layer == 2)
        & (v_layer == 2)
        & (~access_mask)
    )[0]
    access_edges = np.where(access_mask)[0]

    plot_edge_subset(
        ax,
        data,
        sample_indices(backbone_edges, ratio=0.11, rng=rng),
        z_vis,
        color=LAYER_COLORS[2],
        linewidth=0.8,
        alpha=0.38,
    )
    plot_edge_subset(
        ax,
        data,
        sample_indices(branch_edges, ratio=0.05, rng=rng),
        z_vis,
        color=LAYER_COLORS[1],
        linewidth=0.65,
        alpha=0.25,
    )
    plot_edge_subset(
        ax,
        data,
        sample_indices(climb_edges, ratio=0.18, rng=rng),
        z_vis,
        color="#FFB300",
        linewidth=1.0,
        alpha=0.55,
        linestyle="--",
    )
    plot_edge_subset(
        ax,
        data,
        access_edges,
        z_vis,
        color="#7E57C2",
        linewidth=1.1,
        alpha=0.82,
        linestyle="-.",
    )
    plot_edge_subset(
        ax,
        data,
        vertical_edges,
        z_vis,
        color="#455A64",
        linewidth=1.25,
        alpha=0.9,
    )

    terminals = data.terminal_status.get("terminals", {})
    anchor_indices = []
    depot_idx = []
    target_idx = []
    for meta in terminals.values():
        indices = [int(v) for v in meta.get("indices", [])]
        anchor_indices.extend(indices)
        if not indices:
            continue
        if str(meta.get("source")) == "virtual_depot":
            depot_idx.append(indices[0])
        else:
            target_idx.append(indices[0])

    anchor_indices_arr = np.array(sorted(set(anchor_indices)), dtype=int)
    branch_nodes = np.where((data.node_layers == 1) & (~np.isin(np.arange(len(data.nodes)), anchor_indices_arr)))[0]
    backbone_nodes = np.where((data.node_layers == 2) & (~np.isin(np.arange(len(data.nodes)), anchor_indices_arr)))[0]
    branch_nodes_sample = sample_indices(branch_nodes, ratio=0.08, rng=rng)
    backbone_nodes_sample = sample_indices(backbone_nodes, ratio=0.12, rng=rng)

    ax.scatter(
        data.nodes[branch_nodes_sample, 0],
        data.nodes[branch_nodes_sample, 1],
        z_vis[branch_nodes_sample],
        s=5,
        color=LAYER_COLORS[1],
        alpha=0.30,
        depthshade=False,
    )
    ax.scatter(
        data.nodes[backbone_nodes_sample, 0],
        data.nodes[backbone_nodes_sample, 1],
        z_vis[backbone_nodes_sample],
        s=5,
        color=LAYER_COLORS[2],
        alpha=0.35,
        depthshade=False,
    )

    if target_idx:
        ax.scatter(
            data.nodes[target_idx, 0],
            data.nodes[target_idx, 1],
            z_vis[target_idx],
            s=40,
            color="white",
            edgecolor="#263238",
            linewidth=0.9,
            marker="o",
            depthshade=False,
            zorder=10,
        )
    if depot_idx:
        ax.scatter(
            data.nodes[depot_idx, 0],
            data.nodes[depot_idx, 1],
            z_vis[depot_idx],
            s=90,
            color="#F44336",
            edgecolor="white",
            linewidth=0.9,
            marker="*",
            depthshade=False,
            zorder=10,
        )

    legend_handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#F44336", markeredgecolor="white", markersize=11, label="Virtual depots"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#263238", markersize=7, label="Targets"),
        Line2D([0], [0], color="#455A64", lw=1.4, label="Terminal pillars"),
        Line2D([0], [0], color="#7E57C2", lw=1.2, linestyle="-.", label="Safe access edges"),
        Line2D([0], [0], color="#FFB300", lw=1.0, linestyle="--", label="Climb links"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=True)
    ax.set_title("(c) Layered airway topology with terminal pillars", pad=12, fontsize=13, loc="left")


def segment_color_for_path(data: FigureData, path: list[int], index: int) -> str:
    """为路径中的有向线段分配颜色。"""
    current_layer = int(data.node_layers[path[index]])
    next_layer = int(data.node_layers[path[index + 1]])
    if current_layer == next_layer:
        return LAYER_COLORS[current_layer]
    return LAYER_COLORS[next_layer]


def build_profile_samples(data: FigureData, path: list[int]) -> dict[str, np.ndarray]:
    """沿示例路径采样地形、走廊和三层中面，用于剖面 inset。"""
    dist_dense = []
    terrain_vals = []
    floor_vals = []
    ceiling_vals = []
    layer_vals = [[], [], []]

    cumulative_km = 0.0
    prev_dense_xy: tuple[float, float] | None = None
    for idx in range(len(path) - 1):
        u = int(path[idx])
        v = int(path[idx + 1])
        p1 = data.nodes[u]
        p2 = data.nodes[v]
        horizontal_m = float(np.linalg.norm((p2[:2] - p1[:2]) * 1000.0))
        sample_n = max(2, int(math.ceil(horizontal_m / max(data.resolution_m * 1.2, 80.0))))
        ts = np.linspace(0.0, 1.0, sample_n)
        if idx > 0:
            ts = ts[1:]
        for t in ts:
            x_km = float(p1[0] + t * (p2[0] - p1[0]))
            y_km = float(p1[1] + t * (p2[1] - p1[1]))
            if prev_dense_xy is not None:
                step_km = math.sqrt((x_km - prev_dense_xy[0]) ** 2 + (y_km - prev_dense_xy[1]) ** 2)
                cumulative_km += step_km
            dist_dense.append(cumulative_km)
            r, c = km_to_rc(x_km, y_km, data.rows, data.cols, data.resolution_m)
            terrain_vals.append(float(data.z[r, c]))
            floor_vals.append(float(data.floor[r, c]))
            ceiling_vals.append(float(data.ceiling[r, c]))
            for lid in range(3):
                layer_vals[lid].append(float(data.layer_mid[lid, r, c]))
            prev_dense_xy = (x_km, y_km)

    route_dist = [0.0]
    route_z = [float(data.nodes[path[0], 2])]
    for idx in range(len(path) - 1):
        u = int(path[idx])
        v = int(path[idx + 1])
        horizontal_km = float(np.linalg.norm(data.nodes[v, :2] - data.nodes[u, :2]))
        route_dist.append(route_dist[-1] + horizontal_km)
        route_z.append(float(data.nodes[v, 2]))

    return {
        "distance_km": np.asarray(dist_dense, dtype=float),
        "terrain": np.asarray(terrain_vals, dtype=float),
        "floor": np.asarray(floor_vals, dtype=float),
        "ceiling": np.asarray(ceiling_vals, dtype=float),
        "layer0": np.asarray(layer_vals[0], dtype=float),
        "layer1": np.asarray(layer_vals[1], dtype=float),
        "layer2": np.asarray(layer_vals[2], dtype=float),
        "route_distance_km": np.asarray(route_dist, dtype=float),
        "route_z": np.asarray(route_z, dtype=float),
    }


def plot_profile_inset(ax, data: FigureData, route: ExampleRoute) -> None:
    """在子图 D 中加入剖面图 inset。"""
    profile = build_profile_samples(data, route.path)
    inset = ax.inset_axes([0.05, 0.04, 0.50, 0.33])
    inset.set_facecolor((1.0, 1.0, 1.0, 0.94))
    inset.plot(profile["distance_km"], profile["terrain"], color="#6D4C41", lw=1.5, label="Terrain")
    inset.plot(profile["distance_km"], profile["floor"], color=CORRIDOR_FLOOR_COLOR, lw=1.0, alpha=0.9, label="Floor")
    inset.plot(profile["distance_km"], profile["ceiling"], color=CORRIDOR_CEILING_COLOR, lw=1.0, alpha=0.9, label="Ceiling")
    inset.plot(profile["distance_km"], profile["layer0"], color=LAYER_COLORS[0], lw=0.9, alpha=0.9)
    inset.plot(profile["distance_km"], profile["layer1"], color=LAYER_COLORS[1], lw=0.9, alpha=0.9)
    inset.plot(profile["distance_km"], profile["layer2"], color=LAYER_COLORS[2], lw=0.9, alpha=0.9)

    route_dist = profile["route_distance_km"]
    route_z = profile["route_z"]
    for idx in range(len(route.path) - 1):
        color = segment_color_for_path(data, route.path, idx)
        inset.plot(
            route_dist[idx : idx + 2],
            route_z[idx : idx + 2],
            color=color,
            lw=2.1,
        )

    inset.set_title("Altitude profile", fontsize=8, pad=2)
    inset.set_xlabel("Distance along route (km)", fontsize=8)
    inset.set_ylabel("Altitude (m)", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.grid(True, alpha=0.25, linestyle="--")
    inset.legend(loc="upper left", fontsize=6, frameon=True)


def plot_example_route(ax, data: FigureData, route: ExampleRoute) -> None:
    """子图 D：在三层网络上叠加示例路径。"""
    terrain_surface = prepare_surface(data, data.z)
    add_surface(
        ax,
        data.x_plot_km,
        data.y_plot_km,
        terrain_surface,
        cmap="terrain",
        alpha=0.22,
    )

    for lid, color in enumerate(LAYER_COLORS):
        surface = prepare_surface(
            data,
            data.layer_mid[lid],
            offset_m=float(EXPLODE_OFFSETS_M[lid]),
            mask=data.layer_allowed[lid],
        )
        add_surface(
            ax,
            data.x_plot_km,
            data.y_plot_km,
            surface,
            color=color,
            alpha=0.10,
        )

    z_vis = data.nodes[:, 2] + EXPLODE_OFFSETS_M[data.node_layers]
    for idx in range(len(route.path) - 1):
        u = int(route.path[idx])
        v = int(route.path[idx + 1])
        ax.plot(
            [data.nodes[u, 0], data.nodes[v, 0]],
            [data.nodes[u, 1], data.nodes[v, 1]],
            [z_vis[u], z_vis[v]],
            color=segment_color_for_path(data, route.path, idx),
            lw=2.6,
            alpha=0.98,
        )

    start_idx = int(route.path[0])
    goal_idx = int(route.path[-1])
    ax.scatter(
        [data.nodes[start_idx, 0]],
        [data.nodes[start_idx, 1]],
        [z_vis[start_idx]],
        s=120,
        marker="*",
        color="#F44336",
        edgecolor="white",
        linewidth=1.0,
        depthshade=False,
        zorder=12,
    )
    ax.scatter(
        [data.nodes[goal_idx, 0]],
        [data.nodes[goal_idx, 1]],
        [z_vis[goal_idx]],
        s=55,
        marker="o",
        color="white",
        edgecolor="#263238",
        linewidth=1.0,
        depthshade=False,
        zorder=12,
    )
    ax.text(
        float(data.nodes[start_idx, 0]),
        float(data.nodes[start_idx, 1]),
        float(z_vis[start_idx]) + 18.0,
        "Depot",
        fontsize=9,
        color="#C62828",
    )
    ax.text(
        float(data.nodes[goal_idx, 0]),
        float(data.nodes[goal_idx, 1]),
        float(z_vis[goal_idx]) + 18.0,
        "Target",
        fontsize=9,
        color="#263238",
    )

    route_text = (
        f"Example task {route.task_id}\n"
        f"Semantic sequence:\n"
        f"Depot -> Terminal -> Branch -> Backbone -> Branch -> Terminal -> Target\n"
        f"Route length: {route.distance_km:.2f} km"
    )
    ax.text2D(
        0.03,
        0.73,
        route_text,
        transform=ax.transAxes,
        fontsize=9,
        color="#263238",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.86, edgecolor="#CFD8DC"),
    )

    legend_handles = [
        Line2D([0], [0], color=LAYER_COLORS[0], lw=2.4, label="Terminal segment"),
        Line2D([0], [0], color=LAYER_COLORS[1], lw=2.4, label="Branch segment"),
        Line2D([0], [0], color=LAYER_COLORS[2], lw=2.4, label="Backbone segment"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=True)
    plot_profile_inset(ax, data, route)
    ax.set_title("(d) Example route across terminal, branch and backbone layers", pad=12, fontsize=13, loc="left")


def build_figure(data: FigureData, route: ExampleRoute, dpi: int) -> tuple[Path, Path]:
    """生成整张论文主图并保存为 PNG/PDF。"""
    rng = np.random.default_rng(RNG_SEED)
    fig = plt.figure(figsize=(19.5, 14.5), facecolor="white")

    ax_a = fig.add_subplot(2, 2, 1, projection="3d")
    ax_b = fig.add_subplot(2, 2, 2, projection="3d")
    ax_c = fig.add_subplot(2, 2, 3, projection="3d")
    ax_d = fig.add_subplot(2, 2, 4, projection="3d")

    terrain_min = float(np.nanmin(data.z))
    corridor_max = float(np.nanmax(data.ceiling))
    exploded_max = float(np.nanmax(data.layer_mid[2] + EXPLODE_OFFSETS_M[2]))

    plot_terrain_and_corridor(ax_a, data)
    style_3d_axes(ax_a, data, z_min=terrain_min - 20.0, z_max=corridor_max + 35.0, elev=30.0, azim=-132.0)

    plot_layer_surfaces(ax_b, data)
    style_3d_axes(ax_b, data, z_min=terrain_min - 20.0, z_max=exploded_max + 35.0, elev=27.0, azim=-128.0)
    layer_handles = [Patch(facecolor=LAYER_COLORS[i], alpha=0.40, label=LAYER_LABELS[i]) for i in range(3)]
    ax_b.legend(handles=layer_handles, loc="upper right", fontsize=8, frameon=True)

    plot_topology(ax_c, data, rng=rng)
    style_3d_axes(ax_c, data, z_min=terrain_min - 20.0, z_max=exploded_max + 35.0, elev=31.0, azim=-137.0)

    plot_example_route(ax_d, data, route)
    style_3d_axes(ax_d, data, z_min=terrain_min - 20.0, z_max=exploded_max + 35.0, elev=29.0, azim=-122.0)

    fig.suptitle("Figure X. Terrain-aware three-layer airway network construction", fontsize=18, y=0.975)
    fig.text(
        0.5,
        0.018,
        "Note: panels (b)-(d) apply +80 m and +160 m visual offsets to the branch/backbone layers for readability only.",
        ha="center",
        fontsize=10,
        color="#455A64",
    )
    plt.subplots_adjust(left=0.03, right=0.985, top=0.94, bottom=0.05, wspace=0.05, hspace=0.10)

    png_path = data.figure_dir / f"{FIGURE_NAME}.png"
    pdf_path = data.figure_dir / f"{FIGURE_NAME}.pdf"
    fig.savefig(png_path, dpi=int(dpi), facecolor="white")
    fig.savefig(pdf_path, dpi=300, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    data = load_figure_data(args)
    route = build_example_route(data, task_id=args.task_id)

    print(
        f"[读取] 场景={data.scene_name}，DEM={data.rows}x{data.cols}，"
        f"节点数={len(data.nodes)}，边数={len(data.edges)}，曲面步长={data.stride}"
    )
    print(
        f"[示例路径] 任务={route.task_id}，"
        f"{route.start_name} -> {route.goal_name}，"
        f"层序列={route.layer_sequence}，长度={route.distance_km:.2f} km"
    )

    png_path, pdf_path = build_figure(data, route, dpi=args.dpi)
    print(f"[保存] PNG: {png_path}")
    print(f"[保存] PDF: {pdf_path}")


if __name__ == "__main__":
    main()
