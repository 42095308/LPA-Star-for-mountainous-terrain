# -*- coding: utf-8 -*-
"""重绘 aei_v8 实验章节的数据结果图，并生成替换图片后的 DOCX 副本。"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "final_results" / "paper_revision" / "source_data" / "chapter4_python"
FIG_DIR = PROJECT_ROOT / "final_results" / "paper_revision" / "figures" / "aei_v8_experiment"
DEFAULT_DOCX = Path(r"C:\Users\42095\Desktop\小论文资料\aei稿件\aei_v8.docx")
DEFAULT_OUT_DOCX = DEFAULT_DOCX.with_name("aei_v8_figures_optimized.docx")

FONT_SIZE = 8.5

SCENE_ORDER = ["华山", "黄山", "峨眉山"]
SCENE_LABELS = {"华山": "Huashan", "黄山": "Huangshan", "峨眉山": "Emeishan"}
SCENE_COLORS = {"华山": "#2F6EA6", "黄山": "#D9903D", "峨眉山": "#4B9B78"}
SCENE_MARKERS = {"华山": "o", "黄山": "s", "峨眉山": "D"}

METHOD_ORDER = ["MP", "MA", "MF", "MR", "MV", "BI-APF-RRT*", "GWO-DE"]
ABLATION_METHODS = ["MA", "MF", "MR", "MV"]
STRUCTURE_METHODS = ["MP", "MA", "MF", "MR", "MV"]
METHOD_COLORS = {
    "MP": "#2F6EA6",
    "MA": "#D9903D",
    "MF": "#7A67A8",
    "MR": "#4B9B78",
    "MV": "#5C5C5C",
    "BI-APF-RRT*": "#B75A5A",
    "GWO-DE": "#8A7F45",
}
METHOD_MARKERS = {"MP": "o", "MA": "s", "MF": "^", "MR": "D", "MV": "v"}

BASELINE_LABELS = {
    "B4_Proposed_LPA_Layered": "MP",
    "B2_GlobalAstar_Layered": "MA",
}

STRESS_ORDER = [
    "no_fly_radius_0_4",
    "no_fly_radius_0_8",
    "no_fly_radius_1_2",
    "wind_severity_0_5",
    "wind_severity_1_0",
    "wind_severity_1_5",
]
STRESS_LABELS = {
    "no_fly_radius_0_4": "Radius 0.4 km",
    "no_fly_radius_0_8": "Radius 0.8 km",
    "no_fly_radius_1_2": "Radius 1.2 km",
    "wind_severity_0_5": "Wind 0.5",
    "wind_severity_1_0": "Wind 1.0",
    "wind_severity_1_5": "Wind 1.5",
}

QUALITY_METRICS = [
    ("Replanning time", "Replan"),
    ("Path cost", "Cost"),
    ("Path length", "Length"),
    ("Communication coverage", "Coverage"),
    ("Risk exposure", "Risk"),
]

LEGEND_FRAME = {
    "facecolor": "white",
    "edgecolor": "#303030",
    "linewidth": 0.65,
}


def configure_matplotlib() -> None:
    """配置统一的 Times New Roman 图形样式。"""

    for candidate in [
        Path(r"C:\Windows\Fonts\times.ttf"),
        Path(r"C:\Windows\Fonts\timesbd.ttf"),
        Path(r"C:\Windows\Fonts\timesi.ttf"),
        Path(r"C:\Windows\Fonts\timesbi.ttf"),
    ]:
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "legend.title_fontsize": FONT_SIZE,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read_csv_records(path: Path) -> list[dict[str, str]]:
    """按 UTF-8 读取 CSV 源数据。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少源数据文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_float(value: object) -> float:
    """将 CSV 字符串转为浮点数。"""

    text = str(value or "").strip().replace(",", "").replace("%", "")
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def ensure_columns(rows: list[dict[str, str]], columns: list[str], source: Path) -> None:
    """检查源表是否包含绘图所需字段。"""

    if not rows:
        raise ValueError(f"源数据为空：{source}")
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError(f"{source.name} 缺少字段：{', '.join(missing)}")


def sort_by_scene_method(row: dict[str, object]) -> tuple[int, int]:
    """按固定场景和方法顺序排序。"""

    scene = str(row.get("scene", ""))
    method = str(row.get("method", ""))
    scene_index = SCENE_ORDER.index(scene) if scene in SCENE_ORDER else len(SCENE_ORDER)
    method_index = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
    return scene_index, method_index


def style_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    """统一坐标轴、网格和刻度外观。"""

    ax.grid(True, axis=grid_axis, color="#D9D9D9", linewidth=0.45, alpha=0.85)
    ax.tick_params(length=2.4, width=0.55, labelsize=FONT_SIZE)
    ax.xaxis.labelpad = 1.5
    ax.yaxis.labelpad = 2.0
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)


def add_panel_heading(ax: plt.Axes, label: str, title: str, x: float = 0.0, y: float = -0.28) -> None:
    """在每个子图正下方添加统一字号的面板字母和标题。"""

    ax.text(
        0.5,
        y,
        f"{label}  {title}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=FONT_SIZE,
        fontweight="bold",
        clip_on=False,
    )


def frame_legend(legend) -> None:
    """只给图例添加边框。"""

    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor(LEGEND_FRAME["facecolor"])
    frame.set_edgecolor(LEGEND_FRAME["edgecolor"])
    frame.set_linewidth(LEGEND_FRAME["linewidth"])
    frame.set_alpha(1.0)
    if legend.get_title() is not None:
        legend.get_title().set_fontsize(FONT_SIZE)
    for text in legend.get_texts():
        text.set_fontsize(FONT_SIZE)


def top_legend(
    ax: plt.Axes,
    handles: list[object],
    title: str,
    ncol: int,
    columnspacing: float = 1.0,
    handlelength: float = 1.6,
) -> None:
    """在图形正上方的专用坐标轴中放置图例。"""

    ax.axis("off")
    legend = ax.legend(
        handles=handles,
        title=title,
        loc="center",
        ncol=ncol,
        frameon=True,
        handlelength=handlelength,
        handletextpad=0.5,
        columnspacing=columnspacing,
        borderaxespad=0.0,
    )
    frame_legend(legend)


def set_colorbar_style(colorbar, label: str) -> None:
    """统一色标字号。"""

    colorbar.set_label(label, fontsize=FONT_SIZE)
    colorbar.ax.tick_params(labelsize=FONT_SIZE, length=2.4, width=0.55)


def save_pub_figure(fig: plt.Figure, basename: str, dpi: int) -> list[Path]:
    """导出论文排版所需的多格式图件。"""

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        FIG_DIR / f"{basename}.svg",
        FIG_DIR / f"{basename}.pdf",
        FIG_DIR / f"{basename}.tiff",
        FIG_DIR / f"{basename}.png",
    ]
    fig.savefig(paths[0])
    fig.savefig(paths[1])
    fig.savefig(paths[2], dpi=dpi)
    fig.savefig(paths[3], dpi=300)
    plt.close(fig)
    return paths


def range_normalize(values: np.ndarray) -> np.ndarray:
    """把数值压缩到 0 到 1，用于多指标同图比较。"""

    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.full_like(array, np.nan)
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    if abs(high - low) < 1e-12:
        return np.full_like(array, 0.5)
    return (array - low) / (high - low)


def value_label(value: float, digits: int = 1, signed: bool = False) -> str:
    """生成紧凑数值标签。"""

    if not math.isfinite(value):
        return ""
    prefix = "+" if signed and value >= 0 else ""
    return f"{prefix}{value:.{digits}f}"


def load_external_records() -> list[dict[str, object]]:
    """读取图 6 的总体对比源数据。"""

    path = SOURCE_DIR / "figure_4_1_external_augmented_source.csv"
    rows = read_csv_records(path)
    ensure_columns(
        rows,
        ["场景", "方法", "成功率", "重规划时间_ms", "路径代价", "长度_km", "通信覆盖率", "风险暴露"],
        path,
    )
    records: list[dict[str, object]] = []
    for row in rows:
        records.append(
            {
                "scene": row["场景"],
                "method": row["方法"],
                "success": parse_float(row["成功率"]),
                "time_ms": parse_float(row["重规划时间_ms"]),
                "cost": parse_float(row["路径代价"]),
                "length_km": parse_float(row["长度_km"]),
                "coverage": parse_float(row["通信覆盖率"]),
                "risk": parse_float(row["风险暴露"]),
            }
        )
    return sorted(records, key=sort_by_scene_method)


def load_event_records() -> list[dict[str, object]]:
    """读取图 7 的事件重规划源数据。"""

    path = SOURCE_DIR / "figure_4_4_event_replanning_source.csv"
    rows = read_csv_records(path)
    ensure_columns(
        rows,
        ["场景", "事件数_K", "MP累计时间_ms", "MA累计时间_ms", "MA与MP时间比", "MP扩展节点", "MA扩展节点"],
        path,
    )
    records: list[dict[str, object]] = []
    for row in rows:
        records.append(
            {
                "scene": row["场景"],
                "events": parse_float(row["事件数_K"]),
                "mp_time": parse_float(row["MP累计时间_ms"]),
                "ma_time": parse_float(row["MA累计时间_ms"]),
                "time_ratio": parse_float(row["MA与MP时间比"]),
                "mp_nodes": parse_float(row["MP扩展节点"]),
                "ma_nodes": parse_float(row["MA扩展节点"]),
            }
        )
    return sorted(records, key=lambda item: (SCENE_ORDER.index(str(item["scene"])), float(item["events"])))


def load_scale_records() -> list[dict[str, object]]:
    """读取图 9 的图规模敏感性源数据。"""

    path = SOURCE_DIR / "figure_4_5_graph_scale_source.csv"
    rows = read_csv_records(path)
    ensure_columns(
        rows,
        ["图规模", "节点数", "边数", "MP累计时间_ms", "MA累计时间_ms", "MA与MP时间比", "MP成功率", "MA成功率"],
        path,
    )
    records: list[dict[str, object]] = []
    order = {"small": 0, "medium": 1, "large": 2}
    for row in rows:
        records.append(
            {
                "scale": row["图规模"],
                "nodes": parse_float(row["节点数"]),
                "edges": parse_float(row["边数"]),
                "mp_time": parse_float(row["MP累计时间_ms"]),
                "ma_time": parse_float(row["MA累计时间_ms"]),
                "time_ratio": parse_float(row["MA与MP时间比"]),
                "mp_success": parse_float(row["MP成功率"]),
                "ma_success": parse_float(row["MA成功率"]),
            }
        )
    return sorted(records, key=lambda item: order.get(str(item["scale"]), 99))


def load_stress_records() -> list[dict[str, object]]:
    """读取图 8 的事件压力实验源数据。"""

    path = SOURCE_DIR / "figure_4_6_event_stress_source.csv"
    rows = read_csv_records(path)
    ensure_columns(
        rows,
        [
            "stress_label",
            "baseline",
            "n_trials",
            "n_success",
            "success_rate",
            "mean_cumulative_replan_ms",
            "ci95_cumulative_replan_ms",
            "mean_cumulative_expanded",
            "ci95_event_expanded",
        ],
        path,
    )
    records: list[dict[str, object]] = []
    for row in rows:
        stress = row["stress_label"]
        method = BASELINE_LABELS.get(row["baseline"])
        if stress not in STRESS_ORDER or method is None:
            continue
        records.append(
            {
                "stress": stress,
                "method": method,
                "n_trials": int(parse_float(row["n_trials"])),
                "n_success": int(parse_float(row["n_success"])),
                "success": parse_float(row["success_rate"]),
                "time_ms": parse_float(row["mean_cumulative_replan_ms"]),
                "time_ci": parse_float(row["ci95_cumulative_replan_ms"]),
                "expanded": parse_float(row["mean_cumulative_expanded"]),
                "expanded_ci": parse_float(row["ci95_event_expanded"]),
            }
        )
    return sorted(records, key=lambda item: (STRESS_ORDER.index(str(item["stress"])), str(item["method"])))


def load_quality_records() -> list[dict[str, object]]:
    """读取图 10 的路径质量消融源数据。"""

    path = SOURCE_DIR / "figure_4_6_ablation_quality_relative_source.csv"
    rows = read_csv_records(path)
    ensure_columns(rows, ["场景", "方法", "指标", "相对MP变化_越高越好"], path)
    records: list[dict[str, object]] = []
    for row in rows:
        records.append(
            {
                "scene": row["场景"],
                "method": row["方法"],
                "metric": row["指标"],
                "change": parse_float(row["相对MP变化_越高越好"]),
            }
        )
    return records


def load_structural_records() -> list[dict[str, object]]:
    """读取图 11 的搜索工作量消融源数据。"""

    path = SOURCE_DIR / "figure_4_7_structural_ablation_source.csv"
    rows = read_csv_records(path)
    ensure_columns(rows, ["场景", "方法", "mean_replan_ms", "mean_expanded"], path)
    records: list[dict[str, object]] = []
    for row in rows:
        records.append(
            {
                "scene": row["场景"],
                "method": row["方法"],
                "time_ms": parse_float(row["mean_replan_ms"]),
                "expanded": parse_float(row["mean_expanded"]),
            }
        )
    return records


def get_external_value(records: list[dict[str, object]], scene: str, method: str, field: str) -> float:
    """按场景和方法取总体对比数值。"""

    for row in records:
        if row["scene"] == scene and row["method"] == method:
            return float(row[field])
    return float("nan")


def plot_external_metric(
    ax: plt.Axes,
    records: list[dict[str, object]],
    field: str,
    transform,
    xlabel: str,
    label: str,
    title: str,
    xlim: tuple[float, float] | None,
    log_scale: bool = False,
    show_ylabels: bool = False,
) -> None:
    """绘制图 6 的单指标面板。"""

    y_positions = np.arange(len(METHOD_ORDER), dtype=float)
    offsets = {"华山": -0.18, "黄山": 0.0, "峨眉山": 0.18}
    for scene in SCENE_ORDER:
        xs = [transform(get_external_value(records, scene, method, field)) for method in METHOD_ORDER]
        ys = y_positions + offsets[scene]
        ax.scatter(
            xs,
            ys,
            s=25,
            color=SCENE_COLORS[scene],
            marker=SCENE_MARKERS[scene],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(METHOD_ORDER if show_ylabels else [])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if log_scale:
        ax.set_xscale("log")
        ax.set_xticks([10, 30, 100, 300, 1000])
        ax.set_xticklabels(["10", "30", "100", "300", "1000"])
    style_axis(ax, grid_axis="x")
    add_panel_heading(ax, label, title)


def plot_figure_6(records: list[dict[str, object]], dpi: int) -> list[Path]:
    """重绘图 6，总体对比。"""

    fig = plt.figure(figsize=(9.05, 6.0))
    fig.subplots_adjust(left=0.125, right=0.985, top=0.97, bottom=0.18, hspace=0.92, wspace=0.32)
    grid = fig.add_gridspec(3, 3, height_ratios=[0.16, 1.0, 1.0])
    legend_ax = fig.add_subplot(grid[0, :])
    axes = [
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
        fig.add_subplot(grid[2, 0]),
        fig.add_subplot(grid[2, 1]),
        fig.add_subplot(grid[2, 2]),
    ]

    panels = [
        ("success", lambda value: value * 100.0, "Success rate, %", "a", "Success rate", (22, 104), False),
        ("time_ms", lambda value: value, "Replanning time, ms", "b", "Replanning time", (7, 1350), True),
        ("cost", lambda value: value, "Path cost", "c", "Path cost", (8, 104), False),
        ("length_km", lambda value: value, "Path length, km", "d", "Path length", (6.5, 15.8), False),
        ("coverage", lambda value: value, "Communication coverage", "e", "Coverage", (0.22, 0.95), False),
        ("risk", lambda value: value, "Risk exposure", "f", "Risk exposure", (2.45, 4.45), False),
    ]
    for index, panel in enumerate(panels):
        field, transform, xlabel, label, title, xlim, log_scale = panel
        plot_external_metric(
            axes[index],
            records,
            field,
            transform,
            xlabel,
            label,
            title,
            xlim,
            log_scale=log_scale,
            show_ylabels=index in {0, 3},
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker=SCENE_MARKERS[scene],
            color="none",
            markerfacecolor=SCENE_COLORS[scene],
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=5.3,
            label=SCENE_LABELS[scene],
        )
        for scene in SCENE_ORDER
    ]
    top_legend(legend_ax, handles, "Scene", ncol=3)
    return save_pub_figure(fig, "fig_4_1_external_augmented_tnr", dpi)


def event_rows_by_scene(records: list[dict[str, object]], scene: str) -> list[dict[str, object]]:
    """按场景取事件重规划记录。"""

    return [row for row in records if row["scene"] == scene]


def plot_figure_7(records: list[dict[str, object]], dpi: int) -> list[Path]:
    """重绘图 7，连续事件重规划。"""

    scene_positions = {scene: index for index, scene in enumerate(SCENE_ORDER)}
    event_columns = {1.0: 0.0, 5.0: 1.0, 10.0: 2.0}
    event_tick_positions = [event_columns[1.0], event_columns[5.0], event_columns[10.0]]
    event_tick_labels = ["1", "5", "10"]
    x = np.asarray([event_columns[float(row["events"])] for row in records], dtype=float)
    y = np.asarray([scene_positions[str(row["scene"])] for row in records], dtype=float)
    time_ratio = np.asarray([float(row["time_ratio"]) for row in records], dtype=float)
    node_ratio = np.asarray(
        [float(row["ma_nodes"]) / max(float(row["mp_nodes"]), 1e-9) for row in records],
        dtype=float,
    )

    fig = plt.figure(figsize=(8.8, 4.35))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.96, bottom=0.30, hspace=0.20, wspace=0.32)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.18, 1.0], width_ratios=[1.32, 1.0])
    legend_left = fig.add_subplot(grid[0, 0])
    legend_right = fig.add_subplot(grid[0, 1])
    ax_matrix = fig.add_subplot(grid[1, 0])
    ax_summary = fig.add_subplot(grid[1, 1])

    norm = TwoSlopeNorm(vmin=-1.7, vcenter=0.0, vmax=1.7)
    sizes = 50 + 34 * np.clip(node_ratio, 1, 11)
    scatter = ax_matrix.scatter(
        x,
        y,
        s=sizes,
        c=np.log2(time_ratio),
        cmap="RdBu_r",
        norm=norm,
        edgecolor="white",
        linewidth=0.75,
        zorder=3,
    )
    label_offsets = {
        ("峨眉山", 1.0): (0, 12),
        ("峨眉山", 5.0): (0, -15),
        ("峨眉山", 10.0): (0, 12),
        ("黄山", 1.0): (0, 12),
        ("黄山", 5.0): (0, -15),
        ("黄山", 10.0): (0, -15),
        ("华山", 1.0): (0, 12),
        ("华山", 5.0): (0, -15),
        ("华山", 10.0): (0, 14),
    }
    for row, x_value, y_value, ratio in zip(records, x, y, time_ratio):
        offset = label_offsets[(str(row["scene"]), float(row["events"]))]
        ax_matrix.annotate(
            f"{ratio:.2f}x",
            (x_value, y_value),
            textcoords="offset points",
            xytext=offset,
            ha="center",
            va="center",
            fontsize=FONT_SIZE,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.35},
            zorder=4,
        )
    ax_matrix.set_xticks(event_tick_positions)
    ax_matrix.set_xticklabels(event_tick_labels)
    ax_matrix.set_xlabel("Event count K")
    ax_matrix.set_yticks(list(scene_positions.values()))
    ax_matrix.set_yticklabels([SCENE_LABELS[scene] for scene in SCENE_ORDER])
    ax_matrix.set_ylim(-0.55, len(SCENE_ORDER) - 0.45)
    ax_matrix.set_xlim(-0.12, 2.12)
    style_axis(ax_matrix, grid_axis="both")
    add_panel_heading(ax_matrix, "a", "Event burden matrix")
    colorbar = fig.colorbar(scatter, ax=ax_matrix, fraction=0.050, pad=0.025)
    set_colorbar_style(colorbar, "log2 MA/MP time")

    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#BDBDBD",
            markeredgecolor="white",
            markersize=marker_size,
            label=label,
        )
        for marker_size, label in [(5.8, "2x nodes"), (8.4, "6x nodes"), (10.8, "10x nodes")]
    ]
    top_legend(legend_left, size_handles, "Node ratio", ncol=3, columnspacing=0.8, handlelength=1.0)

    summary_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D08C45", markeredgecolor="white", label="Mean time"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#6EA5D8", markeredgecolor="white", label="Mean nodes"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#8A4E2A", markeredgecolor="#8A4E2A", label="K=10 time"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#2E5F87", markeredgecolor="#2E5F87", label="K=10 nodes"),
    ]
    top_legend(legend_right, summary_handles, "Summary", ncol=2, columnspacing=0.65, handlelength=1.0)

    y_base = np.arange(len(SCENE_ORDER), dtype=float)
    offsets_y = [-0.18, -0.06, 0.06, 0.18]
    summary_colors = ["#D08C45", "#6EA5D8", "#8A4E2A", "#2E5F87"]
    summary_markers = ["o", "o", "D", "D"]
    for scene_index, scene in enumerate(SCENE_ORDER):
        subset = event_rows_by_scene(records, scene)
        time_values = np.asarray([float(row["time_ratio"]) for row in subset], dtype=float)
        node_values = np.asarray(
            [float(row["ma_nodes"]) / max(float(row["mp_nodes"]), 1e-9) for row in subset],
            dtype=float,
        )
        values = [np.nanmean(time_values), np.nanmean(node_values), time_values[-1], node_values[-1]]
        for value, dy, color, marker in zip(values, offsets_y, summary_colors, summary_markers):
            ax_summary.scatter(
                math.log2(float(value)),
                y_base[scene_index] + dy,
                s=30,
                marker=marker,
                color=color,
                edgecolor="white" if marker == "o" else color,
                linewidth=0.5,
                zorder=3,
            )
    ax_summary.vlines(0, -0.5, len(SCENE_ORDER) - 0.5, color="#555555", linewidth=0.75)
    ax_summary.set_yticks(y_base)
    ax_summary.set_yticklabels([SCENE_LABELS[scene] for scene in SCENE_ORDER])
    ax_summary.set_xlabel("Fold ratio, MA/MP")
    ax_summary.set_xticks([-1, 0, 1, 2, 3])
    ax_summary.set_xticklabels(["0.5x", "1x", "2x", "4x", "8x"])
    ax_summary.set_xlim(-1.9, 3.7)
    ax_summary.invert_yaxis()
    style_axis(ax_summary, grid_axis="x")
    add_panel_heading(ax_summary, "b", "Scene-level ratio summary")
    return save_pub_figure(fig, "fig_4_4_event_replanning_tnr", dpi)


def stress_pair(records: list[dict[str, object]], stress: str) -> dict[str, dict[str, object]]:
    """取单个压力设置下 MP 和 MA 的记录。"""

    pair: dict[str, dict[str, object]] = {}
    for row in records:
        if row["stress"] == stress:
            pair[str(row["method"])] = row
    return pair


def set_stress_yaxis(ax: plt.Axes, show_labels: bool) -> None:
    """设置事件压力图的 y 轴。"""

    y_positions = np.arange(len(STRESS_ORDER), dtype=float)
    ax.set_yticks(y_positions)
    if show_labels:
        ax.set_yticklabels([STRESS_LABELS[item] for item in STRESS_ORDER])
    else:
        ax.set_yticklabels([])
    ax.set_ylim(len(STRESS_ORDER) - 0.45, -0.55)


def k_formatter(value: float, _pos: int) -> str:
    """把节点坐标格式化为 k。"""

    if abs(value) < 1e-9:
        return "0"
    return f"{value / 1000:.0f}k"


def plot_stress_success(ax: plt.Axes, records: list[dict[str, object]]) -> None:
    """绘制事件压力成功率面板。"""

    y_positions = np.arange(len(STRESS_ORDER), dtype=float)
    offsets = {"MP": -0.15, "MA": 0.15}
    success_values = {
        method: np.asarray([float(stress_pair(records, stress)[method]["success"]) * 100.0 for stress in STRESS_ORDER])
        for method in ["MP", "MA"]
    }
    for method in ["MP", "MA"]:
        xs = success_values[method]
        ys = y_positions + offsets[method]
        ax.hlines(ys, 80.0, xs, color=METHOD_COLORS[method], linewidth=1.35, alpha=0.86)
        ax.scatter(
            xs,
            ys,
            s=26,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
    for i, stress in enumerate(STRESS_ORDER):
        mp_value = float(success_values["MP"][i])
        ma_value = float(success_values["MA"][i])
        if abs(mp_value - ma_value) < 0.05:
            ax.text(
                min(max(mp_value, ma_value) + 0.90, 102.9),
                y_positions[i],
                f"{mp_value:.1f}",
                va="center",
                ha="left",
                fontsize=FONT_SIZE,
                color="#303030",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.20},
            )
        else:
            for method, x_value in [("MP", mp_value), ("MA", ma_value)]:
                ax.text(
                    min(x_value + (0.70 if method == "MP" else 1.40), 102.9),
                    y_positions[i] + offsets[method],
                    f"{x_value:.1f}",
                    va="center",
                    ha="left",
                    fontsize=FONT_SIZE,
                    color="#303030",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.20},
                )
    ax.set_xlim(78, 104)
    set_stress_yaxis(ax, True)
    ax.set_xlabel("Success rate, %")
    style_axis(ax, grid_axis="x")
    add_panel_heading(ax, "a", "Task success rate", y=-0.34)


def plot_stress_dumbbell(
    ax: plt.Axes,
    records: list[dict[str, object]],
    field: str,
    ci_field: str,
    xlabel: str,
    label: str,
    title: str,
    xlim: tuple[float, float],
    formatter=None,
    show_ylabels: bool = False,
) -> None:
    """绘制事件压力下的 MP 与 MA 对照哑铃图。"""

    y_positions = np.arange(len(STRESS_ORDER), dtype=float)
    for y_value, stress in zip(y_positions, STRESS_ORDER):
        pair = stress_pair(records, stress)
        mp_value = float(pair["MP"][field])
        ma_value = float(pair["MA"][field])
        mp_ci = float(pair["MP"][ci_field])
        ma_ci = float(pair["MA"][ci_field])
        ratio = ma_value / max(mp_value, 1e-9)
        ax.plot([mp_value, ma_value], [y_value, y_value], color="#B8B8B8", linewidth=1.0, zorder=1)
        for method, value, ci in [("MP", mp_value, mp_ci), ("MA", ma_value, ma_ci)]:
            ax.errorbar(
                value,
                y_value,
                xerr=ci,
                fmt=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                markeredgecolor="white",
                markeredgewidth=0.45,
                markersize=4.2,
                elinewidth=0.72,
                capsize=2.0,
                zorder=3,
            )
        ax.annotate(
            f"{ratio:.2f}x",
            (ma_value, y_value),
            textcoords="offset points",
            xytext=(8, 8),
            va="center",
            ha="left",
            fontsize=FONT_SIZE,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.30},
            zorder=4,
        )
    ax.set_xlim(*xlim)
    set_stress_yaxis(ax, show_ylabels)
    ax.set_xlabel(xlabel)
    if formatter is not None:
        ax.xaxis.set_major_formatter(FuncFormatter(formatter))
    style_axis(ax, grid_axis="x")
    add_panel_heading(ax, label, title, y=-0.34)


def plot_stress_ratio(ax: plt.Axes, records: list[dict[str, object]]) -> None:
    """绘制事件压力的相对负担面板。"""

    y_positions = np.arange(len(STRESS_ORDER), dtype=float)
    height = 0.30
    time_ratios = []
    node_ratios = []
    for stress in STRESS_ORDER:
        pair = stress_pair(records, stress)
        time_ratios.append(float(pair["MA"]["time_ms"]) / max(float(pair["MP"]["time_ms"]), 1e-9))
        node_ratios.append(float(pair["MA"]["expanded"]) / max(float(pair["MP"]["expanded"]), 1e-9))
    ax.vlines(1.0, -0.5, len(STRESS_ORDER) - 0.5, color="#555555", linewidth=0.75)
    ax.barh(
        y_positions - height / 2,
        np.asarray(time_ratios) - 1.0,
        left=1.0,
        height=height,
        color="#D9903D",
        edgecolor="white",
    )
    ax.barh(
        y_positions + height / 2,
        np.asarray(node_ratios) - 1.0,
        left=1.0,
        height=height,
        color="#6EA5D8",
        edgecolor="white",
    )
    for y_value, time_ratio, node_ratio in zip(y_positions, time_ratios, node_ratios):
        ax.text(time_ratio + 0.12, y_value - height / 2, f"{time_ratio:.2f}", va="center", fontsize=FONT_SIZE)
        ax.text(node_ratio + 0.12, y_value + height / 2, f"{node_ratio:.2f}", va="center", fontsize=FONT_SIZE)
    ax.set_xlim(0.8, 11.3)
    set_stress_yaxis(ax, False)
    ax.set_xlabel("MA/MP ratio")
    style_axis(ax, grid_axis="x")
    add_panel_heading(ax, "d", "Relative burden", y=-0.34)


def plot_figure_8(records: list[dict[str, object]], dpi: int) -> list[Path]:
    """重绘图 8，事件严重度压力实验。"""

    fig = plt.figure(figsize=(9.1, 5.25))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.20, hspace=1.05, wspace=0.28)
    grid = fig.add_gridspec(3, 2, height_ratios=[0.16, 1.0, 1.0])
    legend_ax = fig.add_subplot(grid[0, :])
    ax_success = fig.add_subplot(grid[1, 0])
    ax_time = fig.add_subplot(grid[1, 1])
    ax_nodes = fig.add_subplot(grid[2, 0])
    ax_ratio = fig.add_subplot(grid[2, 1])

    plot_stress_success(ax_success, records)
    time_limit = max(float(row["time_ms"]) + float(row["time_ci"]) for row in records if row["method"] in {"MP", "MA"})
    node_limit = max(float(row["expanded"]) + float(row["expanded_ci"]) for row in records if row["method"] in {"MP", "MA"})

    plot_stress_dumbbell(
        ax_time,
        records,
        "time_ms",
        "time_ci",
        "Cumulative time, ms",
        "b",
        "Cumulative time",
        (0, time_limit * 1.22),
    )
    plot_stress_dumbbell(
        ax_nodes,
        records,
        "expanded",
        "expanded_ci",
        "Expanded nodes",
        "c",
        "Search expansion",
        (0, node_limit * 1.12),
        formatter=k_formatter,
        show_ylabels=True,
    )
    plot_stress_ratio(ax_ratio, records)

    handles = [
        Line2D([0], [0], marker="o", color=METHOD_COLORS["MP"], markerfacecolor=METHOD_COLORS["MP"], markeredgecolor="white", label="MP"),
        Line2D([0], [0], marker="s", color=METHOD_COLORS["MA"], markerfacecolor=METHOD_COLORS["MA"], markeredgecolor="white", label="MA"),
        Patch(facecolor="#D9903D", edgecolor="white", label="Time ratio"),
        Patch(facecolor="#6EA5D8", edgecolor="white", label="Expansion ratio"),
    ]
    top_legend(legend_ax, handles, "Legend", ncol=4, columnspacing=1.1)
    return save_pub_figure(fig, "fig_4_6_event_stress_tnr", dpi)


def plot_figure_9(records: list[dict[str, object]], dpi: int) -> list[Path]:
    """重绘图 9，图规模敏感性分析。"""

    labels = [str(row["scale"]).capitalize() for row in records]
    x = np.arange(len(records), dtype=float)
    nodes = np.asarray([float(row["nodes"]) for row in records], dtype=float)
    edges = np.asarray([float(row["edges"]) for row in records], dtype=float)
    mp_time = np.asarray([float(row["mp_time"]) for row in records], dtype=float)
    ma_time = np.asarray([float(row["ma_time"]) for row in records], dtype=float)
    time_ratio = np.asarray([float(row["time_ratio"]) for row in records], dtype=float)
    mp_success = np.asarray([float(row["mp_success"]) for row in records], dtype=float)
    ma_success = np.asarray([float(row["ma_success"]) for row in records], dtype=float)
    mp_eff = mp_time / np.maximum(nodes / 1000.0, 1e-9)
    ma_eff = ma_time / np.maximum(nodes / 1000.0, 1e-9)

    fig = plt.figure(figsize=(8.8, 4.35))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.96, bottom=0.30, hspace=0.20, wspace=0.32)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.24, 1.0], width_ratios=[1.42, 1.0])
    legend_ax = fig.add_subplot(grid[0, :])
    ax_trend = fig.add_subplot(grid[1, 0])
    ax_regime = fig.add_subplot(grid[1, 1])

    metrics = [
        ("Nodes", nodes, "#5F9EC9", "o", "-", 0.000),
        ("Edges", edges, "#F0A17A", "s", "-", 0.012),
        ("MA/MP time", time_ratio, "#3F3F3F", "D", "-", -0.012),
        ("MP success", mp_success, "#4B9B78", "^", "-", 0.024),
        ("MA success", ma_success, "#7E6AAE", "v", "-", -0.024),
        ("MP time per node", mp_eff, "#5E79B9", "o", "--", 0.036),
        ("MA time per node", ma_eff, "#D08C45", "s", "--", -0.036),
    ]
    handles = []
    for name, values, color, marker, line_style, offset in metrics:
        normalized = np.clip(range_normalize(values) + offset, -0.03, 1.08)
        ax_trend.plot(
            x,
            normalized,
            marker=marker,
            linestyle=line_style,
            color=color,
            linewidth=1.25,
            markersize=4.1,
            label=name,
        )
        handles.append(Line2D([0], [0], marker=marker, color=color, linestyle=line_style, linewidth=1.25, markersize=4.1, label=name))
    ax_trend.set_xlim(-0.12, len(labels) - 0.88)
    ax_trend.set_ylim(-0.06, 1.10)
    ax_trend.set_xticks(x)
    ax_trend.set_xticklabels(labels)
    ax_trend.set_ylabel("Range-normalized value")
    style_axis(ax_trend, grid_axis="y")
    add_panel_heading(ax_trend, "a", "Scale-response trajectories")
    top_legend(legend_ax, handles, "Metric", ncol=4, columnspacing=0.75, handlelength=1.35)

    sizes = 72 + 210 * (edges / np.nanmax(edges))
    scatter = ax_regime.scatter(
        time_ratio,
        mp_eff,
        s=sizes,
        c=mp_success,
        cmap="Greens",
        vmin=0,
        vmax=1,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    label_offsets = {"small": (5, 7), "medium": (5, 8), "large": (-26, -8)}
    for row, x_value, y_value in zip(records, time_ratio, mp_eff):
        scale = str(row["scale"])
        ax_regime.annotate(
            scale,
            (x_value, y_value),
            textcoords="offset points",
            xytext=label_offsets.get(scale, (8, 8)),
            fontsize=FONT_SIZE,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.35},
            zorder=4,
        )
    x_margin = max(0.08, (float(np.nanmax(time_ratio)) - float(np.nanmin(time_ratio))) * 0.16)
    y_margin = max(4.0, (float(np.nanmax(mp_eff)) - float(np.nanmin(mp_eff))) * 0.20)
    ax_regime.set_xlim(float(np.nanmin(time_ratio)) - x_margin, float(np.nanmax(time_ratio)) + x_margin)
    ax_regime.set_ylim(float(np.nanmin(mp_eff)) - y_margin, float(np.nanmax(mp_eff)) + y_margin)
    y_low, y_high = ax_regime.get_ylim()
    ax_regime.vlines(1.0, y_low, y_high, color="#555555", linestyles="--", linewidth=0.75)
    ax_regime.set_xlabel("MA/MP time ratio")
    ax_regime.set_ylabel("MP time per 1k nodes")
    style_axis(ax_regime, grid_axis="both")
    add_panel_heading(ax_regime, "b", "Scale regime map")
    colorbar = fig.colorbar(scatter, ax=ax_regime, fraction=0.052, pad=0.03)
    set_colorbar_style(colorbar, "MP success rate")
    return save_pub_figure(fig, "fig_4_5_graph_scale_tnr", dpi)


def plot_figure_10(records: list[dict[str, object]], dpi: int) -> list[Path]:
    """重绘图 10，路径质量消融分析。"""

    metric_names = [item[0] for item in QUALITY_METRICS]
    metric_labels = [item[1] for item in QUALITY_METRICS]
    matrix = np.full((len(ABLATION_METHODS), len(metric_names)), np.nan, dtype=float)
    for row in records:
        method = str(row["method"])
        metric = str(row["metric"])
        if method in ABLATION_METHODS and metric in metric_names:
            i = ABLATION_METHODS.index(method)
            j = metric_names.index(metric)
            current = matrix[i, j]
            value = float(row["change"])
            if math.isfinite(current):
                matrix[i, j] = np.nanmean([current, value])
            else:
                matrix[i, j] = value

    for i, method in enumerate(ABLATION_METHODS):
        for j, metric in enumerate(metric_names):
            values = [
                float(row["change"])
                for row in records
                if row["method"] == method and row["metric"] == metric and math.isfinite(float(row["change"]))
            ]
            matrix[i, j] = float(np.nanmean(values)) if values else float("nan")

    method_effect = np.nanmean(matrix, axis=1)
    finite = matrix[np.isfinite(matrix)]
    vmax = max(80.0, float(np.nanpercentile(np.abs(finite), 90)) if finite.size else 80.0)

    fig = plt.figure(figsize=(9.35, 3.35))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.32, wspace=0.36)
    grid = fig.add_gridspec(1, 4, width_ratios=[1.22, 1.22, 1.22, 1.08])
    ax_heat = fig.add_subplot(grid[0, 0:3])
    ax_summary = fig.add_subplot(grid[0, 3])

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax_heat.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
    ax_heat.set_xticks(np.arange(len(metric_labels)))
    ax_heat.set_xticklabels(metric_labels)
    ax_heat.set_yticks(np.arange(len(ABLATION_METHODS)))
    ax_heat.set_yticklabels(ABLATION_METHODS)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if math.isfinite(value):
                color = "white" if abs(value) > vmax * 0.52 else "#222222"
                ax_heat.text(j, i, value_label(value, 1, signed=True), ha="center", va="center", fontsize=FONT_SIZE, color=color)
    colorbar = fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.025)
    set_colorbar_style(colorbar, "Change relative to MP, %")
    ax_heat.set_ylabel("Ablation variant")
    ax_heat.tick_params(length=0, labelsize=FONT_SIZE)
    add_panel_heading(ax_heat, "a", "Metric-level ablation effect")

    y = np.arange(len(ABLATION_METHODS), dtype=float)
    colors = [METHOD_COLORS[method] for method in ABLATION_METHODS]
    ax_summary.vlines(0, -0.5, len(ABLATION_METHODS) - 0.5, color="#555555", linewidth=0.75)
    ax_summary.barh(y, method_effect, color=colors, edgecolor="white", linewidth=0.35)
    ax_summary.set_yticks(y)
    ax_summary.set_yticklabels(ABLATION_METHODS)
    ax_summary.invert_yaxis()
    ax_summary.set_xlabel("Mean change, %")
    finite_effect = method_effect[np.isfinite(method_effect)]
    if finite_effect.size:
        margin = max(6.0, float(np.nanmax(np.abs(finite_effect))) * 0.14)
        ax_summary.set_xlim(float(np.nanmin(finite_effect)) - margin, float(np.nanmax(finite_effect)) + margin)
    for yi, value in zip(y, method_effect):
        if math.isfinite(value):
            if value >= 0:
                ax_summary.text(value + 1.0, yi, value_label(value, 1, signed=True), va="center", ha="left", fontsize=FONT_SIZE)
            else:
                if abs(value) < 24:
                    ax_summary.text(
                        value - 2.0,
                        yi,
                        value_label(value, 1, signed=True),
                        va="center",
                        ha="right",
                        fontsize=FONT_SIZE,
                        color="#222222",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.20},
                    )
                else:
                    ax_summary.text(
                        value + max(4.0, abs(value) * 0.035),
                        yi,
                        value_label(value, 1, signed=True),
                        va="center",
                        ha="left",
                        fontsize=FONT_SIZE,
                        color="white",
                    )
    style_axis(ax_summary, grid_axis="x")
    add_panel_heading(ax_summary, "b", "Variant summary")
    return save_pub_figure(fig, "fig_4_6_ablation_quality_tnr", dpi)


def mean_by_method(records: list[dict[str, object]], field: str) -> dict[str, float]:
    """按方法计算跨场景均值。"""

    result: dict[str, float] = {}
    for method in STRUCTURE_METHODS:
        values = [float(row[field]) for row in records if row["method"] == method]
        result[method] = float(np.nanmean(values)) if values else float("nan")
    return result


def plot_figure_11(records: list[dict[str, object]], dpi: int) -> list[Path]:
    """重绘图 11，搜索工作量消融分析。"""

    expanded_map = mean_by_method(records, "expanded")
    time_map = mean_by_method(records, "time_ms")
    methods = STRUCTURE_METHODS
    x = np.arange(len(methods), dtype=float)
    expanded = np.asarray([expanded_map[method] for method in methods], dtype=float)
    replan_time = np.asarray([time_map[method] for method in methods], dtype=float)
    colors = [METHOD_COLORS[method] for method in methods]

    fig = plt.figure(figsize=(8.7, 5.25))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.22, hspace=1.05, wspace=0.32)
    grid = fig.add_gridspec(3, 3, height_ratios=[0.16, 1.0, 1.15], width_ratios=[1.0, 1.0, 1.05])
    legend_ax = fig.add_subplot(grid[0, :])
    ax_nodes = fig.add_subplot(grid[1, 0])
    ax_time = fig.add_subplot(grid[1, 1])
    ax_gain = fig.add_subplot(grid[1, 2])
    ax_map = fig.add_subplot(grid[2, :])

    ax_nodes.bar(x, expanded, color=colors, edgecolor="white", linewidth=0.35)
    ax_nodes.set_xticks(x)
    ax_nodes.set_xticklabels(methods)
    ax_nodes.set_yscale("log")
    ax_nodes.set_ylabel("Mean expanded nodes")
    style_axis(ax_nodes, grid_axis="y")
    add_panel_heading(ax_nodes, "a", "Search workload", y=-0.36)

    ax_time.bar(x, replan_time, color=colors, edgecolor="white", linewidth=0.35)
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(methods)
    ax_time.set_yscale("log")
    ax_time.set_ylabel("Mean replanning time, ms")
    style_axis(ax_time, grid_axis="y")
    add_panel_heading(ax_time, "b", "Replanning time", y=-0.36)

    mv_index = methods.index("MV")
    mv_expanded = expanded[mv_index]
    mv_time = replan_time[mv_index]
    expanded_gain = np.asarray([mv_expanded / value if value > 0 else np.nan for value in expanded], dtype=float)
    time_gain = np.asarray([mv_time / value if value > 0 else np.nan for value in replan_time], dtype=float)
    width = 0.34
    ax_gain.hlines(1.0, -0.5, len(methods) - 0.5, color="#555555", linestyles="--", linewidth=0.75)
    ax_gain.bar(x - width / 2, expanded_gain, width=width, color="#8FBBD9", edgecolor="white", linewidth=0.35)
    ax_gain.bar(x + width / 2, time_gain, width=width, color="#E5A46E", edgecolor="white", linewidth=0.35)
    ax_gain.set_xticks(x)
    ax_gain.set_xticklabels(methods)
    ax_gain.set_ylabel("Gain vs MV, fold")
    style_axis(ax_gain, grid_axis="y")
    add_panel_heading(ax_gain, "c", "Compression gain", y=-0.36)

    finite_expanded = expanded[np.isfinite(expanded) & (expanded > 0)]
    finite_time = replan_time[np.isfinite(replan_time) & (replan_time > 0)]
    sizes = 76 + 130 * (expanded_gain / np.nanmax(expanded_gain)) if np.isfinite(expanded_gain).any() else np.full_like(x, 90.0)
    label_offsets = {
        "MP": (8, 10),
        "MA": (10, -12),
        "MF": (-20, 10),
        "MR": (12, -2),
        "MV": (-24, -12),
    }
    for xi, yi, size, method, color in zip(expanded, replan_time, sizes, methods, colors):
        ax_map.scatter(xi, yi, s=size, color=color, edgecolor="white", linewidth=0.55, zorder=3)
        ax_map.annotate(
            method,
            (xi, yi),
            textcoords="offset points",
            xytext=label_offsets.get(method, (8, 8)),
            fontsize=FONT_SIZE,
            color="#222222",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.35},
            clip_on=False,
            zorder=4,
        )
    if finite_expanded.size and finite_time.size:
        fit = np.polyfit(np.log10(finite_expanded), np.log10(finite_time), 1)
        xs = np.geomspace(float(finite_expanded.min()), float(finite_expanded.max()), 80)
        ys = 10 ** (fit[1] + fit[0] * np.log10(xs))
        ax_map.plot(xs, ys, color="#555555", linewidth=0.9, linestyle="--")
        ax_map.set_xlim(float(finite_expanded.min()) * 0.48, float(finite_expanded.max()) * 1.35)
        ax_map.set_ylim(float(finite_time.min()) * 0.42, float(finite_time.max()) * 1.45)
    ax_map.set_xscale("log")
    ax_map.set_yscale("log")
    ax_map.set_xlabel("Mean expanded nodes")
    ax_map.set_ylabel("Mean replanning time, ms")
    style_axis(ax_map, grid_axis="both")
    add_panel_heading(ax_map, "d", "Workload-efficiency map", y=-0.48)

    handles = [
        Patch(facecolor="#8FBBD9", edgecolor="white", label="Node gain"),
        Patch(facecolor="#E5A46E", edgecolor="white", label="Time gain"),
        Line2D([0], [0], color="#555555", linestyle="--", linewidth=0.9, label="Log-log trend"),
    ]
    top_legend(legend_ax, handles, "Legend", ncol=3, columnspacing=1.0)
    return save_pub_figure(fig, "fig_4_7_ablation_workload_tnr", dpi)


def replace_docx_images(input_docx: Path, output_docx: Path) -> Path:
    """复制 DOCX，并替换实验结果图的媒体文件。"""

    replacements = {
        "word/media/image175.png": FIG_DIR / "fig_4_1_external_augmented_tnr.png",
        "word/media/image176.png": FIG_DIR / "fig_4_4_event_replanning_tnr.png",
        "word/media/image179.png": FIG_DIR / "fig_4_6_event_stress_tnr.png",
        "word/media/image180.png": FIG_DIR / "fig_4_5_graph_scale_tnr.png",
        "word/media/image181.png": FIG_DIR / "fig_4_6_ablation_quality_tnr.png",
        "word/media/image182.png": FIG_DIR / "fig_4_7_ablation_workload_tnr.png",
    }
    missing = [str(path) for path in replacements.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少待替换图件：" + "；".join(missing))
    if not input_docx.exists():
        raise FileNotFoundError(f"缺少输入稿件：{input_docx}")

    output_docx.parent.mkdir(parents=True, exist_ok=True)
    temp_docx = output_docx.with_suffix(".tmp.docx")
    with zipfile.ZipFile(input_docx, "r") as source, zipfile.ZipFile(temp_docx, "w") as target:
        names = set(source.namelist())
        for item in source.infolist():
            data = replacements[item.filename].read_bytes() if item.filename in replacements else source.read(item.filename)
            target.writestr(item, data)
    absent = [name for name in replacements if name not in names]
    if absent:
        temp_docx.unlink(missing_ok=True)
        raise FileNotFoundError("DOCX 中未找到目标媒体文件：" + "；".join(absent))
    shutil.move(str(temp_docx), str(output_docx))
    return output_docx


def assert_english_svg(svg_paths: list[Path]) -> None:
    """检查 SVG 中是否残留中文可见文本。"""

    forbidden = [char for char in "华山黄峨眉场景方法成功率重规划路径长度通信风险事件节点图规模消融"]
    for path in svg_paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        found = sorted({char for char in forbidden if char in text})
        if found:
            raise AssertionError(f"{path.name} 残留中文字符：{''.join(found)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="重绘 aei_v8 实验章节结果图")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--docx", type=Path, default=DEFAULT_DOCX)
    parser.add_argument("--out-docx", type=Path, default=DEFAULT_OUT_DOCX)
    parser.add_argument("--skip-docx", action="store_true", help="只生成图片，不替换 DOCX")
    args = parser.parse_args()

    configure_matplotlib()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    produced: list[Path] = []
    produced += plot_figure_6(load_external_records(), args.dpi)
    produced += plot_figure_7(load_event_records(), args.dpi)
    produced += plot_figure_8(load_stress_records(), args.dpi)
    produced += plot_figure_9(load_scale_records(), args.dpi)
    produced += plot_figure_10(load_quality_records(), args.dpi)
    produced += plot_figure_11(load_structural_records(), args.dpi)

    svg_paths = [path for path in produced if path.suffix.lower() == ".svg"]
    assert_english_svg(svg_paths)

    print("已生成优化后的实验结果图：")
    for path in produced:
        print(path)

    if not args.skip_docx:
        out_docx = replace_docx_images(args.docx, args.out_docx)
        print("已生成替换图片后的 DOCX 副本：")
        print(out_docx)


if __name__ == "__main__":
    main()
