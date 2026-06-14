"""基于新增实验数据生成第四章新版论文图件。

脚本只依赖 csv、numpy 与 matplotlib，不使用 pandas。输出图件采用英文
可编辑文本，并同时导出 SVG、PDF、TIFF 与 PNG，便于论文排版和后续审稿核对。
"""

from __future__ import annotations

import csv
import math
import warnings
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "final_results" / "paper_revision" / "source_data" / "chapter4_python"
TABLE_DIR = PROJECT_ROOT / "final_results" / "paper_revision" / "tables"
FIG_DIR = PROJECT_ROOT / "final_results" / "paper_revision" / "figures" / "chapter4_python"

FIGURE_4_1_SOURCE = SOURCE_DIR / "figure_4_1_external_augmented_source.csv"
FIGURE_4_6_SOURCE = SOURCE_DIR / "figure_4_6_event_stress_source.csv"
TABLE_4_3_SOURCE = TABLE_DIR / "table_4_3_three_scene_main_results.csv"

SCENE_ORDER = ["华山", "黄山", "峨眉山"]
SCENE_LABELS = {
    "华山": "Huashan",
    "黄山": "Huangshan",
    "峨眉山": "Emeishan",
}
SCENE_COLORS = {
    "华山": "#2F6EA6",
    "黄山": "#D9903D",
    "峨眉山": "#4B9B78",
}
SCENE_MARKERS = {
    "华山": "o",
    "黄山": "s",
    "峨眉山": "D",
}

METHOD_ORDER = ["MP", "MA", "MF", "MR", "MV", "BI-APF-RRT*", "GWO-DE"]
METHOD_LABELS = {
    "MP": "MP",
    "MA": "MA",
    "MF": "MF",
    "MR": "MR",
    "MV": "MV",
    "BI-APF-RRT*": "BI-APF-RRT*",
    "GWO-DE": "GWO-DE",
}

BASELINE_LABELS = {
    "B4_Proposed_LPA_Layered": "MP",
    "B2_GlobalAstar_Layered": "MA",
}
METHOD_COLORS = {
    "MP": "#2F6EA6",
    "MA": "#D9903D",
}
METHOD_MARKERS = {
    "MP": "o",
    "MA": "s",
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
    "no_fly_radius_0_4": "No-fly radius 0.4 km",
    "no_fly_radius_0_8": "No-fly radius 0.8 km",
    "no_fly_radius_1_2": "No-fly radius 1.2 km",
    "wind_severity_0_5": "Wind severity 0.5",
    "wind_severity_1_0": "Wind severity 1.0",
    "wind_severity_1_5": "Wind severity 1.5",
}

LEGEND_BOX = {
    "facecolor": "white",
    "edgecolor": "#303030",
    "linewidth": 0.65,
}


def configure_matplotlib() -> None:
    """设置适合论文图件的 matplotlib 样式。"""

    warnings.filterwarnings("ignore", message=".*MERG NOT subset.*")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.6,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def parse_float(value: object) -> float:
    """把 CSV 字符串转换为浮点数。"""

    text = str(value or "").strip().replace(",", "").replace("%", "")
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def read_csv_records(path: Path) -> list[dict[str, str]]:
    """读取 CSV 记录。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少源数据文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def require_columns(rows: list[dict[str, str]], path: Path, columns: list[str]) -> None:
    """确认源数据包含绘图所需字段。"""

    if not rows:
        raise ValueError(f"源数据为空：{path}")
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError(f"{path.name} 缺少字段：{', '.join(missing)}")


def pick_column(row: dict[str, str], candidates: list[str]) -> str:
    """从多个候选表头中选择当前数据实际存在的字段。"""

    for candidate in candidates:
        if candidate in row:
            return candidate
    raise KeyError(f"缺少候选字段：{', '.join(candidates)}")


def style_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    """统一坐标轴网格和刻度样式。"""

    ax.grid(True, axis=grid_axis, color="#D9D9D9", linewidth=0.45, alpha=0.85)
    ax.tick_params(length=2.4, width=0.55)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)


def add_panel_heading(
    ax: plt.Axes,
    label: str,
    title: str,
    x: float = 0.0,
    y: float = 1.045,
    title_dx: float = 0.08,
) -> None:
    """添加无边框面板字母和英文标题。"""

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        x + title_dx,
        y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        clip_on=False,
    )


def frame_legend(legend) -> None:
    """只给真正的图例添加边框。"""

    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor(LEGEND_BOX["facecolor"])
    frame.set_edgecolor(LEGEND_BOX["edgecolor"])
    frame.set_linewidth(LEGEND_BOX["linewidth"])
    frame.set_alpha(1.0)


def save_pub_figure(fig: plt.Figure, basename: str, dpi: int = 600) -> list[Path]:
    """按论文交付要求导出多格式图件。"""

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        FIG_DIR / f"{basename}.svg",
        FIG_DIR / f"{basename}.pdf",
        FIG_DIR / f"{basename}.tiff",
        FIG_DIR / f"{basename}.png",
    ]
    fig.savefig(paths[0], bbox_inches="tight", pad_inches=0.04)
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.04)
    fig.savefig(paths[2], dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(paths[3], dpi=240, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return paths


def method_sort_key(row: dict[str, str]) -> tuple[int, int]:
    """按场景和方法固定顺序排序。"""

    scene = row.get("场景", "")
    method = row.get("方法", "")
    return (
        SCENE_ORDER.index(scene) if scene in SCENE_ORDER else len(SCENE_ORDER),
        METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER),
    )


def audit_figure_4_1_source(figure_rows: list[dict[str, str]], table_rows: list[dict[str, str]]) -> None:
    """核对图 4.1 源数据与主结果表是否一致。"""

    figure_index = {(row["场景"], row["方法"]): row for row in figure_rows}
    table_index = {(row["场景"], row["方法"]): row for row in table_rows}
    if set(figure_index) != set(table_index):
        missing_in_table = sorted(set(figure_index) - set(table_index))
        missing_in_figure = sorted(set(table_index) - set(figure_index))
        raise ValueError(f"图 4.1 源数据与主表方法集合不一致：{missing_in_table}，{missing_in_figure}")

    field_map = {
        "成功率": ["成功率"],
        "重规划时间_ms": ["重规划时间 ms", "重规划或规划时间 ms"],
        "路径代价": ["路径代价"],
        "长度_km": ["长度 km"],
        "风险暴露": ["风险暴露"],
        "通信覆盖率": ["通信覆盖率"],
    }
    mismatches: list[tuple[tuple[str, str], str, float, float]] = []
    for key in sorted(figure_index):
        table_row = table_index[key]
        for figure_field, table_candidates in field_map.items():
            table_field = pick_column(table_row, table_candidates)
            figure_value = parse_float(figure_index[key][figure_field])
            table_value = parse_float(table_row[table_field])
            if abs(figure_value - table_value) > 1e-9:
                mismatches.append((key, figure_field, figure_value, table_value))
    if mismatches:
        first = mismatches[0]
        raise ValueError(f"图 4.1 源数据与主表数值不一致，示例：{first}")


def load_figure_4_1_records() -> list[dict[str, object]]:
    """读取并核实图 4.1 的新增外部基线源数据。"""

    figure_rows = read_csv_records(FIGURE_4_1_SOURCE)
    table_rows = read_csv_records(TABLE_4_3_SOURCE)
    require_columns(
        figure_rows,
        FIGURE_4_1_SOURCE,
        ["场景", "方法", "成功率", "重规划时间_ms", "路径代价", "长度_km", "通信覆盖率", "风险暴露"],
    )
    require_columns(
        table_rows,
        TABLE_4_3_SOURCE,
        ["场景", "方法", "成功率", "路径代价", "长度 km", "风险暴露", "通信覆盖率"],
    )
    pick_column(table_rows[0], ["重规划时间 ms", "重规划或规划时间 ms"])
    audit_figure_4_1_source(figure_rows, table_rows)

    records: list[dict[str, object]] = []
    for row in sorted(figure_rows, key=method_sort_key):
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
    return records


def load_figure_4_6_records() -> list[dict[str, object]]:
    """读取并核实图 4.6 的事件压力源数据。"""

    rows = read_csv_records(FIGURE_4_6_SOURCE)
    require_columns(
        rows,
        FIGURE_4_6_SOURCE,
        [
            "stress_type",
            "stress_label",
            "baseline",
            "n_trials",
            "n_success",
            "success_rate",
            "mean_cumulative_replan_ms",
            "ci95_cumulative_replan_ms",
            "mean_cumulative_expanded",
            "ci95_event_expanded",
            "failure_reason_top",
        ],
    )

    records: list[dict[str, object]] = []
    for row in rows:
        stress_label = row["stress_label"]
        baseline = row["baseline"]
        if stress_label not in STRESS_ORDER:
            raise ValueError(f"未知压力设置：{stress_label}")
        if baseline not in BASELINE_LABELS:
            raise ValueError(f"未知 baseline：{baseline}")
        records.append(
            {
                "stress_label": stress_label,
                "method": BASELINE_LABELS[baseline],
                "n_trials": int(parse_float(row["n_trials"])),
                "n_success": int(parse_float(row["n_success"])),
                "success": parse_float(row["success_rate"]),
                "time_ms": parse_float(row["mean_cumulative_replan_ms"]),
                "time_ci": parse_float(row["ci95_cumulative_replan_ms"]),
                "expanded": parse_float(row["mean_cumulative_expanded"]),
                "expanded_ci": parse_float(row["ci95_event_expanded"]),
                "failure_reason": row.get("failure_reason_top", "").strip(),
            }
        )

    counts: dict[tuple[str, str], int] = {}
    for record in records:
        key = (str(record["stress_label"]), str(record["method"]))
        counts[key] = counts.get(key, 0) + 1
    expected = {(stress, method) for stress in STRESS_ORDER for method in ["MP", "MA"]}
    if set(counts) != expected or any(value != 1 for value in counts.values()):
        raise ValueError("图 4.6 源数据缺少压力设置或方法组合")
    return sorted(records, key=lambda row: (STRESS_ORDER.index(str(row["stress_label"])), str(row["method"])))


def value_by(records: list[dict[str, object]], scene: str, method: str, field: str) -> float:
    """按场景和方法取数值。"""

    for row in records:
        if row["scene"] == scene and row["method"] == method:
            return float(row[field])
    return float("nan")


def plot_metric_panel(
    ax: plt.Axes,
    records: list[dict[str, object]],
    field: str,
    transform: Callable[[float], float],
    xlabel: str,
    label: str,
    title: str,
    xlim: tuple[float, float] | None = None,
    log_scale: bool = False,
    percent_axis: bool = False,
    show_ylabels: bool = False,
) -> None:
    """绘制图 4.1 的单个指标点图面板。"""

    y_positions = np.arange(len(METHOD_ORDER), dtype=float)
    offsets = {"华山": -0.18, "黄山": 0.0, "峨眉山": 0.18}
    for scene in SCENE_ORDER:
        xs = [transform(value_by(records, scene, method, field)) for method in METHOD_ORDER]
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
            label=SCENE_LABELS[scene],
        )
    for y_value in y_positions:
        ax.axhline(y_value + 0.5, color="#EEEEEE", linewidth=0.45, zorder=0)
    ax.set_yticks(y_positions)
    if show_ylabels:
        ax.set_yticklabels([METHOD_LABELS[method] for method in METHOD_ORDER])
    else:
        ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    if xlim:
        ax.set_xlim(*xlim)
    if log_scale:
        ax.set_xscale("log")
        ax.set_xticks([10, 30, 100, 300, 1000])
        ax.set_xticklabels(["10", "30", "100", "300", "1000"])
    if percent_axis:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}"))
    style_axis(ax, grid_axis="x")
    add_panel_heading(ax, label, title)


def plot_figure_4_1(records: list[dict[str, object]]) -> list[Path]:
    """重绘图 4.1，展示七种方法在三场景中的多指标表现。"""

    fig, axes = plt.subplots(2, 3, figsize=(8.35, 5.85), constrained_layout=True)
    axes_flat = list(axes.ravel())
    panels = [
        ("success", lambda value: value * 100.0, "Success rate, %", "a", "Success rate", (22, 104), False, True),
        ("time_ms", lambda value: value, "Replanning time, ms", "b", "Replanning time", (7, 1350), True, False),
        ("cost", lambda value: value, "Path cost", "c", "Path cost", (8, 104), False, False),
        ("length_km", lambda value: value, "Path length, km", "d", "Path length", (6.5, 15.8), False, False),
        ("coverage", lambda value: value, "Communication coverage", "e", "Coverage", (0.22, 0.95), False, False),
        ("risk", lambda value: value, "Risk exposure", "f", "Risk exposure", (2.45, 4.45), False, False),
    ]
    for index, panel in enumerate(panels):
        field, transform, xlabel, label, title, xlim, log_scale, percent_axis = panel
        plot_metric_panel(
            axes_flat[index],
            records,
            field,
            transform,
            xlabel,
            label,
            title,
            xlim=xlim,
            log_scale=log_scale,
            percent_axis=percent_axis,
            show_ylabels=index % 3 == 0,
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
    legend = fig.legend(
        handles=handles,
        labels=[SCENE_LABELS[scene] for scene in SCENE_ORDER],
        title="Scene",
        loc="upper center",
        bbox_to_anchor=(0.52, 1.105),
        ncol=3,
        frameon=True,
        handletextpad=0.45,
        columnspacing=1.0,
    )
    frame_legend(legend)
    return save_pub_figure(fig, "fig_4_1_external_augmented_python")


def event_pair(records: list[dict[str, object]], stress_label: str) -> dict[str, dict[str, object]]:
    """取单个压力设置下 MP 与 MA 的记录。"""

    pair: dict[str, dict[str, object]] = {}
    for row in records:
        if row["stress_label"] == stress_label:
            pair[str(row["method"])] = row
    return pair


def k_formatter(value: float, _pos: int) -> str:
    """把节点数坐标格式化为 k。"""

    if abs(value) < 1e-9:
        return "0"
    return f"{value / 1000:.0f}k"


def set_stress_axis_labels(ax: plt.Axes, show_labels: bool) -> None:
    """设置事件压力图的 y 轴标签，避免共享坐标轴误清空标签。"""

    y_positions = np.arange(len(STRESS_ORDER), dtype=float)
    ax.set_yticks(y_positions)
    if show_labels:
        ax.set_yticklabels([STRESS_LABELS[stress] for stress in STRESS_ORDER])
        ax.tick_params(axis="y", labelleft=True)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_ylim(len(STRESS_ORDER) - 0.45, -0.55)


def plot_success_panel(ax: plt.Axes, records: list[dict[str, object]]) -> None:
    """绘制事件压力实验的成功率面板。"""

    y_positions = np.arange(len(STRESS_ORDER), dtype=float)
    offsets = {"MP": -0.16, "MA": 0.16}
    for method in ["MP", "MA"]:
        xs = [float(event_pair(records, stress)[method]["success"]) * 100.0 for stress in STRESS_ORDER]
        ys = y_positions + offsets[method]
        ax.hlines(ys, 80.0, xs, color=METHOD_COLORS[method], linewidth=1.45, alpha=0.88)
        ax.scatter(
            xs,
            ys,
            s=26,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
            label=method,
        )
        for x_value, y_value in zip(xs, ys):
            ax.text(x_value + 0.55, y_value, f"{x_value:.1f}", va="center", fontsize=5.9, color="#303030")
    ax.set_xlim(78, 103.8)
    set_stress_axis_labels(ax, show_labels=True)
    ax.set_xlabel("Success rate, %")
    style_axis(ax, grid_axis="x")
    add_panel_heading(ax, "a", "Stress reliability")
    ax.text(84.2, 2.48, "5 failures", fontsize=5.9, color="#6B6B6B")


def plot_dumbbell_panel(
    ax: plt.Axes,
    records: list[dict[str, object]],
    field: str,
    ci_field: str,
    xlabel: str,
    label: str,
    title: str,
    xlim: tuple[float, float],
    formatter: Callable[[float, int], str] | None = None,
) -> None:
    """绘制 MP 与 MA 的绝对量哑铃图。"""

    y_positions = np.arange(len(STRESS_ORDER), dtype=float)
    for y_value, stress in zip(y_positions, STRESS_ORDER):
        pair = event_pair(records, stress)
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
                markersize=4.3,
                elinewidth=0.75,
                capsize=2.0,
                zorder=3,
            )
        ax.text(max(mp_value, ma_value) + (xlim[1] - xlim[0]) * 0.018, y_value, f"{ratio:.2f}x", va="center", fontsize=5.9)
    ax.set_xlim(*xlim)
    set_stress_axis_labels(ax, show_labels=False)
    ax.set_xlabel(xlabel)
    if formatter:
        ax.xaxis.set_major_formatter(FuncFormatter(formatter))
    style_axis(ax, grid_axis="x")
    add_panel_heading(ax, label, title)


def plot_ratio_panel(ax: plt.Axes, records: list[dict[str, object]]) -> None:
    """绘制 MA 相对 MP 的累计负担倍率。"""

    y_positions = np.arange(len(STRESS_ORDER), dtype=float)
    height = 0.30
    time_ratios = []
    node_ratios = []
    for stress in STRESS_ORDER:
        pair = event_pair(records, stress)
        time_ratios.append(float(pair["MA"]["time_ms"]) / max(float(pair["MP"]["time_ms"]), 1e-9))
        node_ratios.append(float(pair["MA"]["expanded"]) / max(float(pair["MP"]["expanded"]), 1e-9))

    ax.axvline(1.0, color="#555555", linewidth=0.75)
    ax.barh(
        y_positions - height / 2,
        np.asarray(time_ratios) - 1.0,
        left=1.0,
        height=height,
        color="#D9903D",
        edgecolor="white",
        label="Time ratio",
    )
    ax.barh(
        y_positions + height / 2,
        np.asarray(node_ratios) - 1.0,
        left=1.0,
        height=height,
        color="#6EA5D8",
        edgecolor="white",
        label="Expansion ratio",
    )
    for y_value, time_ratio, node_ratio in zip(y_positions, time_ratios, node_ratios):
        ax.text(time_ratio + 0.10, y_value - height / 2, f"{time_ratio:.2f}", va="center", fontsize=5.8)
        ax.text(node_ratio + 0.10, y_value + height / 2, f"{node_ratio:.2f}", va="center", fontsize=5.8)
    ax.set_xlim(0.8, 11.2)
    set_stress_axis_labels(ax, show_labels=False)
    ax.set_xlabel("MA/MP ratio")
    style_axis(ax, grid_axis="x")
    add_panel_heading(ax, "d", "Relative burden")
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, -0.20),
        title="Metric",
        frameon=True,
        ncol=2,
        handlelength=1.6,
        columnspacing=1.0,
    )
    frame_legend(legend)


def plot_figure_4_6(records: list[dict[str, object]]) -> list[Path]:
    """重绘图 4.6，展示事件压力下的可靠性和效率。"""

    fig = plt.figure(figsize=(8.35, 5.35), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[1.0, 1.0])
    ax_success = fig.add_subplot(grid[0, 0])
    ax_time = fig.add_subplot(grid[0, 1])
    ax_nodes = fig.add_subplot(grid[1, 0])
    ax_ratio = fig.add_subplot(grid[1, 1])

    plot_success_panel(ax_success, records)
    plot_dumbbell_panel(
        ax_time,
        records,
        "time_ms",
        "time_ci",
        "Cumulative replanning time, ms",
        "b",
        "Cumulative time",
        (40, 660),
    )
    plot_dumbbell_panel(
        ax_nodes,
        records,
        "expanded",
        "expanded_ci",
        "Cumulative expanded nodes",
        "c",
        "Search expansion",
        (0, 22500),
        formatter=k_formatter,
    )
    set_stress_axis_labels(ax_nodes, show_labels=True)
    plot_ratio_panel(ax_ratio, records)
    handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS[method],
            color="none",
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=5.2,
            label=method,
        )
        for method in ["MP", "MA"]
    ]
    legend = fig.legend(
        handles=handles,
        labels=["MP", "MA"],
        title="Method",
        loc="upper center",
        bbox_to_anchor=(0.52, 1.065),
        ncol=2,
        frameon=True,
        handletextpad=0.45,
        columnspacing=1.0,
    )
    frame_legend(legend)
    return save_pub_figure(fig, "fig_4_6_event_stress_python")


def print_audit_summary(records_4_1: list[dict[str, object]], records_4_6: list[dict[str, object]]) -> None:
    """输出本次绘图前的数据核验摘要。"""

    scene_counts = {scene: sum(1 for row in records_4_1 if row["scene"] == scene) for scene in SCENE_ORDER}
    method_count = len({str(row["method"]) for row in records_4_1})
    stress_counts = {
        stress: sum(1 for row in records_4_6 if row["stress_label"] == stress) for stress in STRESS_ORDER
    }
    print("数据核验完成")
    print(f"图 4.1 源数据：{len(records_4_1)} 行，{len(SCENE_ORDER)} 个场景，{method_count} 种方法，场景行数 {scene_counts}")
    print(f"图 4.6 源数据：{len(records_4_6)} 行，{len(STRESS_ORDER)} 个压力设置，压力设置行数 {stress_counts}")
    for stress in STRESS_ORDER:
        pair = event_pair(records_4_6, stress)
        time_ratio = float(pair["MA"]["time_ms"]) / max(float(pair["MP"]["time_ms"]), 1e-9)
        node_ratio = float(pair["MA"]["expanded"]) / max(float(pair["MP"]["expanded"]), 1e-9)
        print(f"{stress}: time_ratio={time_ratio:.2f}, expansion_ratio={node_ratio:.2f}")


def main() -> None:
    """执行新增实验数据图件生成流程。"""

    configure_matplotlib()
    records_4_1 = load_figure_4_1_records()
    records_4_6 = load_figure_4_6_records()
    print_audit_summary(records_4_1, records_4_6)

    produced: list[Path] = []
    produced += plot_figure_4_1(records_4_1)
    produced += plot_figure_4_6(records_4_6)

    print("新版实验结果图生成完成")
    for path in produced:
        print(path)


if __name__ == "__main__":
    main()
