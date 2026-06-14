"""生成第四章事件重规划与图规模敏感性的非热图主导重设计版本。"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

import create_chapter4_nature_figures as base


FIG_DIR = base.DEFAULT_FIG_DIR
SOURCE_DIR = base.DEFAULT_SOURCE_DIR


def read_records(path: Path) -> list[dict[str, object]]:
    """读取 CSV 记录，并将数值字段转换为浮点数。"""

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row: dict[str, object] = {}
            for key, value in raw.items():
                text = str(value or "").strip()
                if key in {"图规模", "场景"}:
                    row[key] = text
                else:
                    try:
                        row[key] = float(text)
                    except ValueError:
                        row[key] = float("nan")
            rows.append(row)
    return rows


def range_normalize(values: np.ndarray) -> np.ndarray:
    """将一维数组归一化到 0 到 1，保留趋势形态。"""

    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.full_like(array, np.nan)
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    if abs(high - low) < 1e-12:
        return np.full_like(array, 0.5)
    return (array - low) / (high - low)


def compact_ratio(value: float) -> str:
    """生成适合点内标注的倍率文本。"""

    if not math.isfinite(value):
        return ""
    return f"{value:.2f}x"


def plot_graph_scale_redesigned(scale: list[dict[str, object]], fig_dir: Path, dpi: int = 600) -> list[Path]:
    """用趋势线和状态散点重绘图规模敏感性分析。"""

    scales = [str(row["图规模"]) for row in scale]
    x = np.arange(len(scales), dtype=float)
    labels = [item.capitalize() for item in scales]
    nodes = np.asarray([float(row["节点数"]) for row in scale], dtype=float)
    edges = np.asarray([float(row["边数"]) for row in scale], dtype=float)
    mp_time = np.asarray([float(row["MP累计时间_ms"]) for row in scale], dtype=float)
    ma_time = np.asarray([float(row["MA累计时间_ms"]) for row in scale], dtype=float)
    time_ratio = np.asarray([float(row["MA与MP时间比"]) for row in scale], dtype=float)
    success = np.asarray([float(row["MP成功率"]) for row in scale], dtype=float)
    mp_eff = mp_time / np.maximum(nodes / 1000.0, 1e-9)
    ma_eff = ma_time / np.maximum(nodes / 1000.0, 1e-9)

    fig = plt.figure(figsize=(8.6, 4.05), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    ax_trend = fig.add_subplot(grid[0, 0])
    ax_regime = fig.add_subplot(grid[0, 1])

    metrics = [
        ("Nodes", nodes, "#5F9EC9", "o", "-"),
        ("Edges", edges, "#F0A17A", "s", "-"),
        ("Time ratio", time_ratio, "#3F3F3F", "D", "-"),
        ("Success", success, "#4B9B78", "^", "-"),
        ("MP efficiency", mp_eff, "#5E79B9", "o", "--"),
        ("MA efficiency", ma_eff, "#D08C45", "s", "--"),
    ]
    for name, values, color, marker, line_style in metrics:
        normalized = range_normalize(values)
        ax_trend.plot(
            x,
            normalized,
            marker=marker,
            linestyle=line_style,
            color=color,
            linewidth=1.35,
            markersize=4.0,
            label=name,
        )
    ax_trend.set_xlim(-0.12, len(labels) - 0.88)
    ax_trend.set_ylim(-0.05, 1.08)
    ax_trend.set_xticks(x)
    ax_trend.set_xticklabels(labels)
    ax_trend.set_ylabel("Range-normalized value")
    base.add_panel_heading(ax_trend, "a", "Scale-response trajectories")
    base.style_axis(ax_trend)
    base.add_framed_legend(
        ax_trend,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.13),
        ncol=3,
        title="Metric",
        columnspacing=0.9,
        handlelength=1.6,
        borderaxespad=0.15,
    )

    sizes = 70 + 210 * (edges / np.nanmax(edges))
    scatter = ax_regime.scatter(
        time_ratio,
        mp_eff,
        s=sizes,
        c=success,
        cmap="Greens",
        vmin=0,
        vmax=1,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax_regime.axvline(1.0, color="#555555", linestyle="--", linewidth=0.8)
    for scale_name, x_value, y_value in zip(scales, time_ratio, mp_eff):
        ax_regime.annotate(scale_name, (x_value, y_value), textcoords="offset points", xytext=(5, 4), fontsize=6.4)
    ax_regime.set_xlabel("MA/MP time ratio")
    ax_regime.set_ylabel("MP time per 1k nodes")
    ax_regime.set_xlim(0.34, 1.52)
    base.add_panel_heading(ax_regime, "b", "Scale regime map")
    base.style_axis(ax_regime)
    colorbar = fig.colorbar(scatter, ax=ax_regime, fraction=0.052, pad=0.03)
    colorbar.set_label("Success rate")

    return base.save_pub_figure(fig, fig_dir, "fig_4_5_graph_scale_redesigned_python", dpi)


def plot_event_replanning_redesigned(event: list[dict[str, object]], fig_dir: Path, dpi: int = 600) -> list[Path]:
    """用气泡矩阵和汇总条形重绘事件驱动重规划结果。"""

    rows = sorted(
        event,
        key=lambda row: (base.SCENE_ORDER.index(str(row["场景"])), float(row["事件数_K"])),
    )
    scene_positions = {scene: index for index, scene in enumerate(base.SCENE_ORDER)}
    x = np.asarray([float(row["事件数_K"]) for row in rows], dtype=float)
    y = np.asarray([scene_positions[str(row["场景"])] for row in rows], dtype=float)
    time_ratio = np.asarray([float(row["MA与MP时间比"]) for row in rows], dtype=float)
    mp_nodes = np.asarray([float(row["MP扩展节点"]) for row in rows], dtype=float)
    ma_nodes = np.asarray([float(row["MA扩展节点"]) for row in rows], dtype=float)
    node_ratio = ma_nodes / np.maximum(mp_nodes, 1e-9)
    color_values = np.log2(time_ratio)

    fig = plt.figure(figsize=(8.4, 4.25), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0])
    ax_matrix = fig.add_subplot(grid[0, 0])
    ax_summary = fig.add_subplot(grid[0, 1])

    norm = TwoSlopeNorm(vmin=-1.7, vcenter=0.0, vmax=1.7)
    sizes = 48 + 34 * np.clip(node_ratio, 1, 11)
    scatter = ax_matrix.scatter(
        x,
        y,
        s=sizes,
        c=color_values,
        cmap="RdBu_r",
        norm=norm,
        edgecolor="white",
        linewidth=0.75,
        zorder=3,
    )
    for x_value, y_value, ratio in zip(x, y, time_ratio):
        ax_matrix.text(x_value, y_value, compact_ratio(ratio), ha="center", va="center", fontsize=5.9, color="#1F1F1F", zorder=4)
    ax_matrix.set_xticks([1, 5, 10])
    ax_matrix.set_xlabel("Event count K")
    ax_matrix.set_yticks(list(scene_positions.values()))
    ax_matrix.set_yticklabels([base.SCENE_LABELS[scene] for scene in base.SCENE_ORDER])
    ax_matrix.set_ylim(-0.55, len(base.SCENE_ORDER) - 0.45)
    ax_matrix.set_xlim(0.35, 10.85)
    ax_matrix.grid(True, axis="both", color="#D6D6D6", linewidth=0.45, alpha=0.85)
    base.add_panel_heading(ax_matrix, "a", "Event burden matrix")
    for size_value, label in [(2.0, "2x nodes"), (6.0, "6x nodes"), (10.0, "10x nodes")]:
        ax_matrix.scatter([], [], s=48 + 34 * size_value, facecolor="#B0B0B0", edgecolor="white", label=label)
    base.add_framed_legend(
        ax_matrix,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.14),
        title="MA/MP nodes",
        ncol=3,
        scatterpoints=1,
        labelspacing=0.45,
        handletextpad=0.55,
    )
    colorbar = fig.colorbar(scatter, ax=ax_matrix, fraction=0.052, pad=0.03)
    colorbar.set_label("log2 MA/MP time")

    scene_labels = [base.SCENE_LABELS[scene] for scene in base.SCENE_ORDER]
    mean_time = []
    mean_nodes = []
    final_time = []
    final_nodes = []
    for scene in base.SCENE_ORDER:
        indices = [idx for idx, row in enumerate(rows) if str(row["场景"]) == scene]
        mean_time.append(float(np.nanmean(np.log2(time_ratio[indices]))))
        mean_nodes.append(float(np.nanmean(np.log2(node_ratio[indices]))))
        final_index = indices[-1]
        final_time.append(float(np.log2(time_ratio[final_index])))
        final_nodes.append(float(np.log2(node_ratio[final_index])))

    y_pos = np.arange(len(scene_labels), dtype=float)
    height = 0.34
    ax_summary.axvline(0, color="#555555", linewidth=0.8)
    ax_summary.barh(y_pos - height / 2, mean_time, height=height, color="#D08C45", edgecolor="white", label="Mean time ratio")
    ax_summary.barh(y_pos + height / 2, mean_nodes, height=height, color="#6EA5D8", edgecolor="white", label="Mean node ratio")
    ax_summary.scatter(final_time, y_pos - height / 2, marker="D", color="#8A4E2A", s=28, zorder=4, label="K=10 time")
    ax_summary.scatter(final_nodes, y_pos + height / 2, marker="D", color="#2E5F87", s=28, zorder=4, label="K=10 nodes")
    ax_summary.set_yticks(y_pos)
    ax_summary.set_yticklabels(scene_labels)
    ax_summary.set_xlabel("log2 ratio, MA/MP")
    ax_summary.invert_yaxis()
    base.add_panel_heading(ax_summary, "b", "Scene-level ratio summary")
    base.style_axis(ax_summary, grid_axis="x")
    base.add_framed_legend(ax_summary, loc="lower right", title="Summary")

    return base.save_pub_figure(fig, fig_dir, "fig_4_4_event_replanning_redesigned_python", dpi)


def main() -> None:
    """执行重设计图表生成流程。"""

    base.configure_matplotlib()
    graph_scale = read_records(SOURCE_DIR / "figure_4_5_graph_scale_source.csv")
    event = read_records(SOURCE_DIR / "figure_4_4_event_replanning_source.csv")

    produced: list[Path] = []
    produced += plot_graph_scale_redesigned(graph_scale, FIG_DIR)
    produced += plot_event_replanning_redesigned(event, FIG_DIR)

    print("重设计实验结果图生成完成")
    for path in produced:
        print(path)


if __name__ == "__main__":
    main()
