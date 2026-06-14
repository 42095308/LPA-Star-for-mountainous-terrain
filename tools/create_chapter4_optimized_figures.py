"""生成第四章两张实验结果图的优化版，保留原始脚本与原始图片。"""

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
    """读取 CSV，并尽量把数值字段转换为浮点数。"""

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


def normalize_columns(values: np.ndarray) -> np.ndarray:
    """对每一列做 0 到 1 归一化，用于多量纲指标融合展示。"""

    out = np.full_like(values, np.nan, dtype=float)
    for col in range(values.shape[1]):
        column = values[:, col].astype(float)
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            continue
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
        if abs(high - low) < 1e-12:
            out[:, col] = 0.5
        else:
            out[:, col] = (column - low) / (high - low)
    return out


def add_group_spans(ax: plt.Axes, groups: list[tuple[int, int, str]], y: float) -> None:
    """给热图列组添加轻量分组标签。"""

    for start, end, label in groups:
        center = (start + end) / 2
        ax.text(center, y, label, ha="center", va="bottom", fontsize=6.6, fontweight="bold", clip_on=False)


def format_graph_value(label: str, value: float) -> str:
    """为图规模热图生成紧凑数值标注。"""

    if not math.isfinite(value):
        return ""
    if label in {"Nodes", "Edges"}:
        return f"{value / 1000:.1f}k"
    if label == "Edge/node":
        return f"{value:.2f}"
    if label in {"MP time", "MA time"}:
        return f"{value:.0f}"
    if label == "Time ratio":
        return f"{value:.2f}x"
    if label == "Success":
        return f"{value * 100:.0f}%"
    return f"{value:.1f}"


def plot_graph_scale_optimized(scale: list[dict[str, object]], fig_dir: Path, dpi: int = 600) -> list[Path]:
    """优化图规模敏感性图，使用热图融合规模、速度、成功率与效率。"""

    scales = [str(row["图规模"]) for row in scale]
    nodes = np.asarray([float(row["节点数"]) for row in scale], dtype=float)
    edges = np.asarray([float(row["边数"]) for row in scale], dtype=float)
    mp_time = np.asarray([float(row["MP累计时间_ms"]) for row in scale], dtype=float)
    ma_time = np.asarray([float(row["MA累计时间_ms"]) for row in scale], dtype=float)
    time_ratio = np.asarray([float(row["MA与MP时间比"]) for row in scale], dtype=float)
    success = np.asarray([float(row["MP成功率"]) for row in scale], dtype=float)
    edge_node = edges / np.maximum(nodes, 1.0)
    mp_eff = mp_time / np.maximum(nodes / 1000.0, 1e-9)
    ma_eff = ma_time / np.maximum(nodes / 1000.0, 1e-9)

    labels = ["Nodes", "Edges", "Edge/node", "MP time", "MA time", "Time ratio", "Success", "MP eff.", "MA eff."]
    raw = np.column_stack([nodes, edges, edge_node, mp_time, ma_time, time_ratio, success, mp_eff, ma_eff])
    normalized = normalize_columns(raw)

    fig = plt.figure(figsize=(8.4, 3.75), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[2.25, 1.0])
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_regime = fig.add_subplot(grid[0, 1])

    image = ax_heat.imshow(normalized, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax_heat.set_xticks(np.arange(len(labels)))
    ax_heat.set_xticklabels(labels, rotation=35, ha="right")
    ax_heat.set_yticks(np.arange(len(scales)))
    ax_heat.set_yticklabels([item.capitalize() for item in scales])
    for row in range(raw.shape[0]):
        for col, label in enumerate(labels):
            value = raw[row, col]
            color = "white" if normalized[row, col] > 0.72 else "#1F1F1F"
            ax_heat.text(col, row, format_graph_value(label, value), ha="center", va="center", fontsize=5.8, color=color)
    ax_heat.tick_params(length=0)
    base.add_panel_heading(ax_heat, "a", "Scale-normalized multi-metric profile", x=0.0, y=1.16, title_dx=0.075)
    add_group_spans(ax_heat, [(0, 2, "Graph load"), (3, 5, "Runtime"), (6, 8, "Reliability and efficiency")], y=-0.78)
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.038, pad=0.015)
    colorbar.set_label("Column-normalized value")

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
    base.add_panel_heading(ax_regime, "b", "Scale regime map")
    base.style_axis(ax_regime)
    cbar = fig.colorbar(scatter, ax=ax_regime, fraction=0.052, pad=0.03)
    cbar.set_label("Success rate")

    return base.save_pub_figure(fig, fig_dir, "fig_4_5_graph_scale_optimized_python", dpi)


def format_event_value(label: str, value: float) -> str:
    """为事件重规划热图生成紧凑数值标注。"""

    if not math.isfinite(value):
        return ""
    if label in {"MP time", "MA time"}:
        return f"{value:.0f}"
    if label in {"MP nodes", "MA nodes"}:
        return f"{value / 1000:.1f}k"
    return f"{value:.2f}x"


def plot_event_replanning_optimized(event: list[dict[str, object]], fig_dir: Path, dpi: int = 600) -> list[Path]:
    """优化连续事件重规划图，把时间、节点与相对倍率融合到同一证据框架。"""

    rows = sorted(
        event,
        key=lambda row: (base.SCENE_ORDER.index(str(row["场景"])), float(row["事件数_K"])),
    )
    row_labels = [f"{base.SCENE_LABELS[str(row['场景'])]} K={int(float(row['事件数_K']))}" for row in rows]
    mp_time = np.asarray([float(row["MP累计时间_ms"]) for row in rows], dtype=float)
    ma_time = np.asarray([float(row["MA累计时间_ms"]) for row in rows], dtype=float)
    time_ratio = np.asarray([float(row["MA与MP时间比"]) for row in rows], dtype=float)
    mp_nodes = np.asarray([float(row["MP扩展节点"]) for row in rows], dtype=float)
    ma_nodes = np.asarray([float(row["MA扩展节点"]) for row in rows], dtype=float)
    node_ratio = ma_nodes / np.maximum(mp_nodes, 1e-9)

    labels = ["MP time", "MA time", "Time ratio", "MP nodes", "MA nodes", "Node ratio"]
    raw = np.column_stack([mp_time, ma_time, time_ratio, mp_nodes, ma_nodes, node_ratio])
    normalized = normalize_columns(np.column_stack([np.log10(mp_time), np.log10(ma_time), time_ratio, np.log10(mp_nodes), np.log10(ma_nodes), node_ratio]))

    fig = plt.figure(figsize=(8.7, 4.75), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.48, 1.06])
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_regime = fig.add_subplot(grid[0, 1])

    image = ax_heat.imshow(normalized, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax_heat.set_xticks(np.arange(len(labels)))
    ax_heat.set_xticklabels(labels, rotation=35, ha="right")
    ax_heat.set_yticks(np.arange(len(row_labels)))
    ax_heat.set_yticklabels(row_labels)
    for row in range(raw.shape[0]):
        for col, label in enumerate(labels):
            color = "white" if normalized[row, col] > 0.72 else "#1F1F1F"
            ax_heat.text(col, row, format_event_value(label, raw[row, col]), ha="center", va="center", fontsize=5.5, color=color)
    ax_heat.tick_params(length=0)
    base.add_panel_heading(ax_heat, "a", "Event-normalized multi-metric profile", x=0.0, y=1.05, title_dx=0.075)
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.04, pad=0.015)
    colorbar.set_label("Column-normalized burden")

    scene_positions = {scene: index for index, scene in enumerate(base.SCENE_ORDER)}
    x = np.asarray([float(row["事件数_K"]) for row in rows], dtype=float)
    y = np.asarray([scene_positions[str(row["场景"])] for row in rows], dtype=float)
    sizes = 35 + 30 * np.clip(node_ratio, 1, 11)
    color_values = np.log2(time_ratio)
    norm = TwoSlopeNorm(vmin=-1.7, vcenter=0.0, vmax=1.7)
    scatter = ax_regime.scatter(
        x,
        y,
        s=sizes,
        c=color_values,
        cmap="RdBu_r",
        norm=norm,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    for x_value, y_value, ratio in zip(x, y, time_ratio):
        ax_regime.text(x_value, y_value, f"{ratio:.2f}x", ha="center", va="center", fontsize=5.6, color="#1F1F1F", zorder=4)
    ax_regime.set_xticks([1, 5, 10])
    ax_regime.set_xlabel("Event count K")
    ax_regime.set_yticks(list(scene_positions.values()))
    ax_regime.set_yticklabels([base.SCENE_LABELS[scene] for scene in base.SCENE_ORDER])
    ax_regime.set_ylim(-0.55, len(base.SCENE_ORDER) - 0.45)
    ax_regime.set_xlim(0.25, 11.35)
    ax_regime.grid(True, axis="both", color="#D6D6D6", linewidth=0.45, alpha=0.8)
    base.add_panel_heading(ax_regime, "b", "Time-ratio and node-ratio regime")
    for size_value, label in [(2.0, "2x nodes"), (6.0, "6x nodes"), (10.0, "10x nodes")]:
        ax_regime.scatter([], [], s=35 + 30 * size_value, facecolor="#B0B0B0", edgecolor="white", label=label)
    base.add_framed_legend(
        ax_regime,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.12),
        title="MA/MP nodes",
        ncol=3,
        scatterpoints=1,
        labelspacing=0.45,
        handletextpad=0.55,
    )
    cbar = fig.colorbar(scatter, ax=ax_regime, fraction=0.052, pad=0.03)
    cbar.set_label("log2 MA/MP time")

    return base.save_pub_figure(fig, fig_dir, "fig_4_4_event_replanning_optimized_python", dpi)


def main() -> None:
    """执行优化图生成流程。"""

    base.configure_matplotlib()
    graph_scale = read_records(SOURCE_DIR / "figure_4_5_graph_scale_source.csv")
    event = read_records(SOURCE_DIR / "figure_4_4_event_replanning_source.csv")

    produced: list[Path] = []
    produced += plot_graph_scale_optimized(graph_scale, FIG_DIR)
    produced += plot_event_replanning_optimized(event, FIG_DIR)

    print("优化版实验结果图生成完成")
    for path in produced:
        print(path)


if __name__ == "__main__":
    main()
