"""重绘 aei_v7 稿件中需要修正的实验图，并生成替换图片后的 DOCX 副本。"""

from __future__ import annotations

import argparse
import math
import zipfile
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import create_chapter4_nature_figures as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "final_results" / "paper_revision" / "figures" / "chapter4_python"
SOURCE_DIR = PROJECT_ROOT / "final_results" / "paper_revision" / "source_data" / "chapter4_python"
DEFAULT_DOCX = Path(r"C:\Users\42095\Desktop\小论文资料\aei稿件\aei_v7.docx")
DEFAULT_OUT_DOCX = DEFAULT_DOCX.with_name("aei_v7_figures_fixed.docx")

SCENE_FOLDER = {
    "华山": "huashan",
    "黄山": "huangshan",
    "峨眉山": "emeishan",
}
METHODS = ["MP", "MA", "MF", "MR", "MV"]
ABLATION_METHODS = ["MA", "MF", "MR", "MV"]
QUALITY_METRICS = [
    ("mean_replan_ms", "Replanning time", "Replan", False),
    ("mean_path_cost", "Path cost", "Cost", False),
    ("mean_length_km", "Path length", "Length", False),
    ("mean_comm_coverage_ratio", "Communication coverage", "Coverage", True),
    ("mean_risk_exposure_integral", "Risk exposure", "Risk", False),
]


def method_code(raw: object) -> str:
    """把原始消融表中的 M-P 形式统一为 MP。"""

    return str(raw).strip().replace("M-", "M")


def save_precise_csv(df: pd.DataFrame, path: Path) -> None:
    """保存源数据，数值不预先四舍五入。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.15g")


def format_no_round(value: float, digits: int = 6) -> str:
    """生成不四舍五入的定长小数标注，用于图 9 的热图单元格。"""

    if not math.isfinite(value):
        return ""
    sign = "+" if value >= 0 else "-"
    quant = Decimal("1").scaleb(-digits)
    dec = Decimal(str(abs(float(value)))).quantize(quant, rounding=ROUND_DOWN)
    return f"{sign}{dec:.{digits}f}"


def range_normalize(values: np.ndarray) -> np.ndarray:
    """把一组数值压缩到 0 到 1，便于不同量纲同图比较。"""

    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.full_like(array, np.nan)
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    if abs(high - low) < 1e-12:
        return np.full_like(array, 0.5)
    return (array - low) / (high - low)


def read_raw_structural() -> pd.DataFrame:
    """读取三座山的结构消融原始汇总表。"""

    rows: list[pd.DataFrame] = []
    for scene in base.SCENE_ORDER:
        folder = SCENE_FOLDER[scene]
        path = PROJECT_ROOT / "final_results" / folder / "E1_E2_single_final" / "benchmark_structural_ablation.csv"
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.insert(0, "场景", scene)
        df["方法"] = df["code"].map(method_code)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def rebuild_structural_source(raw: pd.DataFrame) -> pd.DataFrame:
    """用原始结构消融汇总重新生成图 11 的源表。"""

    cols = [
        "场景",
        "方法",
        "success_rate",
        "mean_replan_ms",
        "mean_expanded",
        "mean_path_cost",
        "mean_risk_exposure_integral",
        "mean_comm_coverage_ratio",
    ]
    structural = raw.loc[raw["方法"].isin(METHODS), cols].copy()
    structural["方法"] = pd.Categorical(structural["方法"], categories=METHODS, ordered=True)
    structural["场景"] = pd.Categorical(structural["场景"], categories=base.SCENE_ORDER, ordered=True)
    structural = structural.sort_values(["场景", "方法"]).reset_index(drop=True)
    save_precise_csv(structural, SOURCE_DIR / "figure_4_7_structural_ablation_source.csv")
    return structural


def rebuild_quality_source(raw: pd.DataFrame) -> pd.DataFrame:
    """从原始结构消融表直接计算相对 MP 的变化，避免使用已低精度化的中间表。"""

    rows: list[dict[str, object]] = []
    for scene in base.SCENE_ORDER:
        scene_df = raw.loc[raw["场景"] == scene].set_index("方法")
        mp = scene_df.loc["MP"]
        for method in ABLATION_METHODS:
            item = scene_df.loc[method]
            for column, metric, _short, higher_better in QUALITY_METRICS:
                base_value = float(mp[column])
                value = float(item[column])
                if not math.isfinite(base_value) or abs(base_value) < 1e-12:
                    continue
                relative = (value - base_value) / abs(base_value) * 100.0
                if not higher_better:
                    relative = -relative
                rows.append(
                    {
                        "场景": scene,
                        "方法": method,
                        "指标": metric,
                        "相对MP变化_越高越好": relative,
                    }
                )
    rel = pd.DataFrame(rows)
    save_precise_csv(rel, SOURCE_DIR / "figure_4_6_ablation_quality_relative_source.csv")
    return rel


def add_clean_legend(ax: plt.Axes, handles: list[object], **kwargs):
    """使用统一边框样式放置图例，避免图例与数据主体混在一起。"""

    legend = ax.legend(handles=handles, frameon=True, **kwargs)
    if legend is not None:
        frame = legend.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("#303030")
        frame.set_linewidth(0.65)
        frame.set_alpha(1.0)
    return legend


def plot_event_replanning(fig_dir: Path, dpi: int) -> list[Path]:
    """重绘图 7，重点修正图例组织和占位。"""

    event = pd.read_csv(SOURCE_DIR / "figure_4_4_event_replanning_source.csv", encoding="utf-8-sig")
    event["场景"] = pd.Categorical(event["场景"], categories=base.SCENE_ORDER, ordered=True)
    event = event.sort_values(["场景", "事件数_K"])

    scene_positions = {scene: index for index, scene in enumerate(base.SCENE_ORDER)}
    x = event["事件数_K"].to_numpy(dtype=float)
    y = event["场景"].map(scene_positions).to_numpy(dtype=float)
    time_ratio = event["MA与MP时间比"].to_numpy(dtype=float)
    node_ratio = event["MA扩展节点"].to_numpy(dtype=float) / np.maximum(event["MP扩展节点"].to_numpy(dtype=float), 1e-9)
    color_values = np.log2(time_ratio)

    fig = plt.figure(figsize=(9.0, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.17, 1.0], width_ratios=[1.38, 1.0])
    ax_node_legend = fig.add_subplot(grid[0, 0])
    ax_metric_legend = fig.add_subplot(grid[0, 1])
    ax_node_legend.axis("off")
    ax_metric_legend.axis("off")
    ax_matrix = fig.add_subplot(grid[1, 0])
    ax_summary = fig.add_subplot(grid[1, 1])

    norm = TwoSlopeNorm(vmin=-1.7, vcenter=0.0, vmax=1.7)
    sizes = 54 + 32 * np.clip(node_ratio, 1, 11)
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
        ax_matrix.text(
            x_value,
            y_value,
            f"{ratio:.2f}x",
            ha="center",
            va="center",
            fontsize=5.8,
            color="#1F1F1F",
            zorder=4,
        )
    ax_matrix.set_xticks([1, 5, 10])
    ax_matrix.set_xlabel("Event count K")
    ax_matrix.set_yticks(list(scene_positions.values()))
    ax_matrix.set_yticklabels([base.SCENE_LABELS[scene] for scene in base.SCENE_ORDER])
    ax_matrix.set_ylim(-0.55, len(base.SCENE_ORDER) - 0.45)
    ax_matrix.set_xlim(0.35, 10.85)
    ax_matrix.grid(True, axis="both", color="#D6D6D6", linewidth=0.45, alpha=0.85)
    base.add_panel_heading(ax_matrix, "a", "Event burden matrix")
    base.style_axis(ax_matrix, grid_axis="both")
    colorbar = fig.colorbar(scatter, ax=ax_matrix, fraction=0.050, pad=0.025)
    colorbar.set_label("log2 MA/MP time")

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
    add_clean_legend(
        ax_node_legend,
        size_handles,
        loc="center",
        ncol=3,
        title="Node ratio",
        borderaxespad=0.0,
        columnspacing=1.0,
        handletextpad=0.55,
    )

    mean_time = []
    mean_nodes = []
    final_time = []
    final_nodes = []
    scene_labels = [base.SCENE_LABELS[scene] for scene in base.SCENE_ORDER]
    for scene in base.SCENE_ORDER:
        subset = event.loc[event["场景"] == scene]
        time_values = subset["MA与MP时间比"].to_numpy(dtype=float)
        node_values = subset["MA扩展节点"].to_numpy(dtype=float) / np.maximum(subset["MP扩展节点"].to_numpy(dtype=float), 1e-9)
        mean_time.append(float(np.nanmean(np.log2(time_values))))
        mean_nodes.append(float(np.nanmean(np.log2(node_values))))
        final_time.append(float(np.log2(time_values[-1])))
        final_nodes.append(float(np.log2(node_values[-1])))

    y_pos = np.arange(len(scene_labels), dtype=float)
    height = 0.34
    ax_summary.axvline(0, color="#555555", linewidth=0.8)
    ax_summary.barh(y_pos - height / 2, mean_time, height=height, color="#D08C45", edgecolor="white")
    ax_summary.barh(y_pos + height / 2, mean_nodes, height=height, color="#6EA5D8", edgecolor="white")
    ax_summary.scatter(final_time, y_pos - height / 2, marker="D", color="#8A4E2A", s=28, zorder=4)
    ax_summary.scatter(final_nodes, y_pos + height / 2, marker="D", color="#2E5F87", s=28, zorder=4)
    ax_summary.set_yticks(y_pos)
    ax_summary.set_yticklabels(scene_labels)
    ax_summary.set_xlabel("log2 ratio, MA/MP")
    ax_summary.invert_yaxis()
    base.add_panel_heading(ax_summary, "b", "Scene-level ratio summary")
    base.style_axis(ax_summary, grid_axis="x")
    summary_handles = [
        Patch(facecolor="#D08C45", edgecolor="white", label="Mean time"),
        Patch(facecolor="#6EA5D8", edgecolor="white", label="Mean nodes"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#4B4B4B", markeredgecolor="#4B4B4B", label="K=10"),
    ]
    add_clean_legend(
        ax_metric_legend,
        summary_handles,
        loc="center",
        ncol=3,
        title="Metric",
        borderaxespad=0.0,
        columnspacing=0.75,
        handlelength=1.2,
    )

    return base.save_pub_figure(fig, fig_dir, "fig_4_4_event_replanning_redesigned_python", dpi)


def plot_graph_scale(fig_dir: Path, dpi: int) -> list[Path]:
    """重绘图 8，确保 a 图不出现失败次数文本。"""

    scale = pd.read_csv(SOURCE_DIR / "figure_4_5_graph_scale_source.csv", encoding="utf-8-sig")
    labels = [str(item).capitalize() for item in scale["图规模"]]
    x = np.arange(len(scale), dtype=float)
    nodes = scale["节点数"].to_numpy(dtype=float)
    edges = scale["边数"].to_numpy(dtype=float)
    mp_time = scale["MP累计时间_ms"].to_numpy(dtype=float)
    ma_time = scale["MA累计时间_ms"].to_numpy(dtype=float)
    time_ratio = scale["MA与MP时间比"].to_numpy(dtype=float)
    mp_success = scale["MP成功率"].to_numpy(dtype=float)
    ma_success = scale["MA成功率"].to_numpy(dtype=float)
    mp_eff = mp_time / np.maximum(nodes / 1000.0, 1e-9)
    ma_eff = ma_time / np.maximum(nodes / 1000.0, 1e-9)

    fig = plt.figure(figsize=(8.8, 4.25), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.0])
    ax_trend = fig.add_subplot(grid[0, 0])
    ax_regime = fig.add_subplot(grid[0, 1])

    metrics = [
        ("Nodes", nodes, "#5F9EC9", "o", "-"),
        ("Edges", edges, "#F0A17A", "s", "-"),
        ("MA/MP time", time_ratio, "#3F3F3F", "D", "-"),
        ("MP success", mp_success, "#4B9B78", "^", "-"),
        ("MA success", ma_success, "#7E6AAE", "v", "-"),
        ("MP time per node", mp_eff, "#5E79B9", "o", "--"),
        ("MA time per node", ma_eff, "#D08C45", "s", "--"),
    ]
    for name, values, color, marker, line_style in metrics:
        ax_trend.plot(
            x,
            range_normalize(values),
            marker=marker,
            linestyle=line_style,
            color=color,
            linewidth=1.28,
            markersize=3.9,
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
        bbox_to_anchor=(0.0, 1.10),
        ncol=4,
        title="Metric",
        columnspacing=0.75,
        handlelength=1.45,
        borderaxespad=0.1,
    )

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
    ax_regime.axvline(1.0, color="#555555", linestyle="--", linewidth=0.8)
    for scale_name, x_value, y_value in zip(scale["图规模"], time_ratio, mp_eff):
        ax_regime.annotate(
            str(scale_name),
            (x_value, y_value),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=6.4,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.4},
        )
    x_margin = max(0.08, (float(np.nanmax(time_ratio)) - float(np.nanmin(time_ratio))) * 0.16)
    y_margin = max(4.0, (float(np.nanmax(mp_eff)) - float(np.nanmin(mp_eff))) * 0.18)
    ax_regime.set_xlim(float(np.nanmin(time_ratio)) - x_margin, float(np.nanmax(time_ratio)) + x_margin)
    ax_regime.set_ylim(float(np.nanmin(mp_eff)) - y_margin, float(np.nanmax(mp_eff)) + y_margin)
    ax_regime.set_xlabel("MA/MP time ratio")
    ax_regime.set_ylabel("MP time per 1k nodes")
    base.add_panel_heading(ax_regime, "b", "Scale regime map")
    base.style_axis(ax_regime)
    colorbar = fig.colorbar(scatter, ax=ax_regime, fraction=0.052, pad=0.03)
    colorbar.set_label("MP success rate")

    return base.save_pub_figure(fig, fig_dir, "fig_4_5_graph_scale_redesigned_python", dpi)


def plot_ablation_quality(rel: pd.DataFrame, fig_dir: Path, dpi: int) -> list[Path]:
    """重绘图 9，热图和源表均使用全精度结构消融结果。"""

    metrics = [item[1] for item in QUALITY_METRICS]
    metric_labels = [item[2] for item in QUALITY_METRICS]
    matrix = np.full((len(ABLATION_METHODS), len(metrics)), np.nan, dtype=float)
    for i, method in enumerate(ABLATION_METHODS):
        for j, metric in enumerate(metrics):
            values = rel.loc[(rel["方法"] == method) & (rel["指标"] == metric), "相对MP变化_越高越好"]
            matrix[i, j] = float(values.mean())
    method_effect = np.nanmean(matrix, axis=1)

    fig = plt.figure(figsize=(9.4, 3.25), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=[1.25, 1.25, 1.25, 1.05])
    ax_heat = fig.add_subplot(grid[0, 0:3])
    ax_method = fig.add_subplot(grid[0, 3])

    finite = matrix[np.isfinite(matrix)]
    vmax = max(80.0, float(np.nanpercentile(np.abs(finite), 90)) if finite.size else 80.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax_heat.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
    ax_heat.set_xticks(np.arange(len(metrics)))
    ax_heat.set_xticklabels(metric_labels)
    ax_heat.set_yticks(np.arange(len(ABLATION_METHODS)))
    ax_heat.set_yticklabels(ABLATION_METHODS)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if math.isfinite(value):
                color = "white" if abs(value) > vmax * 0.52 else "#222222"
                ax_heat.text(j, i, format_no_round(value, 6), ha="center", va="center", fontsize=5.15, color=color)
    colorbar = fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.025)
    colorbar.set_label("Change relative to MP, %, higher is better")
    ax_heat.set_ylabel("Ablation variant")
    base.add_panel_heading(ax_heat, "a", "Metric-level ablation effect", x=0.0, y=1.045)
    ax_heat.tick_params(length=0)

    y = np.arange(len(ABLATION_METHODS), dtype=float)
    method_colors = [base.METHOD_COLORS[name] for name in ABLATION_METHODS]
    ax_method.axvline(0, color="#555555", linewidth=0.75)
    ax_method.barh(y, method_effect, color=method_colors, edgecolor="white", linewidth=0.35)
    ax_method.set_yticks(y)
    ax_method.set_yticklabels(ABLATION_METHODS)
    ax_method.invert_yaxis()
    ax_method.set_xlabel("Mean change, %")
    finite_method = method_effect[np.isfinite(method_effect)]
    if finite_method.size:
        margin = max(6.0, float(np.nanmax(np.abs(finite_method))) * 0.12)
        ax_method.set_xlim(float(np.nanmin(finite_method)) - margin, float(np.nanmax(finite_method)) + margin)
    for yi, value in zip(y, method_effect):
        if math.isfinite(value):
            if value >= 0:
                ax_method.text(value + 1.0, yi, format_no_round(value, 4), va="center", ha="left", fontsize=5.8)
            else:
                ax_method.text(-2.0, yi, format_no_round(value, 4), va="center", ha="right", fontsize=5.8)
    base.add_panel_heading(ax_method, "b", "Variant summary")
    base.style_axis(ax_method)

    return base.save_pub_figure(fig, fig_dir, "fig_4_6_ablation_quality_python", dpi)


def plot_ablation_workload(structural: pd.DataFrame, fig_dir: Path, dpi: int) -> list[Path]:
    """重绘图 11，给 MR 点和标签留出坐标轴空间。"""

    summary = (
        structural.groupby("方法", observed=False)[["mean_expanded", "mean_replan_ms"]]
        .mean()
        .reindex(METHODS)
        .reset_index()
    )
    methods = summary["方法"].astype(str).tolist()
    x = np.arange(len(methods), dtype=float)
    colors = [base.METHOD_COLORS[method] for method in methods]
    expanded = summary["mean_expanded"].to_numpy(dtype=float)
    replan_time = summary["mean_replan_ms"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(8.4, 4.95), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], width_ratios=[1, 1, 1.05])
    ax_nodes = fig.add_subplot(grid[0, 0])
    ax_time = fig.add_subplot(grid[0, 1])
    ax_compress = fig.add_subplot(grid[0, 2])
    ax_map = fig.add_subplot(grid[1, :])

    ax_nodes.bar(x, expanded, color=colors, edgecolor="white", linewidth=0.35)
    ax_nodes.set_xticks(x)
    ax_nodes.set_xticklabels(methods)
    ax_nodes.set_yscale("log")
    ax_nodes.set_ylabel("Mean expanded nodes")
    base.add_panel_heading(ax_nodes, "a", "Search workload")
    base.style_axis(ax_nodes)

    ax_time.bar(x, replan_time, color=colors, edgecolor="white", linewidth=0.35)
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(methods)
    ax_time.set_yscale("log")
    ax_time.set_ylabel("Mean replanning time, ms")
    base.add_panel_heading(ax_time, "b", "Replanning time")
    base.style_axis(ax_time)

    mv_index = methods.index("MV")
    mv_expanded = expanded[mv_index]
    mv_time = replan_time[mv_index]
    expanded_gain = np.asarray([mv_expanded / value if value > 0 else np.nan for value in expanded], dtype=float)
    time_gain = np.asarray([mv_time / value if value > 0 else np.nan for value in replan_time], dtype=float)
    width = 0.34
    ax_compress.axhline(1.0, color="#555555", linestyle="--", linewidth=0.75)
    ax_compress.bar(x - width / 2, expanded_gain, width=width, color="#8FBBD9", edgecolor="white", linewidth=0.35, label="Nodes")
    ax_compress.bar(x + width / 2, time_gain, width=width, color="#E5A46E", edgecolor="white", linewidth=0.35, label="Time")
    ax_compress.set_xticks(x)
    ax_compress.set_xticklabels(methods)
    ax_compress.set_ylabel("Gain vs MV, fold")
    base.add_panel_heading(ax_compress, "c", "Compression gain")
    base.add_framed_legend(ax_compress, loc="upper right", title="Baseline")
    base.style_axis(ax_compress)

    finite_expanded = expanded[np.isfinite(expanded) & (expanded > 0)]
    finite_time = replan_time[np.isfinite(replan_time) & (replan_time > 0)]
    sizes = 72 + 130 * (expanded_gain / np.nanmax(expanded_gain)) if np.isfinite(expanded_gain).any() else np.full_like(x, 90.0)
    label_offsets = {
        "MP": (8, 8),
        "MA": (8, -10),
        "MF": (-18, 9),
        "MR": (14, 16),
        "MV": (-22, -12),
    }
    for xi, yi, size, method, color in zip(expanded, replan_time, sizes, methods, colors):
        ax_map.scatter(xi, yi, s=size, color=color, edgecolor="white", linewidth=0.55, zorder=3)
        ax_map.annotate(
            method,
            (xi, yi),
            textcoords="offset points",
            xytext=label_offsets.get(method, (6, 5)),
            fontsize=6.6,
            color="#222222",
            clip_on=False,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.35},
        )
    if finite_expanded.size and finite_time.size:
        fit = np.polyfit(np.log10(finite_expanded), np.log10(finite_time), 1)
        xs = np.geomspace(float(finite_expanded.min()), float(finite_expanded.max()), 80)
        ys = 10 ** (fit[1] + fit[0] * np.log10(xs))
        ax_map.plot(xs, ys, color="#555555", linewidth=0.9, linestyle="--", label="Log-log trend")
        base.add_framed_legend(ax_map, loc="upper left")
        ax_map.set_xlim(float(finite_expanded.min()) * 0.48, float(finite_expanded.max()) * 1.35)
        ax_map.set_ylim(float(finite_time.min()) * 0.42, float(finite_time.max()) * 1.45)
    ax_map.set_xscale("log")
    ax_map.set_yscale("log")
    ax_map.set_xlabel("Mean expanded nodes")
    ax_map.set_ylabel("Mean replanning time, ms")
    base.add_panel_heading(ax_map, "d", "Workload-efficiency map")
    base.style_axis(ax_map)

    mp_expanded = expanded[methods.index("MP")]
    if math.isfinite(mp_expanded) and mp_expanded > 0 and math.isfinite(mv_expanded):
        ax_nodes.text(
            0.02,
            0.94,
            f"MV/MP = {mv_expanded / mp_expanded:.1f}x",
            transform=ax_nodes.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
        )

    return base.save_pub_figure(fig, fig_dir, "fig_4_7_ablation_workload_python", dpi)


def replace_docx_images(input_docx: Path, output_docx: Path, fig_dir: Path) -> Path:
    """复制 DOCX 并替换四张目标图片。"""

    replacements = {
        "word/media/image176.png": fig_dir / "fig_4_4_event_replanning_redesigned_python.png",
        "word/media/image180.png": fig_dir / "fig_4_5_graph_scale_redesigned_python.png",
        "word/media/image181.png": fig_dir / "fig_4_6_ablation_quality_python.png",
        "word/media/image182.png": fig_dir / "fig_4_7_ablation_workload_python.png",
    }
    missing = [str(path) for path in replacements.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少待替换图片：" + "；".join(missing))

    output_docx.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(input_docx, "r") as zin, zipfile.ZipFile(output_docx, "w") as zout:
        names = set(zin.namelist())
        for item in zin.infolist():
            data = replacements[item.filename].read_bytes() if item.filename in replacements else zin.read(item.filename)
            zout.writestr(item, data)
    absent = [name for name in replacements if name not in names]
    if absent:
        raise FileNotFoundError("DOCX 中未找到目标媒体文件：" + "；".join(absent))
    return output_docx


def assert_no_failure_text(svg_path: Path) -> None:
    """检查图 8 矢量文本中没有失败次数说明。"""

    text = svg_path.read_text(encoding="utf-8", errors="ignore").lower()
    forbidden = ["failure", "failures", "failed", "失败", "5 failures", "5次失败"]
    found = [item for item in forbidden if item.lower() in text]
    if found:
        raise AssertionError(f"图 8 仍包含失败说明文本：{found}")


def main() -> None:
    parser = argparse.ArgumentParser(description="重绘 aei_v7 中的图 7、图 8、图 9 和图 11")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--docx", type=Path, default=DEFAULT_DOCX)
    parser.add_argument("--out-docx", type=Path, default=DEFAULT_OUT_DOCX)
    parser.add_argument("--skip-docx", action="store_true", help="只重绘图片，不生成 DOCX 副本")
    args = parser.parse_args()

    base.configure_matplotlib()
    raw = read_raw_structural()
    structural = rebuild_structural_source(raw)
    rel = rebuild_quality_source(raw)

    produced: list[Path] = []
    produced += plot_event_replanning(FIG_DIR, args.dpi)
    produced += plot_graph_scale(FIG_DIR, args.dpi)
    produced += plot_ablation_quality(rel, FIG_DIR, args.dpi)
    produced += plot_ablation_workload(structural, FIG_DIR, args.dpi)
    assert_no_failure_text(FIG_DIR / "fig_4_5_graph_scale_redesigned_python.svg")

    print("重绘完成：")
    for path in produced:
        print(path)
    print("已更新源数据：")
    print(SOURCE_DIR / "figure_4_6_ablation_quality_relative_source.csv")
    print(SOURCE_DIR / "figure_4_7_structural_ablation_source.csv")

    if not args.skip_docx:
        out_docx = replace_docx_images(args.docx, args.out_docx, FIG_DIR)
        print("已生成替换图片后的 DOCX 副本：")
        print(out_docx)


if __name__ == "__main__":
    main()
