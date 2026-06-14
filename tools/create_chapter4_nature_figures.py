"""为第四章实验结果生成论文级 Python 图表。

脚本直接从 Word 初稿的 XML 表格中读取第四章数据，并补充项目正式实验输出中的
结构消融展开量。绘图仅依赖 numpy 与 matplotlib，避免在项目 env 环境中额外要求
pandas、seaborn 或 python-docx。
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import sys
import warnings
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore", message=".*MERG NOT subset.*")
logging.getLogger("fontTools.subset").setLevel(logging.ERROR)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCX = Path(r"C:\Users\42095\Desktop\小论文资料\稿件\初稿4.docx")
DEFAULT_FIG_DIR = PROJECT_ROOT / "final_results" / "paper_revision" / "figures" / "chapter4_python"
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "final_results" / "paper_revision" / "source_data" / "chapter4_python"

WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
SCENE_ORDER = ["华山", "黄山", "峨眉山"]
SCENE_LABELS = {
    "华山": "Huashan",
    "黄山": "Huangshan",
    "峨眉山": "Emeishan",
}
METHOD_ORDER = ["MP", "MA", "MF", "MR", "MV"]
METHOD_LABELS = {
    "MP": "MP, full method",
    "MA": "MA, global replanning",
    "MF": "MF, flat graph",
    "MR": "MR, regular layers",
    "MV": "MV, voxel baseline",
}
METHOD_COLORS = {
    "MP": "#3B6EA8",
    "MA": "#D08C45",
    "MF": "#7E6AAE",
    "MR": "#4B9B78",
    "MV": "#6B6B6B",
}
LEGEND_BOX = {
    "facecolor": "white",
    "edgecolor": "#303030",
    "linewidth": 0.65,
}
def configure_matplotlib() -> None:
    """设置适合中文论文图表的基础样式。"""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Microsoft YaHei", "SimHei"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.2,
            "axes.labelsize": 7.4,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def clean_cell(text: object) -> str:
    """清理 Word 表格单元格中的多余空白。"""

    return re.sub(r"\s+", " ", str(text or "").strip())


def normalize_header(text: object) -> str:
    """归一化表头，消除空格和单位写法差异。"""

    return re.sub(r"\s+", "", str(text or "").strip())


def to_float(value: object) -> float:
    """把表格文本转换为浮点数，无法转换时返回 NaN。"""

    text = str(value or "").strip().replace(",", "").replace("%", "")
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def finite_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def extract_docx_table_rows(docx_path: Path) -> list[list[list[str]]]:
    """使用标准库读取 docx 中的所有表格。"""

    with zipfile.ZipFile(docx_path) as archive:
        document_xml = archive.read("word/document.xml")
    root = ET.fromstring(document_xml)
    tables: list[list[list[str]]] = []
    for table in root.findall(".//w:tbl", WORD_NS):
        table_rows: list[list[str]] = []
        for row in table.findall("./w:tr", WORD_NS):
            cells = []
            for cell in row.findall("./w:tc", WORD_NS):
                text = "".join(node.text or "" for node in cell.findall(".//w:t", WORD_NS))
                cells.append(clean_cell(text))
            if any(cells):
                table_rows.append(cells)
        if table_rows:
            tables.append(table_rows)
    return tables


def rows_to_records(rows: list[list[str]]) -> list[dict[str, str]]:
    """把二维表转换为字典记录。"""

    if not rows:
        return []
    header = rows[0]
    records = []
    for row in rows[1:]:
        padded = row + [""] * max(0, len(header) - len(row))
        records.append({header[i]: padded[i] for i in range(len(header))})
    return records


def extract_docx_tables(docx_path: Path) -> dict[str, list[dict[str, str]]]:
    """从初稿中定位第四章三张可绘图数值表。"""

    extracted: dict[str, list[dict[str, str]]] = {}
    for rows in extract_docx_table_rows(docx_path):
        header = [normalize_header(col) for col in rows[0]]
        header_set = set(header)
        if {"场景", "方法", "成功率", "路径代价"}.issubset(header_set):
            extracted["method"] = rows_to_records(rows)
        elif {"场景", "事件数K", "MP累计时间ms", "MA累计时间ms"}.issubset(header_set):
            extracted["event"] = rows_to_records(rows)
        elif {"图规模", "节点数", "边数", "MP累计时间ms", "MA累计时间ms"}.issubset(header_set):
            extracted["scale"] = rows_to_records(rows)
    missing = [name for name in ["method", "event", "scale"] if name not in extracted]
    if missing:
        missing_text = "、".join(missing)
        raise ValueError(f"未能从稿件中识别以下第四章数据表：{missing_text}")
    return extracted


def rename_record(record: dict[str, object]) -> dict[str, object]:
    """把 Word 表头转换为稳定字段名。"""

    field_map = {
        "重规划时间ms": "重规划时间_ms",
        "长度km": "长度_km",
        "事件数K": "事件数_K",
        "MP累计时间ms": "MP累计时间_ms",
        "MA累计时间ms": "MA累计时间_ms",
        "MA与MP时间比": "MA与MP时间比",
        "MP扩展节点": "MP扩展节点",
        "MA扩展节点": "MA扩展节点",
        "MP成功率": "MP成功率",
        "MA成功率": "MA成功率",
    }
    out: dict[str, object] = {}
    for key, value in record.items():
        normalized = normalize_header(key)
        out[field_map.get(normalized, normalized)] = value
    return out


def convert_numeric(records: list[dict[str, object]], fields: Iterable[str]) -> None:
    """就地转换指定字段为浮点数。"""

    for record in records:
        for field in fields:
            if field in record:
                record[field] = to_float(record[field])


def order_index(order: list[str], value: object) -> int:
    text = str(value)
    return order.index(text) if text in order else len(order)


def save_records_csv(path: Path, records: list[dict[str, object]], fields: list[str]) -> None:
    """保存源数据，便于论文图表审查。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def load_records_csv(path: Path) -> list[dict[str, object]]:
    """读取已经导出的源数据 CSV。"""

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_existing_source_tables(source_dir: Path) -> dict[str, list[dict[str, object]]]:
    """当 Word 初稿仅保留图位时，复用已导出的图表源数据。"""

    paths = {
        "method": source_dir / "figure_4_3_method_comparison_source.csv",
        "event": source_dir / "figure_4_4_event_replanning_source.csv",
        "scale": source_dir / "figure_4_5_graph_scale_source.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        missing_text = "、".join(missing)
        raise FileNotFoundError(f"稿件中未找到数值表，且缺少回退源数据：{missing_text}")

    method = load_records_csv(paths["method"])
    convert_numeric(method, ["成功率", "重规划时间_ms", "路径代价", "长度_km", "通信覆盖率", "风险暴露"])
    event = load_records_csv(paths["event"])
    convert_numeric(event, ["事件数_K", "MP累计时间_ms", "MA累计时间_ms", "MA与MP时间比", "MP扩展节点", "MA扩展节点"])
    scale = load_records_csv(paths["scale"])
    convert_numeric(scale, ["节点数", "边数", "MP累计时间_ms", "MA累计时间_ms", "MA与MP时间比", "MP成功率", "MA成功率"])
    return {"method": method, "event": event, "scale": scale}


def prepare_source_tables(docx_path: Path, source_dir: Path) -> dict[str, list[dict[str, object]]]:
    """抽取并保存图表源数据。"""

    source_dir.mkdir(parents=True, exist_ok=True)
    try:
        tables = extract_docx_tables(docx_path)
    except ValueError:
        return load_existing_source_tables(source_dir)

    method = [rename_record(row) for row in tables["method"]]
    for row in method:
        row["方法"] = str(row.get("方法", "")).replace("-", "")
    convert_numeric(method, ["成功率", "重规划时间_ms", "路径代价", "长度_km", "通信覆盖率", "风险暴露"])
    method.sort(key=lambda r: (order_index(SCENE_ORDER, r.get("场景")), order_index(METHOD_ORDER, r.get("方法"))))

    event = [rename_record(row) for row in tables["event"]]
    convert_numeric(event, ["事件数_K", "MP累计时间_ms", "MA累计时间_ms", "MA与MP时间比", "MP扩展节点", "MA扩展节点"])
    event.sort(key=lambda r: (order_index(SCENE_ORDER, r.get("场景")), float(r.get("事件数_K", 0))))

    scale = [rename_record(row) for row in tables["scale"]]
    convert_numeric(scale, ["节点数", "边数", "MP累计时间_ms", "MA累计时间_ms", "MA与MP时间比", "MP成功率", "MA成功率"])
    scale_order = ["small", "medium", "large"]
    scale.sort(key=lambda r: order_index(scale_order, r.get("图规模")))

    save_records_csv(
        source_dir / "figure_4_3_method_comparison_source.csv",
        method,
        ["场景", "方法", "成功率", "重规划时间_ms", "路径代价", "长度_km", "通信覆盖率", "风险暴露"],
    )
    save_records_csv(
        source_dir / "figure_4_4_event_replanning_source.csv",
        event,
        ["场景", "事件数_K", "MP累计时间_ms", "MA累计时间_ms", "MA与MP时间比", "MP扩展节点", "MA扩展节点"],
    )
    save_records_csv(
        source_dir / "figure_4_5_graph_scale_source.csv",
        scale,
        ["图规模", "节点数", "边数", "MP累计时间_ms", "MA累计时间_ms", "MA与MP时间比", "MP成功率", "MA成功率"],
    )
    return {"method": method, "event": event, "scale": scale}


def load_structural_ablation(root: Path, source_dir: Path) -> list[dict[str, object]]:
    """读取结构消融 CSV，用于图 4.7 的搜索展开量分析。"""

    scene_map = {"huashan": "华山", "huangshan": "黄山", "emeishan": "峨眉山"}
    rows: list[dict[str, object]] = []
    for scene_key, scene_cn in scene_map.items():
        path = root / "final_results" / scene_key / "E1_E2_single_final" / "benchmark_structural_ablation.csv"
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            for item in csv.DictReader(handle):
                method = str(item.get("code", "")).replace("-", "")
                if method not in METHOD_ORDER:
                    continue
                rows.append(
                    {
                        "场景": scene_cn,
                        "方法": method,
                        "success_rate": to_float(item.get("success_rate")),
                        "mean_replan_ms": to_float(item.get("mean_replan_ms")),
                        "mean_expanded": to_float(item.get("mean_expanded")),
                        "mean_path_cost": to_float(item.get("mean_path_cost")),
                        "mean_risk_exposure_integral": to_float(item.get("mean_risk_exposure_integral")),
                        "mean_comm_coverage_ratio": to_float(item.get("mean_comm_coverage_ratio")),
                    }
                )
    if not rows:
        raise FileNotFoundError("未找到结构消融 benchmark_structural_ablation.csv，无法生成图 4.7。")
    rows.sort(key=lambda r: (order_index(SCENE_ORDER, r.get("场景")), order_index(METHOD_ORDER, r.get("方法"))))
    save_records_csv(
        source_dir / "figure_4_7_structural_ablation_source.csv",
        rows,
        [
            "场景",
            "方法",
            "success_rate",
            "mean_replan_ms",
            "mean_expanded",
            "mean_path_cost",
            "mean_risk_exposure_integral",
            "mean_comm_coverage_ratio",
        ],
    )
    return rows


def save_pub_figure(fig: plt.Figure, out_dir: Path, basename: str, dpi: int = 600) -> list[Path]:
    """按论文图表交付要求导出多格式文件。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        out_dir / f"{basename}.svg",
        out_dir / f"{basename}.pdf",
        out_dir / f"{basename}.tiff",
        out_dir / f"{basename}.png",
    ]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], dpi=dpi, bbox_inches="tight")
    fig.savefig(paths[3], dpi=220, bbox_inches="tight")
    plt.close(fig)
    return paths


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, color="#D6D6D6", linewidth=0.45, alpha=0.8)
    ax.tick_params(length=2.5, width=0.55)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)


def add_panel_label(
    ax: plt.Axes,
    text: str,
    x: float = 0.0,
    y: float = 1.035,
    ha: str = "left",
    va: str = "bottom",
    size: float = 7.6,
    weight: str = "bold",
) -> None:
    """添加无边框面板字母标签。"""

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=size,
        fontweight=weight,
        clip_on=False,
    )


def add_panel_heading(
    ax: plt.Axes,
    label: str,
    title: str,
    x: float = 0.0,
    y: float = 1.035,
    title_dx: float = 0.075,
) -> None:
    """添加无边框面板字母标签和无框英文标题。"""

    add_panel_label(ax, label, x=x, y=y)
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


def add_framed_legend(ax: plt.Axes, *args, **kwargs):
    """只给真正的图例添加边框。"""

    kwargs.setdefault("frameon", True)
    legend = ax.legend(*args, **kwargs)
    if legend is not None:
        frame = legend.get_frame()
        frame.set_facecolor(LEGEND_BOX["facecolor"])
        frame.set_edgecolor(LEGEND_BOX["edgecolor"])
        frame.set_linewidth(LEGEND_BOX["linewidth"])
        frame.set_alpha(1.0)
    return legend


def get_row(records: list[dict[str, object]], **conditions: object) -> dict[str, object] | None:
    for record in records:
        if all(record.get(key) == value for key, value in conditions.items()):
            return record
    return None


def as_values(records: list[dict[str, object]], field: str) -> list[float]:
    return [float(record.get(field, float("nan"))) for record in records]


def clipped_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def method_score_matrix(method: list[dict[str, object]]) -> tuple[np.ndarray, list[str]]:
    """计算跨场景归一化综合表现，数值越高表示表现越好。"""

    metric_specs = [
        ("Success", "成功率", True),
        ("Time", "重规划时间_ms", False),
        ("Cost", "路径代价", False),
        ("Length", "长度_km", False),
        ("Coverage", "通信覆盖率", True),
        ("Risk", "风险暴露", False),
    ]
    scores = np.full((len(METHOD_ORDER), len(metric_specs)), np.nan, dtype=float)
    for method_idx, method_name in enumerate(METHOD_ORDER):
        for metric_idx, (_label, field, higher_better) in enumerate(metric_specs):
            scene_scores = []
            for scene in SCENE_ORDER:
                scene_rows = [row for row in method if row.get("场景") == scene]
                values = np.asarray([float(row.get(field, float("nan"))) for row in scene_rows], dtype=float)
                finite = values[np.isfinite(values)]
                row = get_row(method, 场景=scene, 方法=method_name)
                if row is None or not finite.size:
                    continue
                value = float(row.get(field, float("nan")))
                if not math.isfinite(value):
                    continue
                lo = float(np.min(finite))
                hi = float(np.max(finite))
                if abs(hi - lo) < 1e-12:
                    score = 1.0
                else:
                    score = (value - lo) / (hi - lo)
                    if not higher_better:
                        score = 1.0 - score
                scene_scores.append(score)
            scores[method_idx, metric_idx] = clipped_mean(scene_scores)
    return scores, [item[0] for item in metric_specs]


def plot_grouped_metric(
    ax: plt.Axes,
    data: list[dict[str, object]],
    metric: str,
    label: str,
    title: str,
    ylabel: str,
) -> None:
    width = 0.145
    scene_positions = np.arange(len(SCENE_ORDER), dtype=float)
    for idx, method in enumerate(METHOD_ORDER):
        values = []
        for scene in SCENE_ORDER:
            row = get_row(data, 场景=scene, 方法=method)
            values.append(float(row.get(metric, float("nan"))) if row else float("nan"))
        offset = (idx - (len(METHOD_ORDER) - 1) / 2) * width
        ax.bar(
            scene_positions + offset,
            values,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.35,
            label=METHOD_LABELS[method],
        )
    add_panel_heading(ax, label, title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(scene_positions)
    ax.set_xticklabels([SCENE_LABELS[scene] for scene in SCENE_ORDER])
    style_axis(ax)


def plot_method_comparison(method: list[dict[str, object]], fig_dir: Path, dpi: int) -> list[Path]:
    """图 4.3：只用五指标柱状图展示方法对比。"""

    fig = plt.figure(figsize=(8.6, 5.45), constrained_layout=True)
    grid = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 0.20])
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    ]
    legend_ax = fig.add_subplot(grid[2, :])

    metric_specs = [
        ("重规划时间_ms", "a", "Replanning time", "ms"),
        ("路径代价", "b", "Path cost", "Composite cost"),
        ("长度_km", "c", "Path length", "km"),
        ("通信覆盖率", "d", "Communication coverage", "Ratio"),
        ("风险暴露", "e", "Risk exposure", "Integral value"),
    ]
    for ax, (metric, label, title, ylabel) in zip(axes, metric_specs):
        plot_grouped_metric(ax, method, metric, label, title, ylabel)
        ax.tick_params(axis="x", labelrotation=24)
    axes[0].set_ylim(0, max(as_values(method, "重规划时间_ms")) * 1.12)
    axes[3].set_ylim(0, min(1.02, max(as_values(method, "通信覆盖率")) * 1.12))

    handles, labels = axes[0].get_legend_handles_labels()
    add_framed_legend(legend_ax, handles, labels, loc="center", title="Method", ncol=3, columnspacing=1.4, handlelength=1.8)
    legend_ax.set_axis_off()
    return save_pub_figure(fig, fig_dir, "fig_4_3_method_comparison_python", dpi)


def plot_event_replanning(event: list[dict[str, object]], fig_dir: Path, dpi: int) -> list[Path]:
    """图 4.4：连续事件下增量重规划的时间与展开量。"""

    fig, axes = plt.subplots(3, 3, figsize=(7.4, 6.2), sharex=True, constrained_layout=True)
    for col, scene in enumerate(SCENE_ORDER):
        subset = [row for row in event if row.get("场景") == scene]
        subset.sort(key=lambda r: float(r.get("事件数_K", 0)))
        x = np.asarray(as_values(subset, "事件数_K"), dtype=float)
        ax_time = axes[0, col]
        ax_nodes = axes[1, col]
        ax_ratio = axes[2, col]
        ax_time.plot(x, as_values(subset, "MP累计时间_ms"), marker="o", color=METHOD_COLORS["MP"], label="MP", linewidth=1.45, markersize=3.2)
        ax_time.plot(x, as_values(subset, "MA累计时间_ms"), marker="s", color=METHOD_COLORS["MA"], label="MA", linewidth=1.45, markersize=3.2)
        ax_nodes.plot(x, as_values(subset, "MP扩展节点"), marker="o", color=METHOD_COLORS["MP"], label="MP", linewidth=1.45, markersize=3.2)
        ax_nodes.plot(x, as_values(subset, "MA扩展节点"), marker="s", color=METHOD_COLORS["MA"], label="MA", linewidth=1.45, markersize=3.2)
        ratio = np.asarray(as_values(subset, "MA与MP时间比"), dtype=float)
        ax_ratio.axhline(1.0, color="#555555", linestyle="--", linewidth=0.8)
        ax_ratio.fill_between(x, 1.0, ratio, where=ratio >= 1.0, color="#E6A45F", alpha=0.28, interpolate=True)
        ax_ratio.fill_between(x, ratio, 1.0, where=ratio < 1.0, color="#6EA5D8", alpha=0.22, interpolate=True)
        ax_ratio.plot(x, ratio, marker="D", color="#3F3F3F", linewidth=1.35, markersize=3.2, label="MA/MP")
        ax_time.set_title(SCENE_LABELS.get(scene, str(scene)), pad=8, fontweight="bold")
        ax_ratio.set_xlabel("Event count K")
        if col == 0:
            ax_time.set_ylabel("Cumulative time, ms")
            ax_nodes.set_ylabel("Cumulative expanded nodes")
            ax_ratio.set_ylabel("Time ratio")
        for ax in [ax_time, ax_nodes, ax_ratio]:
            ax.set_xticks(x)
            style_axis(ax)
    add_panel_heading(axes[0, 0], "a", "Time accumulation", x=-0.12, y=1.16, title_dx=0.085)
    add_panel_heading(axes[1, 0], "b", "Search expansion", x=-0.12, y=1.08, title_dx=0.085)
    add_panel_heading(axes[2, 0], "c", "Relative speed regime", x=-0.12, y=1.08, title_dx=0.085)
    add_framed_legend(axes[0, 2], loc="upper left", title="Method")
    return save_pub_figure(fig, fig_dir, "fig_4_4_event_replanning_python", dpi)


def plot_graph_scale(scale: list[dict[str, object]], fig_dir: Path, dpi: int) -> list[Path]:
    """图 4.5：图规模变化下的性能敏感性。"""

    labels = [str(row.get("图规模")) for row in scale]
    x = np.arange(len(labels), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)

    ax0 = axes[0, 0]
    ax0.bar(x - 0.16, as_values(scale, "节点数"), width=0.30, color="#74A9CF", label="Nodes")
    ax0.bar(x + 0.16, as_values(scale, "边数"), width=0.30, color="#F4A582", label="Edges")
    ax0b = ax0.twinx()
    density = np.asarray(as_values(scale, "边数"), dtype=float) / np.maximum(np.asarray(as_values(scale, "节点数"), dtype=float), 1.0)
    ax0b.plot(x, density, marker="D", color="#555555", linewidth=1.2, label="Edge/node")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylim(0, max(as_values(scale, "边数")) * 1.58)
    ax0.set_ylabel("Graph elements")
    ax0b.set_ylabel("Edge/node ratio")
    add_panel_heading(ax0, "a", "Graph scale")
    lines0, labels0 = ax0.get_legend_handles_labels()
    lines0b, labels0b = ax0b.get_legend_handles_labels()
    add_framed_legend(
        ax0,
        lines0 + lines0b,
        labels0 + labels0b,
        loc="upper center",
        bbox_to_anchor=(0.72, 0.98),
        ncol=1,
        labelspacing=0.32,
        handlelength=1.2,
        borderaxespad=0.15,
    )
    style_axis(ax0)
    ax0b.spines["top"].set_visible(False)
    ax0b.tick_params(length=2.5, width=0.55)

    ax1 = axes[0, 1]
    ax1.plot(x, as_values(scale, "MP累计时间_ms"), marker="o", color=METHOD_COLORS["MP"], label="MP")
    ax1.plot(x, as_values(scale, "MA累计时间_ms"), marker="s", color=METHOD_COLORS["MA"], label="MA")
    ax1.fill_between(x, as_values(scale, "MP累计时间_ms"), as_values(scale, "MA累计时间_ms"), color="#C9C9C9", alpha=0.25, linewidth=0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Cumulative time, ms")
    add_panel_heading(ax1, "b", "Cumulative time")
    add_framed_legend(ax1, loc="upper right")
    style_axis(ax1)

    ax2 = axes[1, 0]
    success_offset = 0.045
    ax2.plot(
        x - success_offset,
        np.asarray(as_values(scale, "MP成功率")) * 100,
        marker="o",
        color=METHOD_COLORS["MP"],
        label="MP success",
        zorder=4,
    )
    ax2.plot(
        x + success_offset,
        np.asarray(as_values(scale, "MA成功率")) * 100,
        marker="s",
        color=METHOD_COLORS["MA"],
        label="MA success",
        zorder=3,
    )
    ax2b = ax2.twinx()
    ax2b.axhline(1.0, color="#555555", linestyle="--", linewidth=0.75)
    ax2b.plot(x, as_values(scale, "MA与MP时间比"), marker="D", color="#404040", label="MA/MP time ratio", linewidth=1.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Success rate, %")
    ax2b.set_ylabel("Time ratio")
    add_panel_heading(ax2, "c", "Success and time ratio")
    lines, line_labels = ax2.get_legend_handles_labels()
    lines_b, line_labels_b = ax2b.get_legend_handles_labels()
    add_framed_legend(ax2, lines + lines_b, line_labels + line_labels_b, loc="lower right")
    style_axis(ax2)
    ax2b.spines["top"].set_visible(False)
    ax2b.tick_params(length=2.5, width=0.55)

    ax3 = axes[1, 1]
    nodes_k = np.asarray(as_values(scale, "节点数"), dtype=float) / 1000.0
    mp_eff = np.asarray(as_values(scale, "MP累计时间_ms"), dtype=float) / np.maximum(nodes_k, 1e-9)
    ma_eff = np.asarray(as_values(scale, "MA累计时间_ms"), dtype=float) / np.maximum(nodes_k, 1e-9)
    ax3.plot(x, mp_eff, marker="o", color=METHOD_COLORS["MP"], label="MP ms/1k nodes")
    ax3.plot(x, ma_eff, marker="s", color=METHOD_COLORS["MA"], label="MA ms/1k nodes")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel("Time per 1k nodes")
    add_panel_heading(ax3, "d", "Search efficiency")
    add_framed_legend(ax3, loc="upper right")
    style_axis(ax3)

    return save_pub_figure(fig, fig_dir, "fig_4_5_graph_scale_python", dpi)


def relative_to_mp(method: list[dict[str, object]]) -> list[dict[str, object]]:
    """计算相对完整方法 MP 的变化，正值统一表示更优。"""

    metric_specs = [
        ("重规划时间_ms", "Replanning time", False),
        ("路径代价", "Path cost", False),
        ("长度_km", "Path length", False),
        ("通信覆盖率", "Communication coverage", True),
        ("风险暴露", "Risk exposure", False),
    ]
    rows: list[dict[str, object]] = []
    for scene in SCENE_ORDER:
        base = get_row(method, 场景=scene, 方法="MP")
        if base is None:
            continue
        for method_name in ["MA", "MF", "MR", "MV"]:
            item = get_row(method, 场景=scene, 方法=method_name)
            if item is None:
                continue
            for field, label, higher_better in metric_specs:
                base_value = float(base.get(field, float("nan")))
                value = float(item.get(field, float("nan")))
                if not math.isfinite(base_value) or abs(base_value) < 1e-12:
                    continue
                rel = (value - base_value) / abs(base_value) * 100.0
                if not higher_better:
                    rel = -rel
                rows.append({"场景": scene, "方法": method_name, "指标": label, "相对MP变化_越高越好": rel})
    return rows


def plot_ablation_quality(method: list[dict[str, object]], source_dir: Path, fig_dir: Path, dpi: int) -> list[Path]:
    """图 4.6：用热图和方案汇总展示路径质量与速度消融影响。"""

    rel = relative_to_mp(method)
    save_records_csv(
        source_dir / "figure_4_6_ablation_quality_relative_source.csv",
        rel,
        ["场景", "方法", "指标", "相对MP变化_越高越好"],
    )
    methods = ["MA", "MF", "MR", "MV"]
    metrics = ["Replanning time", "Path cost", "Path length", "Communication coverage", "Risk exposure"]
    metric_labels = ["Replan", "Cost", "Length", "Coverage", "Risk"]
    matrix = np.full((len(methods), len(metrics)), np.nan, dtype=float)
    for i, method_name in enumerate(methods):
        for j, metric in enumerate(metrics):
            matrix[i, j] = finite_mean(
                float(row["相对MP变化_越高越好"])
                for row in rel
                if row.get("方法") == method_name and row.get("指标") == metric
            )

    method_effect = np.asarray([finite_mean(matrix[i, :]) for i in range(matrix.shape[0])], dtype=float)

    fig = plt.figure(figsize=(8.2, 2.85), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.15, 1.15, 1.0])
    ax_heat = fig.add_subplot(grid[0, 0:3])
    ax_method = fig.add_subplot(grid[0, 3])

    finite = matrix[np.isfinite(matrix)]
    vmax = max(12.0, float(np.nanpercentile(np.abs(finite), 90)) if finite.size else 12.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax_heat.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
    ax_heat.set_xticks(np.arange(len(metrics)))
    ax_heat.set_xticklabels(metric_labels)
    ax_heat.set_yticks(np.arange(len(methods)))
    ax_heat.set_yticklabels(methods)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if math.isfinite(value):
                color = "white" if abs(value) > vmax * 0.55 else "#222222"
                ax_heat.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=6.6, color=color)
    colorbar = fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.025)
    colorbar.set_label("Change relative to MP, %, higher is better")
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("Ablation variant")
    add_panel_heading(ax_heat, "a", "Metric-level ablation effect", x=0.0, y=1.045)
    ax_heat.tick_params(length=0)

    y = np.arange(len(methods), dtype=float)
    method_colors = [METHOD_COLORS[name] for name in methods]
    ax_method.axvline(0, color="#555555", linewidth=0.75)
    ax_method.barh(y, method_effect, color=method_colors, edgecolor="white", linewidth=0.35)
    ax_method.set_yticks(y)
    ax_method.set_yticklabels(methods)
    ax_method.invert_yaxis()
    ax_method.set_xlabel("Mean change, %")
    add_panel_heading(ax_method, "b", "Variant summary")
    finite_method = method_effect[np.isfinite(method_effect)]
    if finite_method.size:
        margin = max(4.0, float(np.nanmax(np.abs(finite_method))) * 0.12)
        ax_method.set_xlim(float(np.nanmin(finite_method)) - margin, float(np.nanmax(finite_method)) + margin)
    for yi, value in zip(y, method_effect):
        if math.isfinite(value):
            if value >= 0:
                ax_method.text(value + 0.35, yi, f"{value:+.1f}", va="center", ha="left", fontsize=6.2)
            else:
                ax_method.text(-1.0, yi, f"{value:+.1f}", va="center", ha="right", fontsize=6.2)
    style_axis(ax_method)
    return save_pub_figure(fig, fig_dir, "fig_4_6_ablation_quality_python", dpi)


def summarize_structural(records: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    for method in METHOD_ORDER:
        items = [row for row in records if row.get("方法") == method]
        summary.append(
            {
                "方法": method,
                "mean_expanded": finite_mean(float(row.get("mean_expanded", float("nan"))) for row in items),
                "mean_replan_ms": finite_mean(float(row.get("mean_replan_ms", float("nan"))) for row in items),
            }
        )
    return summary


def plot_ablation_workload(ablation: list[dict[str, object]], fig_dir: Path, dpi: int) -> list[Path]:
    """图 4.7：用展开节点解释搜索空间压缩效果。"""

    summary = summarize_structural(ablation)
    x = np.arange(len(summary), dtype=float)
    methods = [str(row["方法"]) for row in summary]
    colors = [METHOD_COLORS[method] for method in methods]
    expanded = as_values(summary, "mean_expanded")
    replan_time = as_values(summary, "mean_replan_ms")

    fig = plt.figure(figsize=(8.2, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.12], width_ratios=[1, 1, 1.05])
    ax_nodes = fig.add_subplot(grid[0, 0])
    ax_time = fig.add_subplot(grid[0, 1])
    ax_compress = fig.add_subplot(grid[0, 2])
    ax_map = fig.add_subplot(grid[1, :])

    ax_nodes.bar(x, expanded, color=colors, edgecolor="white", linewidth=0.35)
    ax_nodes.set_xticks(x)
    ax_nodes.set_xticklabels(methods)
    ax_nodes.set_yscale("log")
    ax_nodes.set_ylabel("Mean expanded nodes")
    add_panel_heading(ax_nodes, "a", "Search workload")
    style_axis(ax_nodes)

    ax_time.bar(x, replan_time, color=colors, edgecolor="white", linewidth=0.35)
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(methods)
    ax_time.set_yscale("log")
    ax_time.set_ylabel("Mean replanning time, ms")
    add_panel_heading(ax_time, "b", "Replanning time")
    style_axis(ax_time)

    mv_index = methods.index("MV") if "MV" in methods else len(methods) - 1
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
    add_panel_heading(ax_compress, "c", "Compression gain")
    add_framed_legend(ax_compress, loc="upper right", title="Baseline")
    style_axis(ax_compress)

    sizes = 70 + 130 * (expanded_gain / np.nanmax(expanded_gain)) if np.isfinite(expanded_gain).any() else np.full_like(x, 90.0)
    for xi, yi, size, method_name, color in zip(expanded, replan_time, sizes, methods, colors):
        ax_map.scatter(xi, yi, s=size, color=color, edgecolor="white", linewidth=0.55, zorder=3)
        ax_map.annotate(method_name, (xi, yi), textcoords="offset points", xytext=(5, 4), fontsize=6.6, color="#222222")
    finite_expanded = np.asarray([value for value in expanded if math.isfinite(value) and value > 0], dtype=float)
    finite_time = np.asarray([value for value in replan_time if math.isfinite(value) and value > 0], dtype=float)
    if finite_expanded.size and finite_time.size:
        fit = np.polyfit(np.log10(finite_expanded), np.log10(finite_time), 1)
        xs = np.geomspace(float(finite_expanded.min()), float(finite_expanded.max()), 80)
        ys = 10 ** (fit[1] + fit[0] * np.log10(xs))
        ax_map.plot(xs, ys, color="#555555", linewidth=0.9, linestyle="--", label="Log-log trend")
        add_framed_legend(ax_map, loc="upper left")
    ax_map.set_xscale("log")
    ax_map.set_yscale("log")
    ax_map.set_xlabel("Mean expanded nodes")
    ax_map.set_ylabel("Mean replanning time, ms")
    add_panel_heading(ax_map, "d", "Workload-efficiency map")
    style_axis(ax_map)

    mp_expanded = expanded[METHOD_ORDER.index("MP")]
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

    return save_pub_figure(fig, fig_dir, "fig_4_7_ablation_workload_python", dpi)


def write_manifest(fig_dir: Path, source_dir: Path, produced: list[Path]) -> Path:
    """写出图表契约与交付清单。"""

    manifest = fig_dir / "chapter4_figure_manifest.md"
    lines = [
        "# 第四章图表生成说明",
        "",
        "核心结论：完整方法 MP 在山地无人机路径规划中同时保持可行性、路径质量和事件驱动重规划效率，三层航线网络与增量重规划分别贡献搜索空间压缩和连续事件下的重复展开减少。",
        "",
        "图表类型：quantitative grid。",
        "",
        "后端：Python，matplotlib。",
        "",
        "输出约定：每张图均导出 SVG、PDF、TIFF、PNG，SVG 保留可编辑文本，PDF 使用 TrueType 字体嵌入，TIFF 按 600 dpi 导出。",
        "",
        "证据链：图 4.3 展示方法对比，图 4.4 展示连续事件下的累计时间与扩展节点，图 4.5 展示图规模敏感性，图 4.6 展示路径质量与速度消融贡献，图 4.7 展示搜索工作量机制。",
        "",
        "统计说明：稿件表格提供的是汇总均值或比例，当前图表不添加误差条；图 4.7 的展开节点来自项目结构消融 CSV 的跨场景均值。",
        "",
        "源数据目录：",
        f"{source_dir}",
        "",
        "生成文件：",
    ]
    for path in produced:
        lines.append(f"、{path.name}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成第四章 Python 论文图表。")
    parser.add_argument("--docx", type=Path, default=DEFAULT_DOCX, help="当前论文初稿 docx 路径。")
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR, help="图表输出目录。")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="源数据输出目录。")
    parser.add_argument("--dpi", type=int, default=600, help="TIFF 输出分辨率。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    tables = prepare_source_tables(args.docx, args.source_dir)
    structural = load_structural_ablation(PROJECT_ROOT, args.source_dir)

    produced: list[Path] = []
    produced += plot_method_comparison(tables["method"], args.fig_dir, args.dpi)
    produced += plot_event_replanning(tables["event"], args.fig_dir, args.dpi)
    produced += plot_graph_scale(tables["scale"], args.fig_dir, args.dpi)
    produced += plot_ablation_quality(tables["method"], args.source_dir, args.fig_dir, args.dpi)
    produced += plot_ablation_workload(structural, args.fig_dir, args.dpi)
    manifest = write_manifest(args.fig_dir, args.source_dir, produced)

    print("第四章图表生成完成")
    print(f"图表目录：{args.fig_dir}")
    print(f"源数据目录：{args.source_dir}")
    print(f"生成说明：{manifest}")
    for path in produced:
        print(f"  {path}")


if __name__ == "__main__":
    main()
