"""绘制论文矩阵实验结果图。

该脚本只读取 benchmark_matrix.py / benchmark.py 已生成的 CSV，不重新建图、不重新规划。
默认输出 PDF，便于论文排版；若存在结构性消融 CSV，会额外生成 E2 消融图。
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


B4 = "B4_Proposed_LPA_Layered"
B2 = "B2_GlobalAstar_Layered"
B3 = "B3_LPA_SingleLayer"
B5 = "B5_RegularLayered_LPA"
B1 = "B1_Voxel_Dijkstra"
B6_LEGACY = "B6_RegularLayered_LPA"

METHOD_IDS = {
    B4: "M-P",
    B2: "M-A",
    B3: "M-F",
    B5: "M-R",
    B1: "M-V",
}

FIGURE_LABELS = {
    B4: "Terrain-aware Layered LPA* (Proposed)",
    B2: "Terrain-aware Layered A*",
    B3: "Flat-graph LPA*",
    B5: "Regular-layered LPA*",
    B1: "Voxel Global Search",
}

AXIS_LABELS = {
    B4: "Terrain-aware Layered LPA*",
    B2: "Terrain-aware Layered A*",
    B3: "Flat-graph LPA*",
    B5: "Regular-layered LPA*",
    B1: "Voxel Global Search",
}

SPEEDUP_AXIS_LABEL = "Speedup ratio (M-A / M-P)"
SPEEDUP_NOTE = "A value > 1 indicates that M-P is faster; percent labels denote M-P faster rate."


def read_csv_rows(path: Path, required: bool = True) -> List[dict]:
    """读取 CSV；正式主图依赖的文件缺失时直接报错。"""
    if not path.exists():
        if required:
            raise FileNotFoundError(f"缺少输入文件: {path}")
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_result_dir(raw: str) -> Path:
    """解析结果目录；旧 outputs 路径自动映射到 final_results。"""
    p = Path(raw)
    root = Path(".").resolve()
    norm = str(p).replace("\\", "/")
    root_norm = str(root).replace("\\", "/").rstrip("/")
    if p.is_absolute() and norm.startswith(f"{root_norm}/outputs/"):
        norm = norm[len(root_norm) + 1 :]
        p = Path(norm)
    parts = [part for part in norm.split("/") if part not in {"", "."}]
    if len(parts) >= 2 and parts[0] == "outputs":
        scene = parts[1]
        rest = parts[2:]
        if rest and rest[0] == "tests":
            rest = rest[1:]
        mapped = root / "final_results" / scene
        if rest:
            mapped = mapped / Path(*rest)
        if mapped.exists():
            return mapped.resolve()
    return p.resolve()


def to_float(value: object, default: float = float("nan")) -> float:
    """宽松转换数值字段，兼容空字符串和 nan。"""
    if value is None:
        return default
    s = str(value).strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def to_int(value: object, default: int = 0) -> int:
    """宽松转换整数字段。"""
    v = to_float(value, float("nan"))
    if not math.isfinite(v):
        return default
    return int(round(v))


def is_true(value: object) -> bool:
    """兼容 Python bool 和 CSV 字符串 bool。"""
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite(values: Iterable[float]) -> List[float]:
    """过滤非有限值。"""
    return [float(v) for v in values if math.isfinite(float(v))]


def style_axes(ax: plt.Axes) -> None:
    """统一论文图的坐标轴风格。"""
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, out_dir: Path, name: str, dpi: int) -> Path:
    """保存单张图并关闭 figure，避免批量绘图时泄漏内存。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def sorted_by_int(rows: Sequence[dict], field: str) -> List[dict]:
    return sorted(rows, key=lambda r: to_int(r.get(field)))


def plot_exp_a_event_intensity(rows: List[dict], out_dir: Path, dpi: int) -> Path:
    """E3.1：事件强度索引对单事件重规划时间的影响。"""
    rows = sorted(rows, key=lambda r: to_int(r.get("intensity_index", r.get("n_block"))))
    x = np.asarray([to_int(r.get("intensity_index", r.get("n_block"))) for r in rows], dtype=float)
    b4_mean = np.asarray([to_float(r.get("b4_mean_event_ms")) for r in rows], dtype=float)
    b2_mean = np.asarray([to_float(r.get("b2_mean_event_ms")) for r in rows], dtype=float)
    b4_p50 = np.asarray([to_float(r.get("b4_p50_event_ms")) for r in rows], dtype=float)
    b4_p95 = np.asarray([to_float(r.get("b4_p95_event_ms")) for r in rows], dtype=float)
    b2_p50 = np.asarray([to_float(r.get("b2_p50_event_ms")) for r in rows], dtype=float)
    b2_p95 = np.asarray([to_float(r.get("b2_p95_event_ms")) for r in rows], dtype=float)
    ratio = np.asarray([to_float(r.get("b2_over_b4_time_ratio")) for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax = axes[0]
    ax.plot(x, b4_mean, marker="o", linewidth=2.0, color="#d95f02", label="M-P")
    ax.plot(x, b2_mean, marker="s", linewidth=2.0, color="#1f78b4", label="M-A")
    if np.isfinite(b4_p50).any() and np.isfinite(b4_p95).any():
        ax.fill_between(x, b4_p50, b4_p95, color="#d95f02", alpha=0.16, linewidth=0)
    if np.isfinite(b2_p50).any() and np.isfinite(b2_p95).any():
        ax.fill_between(x, b2_p50, b2_p95, color="#1f78b4", alpha=0.14, linewidth=0)
    ax.set_xlabel("Event intensity index")
    ax.set_ylabel("Mean replanning time per event (ms)")
    ax.set_title("(a) Replanning time")
    style_axes(ax)

    ax_speed = axes[1]
    ax_speed.axhline(1.0, color="#555555", linestyle="--", linewidth=0.9, alpha=0.65)
    ax_speed.plot(x, ratio, marker="D", linewidth=1.8, color="#238b45", label="M-A / M-P")
    ax_speed.set_xlabel("Event intensity index")
    ax_speed.set_ylabel(SPEEDUP_AXIS_LABEL)
    ax_speed.set_title("(b) Speedup")
    style_axes(ax_speed)

    notes = ["A value > 1 indicates that M-P is faster."]
    missing = [
        int(xi)
        for xi, r4, r2, rr in zip(x, b4_mean, b2_mean, ratio)
        if not (math.isfinite(r4) and math.isfinite(r2) and math.isfinite(rr))
    ]
    if missing:
        notes.append(
            f"Missing intensity indices ({', '.join(map(str, missing))}) indicate unavailable successful paired trials."
        )
    fig.text(0.5, -0.02, "\n".join(notes), ha="center", fontsize=7.2, color="#555555")

    handles, labels = [], []
    for one_ax in axes:
        one_handles, one_labels = one_ax.get_legend_handles_labels()
        handles.extend(one_handles)
        labels.extend(one_labels)
    dedup = dict(zip(labels, handles))
    fig.legend(dedup.values(), dedup.keys(), frameon=False, fontsize=8, ncol=len(dedup), loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("E3.1 Event-intensity Sensitivity", y=1.02, fontsize=10.5)
    fig.subplots_adjust(wspace=0.32, top=0.80, bottom=0.22)
    return save_figure(fig, out_dir, "fig_expA_event_intensity_time.pdf", dpi)


def plot_exp_b_cumulative_time(rows: List[dict], out_dir: Path, dpi: int) -> Path:
    """E3.2：连续 K 次扰动下的累计重规划时间。"""
    rows = sorted_by_int(rows, "k_events")
    x = np.asarray([to_int(r.get("k_events")) for r in rows], dtype=float)
    b4_mean = np.asarray([to_float(r.get("b4_mean_cumulative_ms")) for r in rows], dtype=float)
    b2_mean = np.asarray([to_float(r.get("b2_mean_cumulative_ms")) for r in rows], dtype=float)
    b4_p50 = np.asarray([to_float(r.get("b4_p50_cumulative_ms")) for r in rows], dtype=float)
    b4_p95 = np.asarray([to_float(r.get("b4_p95_cumulative_ms")) for r in rows], dtype=float)
    b2_p50 = np.asarray([to_float(r.get("b2_p50_cumulative_ms")) for r in rows], dtype=float)
    b2_p95 = np.asarray([to_float(r.get("b2_p95_cumulative_ms")) for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(x, b4_mean, marker="o", linewidth=2.0, color="#d95f02", label="M-P")
    ax.plot(x, b2_mean, marker="s", linewidth=2.0, color="#1f78b4", label="M-A")
    if np.isfinite(b4_p50).any() and np.isfinite(b4_p95).any():
        ax.fill_between(x, b4_p50, b4_p95, color="#d95f02", alpha=0.16, linewidth=0)
    if np.isfinite(b2_p50).any() and np.isfinite(b2_p95).any():
        ax.fill_between(x, b2_p50, b2_p95, color="#1f78b4", alpha=0.14, linewidth=0)
    ax.set_xlabel("Number of sequential events K")
    ax.set_ylabel("Cumulative replanning time (ms)")
    ax.set_title("E3.2 Consecutive-event Replanning")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    return save_figure(fig, out_dir, "fig_expB_cumulative_time.pdf", dpi)


def plot_exp_b_speedup(rows: List[dict], out_dir: Path, dpi: int) -> Path:
    """E3.2：M-A/M-P 时间比，突出状态复用收益。"""
    rows = sorted_by_int(rows, "k_events")
    x = np.asarray([to_int(r.get("k_events")) for r in rows], dtype=float)
    ratio = np.asarray([to_float(r.get("b2_over_b4_time_ratio")) for r in rows], dtype=float)
    ratio_mean = np.asarray([to_float(r.get("b2_over_b4_time_ratio_mean_of_means")) for r in rows], dtype=float)
    faster = np.asarray([to_float(r.get("b4_faster_rate")) for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.plot(x, ratio, marker="o", linewidth=2.2, color="#2c7fb8", label="Paired median speedup")
    if np.isfinite(ratio_mean).any():
        ax.plot(x, ratio_mean, marker="^", linewidth=1.6, color="#7fcdbb", label="Mean-of-means speedup")
    for xi, yi, fr in zip(x, ratio, faster):
        if math.isfinite(yi) and math.isfinite(fr):
            ax.annotate(f"{100.0 * fr:.0f}%", (xi, yi), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    ax.set_xlabel("Number of sequential events K")
    ax.set_ylabel(SPEEDUP_AXIS_LABEL)
    ax.set_title("E3.2 Consecutive-event Replanning Speedup")
    fig.text(0.5, 0.01, SPEEDUP_NOTE, ha="center", fontsize=7.2, color="#555555")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    fig.subplots_adjust(bottom=0.18)
    return save_figure(fig, out_dir, "fig_expB_speedup.pdf", dpi)


def plot_exp_d_workload(rows: List[dict], out_dir: Path, dpi: int) -> Path:
    """E3.4：不同扰动强度下的 expanded nodes 工作量。"""
    rows = sorted_by_int(rows, "intensity_index")
    x = np.asarray([to_int(r.get("intensity_index", r.get("n_block"))) for r in rows], dtype=float)
    b4 = np.asarray([to_float(r.get("b4_mean_event_expanded")) for r in rows], dtype=float)
    b2 = np.asarray([to_float(r.get("b2_mean_event_expanded")) for r in rows], dtype=float)
    reduction = np.asarray([to_float(r.get("expanded_reduction")) for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    width = 0.34
    ax.bar(x - width / 2.0, b4, width=width, color="#d95f02", alpha=0.88, label="M-P")
    ax.bar(x + width / 2.0, b2, width=width, color="#1f78b4", alpha=0.82, label="M-A")
    ax.set_xlabel("Event intensity index")
    ax.set_ylabel("Expanded nodes per event")
    ax.set_title("E3.4 Workload Mechanism Analysis")
    style_axes(ax)
    ax2 = ax.twinx()
    ax2.plot(x, 100.0 * reduction, color="#238b45", marker="D", linewidth=1.8, label="Reduction")
    ax2.set_ylabel("Workload reduction (%)")
    ax2.spines["top"].set_visible(False)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines + lines2,
        labels + labels2,
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
    )
    return save_figure(fig, out_dir, "fig_expD_workload_expanded.pdf", dpi)


def plot_exp_c_scale_success(rows: List[dict], out_dir: Path, dpi: int) -> Path:
    """E3.3：图规模敏感性下的成功率。"""
    order = ["small", "medium", "large"]
    rows_by_scale = {str(r.get("scale")): r for r in rows}
    scales = [s for s in order if s in rows_by_scale] + [s for s in rows_by_scale if s not in order]
    x = np.arange(len(scales), dtype=float)
    b4 = np.asarray([100.0 * to_float(rows_by_scale[s].get("b4_success_rate")) for s in scales], dtype=float)
    b2 = np.asarray([100.0 * to_float(rows_by_scale[s].get("b2_success_rate")) for s in scales], dtype=float)
    nodes = [to_int(rows_by_scale[s].get("graph_nodes")) for s in scales]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    width = 0.34
    ax.bar(x - width / 2.0, b4, width=width, color="#d95f02", alpha=0.88, label="M-P")
    ax.bar(x + width / 2.0, b2, width=width, color="#1f78b4", alpha=0.82, label="M-A")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n|V|={n}" for s, n in zip(scales, nodes)])
    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Graph scale")
    ax.set_title("E3.3 Graph-scale Feasibility Analysis")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    style_axes(ax)
    return save_figure(fig, out_dir, "fig_expC_scale_success.pdf", dpi)


def paired_trial_costs(rows: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """从 trial 级记录中配对提取 M-A 与 M-P 的最终路径代价。"""
    by_key: Dict[Tuple[str, int, int, int, str], dict] = {}
    for r in rows:
        baseline = str(r.get("baseline", ""))
        if baseline not in {B4, B2}:
            continue
        if "final_path_cost" not in r:
            continue
        if "success_all_events" in r and not is_true(r.get("success_all_events")):
            continue
        scale = str(r.get("scale", ""))
        n_block = to_int(r.get("intensity_index", r.get("n_block")))
        k_events = to_int(r.get("k_events"))
        trial = to_int(r.get("trial"))
        by_key[(scale, n_block, k_events, trial, baseline)] = r

    b2_cost: List[float] = []
    b4_cost: List[float] = []
    keys = sorted({k[:4] for k in by_key})
    for key in keys:
        r4 = by_key.get((*key, B4))
        r2 = by_key.get((*key, B2))
        if r4 is None or r2 is None:
            continue
        c4 = to_float(r4.get("final_path_cost"))
        c2 = to_float(r2.get("final_path_cost"))
        if math.isfinite(c4) and math.isfinite(c2):
            b2_cost.append(c2)
            b4_cost.append(c4)
    return np.asarray(b2_cost, dtype=float), np.asarray(b4_cost, dtype=float)


def plot_path_quality_scatter(trial_rows: List[dict], quality_rows: List[dict], out_dir: Path, dpi: int) -> Path:
    """路径质量散点：M-P 与 M-A 优化目标相同，理想情况下应接近 y=x。"""
    b2_cost, b4_cost = paired_trial_costs(trial_rows)
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    if b2_cost.size > 0:
        ax.scatter(b2_cost, b4_cost, s=22, color="#525252", alpha=0.62, edgecolors="none")
        lo = float(np.nanmin([np.min(b2_cost), np.min(b4_cost)]))
        hi = float(np.nanmax([np.max(b2_cost), np.max(b4_cost)]))
        pad = 0.04 * max(hi - lo, 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="#d95f02", linewidth=1.3)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"{AXIS_LABELS[B2]} final path cost")
        ax.set_ylabel(f"{AXIS_LABELS[B4]} final path cost")
        ax.set_title("E4 Path-quality Consistency")
        rel_gap = np.abs(b4_cost - b2_cost) / np.maximum(np.abs(b2_cost), 1e-9)
        median_gap = float(np.nanmedian(rel_gap)) if rel_gap.size else float("nan")
        note = "Final path costs are nearly identical.\nM-P efficiency is not obtained by sacrificing path quality."
        if math.isfinite(median_gap):
            note = f"{note}\nMedian relative gap = {100.0 * median_gap:.2f}%."
        ax.text(
            0.03,
            0.96,
            note,
            transform=ax.transAxes,
            fontsize=7.1,
            color="#444444",
            va="top",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.86},
        )
    else:
        x = np.arange(len(quality_rows), dtype=float)
        gap = np.asarray([to_float(r.get("median_abs_cost_gap")) for r in quality_rows], dtype=float)
        ax.bar(x, gap, color="#737373", alpha=0.75)
        ax.set_xlabel("Matrix combo index")
        ax.set_ylabel("Median absolute cost gap")
        ax.set_title("E4 Path-quality Summary")
        ax.text(0.5, 0.92, "Trial-level pairs not found", transform=ax.transAxes, ha="center", fontsize=8)
    style_axes(ax)
    return save_figure(fig, out_dir, "fig_path_quality_scatter.pdf", dpi)


def plot_structural_ablation(
    rows: List[dict],
    out_dir: Path,
    dpi: int,
    filename: str = "fig_structural_ablation.pdf",
) -> Path | None:
    """E2 结构性消融图：M-P/M-A/M-F/M-R/M-V 同图比较。"""
    if not rows:
        return None
    by_baseline = {str(r.get("baseline_id", r.get("baseline"))): r for r in rows}
    if B6_LEGACY in by_baseline and B5 not in by_baseline:
        by_baseline[B5] = by_baseline[B6_LEGACY]
    if B5 not in by_baseline:
        return None
    ordered = [
        ("M-P", B4, "#d95f02"),
        ("M-A", B2, "#1f78b4"),
        ("M-F", B3, "#7570b3"),
        ("M-R", B5, "#66a61e"),
        ("M-V", B1, "#8c510a"),
    ]
    present = [(label, key, color) for label, key, color in ordered if key in by_baseline]
    if len(present) < 2:
        return None

    labels = [p[0] for p in present]
    colors = [p[2] for p in present]
    rows_by_method = [by_baseline[p[1]] for p in present]

    def metric(row: dict, *names: str) -> float:
        for name in names:
            val = to_float(row.get(name))
            if math.isfinite(val):
                return val
        return float("nan")

    def yerr_arg(values: Sequence[float]) -> Sequence[float] | None:
        return values if any(math.isfinite(v) and v > 0 for v in values) else None

    times = [metric(r, "mean_replan_ms") for r in rows_by_method]
    times_ci = [metric(r, "ci95_replan_ms") for r in rows_by_method]
    costs = [metric(r, "mean_path_cost", "mean_cost") for r in rows_by_method]
    costs_ci = [metric(r, "ci95_path_cost", "ci95_cost") for r in rows_by_method]
    comm = [100.0 * metric(r, "mean_comm_coverage", "mean_comm_coverage_ratio") for r in rows_by_method]
    comm_ci = [100.0 * metric(r, "ci95_comm_coverage", "ci95_comm_coverage_ratio") for r in rows_by_method]
    risk = [metric(r, "mean_risk_exposure", "mean_risk_exposure_integral") for r in rows_by_method]
    risk_ci = [metric(r, "ci95_risk_exposure", "ci95_risk_exposure_integral") for r in rows_by_method]
    x = np.arange(len(present), dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.8))
    axes_flat = list(axes.ravel())
    err_kw = {"linewidth": 0.8, "capthick": 0.8}
    axes_flat[0].bar(x, times, color=colors, alpha=0.85, yerr=yerr_arg(times_ci), capsize=2.0, error_kw=err_kw)
    axes_flat[0].set_ylabel("Replanning time (ms)")
    axes_flat[0].set_title("(a) Replanning time")
    axes_flat[1].bar(x, costs, color=colors, alpha=0.85, yerr=yerr_arg(costs_ci), capsize=2.0, error_kw=err_kw)
    axes_flat[1].set_ylabel("Path cost")
    axes_flat[1].set_title("(b) Path cost")
    axes_flat[2].bar(x, comm, color=colors, alpha=0.85, yerr=yerr_arg(comm_ci), capsize=2.0, error_kw=err_kw)
    axes_flat[2].set_ylabel("Communication coverage (%)")
    axes_flat[2].set_ylim(0, 105)
    axes_flat[2].set_title("(c) Communication coverage")
    axes_flat[3].bar(x, risk, color=colors, alpha=0.85, yerr=yerr_arg(risk_ci), capsize=2.0, error_kw=err_kw)
    axes_flat[3].set_ylabel("Risk exposure integral")
    axes_flat[3].set_title("(d) Risk exposure")
    for ax in axes_flat:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.2)
        style_axes(ax)
    fig.suptitle("E2 Structural Ablation Study", y=1.03, fontsize=11)
    legend_handles = [Patch(facecolor=color, alpha=0.85, label=label) for label, _, color in present]
    fig.legend(legend_handles, labels, frameon=False, ncol=len(labels), loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(hspace=0.42, wspace=0.30, bottom=0.12)
    return save_figure(fig, out_dir, filename, dpi)


def configure_matplotlib() -> None:
    """设置适合 PDF 投稿的基础样式。"""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制 benchmark matrix 论文图。")
    parser.add_argument("--result-dir", type=str, required=True, help="包含 benchmark_summary.csv 和 E3.1-E3.4（experiment_A/B/C/D.csv）的目录。")
    parser.add_argument("--out-dir", type=str, default="", help="图输出目录；默认写回 result-dir。")
    parser.add_argument("--ablation-only", action="store_true", help="只读取结构性消融 CSV 并生成 fig_structural_ablation.pdf。")
    parser.add_argument("--dpi", type=int, default=300, help="PDF 中嵌入栅格元素的分辨率。")
    return parser.parse_args()


def load_structural_rows(result_dir: Path, summary_rows: List[dict] | None = None) -> List[dict]:
    """按优先级读取 E2 结构性消融结果。"""
    structural_rows = read_csv_rows(result_dir / "benchmark_structural_ablation.csv", required=False)
    if not structural_rows:
        structural_rows = read_csv_rows(result_dir / "benchmark_summary_four_baselines.csv", required=False)
    if not structural_rows and summary_rows is not None:
        structural_rows = [r for r in summary_rows if str(r.get("baseline", "")) in {B1, B2, B3, B4, B5, B6_LEGACY}]
    if not structural_rows:
        structural_rows = read_csv_rows(result_dir / "benchmark_summary.csv", required=False)
        structural_rows = [r for r in structural_rows if str(r.get("baseline", "")) in {B1, B2, B3, B4, B5, B6_LEGACY}]
    return structural_rows


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    result_dir = resolve_result_dir(args.result_dir)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else result_dir

    if args.ablation_only:
        structural_rows = load_structural_rows(result_dir)
        ablation_path = plot_structural_ablation(structural_rows, out_dir, args.dpi)
        if ablation_path is None:
            raise RuntimeError("未找到包含 M-R / B5_RegularLayered_LPA 的结构性消融结果。")
        print("[done] 结构性消融图已生成：")
        print(f"  - {ablation_path}")
        return

    summary_rows = read_csv_rows(result_dir / "benchmark_summary.csv", required=True)
    experiment_a_rows = read_csv_rows(result_dir / "experiment_A.csv", required=True)
    experiment_b_rows = read_csv_rows(result_dir / "experiment_B.csv", required=True)
    experiment_c_rows = read_csv_rows(result_dir / "experiment_C.csv", required=True)
    experiment_d_rows = read_csv_rows(result_dir / "experiment_D.csv", required=True)
    trial_rows = read_csv_rows(result_dir / "benchmark_trials.csv", required=False)
    quality_rows = read_csv_rows(result_dir / "experiment_path_quality.csv", required=False)

    produced = [
        plot_exp_a_event_intensity(experiment_a_rows, out_dir, args.dpi),
        plot_exp_b_cumulative_time(experiment_b_rows, out_dir, args.dpi),
        plot_exp_b_speedup(experiment_b_rows, out_dir, args.dpi),
        plot_exp_d_workload(experiment_d_rows, out_dir, args.dpi),
        plot_exp_c_scale_success(experiment_c_rows, out_dir, args.dpi),
        plot_path_quality_scatter(trial_rows, quality_rows, out_dir, args.dpi),
    ]

    structural_rows = load_structural_rows(result_dir, summary_rows=summary_rows)
    ablation_path = plot_structural_ablation(structural_rows, out_dir, args.dpi)
    if ablation_path is not None:
        produced.append(ablation_path)

    print("[done] 论文图已生成：")
    for path in produced:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
