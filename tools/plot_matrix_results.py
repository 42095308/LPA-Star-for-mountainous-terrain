"""绘制论文矩阵实验结果图。

该脚本只读取 benchmark_matrix.py / benchmark.py 已生成的 CSV，不重新建图、不重新规划。
默认输出 PDF，便于论文排版；若存在结构性消融 CSV，会额外生成 B5 消融图。
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
import numpy as np


B4 = "B4_Proposed_LPA_Layered"
B2 = "B2_GlobalAstar_Layered"
B3 = "B3_LPA_SingleLayer"
B5 = "B5_RegularLayered_LPA"
B6_LEGACY = "B6_RegularLayered_LPA"


def read_csv_rows(path: Path, required: bool = True) -> List[dict]:
    """读取 CSV；正式主图依赖的文件缺失时直接报错。"""
    if not path.exists():
        if required:
            raise FileNotFoundError(f"缺少输入文件: {path}")
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
    """Experiment A：事件强度索引对单事件重规划时间的影响。"""
    rows = sorted(rows, key=lambda r: to_int(r.get("intensity_index", r.get("n_block"))))
    x = np.asarray([to_int(r.get("intensity_index", r.get("n_block"))) for r in rows], dtype=float)
    b4_mean = np.asarray([to_float(r.get("b4_mean_event_ms")) for r in rows], dtype=float)
    b2_mean = np.asarray([to_float(r.get("b2_mean_event_ms")) for r in rows], dtype=float)
    b4_p50 = np.asarray([to_float(r.get("b4_p50_event_ms")) for r in rows], dtype=float)
    b4_p95 = np.asarray([to_float(r.get("b4_p95_event_ms")) for r in rows], dtype=float)
    b2_p50 = np.asarray([to_float(r.get("b2_p50_event_ms")) for r in rows], dtype=float)
    b2_p95 = np.asarray([to_float(r.get("b2_p95_event_ms")) for r in rows], dtype=float)
    ratio = np.asarray([to_float(r.get("b2_over_b4_time_ratio")) for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(5.3, 3.4))
    ax.plot(x, b4_mean, marker="o", linewidth=2.0, color="#d95f02", label="B4 terrain-aware LPA*")
    ax.plot(x, b2_mean, marker="s", linewidth=2.0, color="#1f78b4", label="B2 global A*")
    if np.isfinite(b4_p50).any() and np.isfinite(b4_p95).any():
        ax.fill_between(x, b4_p50, b4_p95, color="#d95f02", alpha=0.16, linewidth=0)
    if np.isfinite(b2_p50).any() and np.isfinite(b2_p95).any():
        ax.fill_between(x, b2_p50, b2_p95, color="#1f78b4", alpha=0.14, linewidth=0)
    ax.set_xlabel("Event intensity index")
    ax.set_ylabel("Mean replanning time per event (ms)")
    ax.set_title("Experiment A: event-intensity sensitivity")
    style_axes(ax)
    ax2 = ax.twinx()
    ax2.axhline(1.0, color="#555555", linestyle="--", linewidth=0.9, alpha=0.55)
    ax2.plot(x, ratio, marker="D", linewidth=1.5, color="#238b45", label="B2/B4 speedup")
    ax2.set_ylabel("Speedup ratio (B2 / B4)")
    ax2.spines["top"].set_visible(False)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="best")
    return save_figure(fig, out_dir, "fig_expA_event_intensity_time.pdf", dpi)


def plot_exp_b_cumulative_time(rows: List[dict], out_dir: Path, dpi: int) -> Path:
    """Experiment B：连续 K 次扰动下的累计重规划时间。"""
    rows = sorted_by_int(rows, "k_events")
    x = np.asarray([to_int(r.get("k_events")) for r in rows], dtype=float)
    b4_mean = np.asarray([to_float(r.get("b4_mean_cumulative_ms")) for r in rows], dtype=float)
    b2_mean = np.asarray([to_float(r.get("b2_mean_cumulative_ms")) for r in rows], dtype=float)
    b4_p50 = np.asarray([to_float(r.get("b4_p50_cumulative_ms")) for r in rows], dtype=float)
    b4_p95 = np.asarray([to_float(r.get("b4_p95_cumulative_ms")) for r in rows], dtype=float)
    b2_p50 = np.asarray([to_float(r.get("b2_p50_cumulative_ms")) for r in rows], dtype=float)
    b2_p95 = np.asarray([to_float(r.get("b2_p95_cumulative_ms")) for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(x, b4_mean, marker="o", linewidth=2.0, color="#d95f02", label="B4 terrain-aware LPA*")
    ax.plot(x, b2_mean, marker="s", linewidth=2.0, color="#1f78b4", label="B2 global A*")
    if np.isfinite(b4_p50).any() and np.isfinite(b4_p95).any():
        ax.fill_between(x, b4_p50, b4_p95, color="#d95f02", alpha=0.16, linewidth=0)
    if np.isfinite(b2_p50).any() and np.isfinite(b2_p95).any():
        ax.fill_between(x, b2_p50, b2_p95, color="#1f78b4", alpha=0.14, linewidth=0)
    ax.set_xlabel("Number of sequential events K")
    ax.set_ylabel("Cumulative replanning time (ms)")
    ax.set_title("Experiment B: cumulative replanning time")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    return save_figure(fig, out_dir, "fig_expB_cumulative_time.pdf", dpi)


def plot_exp_b_speedup(rows: List[dict], out_dir: Path, dpi: int) -> Path:
    """Experiment B：B2/B4 时间比，突出状态复用收益。"""
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
    ax.set_ylabel("Speedup ratio (B2 / B4)")
    ax.set_title("Experiment B: state-reuse speedup")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    return save_figure(fig, out_dir, "fig_expB_speedup.pdf", dpi)


def plot_exp_d_workload(rows: List[dict], out_dir: Path, dpi: int) -> Path:
    """Experiment D：不同扰动强度下的 expanded nodes 工作量。"""
    rows = sorted_by_int(rows, "intensity_index")
    x = np.asarray([to_int(r.get("intensity_index", r.get("n_block"))) for r in rows], dtype=float)
    b4 = np.asarray([to_float(r.get("b4_mean_event_expanded")) for r in rows], dtype=float)
    b2 = np.asarray([to_float(r.get("b2_mean_event_expanded")) for r in rows], dtype=float)
    reduction = np.asarray([to_float(r.get("expanded_reduction")) for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    width = 0.34
    ax.bar(x - width / 2.0, b4, width=width, color="#d95f02", alpha=0.88, label="B4 terrain-aware LPA*")
    ax.bar(x + width / 2.0, b2, width=width, color="#1f78b4", alpha=0.82, label="B2 global A*")
    ax.set_xlabel("Event intensity index")
    ax.set_ylabel("Expanded nodes per event")
    ax.set_title("Experiment D: workload reduction")
    style_axes(ax)
    ax2 = ax.twinx()
    ax2.plot(x, 100.0 * reduction, color="#238b45", marker="D", linewidth=1.8, label="Reduction")
    ax2.set_ylabel("B4 workload reduction (%)")
    ax2.spines["top"].set_visible(False)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="upper left")
    return save_figure(fig, out_dir, "fig_expD_workload_expanded.pdf", dpi)


def plot_exp_c_scale_success(rows: List[dict], out_dir: Path, dpi: int) -> Path:
    """Experiment C：图规模敏感性下的成功率。"""
    order = ["small", "medium", "large"]
    rows_by_scale = {str(r.get("scale")): r for r in rows}
    scales = [s for s in order if s in rows_by_scale] + [s for s in rows_by_scale if s not in order]
    x = np.arange(len(scales), dtype=float)
    b4 = np.asarray([100.0 * to_float(rows_by_scale[s].get("b4_success_rate")) for s in scales], dtype=float)
    b2 = np.asarray([100.0 * to_float(rows_by_scale[s].get("b2_success_rate")) for s in scales], dtype=float)
    nodes = [to_int(rows_by_scale[s].get("graph_nodes")) for s in scales]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    width = 0.34
    ax.bar(x - width / 2.0, b4, width=width, color="#d95f02", alpha=0.88, label="B4 terrain-aware LPA*")
    ax.bar(x + width / 2.0, b2, width=width, color="#1f78b4", alpha=0.82, label="B2 global A*")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n|V|={n}" for s, n in zip(scales, nodes)])
    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Graph scale")
    ax.set_title("Experiment C: scale sensitivity")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axes(ax)
    return save_figure(fig, out_dir, "fig_expC_scale_success.pdf", dpi)


def paired_trial_costs(rows: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """从 trial 级记录中配对提取 B2 与 B4 的最终路径代价。"""
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
    """路径质量散点：B4 与 B2 优化目标相同，理想情况下应接近 y=x。"""
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
        ax.set_xlabel("B2 final path cost")
        ax.set_ylabel("B4 final path cost")
        ax.set_title("Path quality consistency")
    else:
        x = np.arange(len(quality_rows), dtype=float)
        gap = np.asarray([to_float(r.get("median_abs_cost_gap")) for r in quality_rows], dtype=float)
        ax.bar(x, gap, color="#737373", alpha=0.75)
        ax.set_xlabel("Matrix combo index")
        ax.set_ylabel("Median absolute cost gap")
        ax.set_title("Path quality summary")
        ax.text(0.5, 0.92, "Trial-level pairs not found", transform=ax.transAxes, ha="center", fontsize=8)
    style_axes(ax)
    return save_figure(fig, out_dir, "fig_path_quality_scatter.pdf", dpi)


def plot_structural_ablation(rows: List[dict], out_dir: Path, dpi: int) -> Path | None:
    """B5 结构性消融图：B4/B5/B3/B2 四类方法同图比较。"""
    if not rows:
        return None
    by_baseline = {str(r.get("baseline_id", r.get("baseline"))): r for r in rows}
    if B6_LEGACY in by_baseline and B5 not in by_baseline:
        by_baseline[B5] = by_baseline[B6_LEGACY]
    if B5 not in by_baseline:
        return None
    ordered = [
        ("B2\nLayered A*", B2, "#1f78b4"),
        ("B3\nFlat LPA*", B3, "#7570b3"),
        ("B5\nRegular LPA*", B5, "#66a61e"),
        ("B4\nTerrain LPA*", B4, "#d95f02"),
    ]
    present = [(label, key, color) for label, key, color in ordered if key in by_baseline]
    if len(present) < 2:
        return None

    labels = [str(by_baseline[p[1]].get("method", p[0])).replace(" ", "\n", 1) for p in present]
    colors = [p[2] for p in present]
    times = [to_float(by_baseline[p[1]].get("mean_replan_ms")) for p in present]
    costs = [to_float(by_baseline[p[1]].get("mean_cost")) for p in present]
    success = [100.0 * to_float(by_baseline[p[1]].get("success_rate")) for p in present]
    x = np.arange(len(present), dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0))
    axes[0].bar(x, times, color=colors, alpha=0.85)
    axes[0].set_ylabel("Replanning time (ms)")
    axes[0].set_title("Time")
    axes[1].bar(x, costs, color=colors, alpha=0.85)
    axes[1].set_ylabel("Path cost")
    axes[1].set_title("Quality")
    axes[2].bar(x, success, color=colors, alpha=0.85)
    axes[2].set_ylabel("Success rate (%)")
    axes[2].set_ylim(0, 105)
    axes[2].set_title("Robustness")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        style_axes(ax)
    fig.suptitle("Structural ablation: terrain-aware layering vs regular layering", y=1.03, fontsize=11)
    return save_figure(fig, out_dir, "fig_structural_ablation.pdf", dpi)


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
    parser.add_argument("--result-dir", type=str, required=True, help="包含 benchmark_summary.csv 和 experiment_A/B/C/D.csv 的目录。")
    parser.add_argument("--out-dir", type=str, default="", help="图输出目录；默认写回 result-dir。")
    parser.add_argument("--ablation-only", action="store_true", help="只读取结构性消融 CSV 并生成 fig_structural_ablation.pdf。")
    parser.add_argument("--dpi", type=int, default=300, help="PDF 中嵌入栅格元素的分辨率。")
    return parser.parse_args()


def load_structural_rows(result_dir: Path, summary_rows: List[dict] | None = None) -> List[dict]:
    """按优先级读取 B5 结构性消融结果。"""
    structural_rows = read_csv_rows(result_dir / "benchmark_structural_ablation.csv", required=False)
    if not structural_rows:
        structural_rows = read_csv_rows(result_dir / "benchmark_summary_four_baselines.csv", required=False)
    if not structural_rows and summary_rows is not None:
        structural_rows = [r for r in summary_rows if str(r.get("baseline", "")) in {B2, B3, B4, B5, B6_LEGACY}]
    if not structural_rows:
        structural_rows = read_csv_rows(result_dir / "benchmark_summary.csv", required=False)
        structural_rows = [r for r in structural_rows if str(r.get("baseline", "")) in {B2, B3, B4, B5, B6_LEGACY}]
    return structural_rows


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    result_dir = Path(args.result_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else result_dir

    if args.ablation_only:
        structural_rows = load_structural_rows(result_dir)
        ablation_path = plot_structural_ablation(structural_rows, out_dir, args.dpi)
        if ablation_path is None:
            raise RuntimeError("未找到包含 B5_RegularLayered_LPA 的结构性消融结果。")
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
