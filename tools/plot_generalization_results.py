"""绘制 E1 跨地形泛化实验图。

该脚本读取 run_multi_scene.py 汇总出的多场景 single benchmark CSV，
不重新建图、不重新规划。若汇总 CSV 缺少风险暴露字段，会尝试从每行
`benchmark_out_dir/benchmark_summary.csv` 中补读对应方法的风险暴露指标。
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
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
B1 = "B1_Voxel_Dijkstra"

METHOD_ORDER = [B4, B2, B3, B5, B1]
METHOD_IDS = {
    B4: "M-P",
    B2: "M-A",
    B3: "M-F",
    B5: "M-R",
    B6_LEGACY: "M-R",
    B1: "M-V",
}
METHOD_LABELS = {
    B4: "Terrain-aware Layered LPA* (Proposed)",
    B2: "Terrain-aware Layered A*",
    B3: "Flat-graph LPA*",
    B5: "Regular-layered LPA*",
    B6_LEGACY: "Regular-layered LPA*",
    B1: "Voxel Global Search",
}
METHOD_COLORS = {
    B4: "#d95f02",
    B2: "#1f78b4",
    B3: "#7570b3",
    B5: "#66a61e",
    B1: "#8c510a",
}

SCENE_ORDER = ["huashan", "huangshan", "emeishan"]
SCENE_LABELS = {
    "huashan": "Huashan",
    "huangshan": "Huangshan",
    "emeishan": "Emeishan",
}

METRIC_CANDIDATES = {
    "success_rate": ["success_rate"],
    "mean_replan_ms": ["mean_replan_ms", "mean_cumulative_replan_ms"],
    "mean_comm_coverage": ["mean_comm_coverage", "mean_comm_coverage_ratio", "mean_final_comm_coverage_ratio"],
    "mean_risk_exposure": [
        "mean_risk_exposure",
        "mean_risk_exposure_integral",
        "mean_final_risk_exposure_integral",
        "risk_exposure_integral",
    ],
}

CI95_CANDIDATES = {
    "mean_replan_ms": ["ci95_replan_ms", "ci95_cumulative_replan_ms"],
    "mean_comm_coverage": ["ci95_comm_coverage", "ci95_comm_coverage_ratio"],
    "mean_risk_exposure": ["ci95_risk_exposure", "ci95_risk_exposure_integral"],
}


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


def read_csv_rows(path: Path, required: bool = True) -> List[dict]:
    """读取 CSV，兼容带 BOM 的 UTF-8 文件。"""
    if not path.exists():
        if required:
            raise FileNotFoundError(f"缺少输入文件: {path}")
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def to_float(value: object, default: float = float("nan")) -> float:
    """宽松转换数值字段。"""
    if value is None:
        return default
    s = str(value).strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def normalize_baseline(value: object) -> str:
    """兼容旧 B6 命名，统一映射到 M-R 使用的 B5 内部编号。"""
    baseline = str(value or "").strip()
    if baseline == B6_LEGACY:
        return B5
    return baseline


def first_metric(row: dict, metric: str) -> float:
    """按候选字段读取指标，兼容 single 与 matrix 汇总字段差异。"""
    for key in METRIC_CANDIDATES.get(metric, [metric]):
        if key in row:
            val = to_float(row.get(key))
            if math.isfinite(val):
                return val
    return float("nan")


def first_ci95(row: dict, metric: str) -> float:
    """读取指标对应的 95% CI 字段。"""
    for key in CI95_CANDIDATES.get(metric, []):
        if key in row:
            val = to_float(row.get(key))
            if math.isfinite(val):
                return val
    return float("nan")


def pretty_scene(scene: str) -> str:
    """将场景内部名转成论文图上的英文场景名。"""
    s = str(scene or "").strip()
    key = s.lower()
    if key in SCENE_LABELS:
        return SCENE_LABELS[key]
    return s.replace("_", " ").title()


def method_label(baseline: str) -> str:
    """生成图中使用的短 Method ID 标签。"""
    bid = normalize_baseline(baseline)
    return METHOD_IDS.get(bid, bid)


def load_benchmark_summary_cache(rows: Sequence[dict]) -> Dict[Path, List[dict]]:
    """缓存各场景 single benchmark 输出目录下的 benchmark_summary.csv。"""
    cache: Dict[Path, List[dict]] = {}
    for row in rows:
        raw = str(row.get("benchmark_out_dir", "")).strip()
        if not raw:
            continue
        path = Path(raw) / "benchmark_summary.csv"
        if path not in cache:
            cache[path] = read_csv_rows(path, required=False)
    return cache


def enrich_missing_metrics(rows: List[dict]) -> None:
    """从原始 benchmark_summary.csv 回填多场景汇总中缺失的指标。"""
    cache = load_benchmark_summary_cache(rows)
    for row in rows:
        baseline = normalize_baseline(row.get("baseline"))
        raw = str(row.get("benchmark_out_dir", "")).strip()
        if not baseline or not raw:
            continue
        source_rows = cache.get(Path(raw) / "benchmark_summary.csv", [])
        source = next((r for r in source_rows if normalize_baseline(r.get("baseline")) == baseline), None)
        if source is None:
            continue
        for metric, keys in METRIC_CANDIDATES.items():
            if math.isfinite(first_metric(row, metric)):
                continue
            for key in keys:
                if key in source and str(source.get(key, "")).strip():
                    row[key] = source.get(key, "")
                    break


def finite_mean(values: Iterable[float]) -> float:
    """只对有限值求均值，避免空组触发 NumPy 警告。"""
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def aggregate_rows(rows: Sequence[dict]) -> List[dict]:
    """按 scene_name 和 baseline 聚合多场景汇总行。"""
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in rows:
        if str(row.get("status", "ok")).strip().lower() not in {"", "ok"}:
            continue
        baseline = normalize_baseline(row.get("baseline"))
        scene = str(row.get("scene_name", "")).strip()
        if baseline not in set(METHOD_ORDER) or not scene:
            continue
        groups[(scene, baseline)].append(row)

    out: List[dict] = []
    for (scene, baseline), items in groups.items():
        n_trials = sum(to_float(r.get("n_trials"), 0.0) for r in items)
        n_success = sum(to_float(r.get("n_success"), 0.0) for r in items)
        if n_trials > 0:
            success_rate = float(n_success / n_trials)
        else:
            success_rate = finite_mean(first_metric(r, "success_rate") for r in items)
        out.append(
            {
                "scene_name": scene,
                "baseline": baseline,
                "success_rate": success_rate,
                "mean_replan_ms": finite_mean(first_metric(r, "mean_replan_ms") for r in items),
                "ci95_replan_ms": finite_mean(first_ci95(r, "mean_replan_ms") for r in items),
                "mean_comm_coverage": finite_mean(first_metric(r, "mean_comm_coverage") for r in items),
                "ci95_comm_coverage": finite_mean(first_ci95(r, "mean_comm_coverage") for r in items),
                "mean_risk_exposure": finite_mean(first_metric(r, "mean_risk_exposure") for r in items),
                "ci95_risk_exposure": finite_mean(first_ci95(r, "mean_risk_exposure") for r in items),
            }
        )
    return out


def ordered_scenes(rows: Sequence[dict]) -> List[str]:
    """按论文推荐顺序排列场景，其他场景追加到末尾。"""
    present = {str(r.get("scene_name", "")) for r in rows}
    ordered = [s for s in SCENE_ORDER if s in present]
    ordered.extend(sorted(s for s in present if s and s not in set(ordered)))
    return ordered


def save_figure(fig: plt.Figure, out_dir: Path, filename: str, dpi: int) -> Path:
    """保存 PDF 图并关闭 figure。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return path


def plot_grouped_metric(
    rows: Sequence[dict],
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    out_dir: Path,
    dpi: int,
    percent: bool = False,
    ci_metric: str | None = None,
) -> Path:
    """按场景分组、按方法着色绘制 E1 泛化柱状图。"""
    scenes = ordered_scenes(rows)
    methods = [m for m in METHOD_ORDER if any(r["baseline"] == m for r in rows)]
    data = {(r["scene_name"], r["baseline"]): float(r.get(metric, float("nan"))) for r in rows}

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    if not scenes or not methods:
        ax.text(0.5, 0.5, "No valid E1 rows found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return save_figure(fig, out_dir, filename, dpi)

    x = np.arange(len(scenes), dtype=float)
    width = min(0.16, 0.82 / max(1, len(methods)))
    offset0 = -0.5 * width * (len(methods) - 1)
    has_finite = False
    for i, method in enumerate(methods):
        values = []
        yerr = []
        for scene in scenes:
            value = data.get((scene, method), float("nan"))
            ci_value = float("nan")
            if ci_metric:
                row = next((r for r in rows if r["scene_name"] == scene and r["baseline"] == method), None)
                if row is not None:
                    ci_value = to_float(row.get(ci_metric))
            if percent and math.isfinite(value):
                value *= 100.0
            if percent and math.isfinite(ci_value):
                ci_value *= 100.0
            values.append(value)
            yerr.append(ci_value)
        has_finite = has_finite or any(math.isfinite(v) for v in values)
        finite_yerr = any(math.isfinite(v) and v > 0 for v in yerr)
        ax.bar(
            x + offset0 + i * width,
            values,
            width=width,
            color=METHOD_COLORS.get(method, "#777777"),
            alpha=0.86,
            label=method_label(method),
            yerr=yerr if finite_yerr else None,
            capsize=2.0 if finite_yerr else 0.0,
            error_kw={"linewidth": 0.8, "capthick": 0.8},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([pretty_scene(s) for s in scenes])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if percent:
        ax.set_ylim(0, 105)
    if not has_finite:
        ax.text(0.5, 0.86, "Metric not found in summary CSV", ha="center", transform=ax.transAxes, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7.2, ncol=len(methods), loc="upper center", bbox_to_anchor=(0.5, 1.20))
    return save_figure(fig, out_dir, filename, dpi)


def draw_grouped_metric(
    ax: plt.Axes,
    rows: Sequence[dict],
    metric: str,
    ylabel: str,
    title: str,
    percent: bool = False,
    ci_metric: str | None = None,
) -> None:
    """在指定子图上绘制分组柱状指标，用于 E1 综合图。"""
    scenes = ordered_scenes(rows)
    methods = [m for m in METHOD_ORDER if any(r["baseline"] == m for r in rows)]
    data = {(r["scene_name"], r["baseline"]): float(r.get(metric, float("nan"))) for r in rows}
    if not scenes or not methods:
        ax.text(0.5, 0.5, "No valid E1 rows found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    x = np.arange(len(scenes), dtype=float)
    width = min(0.16, 0.82 / max(1, len(methods)))
    offset0 = -0.5 * width * (len(methods) - 1)
    has_finite = False
    for i, method in enumerate(methods):
        values = []
        yerr = []
        for scene in scenes:
            value = data.get((scene, method), float("nan"))
            ci_value = float("nan")
            if ci_metric:
                row = next((r for r in rows if r["scene_name"] == scene and r["baseline"] == method), None)
                if row is not None:
                    ci_value = to_float(row.get(ci_metric))
            if percent and math.isfinite(value):
                value *= 100.0
            if percent and math.isfinite(ci_value):
                ci_value *= 100.0
            values.append(value)
            yerr.append(ci_value)
        has_finite = has_finite or any(math.isfinite(v) for v in values)
        finite_yerr = any(math.isfinite(v) and v > 0 for v in yerr)
        ax.bar(
            x + offset0 + i * width,
            values,
            width=width,
            color=METHOD_COLORS.get(method, "#777777"),
            alpha=0.86,
            label=method_label(method),
            yerr=yerr if finite_yerr else None,
            capsize=2.0 if finite_yerr else 0.0,
            error_kw={"linewidth": 0.8, "capthick": 0.8},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([pretty_scene(s) for s in scenes])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if percent:
        ax.set_ylim(0, 105)
    if not has_finite:
        ax.text(0.5, 0.86, "Metric not found in summary CSV", ha="center", transform=ax.transAxes, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_overall_dashboard(rows: Sequence[dict], out_dir: Path, dpi: int) -> Path:
    """绘制 E1 跨地形泛化综合 2x2 图。"""
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.8))
    axes_flat = list(axes.ravel())
    draw_grouped_metric(
        axes_flat[0],
        rows,
        "success_rate",
        "Success rate (%)",
        "(a) Success rate",
        percent=True,
    )
    draw_grouped_metric(
        axes_flat[1],
        rows,
        "mean_replan_ms",
        "Mean replanning time (ms)",
        "(b) Replanning time",
        ci_metric="ci95_replan_ms",
    )
    draw_grouped_metric(
        axes_flat[2],
        rows,
        "mean_comm_coverage",
        "Communication coverage (%)",
        "(c) Communication coverage",
        percent=True,
        ci_metric="ci95_comm_coverage",
    )
    draw_grouped_metric(
        axes_flat[3],
        rows,
        "mean_risk_exposure",
        "Risk exposure integral",
        "(d) Risk exposure",
        ci_metric="ci95_risk_exposure",
    )
    fig.suptitle("E1 Cross-terrain Generalization", y=1.02, fontsize=11)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(labels), loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(hspace=0.42, wspace=0.28, bottom=0.13)
    return save_figure(fig, out_dir, "fig_E1_cross_terrain_overall.pdf", dpi)


def resolve_summary_path(raw: str, workdir: Path) -> Path:
    """解析输入汇总 CSV；默认正式文件不存在时回退到 multi_scene_summary.csv。"""
    path = Path(raw)
    if not path.is_absolute():
        path = workdir / path
    if path.exists():
        return path
    fallback = workdir / "outputs" / "_summaries" / "multi_scene_summary.csv"
    if path.name == "E1_E2_three_mountain_single_final.csv" and fallback.exists():
        print(f"[warn] 未找到 {path}，回退读取 {fallback}")
        return fallback
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制 E1 Cross-terrain Generalization 论文图。")
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="outputs/_summaries/E1_E2_three_mountain_single_final.csv",
        help="run_multi_scene.py 汇总 CSV。",
    )
    parser.add_argument("--workdir", type=str, default=".", help="项目根目录。")
    parser.add_argument("--out-dir", type=str, default="", help="图输出目录；默认写到 summary-csv 所在目录。")
    parser.add_argument("--dpi", type=int, default=300, help="PDF 中嵌入栅格元素的分辨率。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    workdir = Path(args.workdir).resolve()
    summary_path = resolve_summary_path(args.summary_csv, workdir)
    rows = read_csv_rows(summary_path, required=True)
    enrich_missing_metrics(rows)
    aggregated = aggregate_rows(rows)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else summary_path.parent

    produced = [
        plot_overall_dashboard(aggregated, out_dir, args.dpi),
        plot_grouped_metric(
            aggregated,
            "success_rate",
            "Success rate (%)",
            "E1 Cross-terrain Success Rate",
            "fig_E1_cross_terrain_success.pdf",
            out_dir,
            args.dpi,
            percent=True,
        ),
        plot_grouped_metric(
            aggregated,
            "mean_replan_ms",
            "Mean replanning time (ms)",
            "E1 Cross-terrain Replanning Time",
            "fig_E1_cross_terrain_replan_time.pdf",
            out_dir,
            args.dpi,
            ci_metric="ci95_replan_ms",
        ),
        plot_grouped_metric(
            aggregated,
            "mean_comm_coverage",
            "Communication coverage (%)",
            "E1 Cross-terrain Communication Coverage",
            "fig_E1_cross_terrain_comm_coverage.pdf",
            out_dir,
            args.dpi,
            percent=True,
            ci_metric="ci95_comm_coverage",
        ),
        plot_grouped_metric(
            aggregated,
            "mean_risk_exposure",
            "Risk exposure integral",
            "E1 Cross-terrain Risk Exposure",
            "fig_E1_cross_terrain_risk_exposure.pdf",
            out_dir,
            args.dpi,
            ci_metric="ci95_risk_exposure",
        ),
    ]

    print("[done] E1 跨地形泛化图已生成：")
    for path in produced:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
