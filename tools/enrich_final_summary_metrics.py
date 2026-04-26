"""补齐正式结果包中 E1/E2 绘图所需的 summary 统计字段。

该脚本只读取已完成的 single benchmark trial 记录，并把均值、标准差、
95% CI 等字段回填到 final_results 下的 summary CSV，不重新建图、
不重新规划，也不改变 trial 级原始记录。
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

B4 = "B4_Proposed_LPA_Layered"
B2 = "B2_GlobalAstar_Layered"
B3 = "B3_LPA_SingleLayer"
B5 = "B5_RegularLayered_LPA"
B6_LEGACY = "B6_RegularLayered_LPA"
B1 = "B1_Voxel_Dijkstra"

METHODS = {B4, B2, B3, B5, B6_LEGACY, B1}


def read_csv_rows(path: Path, required: bool = True) -> List[dict]:
    """读取 CSV，兼容 UTF-8 BOM。"""
    if not path.exists():
        if required:
            raise FileNotFoundError(f"缺少输入文件: {path}")
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    """写回 CSV，并保留稳定字段顺序。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_baseline(value: object) -> str:
    """兼容旧 B6 命名，统一映射为 M-R 使用的 B5 内部编号。"""
    baseline = str(value or "").strip()
    if baseline == B6_LEGACY:
        return B5
    return baseline


def to_float(value: object, default: float = float("nan")) -> float:
    """宽松转换数值字段。"""
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def is_success(row: Mapping[str, object]) -> bool:
    """判断 trial 是否成功，兼容 single 与 matrix trial 字段。"""
    if "success" in row:
        return str(row.get("success", "")).strip().lower() in {"1", "true", "yes", "y"}
    if "success_all_events" in row:
        return str(row.get("success_all_events", "")).strip().lower() in {"1", "true", "yes", "y"}
    return False


def finite_values(rows: Iterable[Mapping[str, object]], field: str) -> List[float]:
    """提取某个字段的有限数值。"""
    values: List[float] = []
    for row in rows:
        val = to_float(row.get(field))
        if math.isfinite(val):
            values.append(val)
    return values


def sample_std(values: Sequence[float]) -> float:
    """计算样本标准差；样本不足时返回 0。"""
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(max(var, 0.0))


def mean_std_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    """返回 mean/std/ci95，CI 使用 1.96 * std / sqrt(n)。"""
    if not values:
        return float("nan"), float("nan"), float("nan")
    mean = sum(values) / len(values)
    std = sample_std(values)
    ci95 = 1.96 * std / math.sqrt(len(values)) if len(values) > 0 else float("nan")
    return mean, std, ci95


def fmt(value: float) -> str:
    """CSV 中使用紧凑但可复现的浮点表示。"""
    if not math.isfinite(value):
        return ""
    return f"{value:.12g}"


def compute_scene_stats(trial_rows: Sequence[dict]) -> Dict[str, Dict[str, str]]:
    """按 baseline 从 trial 级记录计算 E1/E2 summary 字段。"""
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in trial_rows:
        baseline = normalize_baseline(row.get("baseline"))
        if baseline in METHODS:
            grouped[baseline].append(row)

    out: Dict[str, Dict[str, str]] = {}
    for baseline, rows in grouped.items():
        success_rows = [r for r in rows if is_success(r)]
        n_trials = len(rows)
        n_success = len(success_rows)
        stats: Dict[str, str] = {
            "n_trials": str(n_trials),
            "n_success": str(n_success),
            "n_failed": str(max(n_trials - n_success, 0)),
            "success_rate": fmt(n_success / n_trials) if n_trials else "",
        }

        metric_map = {
            "replan_ms": ("mean_replan_ms", "std_replan_ms", "ci95_replan_ms"),
            "path_cost": ("mean_path_cost", "std_path_cost", "ci95_path_cost"),
            "risk_exposure_integral": (
                "mean_risk_exposure",
                "std_risk_exposure",
                "ci95_risk_exposure",
            ),
            "comm_coverage_ratio": (
                "mean_comm_coverage",
                "std_comm_coverage",
                "ci95_comm_coverage",
            ),
        }
        for source, targets in metric_map.items():
            mean, std, ci95 = mean_std_ci(finite_values(success_rows, source))
            stats[targets[0]] = fmt(mean)
            stats[targets[1]] = fmt(std)
            stats[targets[2]] = fmt(ci95)

        stats["mean_cost"] = stats.get("mean_path_cost", "")
        stats["std_cost"] = stats.get("std_path_cost", "")
        stats["ci95_cost"] = stats.get("ci95_path_cost", "")
        stats["mean_risk_exposure_integral"] = stats.get("mean_risk_exposure", "")
        stats["std_risk_exposure_integral"] = stats.get("std_risk_exposure", "")
        stats["ci95_risk_exposure_integral"] = stats.get("ci95_risk_exposure", "")
        stats["mean_comm_coverage_ratio"] = stats.get("mean_comm_coverage", "")
        stats["std_comm_coverage_ratio"] = stats.get("std_comm_coverage", "")
        stats["ci95_comm_coverage_ratio"] = stats.get("ci95_comm_coverage", "")
        out[baseline] = stats
    return out


def baseline_from_row(row: Mapping[str, object]) -> str:
    """从 summary/structural row 中提取 baseline 内部编号。"""
    return normalize_baseline(row.get("baseline_id") or row.get("baseline"))


def append_fields(fieldnames: Sequence[str], extra: Iterable[str]) -> List[str]:
    """在保留原字段顺序的基础上追加缺失字段。"""
    ordered = list(fieldnames)
    seen = set(ordered)
    for name in extra:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def update_scene_csv(path: Path, stats_by_baseline: Mapping[str, Mapping[str, str]]) -> bool:
    """回填某个场景目录下的 benchmark summary CSV。"""
    rows = read_csv_rows(path, required=False)
    if not rows:
        return False
    original_fields = list(rows[0].keys())
    extra_fields = set()
    updated = False
    for row in rows:
        stats = stats_by_baseline.get(baseline_from_row(row))
        if not stats:
            continue
        row.update(stats)
        extra_fields.update(stats.keys())
        updated = True
    if updated:
        write_csv_rows(path, rows, append_fields(original_fields, sorted(extra_fields)))
    return updated


def collect_scene_stats(final_dir: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    """遍历 final_results/<scene>/E1_E2_single_final 并回填 per-scene CSV。"""
    collected: Dict[Tuple[str, str], Dict[str, str]] = {}
    for scene_dir in sorted(p for p in final_dir.iterdir() if p.is_dir() and not p.name.startswith("_") and p.name != "logs"):
        result_dir = scene_dir / "E1_E2_single_final"
        trial_rows = read_csv_rows(result_dir / "benchmark_trials.csv", required=False)
        if not trial_rows:
            continue
        stats_by_baseline = compute_scene_stats(trial_rows)
        for csv_name in ("benchmark_summary.csv", "benchmark_structural_ablation.csv"):
            update_scene_csv(result_dir / csv_name, stats_by_baseline)
        for baseline, stats in stats_by_baseline.items():
            collected[(scene_dir.name, baseline)] = stats
    return collected


def update_global_summary(final_dir: Path, scene_stats: Mapping[Tuple[str, str], Mapping[str, str]]) -> bool:
    """回填 final_results/_summaries 下的 E1/E2 三山汇总 CSV。"""
    summary_path = final_dir / "_summaries" / "E1_E2_three_mountain_single_final.csv"
    rows = read_csv_rows(summary_path, required=False)
    if not rows:
        return False
    original_fields = list(rows[0].keys())
    extra_fields = set()
    updated = False
    for row in rows:
        scene = str(row.get("scene_name", "")).strip()
        baseline = baseline_from_row(row)
        stats = scene_stats.get((scene, baseline))
        if not stats:
            continue
        row.update(stats)
        extra_fields.update(stats.keys())
        updated = True
    if updated:
        write_csv_rows(summary_path, rows, append_fields(original_fields, sorted(extra_fields)))
    return updated


def combo_from_trial(row: Mapping[str, object]) -> Tuple[str, int, int, str]:
    """提取 matrix trial 的组合键：scale/intensity/K/baseline。"""
    scale = str(row.get("scale", "")).strip()
    intensity = int(round(to_float(row.get("intensity_index", row.get("n_block")), 0.0)))
    k_events = int(round(to_float(row.get("k_events"), 0.0)))
    baseline = normalize_baseline(row.get("baseline"))
    return scale, intensity, k_events, baseline


def combo_from_summary(row: Mapping[str, object]) -> Tuple[str, int, int, str]:
    """提取 matrix summary 的组合键：scale/intensity/K/baseline。"""
    scale = str(row.get("scale", "")).strip()
    intensity = int(round(to_float(row.get("intensity_index", row.get("n_block")), 0.0)))
    k_events = int(round(to_float(row.get("k_events"), 0.0)))
    baseline = baseline_from_row(row)
    return scale, intensity, k_events, baseline


def compute_matrix_stats(trial_rows: Sequence[dict]) -> Dict[Tuple[str, int, int, str], Dict[str, str]]:
    """从 matrix trial 级记录计算每个组合的 summary 字段。"""
    grouped: Dict[Tuple[str, int, int, str], List[dict]] = defaultdict(list)
    for row in trial_rows:
        key = combo_from_trial(row)
        if key[3] in METHODS:
            grouped[key].append(row)

    out: Dict[Tuple[str, int, int, str], Dict[str, str]] = {}
    for key, rows in grouped.items():
        success_rows = [r for r in rows if is_success(r)]
        n_trials = len(rows)
        n_success = len(success_rows)
        stats: Dict[str, str] = {
            "n_trials": str(n_trials),
            "n_success": str(n_success),
            "n_failed": str(max(n_trials - n_success, 0)),
            "success_rate": fmt(n_success / n_trials) if n_trials else "",
        }

        metric_map = {
            "cumulative_replan_ms": ("mean_replan_ms", "std_replan_ms", "ci95_replan_ms"),
            "mean_event_replan_ms": (
                "mean_event_replan_ms",
                "std_event_replan_ms",
                "ci95_event_replan_ms",
            ),
            "final_path_cost": ("mean_path_cost", "std_path_cost", "ci95_path_cost"),
            "final_risk_exposure_integral": (
                "mean_risk_exposure",
                "std_risk_exposure",
                "ci95_risk_exposure",
            ),
            "final_comm_coverage_ratio": (
                "mean_comm_coverage",
                "std_comm_coverage",
                "ci95_comm_coverage",
            ),
        }
        for source, targets in metric_map.items():
            mean, std, ci95 = mean_std_ci(finite_values(success_rows, source))
            stats[targets[0]] = fmt(mean)
            stats[targets[1]] = fmt(std)
            stats[targets[2]] = fmt(ci95)

        stats["mean_cumulative_replan_ms"] = stats.get("mean_replan_ms", "")
        stats["std_cumulative_replan_ms"] = stats.get("std_replan_ms", "")
        stats["ci95_cumulative_replan_ms"] = stats.get("ci95_replan_ms", "")
        stats["mean_final_path_cost"] = stats.get("mean_path_cost", "")
        stats["std_final_path_cost"] = stats.get("std_path_cost", "")
        stats["ci95_final_path_cost"] = stats.get("ci95_path_cost", "")
        stats["mean_final_risk_exposure_integral"] = stats.get("mean_risk_exposure", "")
        stats["std_final_risk_exposure_integral"] = stats.get("std_risk_exposure", "")
        stats["ci95_final_risk_exposure_integral"] = stats.get("ci95_risk_exposure", "")
        stats["mean_final_comm_coverage_ratio"] = stats.get("mean_comm_coverage", "")
        stats["std_final_comm_coverage_ratio"] = stats.get("std_comm_coverage", "")
        stats["ci95_final_comm_coverage_ratio"] = stats.get("ci95_comm_coverage", "")
        out[key] = stats
    return out


def update_matrix_scene_csv(path: Path, stats_by_combo: Mapping[Tuple[str, int, int, str], Mapping[str, str]]) -> bool:
    """回填单场景 E3/E4 matrix benchmark_summary.csv。"""
    rows = read_csv_rows(path, required=False)
    if not rows:
        return False
    original_fields = list(rows[0].keys())
    extra_fields = set()
    updated = False
    for row in rows:
        stats = stats_by_combo.get(combo_from_summary(row))
        if not stats:
            continue
        row.update(stats)
        extra_fields.update(stats.keys())
        updated = True
    if updated:
        write_csv_rows(path, rows, append_fields(original_fields, sorted(extra_fields)))
    return updated


def collect_matrix_stats(final_dir: Path) -> Dict[Tuple[str, str, int, int, str], Dict[str, str]]:
    """遍历 final_results/<scene>/E3_E4_matrix_final 并回填 matrix summary。"""
    collected: Dict[Tuple[str, str, int, int, str], Dict[str, str]] = {}
    for scene_dir in sorted(p for p in final_dir.iterdir() if p.is_dir() and not p.name.startswith("_") and p.name != "logs"):
        result_dir = scene_dir / "E3_E4_matrix_final"
        trial_rows = read_csv_rows(result_dir / "benchmark_trials.csv", required=False)
        if not trial_rows:
            continue
        stats_by_combo = compute_matrix_stats(trial_rows)
        update_matrix_scene_csv(result_dir / "benchmark_summary.csv", stats_by_combo)
        for (scale, intensity, k_events, baseline), stats in stats_by_combo.items():
            collected[(scene_dir.name, scale, intensity, k_events, baseline)] = stats
    return collected


def update_matrix_global_summary(
    final_dir: Path,
    matrix_stats: Mapping[Tuple[str, str, int, int, str], Mapping[str, str]],
) -> bool:
    """回填 final_results/_summaries 下的 E3/E4 三山矩阵汇总 CSV。"""
    summary_path = final_dir / "_summaries" / "E3_E4_three_mountain_matrix_final.csv"
    rows = read_csv_rows(summary_path, required=False)
    if not rows:
        return False
    original_fields = list(rows[0].keys())
    extra_fields = set()
    updated = False
    for row in rows:
        scene = str(row.get("scene_name", "")).strip()
        scale, intensity, k_events, baseline = combo_from_summary(row)
        stats = matrix_stats.get((scene, scale, intensity, k_events, baseline))
        if not stats:
            continue
        row.update(stats)
        extra_fields.update(stats.keys())
        updated = True
    if updated:
        write_csv_rows(summary_path, rows, append_fields(original_fields, sorted(extra_fields)))
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="补齐 final_results 中 E1/E2 summary 的 CI 与风险/通信字段。")
    parser.add_argument("--final-dir", type=str, default="final_results", help="正式结果根目录。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_dir = Path(args.final_dir).resolve()
    scene_stats = collect_scene_stats(final_dir)
    update_global = update_global_summary(final_dir, scene_stats)
    matrix_stats = collect_matrix_stats(final_dir)
    update_matrix_global = update_matrix_global_summary(final_dir, matrix_stats)
    print("[done] E1/E2 summary 统计字段已补齐：")
    print(f"  - 场景/方法组合数: {len(scene_stats)}")
    print(f"  - 全局 summary 更新: {'yes' if update_global else 'no'}")
    print(f"  - E3/E4 matrix 组合数: {len(matrix_stats)}")
    print(f"  - E3/E4 全局 summary 更新: {'yes' if update_matrix_global else 'no'}")


if __name__ == "__main__":
    main()
