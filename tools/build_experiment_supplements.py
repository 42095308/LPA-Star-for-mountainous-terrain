from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats


SCENES = ["huashan", "huangshan", "emeishan"]
METHOD_ORDER = ["MP", "MA", "MF", "MR", "MV", "BI-APF-RRT*", "GWO-DE"]
BASELINE_TO_METHOD = {
    "B4_Proposed_LPA_Layered": "MP",
    "B2_GlobalAstar_Layered": "MA",
    "B3_LPA_SingleLayer": "MF",
    "B5_RegularLayered_LPA": "MR",
    "B1_Voxel_Dijkstra": "MV",
    "BI_APF_RRT_STAR": "BI-APF-RRT*",
    "GWO_DE": "GWO-DE",
}
MAIN_METRICS = [
    ("replan_ms", "重规划时间_ms", "lower"),
    ("expanded", "扩展节点数", "lower"),
    ("path_cost", "路径代价", "lower"),
    ("path_len_km", "路径长度_km", "lower"),
    ("risk_exposure_integral", "风险暴露", "lower"),
    ("comm_coverage_ratio", "通信覆盖率", "higher"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(value: object) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean_std_ci(values: Iterable[float]) -> tuple[float, float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    ci = float(1.96 * std / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, std, ci


def fmt_num(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def holm_symbols(p_values: list[float]) -> list[str]:
    indexed = [(i, p) for i, p in enumerate(p_values) if np.isfinite(p)]
    indexed.sort(key=lambda x: x[1])
    significant = [False] * len(p_values)
    m = len(indexed)
    for rank, (idx, p_value) in enumerate(indexed):
        threshold = 0.05 / max(m - rank, 1)
        if p_value <= threshold:
            significant[idx] = True
        else:
            break
    symbols = []
    for i, p_value in enumerate(p_values):
        if not np.isfinite(p_value):
            symbols.append("")
        elif significant[i] and p_value < 0.001:
            symbols.append("***")
        elif significant[i] and p_value < 0.01:
            symbols.append("**")
        elif significant[i] and p_value < 0.05:
            symbols.append("*")
        else:
            symbols.append("ns")
    return symbols


def load_main_trials(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for scene in SCENES:
        path = root / "final_results" / scene / "E1_E2_single_final" / "benchmark_trials.csv"
        for row in read_csv(path):
            row = dict(row)
            row["scene"] = scene
            row["method"] = BASELINE_TO_METHOD.get(row.get("baseline", ""), row.get("baseline", ""))
            rows.append(row)
    return rows


def load_external_trials(root: Path) -> list[dict[str, str]]:
    path = root / "final_results" / "external_baselines" / "external_baseline_trials.csv"
    rows: list[dict[str, str]] = []
    for row in read_csv(path):
        row = dict(row)
        row["scene"] = str(row.get("scene", "")).lower()
        row["method"] = BASELINE_TO_METHOD.get(row.get("baseline", ""), row.get("method", ""))
        rows.append(row)
    return rows


def build_statistical_stability(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if str(row.get("success", "True")).lower() not in {"true", "1"}:
            continue
        grouped[(row.get("scene", ""), row.get("method", ""))].append(row)

    for scene in SCENES:
        for method in METHOD_ORDER:
            subset = grouped.get((scene, method), [])
            if not subset:
                continue
            for metric, label, direction in MAIN_METRICS:
                values = [to_float(r.get(metric)) for r in subset]
                mean, std, ci = mean_std_ci(values)
                out.append(
                    {
                        "场景": scene,
                        "方法": method,
                        "指标": label,
                        "方向": direction,
                        "样本数": len([v for v in values if np.isfinite(v)]),
                        "均值": fmt_num(mean),
                        "标准差": fmt_num(std),
                        "95CI": fmt_num(ci),
                        "均值±标准差": f"{fmt_num(mean)} ± {fmt_num(std)}" if np.isfinite(mean) else "",
                    }
                )
    return out


def paired_rows(rows: list[dict[str, str]], scene: str, metric: str, method_a: str, method_b: str) -> tuple[np.ndarray, np.ndarray]:
    by_key: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.get("scene") != scene:
            continue
        if str(row.get("success", "True")).lower() not in {"true", "1"}:
            continue
        method = row.get("method", "")
        if method not in {method_a, method_b}:
            continue
        key = (str(row.get("trial", "")), str(row.get("task_id", "")))
        value = to_float(row.get(metric))
        if np.isfinite(value):
            by_key[key][method] = value
    a_values: list[float] = []
    b_values: list[float] = []
    for values in by_key.values():
        if method_a in values and method_b in values:
            a_values.append(values[method_a])
            b_values.append(values[method_b])
    return np.asarray(a_values, dtype=float), np.asarray(b_values, dtype=float)


def build_significance(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    comparisons = [m for m in METHOD_ORDER if m != "MP"]
    pending_symbol_indices: list[int] = []
    pending_p_values: list[float] = []

    for scene in SCENES:
        for metric, label, direction in MAIN_METRICS:
            for method in comparisons:
                mp, other = paired_rows(rows, scene, metric, "MP", method)
                if mp.size < 3:
                    continue
                diff = mp - other
                try:
                    p_value = float(stats.wilcoxon(mp, other, zero_method="wilcox").pvalue)
                    test_name = "Wilcoxon"
                except ValueError:
                    p_value = 1.0 if np.allclose(diff, 0.0) else float("nan")
                    test_name = "Wilcoxon"
                row = {
                    "场景": scene,
                    "指标": label,
                    "方向": direction,
                    "比较": f"MP vs {method}",
                    "检验": test_name,
                    "配对样本数": int(mp.size),
                    "MP均值": fmt_num(float(np.mean(mp))),
                    "对照均值": fmt_num(float(np.mean(other))),
                    "p值": fmt_num(p_value, 6),
                    "Holm显著性": "",
                }
                pending_symbol_indices.append(len(out))
                pending_p_values.append(p_value)
                out.append(row)

            matrix: list[np.ndarray] = []
            valid_methods: list[str] = []
            for method in METHOD_ORDER:
                values = []
                for row in rows:
                    if row.get("scene") == scene and row.get("method") == method and str(row.get("success", "True")).lower() in {"true", "1"}:
                        value = to_float(row.get(metric))
                        if np.isfinite(value):
                            values.append(value)
                if len(values) >= 3:
                    valid_methods.append(method)
                    matrix.append(np.asarray(values[: min(len(v) for v in matrix + [np.asarray(values)])], dtype=float))
            if len(valid_methods) >= 3:
                n = min(len(v) for v in matrix)
                try:
                    p_value = float(stats.friedmanchisquare(*[v[:n] for v in matrix]).pvalue)
                except ValueError:
                    p_value = float("nan")
                out.append(
                    {
                        "场景": scene,
                        "指标": label,
                        "方向": direction,
                        "比较": " / ".join(valid_methods),
                        "检验": "Friedman",
                        "配对样本数": int(n),
                        "MP均值": "",
                        "对照均值": "",
                        "p值": fmt_num(p_value, 6),
                        "Holm显著性": "",
                    }
                )

    symbols = holm_symbols(pending_p_values)
    for idx, symbol in zip(pending_symbol_indices, symbols):
        out[idx]["Holm显著性"] = symbol
    return out


def build_failure_summary(root: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for scene in SCENES:
        path = root / "final_results" / scene / "E3_E4_matrix_final" / "benchmark_failure_reasons.csv"
        rows = read_csv(path)
        grouped: Counter[tuple[str, str, str]] = Counter()
        meta: dict[tuple[str, str, str], dict[str, str]] = {}
        for row in rows:
            key = (row.get("scale", ""), row.get("stage", ""), row.get("failure_reason", ""))
            grouped[key] += int(float(row.get("count", "0") or 0))
            meta[key] = row
        for (scale, stage, reason), count in sorted(grouped.items()):
            row = meta[(scale, stage, reason)]
            out.append(
                {
                    "场景": scene,
                    "图规模": scale,
                    "阶段": stage,
                    "失败原因": reason,
                    "次数": count,
                    "节点数": row.get("graph_nodes", ""),
                    "边数": row.get("graph_edges", ""),
                    "解释": explain_failure(scale, stage, reason),
                }
            )
    return out


def explain_failure(scale: str, stage: str, reason: str) -> str:
    if scale == "small" and reason in {"event_path_disconnected", "small_graph_task_unreachable", "start_goal_not_connected"}:
        return "图规模过小削弱任务连通性或事件后绕行空间，不解释为搜索算法本身失效"
    if reason == "start_goal_not_connected":
        return "候选任务端点在当前图规模下未形成可用连通路径"
    if reason == "event_path_disconnected":
        return "区域事件阻断后缺少可行替代边"
    if reason == "event_schedule_unavailable":
        return "事件序列无法构造足够可用扰动"
    return "按实验日志保留原始失败标签"


def build_external_table(root: Path) -> list[dict[str, object]]:
    rows = read_csv(root / "final_results" / "external_baselines" / "external_baseline_summary.csv")
    out: list[dict[str, object]] = []
    for row in rows:
        method = BASELINE_TO_METHOD.get(row.get("baseline", ""), row.get("method", ""))
        out.append(
            {
                "场景": row.get("scene", ""),
                "方法": method,
                "成功率": row.get("success_rate", ""),
                "规划时间_ms_均值±标准差": f"{row.get('mean_replan_ms', '')} ± {row.get('std_replan_ms', '')}",
                "路径代价": row.get("mean_path_cost", ""),
                "路径长度_km": row.get("mean_path_len_km", ""),
                "通信覆盖率": row.get("mean_comm_coverage_ratio", ""),
                "风险暴露": row.get("mean_risk_exposure_integral", ""),
                "采样或评估次数": row.get("mean_work_units", ""),
            }
        )
    return out


def build_external_significance(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    p_values: list[float] = []
    row_indices: list[int] = []
    for scene in SCENES:
        for metric, label, direction in MAIN_METRICS:
            for method in ["BI-APF-RRT*", "GWO-DE"]:
                mp, other = paired_rows(rows, scene, metric, "MP", method)
                if mp.size < 3:
                    continue
                try:
                    p_value = float(stats.wilcoxon(mp, other, zero_method="wilcox").pvalue)
                except ValueError:
                    p_value = 1.0 if np.allclose(mp - other, 0.0) else float("nan")
                row_indices.append(len(out))
                p_values.append(p_value)
                out.append(
                    {
                        "场景": scene,
                        "指标": label,
                        "方向": direction,
                        "比较": f"MP vs {method}",
                        "检验": "Wilcoxon",
                        "配对样本数": int(mp.size),
                        "MP均值": fmt_num(float(np.mean(mp))),
                        "外部基线均值": fmt_num(float(np.mean(other))),
                        "p值": fmt_num(p_value, 6),
                        "Holm显著性": "",
                    }
                )
    for idx, symbol in zip(row_indices, holm_symbols(p_values)):
        out[idx]["Holm显著性"] = symbol
    return out


def build_augmented_figure_source(root: Path, all_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        if str(row.get("success", "True")).lower() in {"true", "1"}:
            grouped[(row.get("scene", ""), row.get("method", ""))].append(row)
    scene_cn = {"huashan": "华山", "huangshan": "黄山", "emeishan": "峨眉山"}
    for scene in SCENES:
        for method in METHOD_ORDER:
            subset = grouped.get((scene, method), [])
            if not subset:
                continue
            def metric_mean(key: str) -> float:
                return mean_std_ci(to_float(r.get(key)) for r in subset)[0]
            out.append(
                {
                    "场景": scene_cn.get(scene, scene),
                    "方法": method,
                    "成功率": fmt_num(len(subset) / max(1, len([r for r in all_rows if r.get("scene") == scene and r.get("method") == method]))),
                    "重规划时间_ms": fmt_num(metric_mean("replan_ms"), 2),
                    "路径代价": fmt_num(metric_mean("path_cost"), 3),
                    "长度_km": fmt_num(metric_mean("path_len_km"), 3),
                    "通信覆盖率": fmt_num(metric_mean("comm_coverage_ratio"), 3),
                    "风险暴露": fmt_num(metric_mean("risk_exposure_integral"), 3),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 ESWA 实验补充统计表和源数据。")
    parser.add_argument("--workdir", type=str, default=".", help="项目根目录。")
    args = parser.parse_args()
    root = Path(args.workdir).resolve()

    main_rows = load_main_trials(root)
    external_rows = load_external_trials(root)
    all_rows = main_rows + external_rows

    table_dir = root / "final_results" / "paper_revision" / "tables"
    source_dir = root / "final_results" / "paper_revision" / "source_data" / "chapter4_python"

    stability = build_statistical_stability(all_rows)
    write_csv(
        table_dir / "table_S1_statistical_stability.csv",
        stability,
        ["场景", "方法", "指标", "方向", "样本数", "均值", "标准差", "95CI", "均值±标准差"],
    )

    significance = build_significance(all_rows)
    write_csv(
        table_dir / "table_S2_significance_tests.csv",
        significance,
        ["场景", "指标", "方向", "比较", "检验", "配对样本数", "MP均值", "对照均值", "p值", "Holm显著性"],
    )

    external_significance = build_external_significance(all_rows)
    write_csv(
        root / "final_results" / "external_baselines" / "external_baseline_significance.csv",
        external_significance,
        ["场景", "指标", "方向", "比较", "检验", "配对样本数", "MP均值", "外部基线均值", "p值", "Holm显著性"],
    )

    external_table = build_external_table(root)
    write_csv(
        table_dir / "table_S3_external_baselines.csv",
        external_table,
        ["场景", "方法", "成功率", "规划时间_ms_均值±标准差", "路径代价", "路径长度_km", "通信覆盖率", "风险暴露", "采样或评估次数"],
    )

    failures = build_failure_summary(root)
    write_csv(
        table_dir / "table_S4_failure_reason_summary.csv",
        failures,
        ["场景", "图规模", "阶段", "失败原因", "次数", "节点数", "边数", "解释"],
    )

    figure_source = build_augmented_figure_source(root, all_rows)
    write_csv(
        source_dir / "figure_4_1_external_augmented_source.csv",
        figure_source,
        ["场景", "方法", "成功率", "重规划时间_ms", "路径代价", "长度_km", "通信覆盖率", "风险暴露"],
    )

    print("[完成] 已生成统计稳定性、显著性检验、外部基线和失败原因补充表。")
    print(f"[输出] {table_dir / 'table_S1_statistical_stability.csv'}")
    print(f"[输出] {table_dir / 'table_S2_significance_tests.csv'}")
    print(f"[输出] {table_dir / 'table_S3_external_baselines.csv'}")
    print(f"[输出] {table_dir / 'table_S4_failure_reason_summary.csv'}")
    print(f"[输出] {source_dir / 'figure_4_1_external_augmented_source.csv'}")
    print(f"[输出] {root / 'final_results' / 'external_baselines' / 'external_baseline_significance.csv'}")


if __name__ == "__main__":
    main()
