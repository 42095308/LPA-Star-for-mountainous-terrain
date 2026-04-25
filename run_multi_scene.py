"""
多 DEM 场景批量执行器。

输入一个或多个 scenarios/*.json，按统一流程生成场景缓存、风险场、分层图、
物流任务与 benchmark 输出，最后汇总每个场景的 benchmark_summary.csv。
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from article_planner.scenario_config import load_scenario_config, scenario_output_dir


EXAMPLE_SCENARIO_SUFFIX = ".example.json"


@dataclass
class StepResult:
    name: str
    command: List[str]
    returncode: int
    elapsed_s: float
    skipped: bool = False
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "returncode": self.returncode,
            "elapsed_s": round(float(self.elapsed_s), 3),
            "skipped": bool(self.skipped),
            "note": self.note,
            "command": " ".join(self.command),
        }


def expand_scenario_paths(patterns: Sequence[str], workdir: Path) -> List[Path]:
    paths: List[Path] = []
    for raw in patterns:
        p = Path(raw)
        pattern = str(p if p.is_absolute() else workdir / p)
        matches = [Path(v).resolve() for v in glob.glob(pattern)]
        if matches:
            paths.extend(matches)
        else:
            candidate = p if p.is_absolute() else workdir / p
            if candidate.exists():
                paths.append(candidate.resolve())
    dedup: List[Path] = []
    seen = set()
    for p in paths:
        if p.name.endswith(EXAMPLE_SCENARIO_SUFFIX):
            print(f"[跳过] 示例场景配置不会纳入多场景实验: {p}")
            continue
        key = str(p)
        if key not in seen:
            dedup.append(p)
            seen.add(key)
    return sorted(dedup)


def resolve_optional_path(raw: str, workdir: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return workdir / p


def resolve_scene_out_dir(raw_out_dir: str, scene_out: Path, workdir: Path) -> Path:
    p = Path(raw_out_dir)
    if p.is_absolute():
        return p
    if str(p).replace("\\", "/").startswith("outputs/"):
        return (workdir / p).resolve()
    return (scene_out / p).resolve()


def run_step(name: str, cmd: List[str], cwd: Path, dry_run: bool = False, note: str = "") -> StepResult:
    print(f"\n[步骤] {name}")
    print("[命令] " + " ".join(cmd))
    sys.stdout.flush()
    if dry_run:
        return StepResult(name=name, command=cmd, returncode=0, elapsed_s=0.0, skipped=True, note="dry-run")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(cwd))
    elapsed = time.perf_counter() - t0
    print(f"[结果] {name}: returncode={result.returncode}, elapsed={elapsed:.2f}s")
    return StepResult(name=name, command=cmd, returncode=int(result.returncode), elapsed_s=elapsed, note=note)


def script_cmd(python_exe: str, script: str, scenario_path: Path, workdir: Path) -> List[str]:
    return [
        python_exe,
        str(workdir / script),
        "--scenario-config",
        str(scenario_path),
        "--workdir",
        str(workdir),
    ]


def read_benchmark_summary(summary_path: Path) -> List[Dict[str, str]]:
    if not summary_path.exists():
        return []
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def first_existing_key(row: Dict[str, str], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key, "")
        if value != "":
            return value
    return ""


def scene_summary_rows(
    scenario_path: Path,
    cfg: Dict[str, object],
    scene_out: Path,
    benchmark_out: Path,
    status: str,
    failed_step: str,
    steps: Sequence[StepResult],
    benchmark_mode: str,
    error: str = "",
) -> List[Dict[str, object]]:
    scene_name = str(cfg.get("scene_name") or scenario_path.stem)
    elapsed_s = sum(float(s.elapsed_s) for s in steps)
    step_json = json.dumps([s.to_dict() for s in steps], ensure_ascii=False)
    summary_rows = read_benchmark_summary(benchmark_out / "benchmark_summary.csv")
    if not summary_rows:
        return [
            {
                "scene_name": scene_name,
                "scenario_config": str(scenario_path),
                "scene_output_dir": str(scene_out),
                "benchmark_out_dir": str(benchmark_out),
                "status": status,
                "failed_step": failed_step,
                "benchmark_mode": benchmark_mode,
                "baseline": "",
                "scale": "",
                "n_block": "",
                "intensity_index": "",
                "k_events": "",
                "n_trials": "",
                "n_success": "",
                "success_rate": "",
                "mean_replan_ms": "",
                "mean_event_replan_ms": "",
                "mean_path_cost": "",
                "mean_energy_kj": "",
                "mean_length_km": "",
                "mean_comm_coverage_ratio": "",
                "graph_nodes": "",
                "graph_edges": "",
                "elapsed_s": round(elapsed_s, 3),
                "error": error,
                "step_logs_json": step_json,
            }
        ]

    rows: List[Dict[str, object]] = []
    for row in summary_rows:
        rows.append(
            {
                "scene_name": scene_name,
                "scenario_config": str(scenario_path),
                "scene_output_dir": str(scene_out),
                "benchmark_out_dir": str(benchmark_out),
                "status": status,
                "failed_step": failed_step,
                "benchmark_mode": benchmark_mode,
                "baseline": row.get("baseline", ""),
                "scale": row.get("scale", ""),
                "n_block": row.get("n_block", ""),
                "intensity_index": first_existing_key(row, ["intensity_index", "n_block"]),
                "k_events": row.get("k_events", ""),
                "n_trials": row.get("n_trials", ""),
                "n_success": row.get("n_success", ""),
                "success_rate": row.get("success_rate", ""),
                "mean_replan_ms": first_existing_key(row, ["mean_replan_ms", "mean_cumulative_replan_ms"]),
                "mean_event_replan_ms": row.get("mean_event_replan_ms", ""),
                "mean_path_cost": first_existing_key(row, ["mean_cost", "mean_final_path_cost"]),
                "mean_energy_kj": first_existing_key(row, ["mean_energy_kj", "mean_final_path_energy_kj"]),
                "mean_length_km": first_existing_key(row, ["mean_length_km", "mean_final_path_len_km"]),
                "mean_comm_coverage_ratio": first_existing_key(
                    row,
                    ["mean_comm_coverage_ratio", "mean_final_comm_coverage_ratio"],
                ),
                "graph_nodes": row.get("graph_nodes", ""),
                "graph_edges": row.get("graph_edges", ""),
                "elapsed_s": round(elapsed_s, 3),
                "error": error,
                "step_logs_json": step_json,
            }
        )
    return rows


def write_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scene_name",
        "scenario_config",
        "scene_output_dir",
        "benchmark_out_dir",
        "status",
        "failed_step",
        "benchmark_mode",
        "baseline",
        "scale",
        "n_block",
        "intensity_index",
        "k_events",
        "n_trials",
        "n_success",
        "success_rate",
        "mean_replan_ms",
        "mean_event_replan_ms",
        "mean_path_cost",
        "mean_energy_kj",
        "mean_length_km",
        "mean_comm_coverage_ratio",
        "graph_nodes",
        "graph_edges",
        "elapsed_s",
        "error",
        "step_logs_json",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def should_run_osm(cfg: Dict[str, object], workdir: Path, skip_osm: bool, require_osm: bool) -> tuple[bool, str]:
    if skip_osm:
        return False, "用户跳过 OSM 风险生成"
    osm_file = str(cfg.get("osm_file") or "")
    if not osm_file:
        if require_osm:
            raise FileNotFoundError("场景配置缺少 osm_file")
        return False, "场景配置缺少 osm_file，跳过 human_risk_osm.py"
    osm_path = resolve_optional_path(osm_file, workdir)
    if osm_path.exists():
        return True, ""
    if require_osm:
        raise FileNotFoundError(f"场景声明的 OSM 文件不存在: {osm_path}")
    return False, f"未找到 OSM 文件，跳过 human_risk_osm.py: {osm_path}"


def append_matrix_benchmark_args(
    cmd: List[str],
    args: argparse.Namespace,
    benchmark_runner: str,
    benchmark_mode: str,
) -> None:
    """把 run_multi_scene 的矩阵参数映射到具体 benchmark runner。"""
    if benchmark_runner == "benchmark_matrix":
        mapping = [
            ("n_block_grid", "--n-block-grid"),
            ("k_events_grid", "--k-events-grid"),
            ("scales", "--scales"),
            ("scale_fractions", "--scale-fractions"),
            ("focus_scale", "--focus-scale"),
            ("focus_k_intensity", "--focus-k-intensity"),
            ("focus_n_block_cont", "--focus-n-block-cont"),
            ("focus_k_scale", "--focus-k-scale"),
            ("focus_n_block_scale", "--focus-n-block-scale"),
            ("focus_k_distribution", "--focus-k-distribution"),
            ("plot_scale", "--plot-scale"),
            ("plot_k_intensity", "--plot-k-intensity"),
            ("plot_n_block_cont", "--plot-n-block-cont"),
            ("plot_k_distribution", "--plot-k-distribution"),
            ("event_pool_factor", "--event-pool-factor"),
        ]
    elif benchmark_mode == "matrix":
        mapping = [
            ("n_block_grid", "--matrix-n-block-grid"),
            ("k_events_grid", "--matrix-k-events-grid"),
            ("scales", "--matrix-scales"),
            ("scale_fractions", "--matrix-scale-fractions"),
            ("focus_scale", "--matrix-focus-scale"),
            ("focus_k_intensity", "--matrix-focus-k-intensity"),
            ("focus_n_block_cont", "--matrix-focus-n-block-cont"),
            ("focus_k_scale", "--matrix-focus-k-scale"),
            ("focus_n_block_scale", "--matrix-focus-n-block-scale"),
            ("focus_k_distribution", "--matrix-focus-k-distribution"),
            ("plot_scale", "--matrix-plot-scale"),
            ("plot_k_intensity", "--matrix-plot-k-intensity"),
            ("plot_n_block_cont", "--matrix-plot-n-block-cont"),
            ("plot_k_distribution", "--matrix-plot-k-distribution"),
            ("event_pool_factor", "--matrix-event-pool-factor"),
        ]
    else:
        return

    for attr, flag in mapping:
        value = getattr(args, attr, None)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        cmd.extend([flag, str(value)])


def run_scene(
    scenario_path: Path,
    args: argparse.Namespace,
    workdir: Path,
) -> List[Dict[str, object]]:
    cfg = load_scenario_config(scenario_path, workdir)
    scene_name = str(cfg.get("scene_name") or scenario_path.stem)
    scene_out = scenario_output_dir(cfg, workdir)
    benchmark_out = resolve_scene_out_dir(args.benchmark_out_name, scene_out, workdir)
    benchmark_runner = "benchmark_matrix" if args.use_benchmark_matrix else args.benchmark_runner
    steps: List[StepResult] = []
    status = "ok"
    failed_step = ""
    error = ""

    print(f"\n========== 场景: {scene_name} ==========")
    print(f"[场景配置] {scenario_path}")
    print(f"[场景输出] {scene_out}")
    print(f"[benchmark输出] {benchmark_out}")
    sys.stdout.flush()

    try:
        init_cmd = script_cmd(args.python, "init_graph.py", scenario_path, workdir)
        if args.force_recrop:
            init_cmd.append("--force-recrop")
        steps.append(run_step("init_graph", init_cmd, workdir, args.dry_run))
        if steps[-1].returncode != 0:
            raise RuntimeError("init_graph 失败")

        run_osm, osm_note = should_run_osm(cfg, workdir, args.skip_osm_risk, args.require_osm_risk)
        if run_osm:
            cmd = script_cmd(args.python, "human_risk_osm.py", scenario_path, workdir)
            steps.append(run_step("human_risk_osm", cmd, workdir, args.dry_run))
            if steps[-1].returncode != 0:
                raise RuntimeError("human_risk_osm 失败")
        else:
            print(f"[跳过] human_risk_osm: {osm_note}")
            steps.append(
                StepResult(
                    name="human_risk_osm",
                    command=[],
                    returncode=0,
                    elapsed_s=0.0,
                    skipped=True,
                    note=osm_note,
                )
            )

        steps.append(run_step("safe_corridor", script_cmd(args.python, "safe_corridor.py", scenario_path, workdir), workdir, args.dry_run))
        if steps[-1].returncode != 0:
            raise RuntimeError("safe_corridor 失败")

        comm_cfg = cfg.get("communication") or {}
        if args.skip_communication or comm_cfg.get("enabled") is False:
            note = "用户跳过通信风险生成" if args.skip_communication else "场景配置关闭通信风险"
            print(f"[跳过] communication_risk: {note}")
            steps.append(
                StepResult(
                    name="communication_risk",
                    command=[],
                    returncode=0,
                    elapsed_s=0.0,
                    skipped=True,
                    note=note,
                )
            )
        else:
            steps.append(run_step("communication_risk", script_cmd(args.python, "communication_risk.py", scenario_path, workdir), workdir, args.dry_run))
            if steps[-1].returncode != 0:
                raise RuntimeError("communication_risk 失败")

        graph_cmd = script_cmd(args.python, "layered_graph.py", scenario_path, workdir)
        if args.skip_layered_plot:
            graph_cmd.append("--skip-plot")
        steps.append(run_step("layered_graph", graph_cmd, workdir, args.dry_run))
        if steps[-1].returncode != 0:
            raise RuntimeError("layered_graph 失败")

        task_cmd = script_cmd(args.python, "task_generator.py", scenario_path, workdir)
        if args.task_target_count >= 0:
            task_cmd.extend(["--target-count", str(args.task_target_count)])
        if args.task_pair_count >= 0:
            task_cmd.extend(["--pair-count", str(args.task_pair_count)])
        steps.append(run_step("task_generator", task_cmd, workdir, args.dry_run))
        if steps[-1].returncode != 0:
            raise RuntimeError("task_generator 失败")

        if benchmark_runner == "benchmark_matrix":
            bench_cmd = [
                args.python,
                str(workdir / "benchmark_matrix.py"),
                "--scenario-config",
                str(scenario_path),
                "--workdir",
                str(workdir),
                "--trials",
                str(args.trials),
                "--key-trials",
                str(args.key_trials),
                "--seed",
                str(args.seed),
                "--out-dir",
                str(benchmark_out),
                "--event-type",
                args.event_type,
                "--event-radius-km",
                str(args.event_radius_km),
                "--event-severity",
                str(args.event_severity),
                "--min-start-goal-dist-km",
                str(args.min_start_goal_dist_km),
            ]
        else:
            bench_cmd = [
                args.python,
                str(workdir / "benchmark.py"),
                "--mode",
                args.benchmark_mode,
                "--scenario-config",
                str(scenario_path),
                "--workdir",
                str(workdir),
                "--trials",
                str(args.trials),
                "--seed",
                str(args.seed),
                "--out-dir",
                str(benchmark_out),
                "--event-type",
                args.event_type,
                "--event-radius-km",
                str(args.event_radius_km),
                "--event-severity",
                str(args.event_severity),
                "--min-start-goal-dist-km",
                str(args.min_start_goal_dist_km),
            ]
            if args.key_trials > 0:
                bench_cmd.extend(["--matrix-key-trials", str(args.key_trials)])
            if args.skip_b1:
                bench_cmd.append("--skip-b1")
        if args.disable_plots:
            bench_cmd.append("--disable-plots")
        append_matrix_benchmark_args(bench_cmd, args, benchmark_runner, args.benchmark_mode)
        if args.benchmark_extra_args:
            bench_cmd.extend(shlex.split(args.benchmark_extra_args))
        steps.append(run_step(benchmark_runner, bench_cmd, workdir, args.dry_run))
        if steps[-1].returncode != 0:
            raise RuntimeError(f"{benchmark_runner} 失败")

    except Exception as exc:
        status = "failed"
        error = str(exc)
        failed_step = steps[-1].name if steps else "prepare"
        print(f"[失败] 场景 {scene_name}: {error}")

    return scene_summary_rows(
        scenario_path,
        cfg,
        scene_out,
        benchmark_out,
        status,
        failed_step,
        steps,
        benchmark_runner if benchmark_runner == "benchmark_matrix" else args.benchmark_mode,
        error,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量运行多 DEM 场景实验并汇总 benchmark CSV。")
    parser.add_argument("--scenario-configs", nargs="+", default=["scenarios/*.json"], help="场景 JSON 路径或 glob。")
    parser.add_argument("--workdir", type=str, default=".", help="项目根目录。")
    parser.add_argument("--python", type=str, default=sys.executable, help="用于执行各脚本的 Python 解释器。")
    parser.add_argument("--benchmark-runner", choices=["benchmark", "benchmark_matrix"], default="benchmark")
    parser.add_argument("--use-benchmark-matrix", action="store_true", help="等价于 --benchmark-runner benchmark_matrix。")
    parser.add_argument("--benchmark-mode", choices=["single", "matrix"], default="single")
    parser.add_argument("--benchmark-out-name", type=str, default="tests/benchmark_multi_scene")
    parser.add_argument("--summary-csv", type=str, default="outputs/_summaries/multi_scene_summary.csv")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--key-trials", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260309)
    parser.add_argument("--event-type", choices=["no_fly", "wind", "comm_risk"], default="no_fly")
    parser.add_argument("--event-radius-km", type=float, default=0.8)
    parser.add_argument("--event-severity", type=float, default=1.0)
    parser.add_argument("--min-start-goal-dist-km", type=float, default=1.5)
    parser.add_argument("--n-block-grid", type=str, default="", help="矩阵实验事件强度索引网格，例如 2,4,6,8。")
    parser.add_argument("--k-events-grid", type=str, default="", help="矩阵实验连续事件数网格，例如 1,3,5,7,10。")
    parser.add_argument("--scales", type=str, default="", help="矩阵实验图规模列表，例如 small,medium,large。")
    parser.add_argument("--scale-fractions", type=str, default="", help="矩阵实验图规模裁剪比例，例如 small:0.55,medium:0.78,large:1.0。")
    parser.add_argument("--focus-scale", type=str, default="", help="Experiment A/B/D 的焦点图规模。")
    parser.add_argument("--focus-k-intensity", type=int, default=None, help="Experiment A 的焦点 K。")
    parser.add_argument("--focus-n-block-cont", type=int, default=None, help="Experiment B 的焦点强度索引。")
    parser.add_argument("--focus-k-scale", type=int, default=None, help="Experiment C 的焦点 K。")
    parser.add_argument("--focus-n-block-scale", type=int, default=None, help="Experiment C 的焦点强度索引。")
    parser.add_argument("--focus-k-distribution", type=int, default=None, help="路径质量分布诊断的焦点 K。")
    parser.add_argument("--plot-scale", type=str, default="", help="矩阵内置绘图的焦点图规模。")
    parser.add_argument("--plot-k-intensity", type=int, default=None, help="矩阵内置绘图的焦点 K。")
    parser.add_argument("--plot-n-block-cont", type=int, default=None, help="矩阵内置绘图的焦点强度索引。")
    parser.add_argument("--plot-k-distribution", type=int, default=None, help="矩阵内置路径质量图的焦点 K。")
    parser.add_argument("--event-pool-factor", type=int, default=None, help="矩阵区域事件候选池倍数。")
    parser.add_argument("--task-target-count", type=int, default=-1)
    parser.add_argument("--task-pair-count", type=int, default=-1)
    parser.add_argument("--skip-b1", action="store_true")
    parser.add_argument("--disable-plots", action="store_true")
    parser.add_argument("--skip-layered-plot", action="store_true")
    parser.add_argument("--skip-osm-risk", action="store_true")
    parser.add_argument("--require-osm-risk", action="store_true")
    parser.add_argument("--skip-communication", action="store_true")
    parser.add_argument("--force-recrop", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--benchmark-extra-args", type=str, default="", help="追加传给 benchmark.py 的参数字符串。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    scenario_paths = expand_scenario_paths(args.scenario_configs, workdir)
    if not scenario_paths:
        raise FileNotFoundError(f"未找到场景配置: {args.scenario_configs}")

    all_rows: List[Dict[str, object]] = []
    for scenario_path in scenario_paths:
        rows = run_scene(scenario_path, args, workdir)
        all_rows.extend(rows)
        if args.stop_on_error and any(str(row.get("status")) == "failed" for row in rows):
            print("[停止] 已按 --stop-on-error 在首个失败场景后停止。")
            break

    summary_path = resolve_optional_path(args.summary_csv, workdir)
    write_summary(summary_path, all_rows)
    print(f"\n[完成] 多场景汇总 CSV: {summary_path}")
    print(f"[完成] 场景数: {len(scenario_paths)}, 汇总行数: {len(all_rows)}")


if __name__ == "__main__":
    main()
