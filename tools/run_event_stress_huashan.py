from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


SETTINGS = [
    {"stress_type": "radius", "label": "no_fly_radius_0_4", "event_type": "no_fly", "radius": 0.4, "severity": 1.0},
    {"stress_type": "radius", "label": "no_fly_radius_0_8", "event_type": "no_fly", "radius": 0.8, "severity": 1.0},
    {"stress_type": "radius", "label": "no_fly_radius_1_2", "event_type": "no_fly", "radius": 1.2, "severity": 1.0},
    {"stress_type": "severity", "label": "wind_severity_0_5", "event_type": "wind", "radius": 0.8, "severity": 0.5},
    {"stress_type": "severity", "label": "wind_severity_1_0", "event_type": "wind", "radius": 0.8, "severity": 1.0},
    {"stress_type": "severity", "label": "wind_severity_1_5", "event_type": "wind", "radius": 0.8, "severity": 1.5},
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


def run_setting(args: argparse.Namespace, workdir: Path, setting: dict[str, object], run_dir: Path) -> None:
    summary_path = run_dir / "benchmark_summary.csv"
    if summary_path.exists() and not args.force:
        print(f"[跳过] 已存在分组结果: {run_dir}")
        return
    out_arg = str(run_dir.relative_to(workdir)) if run_dir.is_relative_to(workdir) else str(run_dir)
    cmd = [
        sys.executable,
        "benchmark_matrix.py",
        "--scenario-config",
        "scenarios/huashan.json",
        "--workdir",
        str(workdir),
        "--trials",
        str(args.trials),
        "--seed",
        str(args.seed + int(float(setting["radius"]) * 1000) + int(float(setting["severity"]) * 100)),
        "--out-dir",
        out_arg,
        "--n-block-grid",
        str(args.n_block),
        "--k-events-grid",
        str(args.k_events),
        "--scales",
        "large",
        "--event-type",
        str(setting["event_type"]),
        "--event-radius-km",
        str(setting["radius"]),
        "--event-severity",
        str(setting["severity"]),
        "--focus-scale",
        "large",
        "--focus-k-intensity",
        str(args.k_events),
        "--focus-n-block-cont",
        str(args.n_block),
        "--focus-k-scale",
        str(args.k_events),
        "--focus-n-block-scale",
        str(args.n_block),
        "--plot-scale",
        "large",
        "--plot-k-intensity",
        str(args.k_events),
        "--plot-n-block-cont",
        str(args.n_block),
        "--disable-plots",
        "--progress-every",
        str(args.progress_every),
    ]
    print("[运行] " + " ".join(cmd))
    subprocess.run(cmd, cwd=workdir, check=True)


def collect(args: argparse.Namespace, base_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    trial_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for setting in SETTINGS:
        run_dir = base_dir / "_runs" / str(setting["label"])
        for row in read_csv(run_dir / "benchmark_trials.csv"):
            out = dict(row)
            out.update(
                {
                    "stress_type": setting["stress_type"],
                    "stress_label": setting["label"],
                    "event_type": setting["event_type"],
                    "event_radius_km_setting": setting["radius"],
                    "event_severity_setting": setting["severity"],
                }
            )
            trial_rows.append(out)
        for row in read_csv(run_dir / "benchmark_summary.csv"):
            if row.get("scale") != "large" or row.get("n_block") != str(args.n_block) or row.get("k_events") != str(args.k_events):
                continue
            out = {
                "stress_type": setting["stress_type"],
                "stress_label": setting["label"],
                "event_type": setting["event_type"],
                "event_radius_km": setting["radius"],
                "event_severity": setting["severity"],
                "scale": row.get("scale", ""),
                "n_block": row.get("n_block", ""),
                "k_events": row.get("k_events", ""),
                "baseline": row.get("baseline", ""),
                "n_trials": row.get("n_trials", ""),
                "n_success": row.get("n_success", ""),
                "success_rate": row.get("success_rate", ""),
                "mean_cumulative_replan_ms": row.get("mean_cumulative_replan_ms", ""),
                "std_cumulative_replan_ms": row.get("std_cumulative_replan_ms", ""),
                "ci95_cumulative_replan_ms": row.get("ci95_cumulative_replan_ms", ""),
                "mean_cumulative_expanded": row.get("mean_cumulative_expanded", ""),
                "std_event_expanded": row.get("std_event_expanded", ""),
                "ci95_event_expanded": row.get("ci95_event_expanded", ""),
                "mean_cumulative_affected_edges": row.get("mean_cumulative_affected_edges", ""),
                "mean_cumulative_affected_vertices": row.get("mean_cumulative_affected_vertices", ""),
                "failure_reason_top": row.get("failure_reason_top", ""),
            }
            summary_rows.append(out)
    return trial_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="运行华山 large 图事件半径与事件强度压力实验。")
    parser.add_argument("--workdir", type=str, default=".", help="项目根目录。")
    parser.add_argument("--out-dir", type=str, default="final_results/huashan/E5_event_severity_stress")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--n-block", type=int, default=4)
    parser.add_argument("--k-events", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="强制重跑已存在的分组。")
    parser.add_argument("--collect-only", action="store_true", help="只合并已有分组，不启动 benchmark。")
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    base_dir = (workdir / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if not args.collect_only:
        for setting in SETTINGS:
            run_setting(args, workdir, setting, base_dir / "_runs" / str(setting["label"]))

    trial_rows, summary_rows = collect(args, base_dir)
    write_csv(base_dir / "event_stress_trials.csv", trial_rows, list(trial_rows[0].keys()) if trial_rows else ["stress_type"])
    write_csv(
        base_dir / "event_stress_summary.csv",
        summary_rows,
        [
            "stress_type",
            "stress_label",
            "event_type",
            "event_radius_km",
            "event_severity",
            "scale",
            "n_block",
            "k_events",
            "baseline",
            "n_trials",
            "n_success",
            "success_rate",
            "mean_cumulative_replan_ms",
            "std_cumulative_replan_ms",
            "ci95_cumulative_replan_ms",
            "mean_cumulative_expanded",
            "std_event_expanded",
            "ci95_event_expanded",
            "mean_cumulative_affected_edges",
            "mean_cumulative_affected_vertices",
            "failure_reason_top",
        ],
    )

    source_dir = workdir / "final_results" / "paper_revision" / "source_data" / "chapter4_python"
    write_csv(
        source_dir / "figure_4_6_event_stress_source.csv",
        summary_rows,
        [
            "stress_type",
            "stress_label",
            "event_type",
            "event_radius_km",
            "event_severity",
            "scale",
            "n_block",
            "k_events",
            "baseline",
            "n_trials",
            "n_success",
            "success_rate",
            "mean_cumulative_replan_ms",
            "std_cumulative_replan_ms",
            "ci95_cumulative_replan_ms",
            "mean_cumulative_expanded",
            "std_event_expanded",
            "ci95_event_expanded",
            "mean_cumulative_affected_edges",
            "mean_cumulative_affected_vertices",
            "failure_reason_top",
        ],
    )
    print(f"[完成] {base_dir / 'event_stress_trials.csv'}")
    print(f"[完成] {base_dir / 'event_stress_summary.csv'}")
    print(f"[完成] {source_dir / 'figure_4_6_event_stress_source.csv'}")


if __name__ == "__main__":
    main()
