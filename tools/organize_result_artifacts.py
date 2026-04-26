"""按论文用途归档正式结果、日志和中间产物。

默认策略：
- 正式 E1/E2/E3/E4 结果复制到 `final_results/`；
- smoke、precheck 和大体量事件明细移动到 `final_results/logs/`；
- 场景中间数据复制到 `intermediate_artifacts/data/`；
- 场景中间步骤图片复制到 `intermediate_artifacts/figures/`。

保留 `outputs/` 中的核心场景缓存，避免破坏现有脚本的默认输入路径。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, Sequence


SCENES = ("huashan", "huangshan", "emeishan")

SUMMARY_CORE_FILES = (
    "E1_E2_three_mountain_single_final.csv",
    "E3_E4_three_mountain_matrix_final.csv",
    "fig_E1_cross_terrain_overall.pdf",
)

SUMMARY_DETAIL_FIGURES = (
    "fig_E1_cross_terrain_success.pdf",
    "fig_E1_cross_terrain_replan_time.pdf",
    "fig_E1_cross_terrain_comm_coverage.pdf",
    "fig_E1_cross_terrain_risk_exposure.pdf",
)

SINGLE_CORE_FILES = (
    "benchmark_summary.csv",
    "benchmark_structural_ablation.csv",
    "benchmark_trials.csv",
    "fig_E2_structural_ablation.pdf",
)

MATRIX_CORE_FILES = (
    "benchmark_summary.csv",
    "experiment_A.csv",
    "experiment_B.csv",
    "experiment_C.csv",
    "experiment_D.csv",
    "experiment_path_quality.csv",
    "benchmark_trials.csv",
    "benchmark_failure_reasons.csv",
    "benchmark_trial_failures.csv",
)

MATRIX_CORE_GLOBS = ("fig_*.pdf",)

SMOKE_SUMMARY_FILES = (
    "E1_E2_single_smoke.csv",
    "E3_E4_matrix_smoke.csv",
    "matrix_smoke_check.csv",
)

PRECHECK_SUMMARY_FILES = (
    "huangshan_precheck.csv",
    "multi_scene_summary.csv",
    "multi_scene_summary(3).csv",
)

INTERMEDIATE_DATA_SUFFIXES = (".npy", ".npz", ".json")
INTERMEDIATE_FIGURE_SUFFIXES = (".png", ".jpg", ".jpeg", ".pdf", ".svg")


def copy_file(src: Path, dst: Path, dry_run: bool = False) -> bool:
    """复制单个文件；缺失文件跳过。"""
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[dry-run] copy {src} -> {dst}")
        return True
    shutil.copy2(src, dst)
    print(f"[copy] {src} -> {dst}")
    return True


def move_path(src: Path, dst: Path, dry_run: bool = False) -> bool:
    """移动文件或目录；目标已存在时跳过，避免覆盖已有日志。"""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[skip] 目标已存在，未移动: {dst}")
        return False
    if dry_run:
        print(f"[dry-run] move {src} -> {dst}")
        return True
    shutil.move(str(src), str(dst))
    print(f"[move] {src} -> {dst}")
    return True


def copy_named_files(src_dir: Path, dst_dir: Path, names: Sequence[str], dry_run: bool = False) -> int:
    """按文件名白名单复制文件。"""
    count = 0
    for name in names:
        if copy_file(src_dir / name, dst_dir / name, dry_run=dry_run):
            count += 1
    return count


def copy_globs(src_dir: Path, dst_dir: Path, patterns: Iterable[str], dry_run: bool = False) -> int:
    """按 glob 白名单复制文件。"""
    count = 0
    for pattern in patterns:
        for src in sorted(src_dir.glob(pattern)):
            if copy_file(src, dst_dir / src.name, dry_run=dry_run):
                count += 1
    return count


def organize_formal_results(outputs: Path, final_root: Path, dry_run: bool = False) -> int:
    """复制正式 E1/E2/E3/E4 核心结果。"""
    count = 0
    summaries_src = outputs / "_summaries"
    summaries_dst = final_root / "_summaries"
    count += copy_named_files(summaries_src, summaries_dst, SUMMARY_CORE_FILES, dry_run=dry_run)
    count += copy_named_files(summaries_src, summaries_dst, SUMMARY_DETAIL_FIGURES, dry_run=dry_run)

    for scene in SCENES:
        scene_src = outputs / scene / "tests"
        scene_dst = final_root / scene
        count += copy_named_files(
            scene_src / "E1_E2_single_final",
            scene_dst / "E1_E2_single_final",
            SINGLE_CORE_FILES,
            dry_run=dry_run,
        )
        matrix_src = scene_src / "E3_E4_matrix_final"
        matrix_dst = scene_dst / "E3_E4_matrix_final"
        count += copy_named_files(matrix_src, matrix_dst, MATRIX_CORE_FILES, dry_run=dry_run)
        count += copy_globs(matrix_src, matrix_dst, MATRIX_CORE_GLOBS, dry_run=dry_run)
    return count


def organize_logs(outputs: Path, logs_root: Path, dry_run: bool = False) -> int:
    """移动 smoke、precheck 和事件明细日志。"""
    count = 0
    summaries_src = outputs / "_summaries"
    for name in SMOKE_SUMMARY_FILES:
        if move_path(summaries_src / name, logs_root / "smoke" / name, dry_run=dry_run):
            count += 1
    for name in PRECHECK_SUMMARY_FILES:
        if move_path(summaries_src / name, logs_root / "precheck" / name, dry_run=dry_run):
            count += 1

    for scene in SCENES:
        tests_dir = outputs / scene / "tests"
        if tests_dir.exists():
            for child in sorted(tests_dir.iterdir()):
                if not child.is_dir():
                    continue
                lname = child.name.lower()
                if "smoke" in lname:
                    if move_path(child, logs_root / "smoke" / scene / child.name, dry_run=dry_run):
                        count += 1
                elif "precheck" in lname:
                    if move_path(child, logs_root / "precheck" / scene / child.name, dry_run=dry_run):
                        count += 1

        event_file = tests_dir / "E3_E4_matrix_final" / "benchmark_events.csv"
        if move_path(
            event_file,
            logs_root / "event_records" / scene / "E3_E4_matrix_final" / "benchmark_events.csv",
            dry_run=dry_run,
        ):
            count += 1
    return count


def organize_intermediate_artifacts(outputs: Path, intermediate_root: Path, dry_run: bool = False) -> int:
    """复制场景根目录下的中间数据和中间步骤图片。"""
    count = 0
    for scene in SCENES:
        scene_dir = outputs / scene
        if not scene_dir.exists():
            continue
        for src in sorted(scene_dir.iterdir()):
            if not src.is_file():
                continue
            suffix = src.suffix.lower()
            if suffix in INTERMEDIATE_DATA_SUFFIXES:
                if copy_file(src, intermediate_root / "data" / scene / src.name, dry_run=dry_run):
                    count += 1
            elif suffix in INTERMEDIATE_FIGURE_SUFFIXES:
                if copy_file(src, intermediate_root / "figures" / scene / src.name, dry_run=dry_run):
                    count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理正式结果、日志和中间产物目录。")
    parser.add_argument("--workdir", type=str, default=".", help="项目根目录。")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="现有 outputs 目录。")
    parser.add_argument("--final-dir", type=str, default="final_results", help="正式结果归档目录。")
    parser.add_argument(
        "--intermediate-dir",
        type=str,
        default="intermediate_artifacts",
        help="中间数据和中间步骤图归档目录。",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="final_results/logs",
        help="smoke、precheck 和事件明细日志目录。",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的整理动作。")
    parser.add_argument("--skip-log-move", action="store_true", help="不移动 smoke/precheck/事件明细日志。")
    parser.add_argument("--skip-intermediate", action="store_true", help="不复制中间数据和中间步骤图。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    outputs = (workdir / args.outputs_dir).resolve()
    final_root = (workdir / args.final_dir).resolve()
    intermediate_root = (workdir / args.intermediate_dir).resolve()
    logs_root = (workdir / args.logs_dir).resolve()

    if not outputs.exists():
        raise FileNotFoundError(f"outputs 目录不存在: {outputs}")

    formal_count = organize_formal_results(outputs, final_root, dry_run=args.dry_run)
    log_count = 0 if args.skip_log_move else organize_logs(outputs, logs_root, dry_run=args.dry_run)
    intermediate_count = (
        0
        if args.skip_intermediate
        else organize_intermediate_artifacts(outputs, intermediate_root, dry_run=args.dry_run)
    )

    print("[done] 结果目录整理完成")
    print(f"  - 正式结果文件: {formal_count}")
    print(f"  - 日志移动项: {log_count}")
    print(f"  - 中间产物文件: {intermediate_count}")
    print(f"  - final_results: {final_root}")
    print(f"  - logs: {logs_root}")
    print(f"  - intermediate_artifacts: {intermediate_root}")


if __name__ == "__main__":
    main()
