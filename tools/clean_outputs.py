"""
整理 outputs 目录。

规则：
1. 每个场景只保留 `outputs/<scene>/tests/` 下的测试结果。
2. 场景根目录中的 DEM 裁剪、风险场、走廊、图、路径图等中间结果会被删除。
3. 根目录多场景汇总 CSV 移动到 `outputs/_summaries/`。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable


INTERMEDIATE_FILES = {
    "Z_crop.npy",
    "Z_crop_geo.npz",
    "Z_crop_meta.json",
    "ceiling.npy",
    "floor.npy",
    "layer_allowed.npy",
    "layer_mid.npy",
    "communication_summary.json",
    "corridor_meta.json",
    "corridor_vis.png",
    "generated_depots.json",
    "generated_tasks.json",
    "graph_edges.npy",
    "graph_node_roles.json",
    "graph_nodes.npy",
    "graph_terminal_status.json",
    "graph_vis.png",
    "huashan_final.png",
    "lpa_path_summary.json",
    "lpa_result.png",
    "osm_feature_summary.json",
    "osm_human_risk_preview.png",
    "path_cost_profile.png",
    "path_vis.png",
    "risk_comm.npy",
    "risk_hotspot.npy",
    "risk_human.npy",
    "risk_l1.npy",
    "risk_l2.npy",
    "risk_l3.npy",
    "risk_l4.npy",
    "risk_trail.npy",
    "target_locations.json",
}


def safe_child(root: Path, path: Path) -> Path:
    """确保待操作路径位于 outputs 内。"""
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise RuntimeError(f"拒绝操作 outputs 外路径: {resolved_path}")
    return resolved_path


def unique_destination(path: Path) -> Path:
    """避免移动测试目录时覆盖已有结果。"""
    if not path.exists():
        return path
    base = path
    idx = 2
    while True:
        candidate = base.with_name(f"{base.name}_{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


def move_path(src: Path, dst: Path, dry_run: bool) -> None:
    dst = unique_destination(dst)
    print(f"[移动] {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def remove_file(path: Path, dry_run: bool) -> None:
    print(f"[删除中间文件] {path}")
    if not dry_run:
        path.unlink()


def remove_empty_dirs(paths: Iterable[Path], dry_run: bool) -> None:
    for path in sorted(paths, key=lambda p: len(p.parts), reverse=True):
        if not path.exists() or not path.is_dir():
            continue
        try:
            next(path.iterdir())
        except StopIteration:
            print(f"[删除空目录] {path}")
            if not dry_run:
                path.rmdir()


def organise_scene(scene_dir: Path, outputs_root: Path, dry_run: bool) -> None:
    tests_dir = scene_dir / "tests"
    if not dry_run:
        tests_dir.mkdir(parents=True, exist_ok=True)

    # benchmark 目录是测试结果，统一归入 tests。
    for child in sorted(scene_dir.iterdir()):
        if child.is_dir() and child.name.startswith("benchmark_") and child.parent != tests_dir:
            move_path(child, tests_dir / child.name, dry_run)

    # 修正历史误传 out-dir 造成的嵌套 outputs/<scene>/outputs/<scene>/benchmark_*。
    nested_root = scene_dir / "outputs" / scene_dir.name
    if nested_root.exists():
        for child in sorted(nested_root.iterdir()):
            if child.is_dir() and child.name.startswith("benchmark_"):
                move_path(child, tests_dir / f"{child.name}_from_nested", dry_run)

    for child in sorted(scene_dir.iterdir()):
        if child.is_file() and (child.name in INTERMEDIATE_FILES or child.name.endswith("_final.png")):
            safe_child(outputs_root, child)
            remove_file(child, dry_run)

    remove_empty_dirs([scene_dir / "outputs" / scene_dir.name, scene_dir / "outputs"], dry_run)


def organise_summaries(outputs_root: Path, dry_run: bool) -> None:
    summary_dir = outputs_root / "_summaries"
    for file_path in sorted(outputs_root.glob("multi_scene*.csv")):
        move_path(file_path, summary_dir / file_path.name, dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理 outputs：保留测试结果，删除可重跑中间结果。")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="outputs 根目录。")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不实际移动或删除。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_dir).resolve()
    if not outputs_root.exists():
        print(f"[跳过] outputs 目录不存在: {outputs_root}")
        return

    organise_summaries(outputs_root, args.dry_run)
    for scene_dir in sorted(outputs_root.iterdir()):
        if not scene_dir.is_dir() or scene_dir.name.startswith("_"):
            continue
        organise_scene(scene_dir, outputs_root, args.dry_run)

    print("[完成] outputs 已整理。")


if __name__ == "__main__":
    main()
