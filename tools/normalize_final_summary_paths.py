"""规范 final_results/_summaries 中的路径字段。

正式论文 summary 只保留相对路径：
- scenario_config -> scenarios/<scene>.json
- scene_output_dir -> intermediate_artifacts/data/<scene>
- benchmark_out_dir -> final_results/<scene>/<run_name>

step_logs_json 中包含本机绝对命令路径，正文级 summary 默认清空。
详细命令日志应放入 final_results/logs/。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List


def read_rows(path: Path) -> List[dict]:
    """读取 CSV。"""
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    """写回 CSV。"""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_benchmark_dir(raw: str, scene: str, default_name: str) -> str:
    """将旧 outputs 或绝对 benchmark 路径规范为 final_results 相对路径。"""
    text = str(raw or "").replace("\\", "/").strip()
    if not text:
        return f"final_results/{scene}/{default_name}"
    marker = f"/outputs/{scene}/tests/"
    if marker in text:
        return f"final_results/{scene}/{text.split(marker, 1)[1].strip('/')}"
    marker = f"outputs/{scene}/tests/"
    if marker in text:
        return f"final_results/{scene}/{text.split(marker, 1)[1].strip('/')}"
    marker = f"/outputs/{scene}/"
    if marker in text:
        rest = text.split(marker, 1)[1].strip("/")
        if rest.startswith("tests/"):
            rest = rest[len("tests/") :]
        return f"final_results/{scene}/{rest}" if rest else f"final_results/{scene}/{default_name}"
    marker = f"outputs/{scene}/"
    if marker in text:
        rest = text.split(marker, 1)[1].strip("/")
        if rest.startswith("tests/"):
            rest = rest[len("tests/") :]
        return f"final_results/{scene}/{rest}" if rest else f"final_results/{scene}/{default_name}"
    marker = f"/final_results/{scene}/"
    if marker in text:
        return f"final_results/{scene}/{text.split(marker, 1)[1].strip('/')}"
    marker = f"final_results/{scene}/"
    if marker in text:
        return f"final_results/{scene}/{text.split(marker, 1)[1].strip('/')}"
    return f"final_results/{scene}/{default_name}"


def normalize_summary(path: Path, default_name: str) -> int:
    """规范单个 summary CSV，返回更新行数。"""
    if not path.exists():
        return 0
    rows = read_rows(path)
    if not rows:
        return 0
    fieldnames = list(rows[0].keys())
    changed = 0
    for row in rows:
        scene = str(row.get("scene_name", "")).strip()
        if not scene:
            continue
        updates = {
            "scenario_config": f"scenarios/{scene}.json",
            "scene_output_dir": f"intermediate_artifacts/data/{scene}",
            "benchmark_out_dir": normalize_benchmark_dir(row.get("benchmark_out_dir", ""), scene, default_name),
            "step_logs_json": "",
        }
        for key, value in updates.items():
            if key in row and row.get(key, "") != value:
                row[key] = value
                changed += 1
    write_rows(path, rows, fieldnames)
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="规范 final_results/_summaries 中的路径字段。")
    parser.add_argument("--summary-dir", type=str, default="final_results/_summaries", help="summary CSV 所在目录。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir).resolve()
    targets = [
        ("E1_E2_three_mountain_single_final.csv", "E1_E2_single_final"),
        ("E3_E4_three_mountain_matrix_final.csv", "E3_E4_matrix_final"),
        ("multi_scene_summary.csv", "benchmark_multi_scene"),
    ]
    total = 0
    for filename, default_name in targets:
        changed = normalize_summary(summary_dir / filename, default_name)
        total += changed
        if changed:
            print(f"[update] {filename}: {changed} fields")
    print(f"[done] summary 路径规范化完成，更新字段数: {total}")


if __name__ == "__main__":
    main()
