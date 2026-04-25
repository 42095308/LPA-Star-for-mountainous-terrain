"""绘制 E2 结构性消融实验图。

该脚本只读取 single benchmark 已生成的结构性消融 CSV，不重新建图、不重新规划。
推荐输入为 `benchmark_structural_ablation.csv` 所在目录。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from plot_matrix_results import configure_matplotlib, load_structural_rows, plot_structural_ablation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制 E2 Structural Ablation Study 论文图。")
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="包含 benchmark_structural_ablation.csv 或 benchmark_summary.csv 的 single benchmark 输出目录。",
    )
    parser.add_argument("--out-dir", type=str, default="", help="图输出目录；默认写回 result-dir。")
    parser.add_argument("--dpi", type=int, default=300, help="PDF 中嵌入栅格元素的分辨率。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    result_dir = Path(args.result_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else result_dir
    structural_rows = load_structural_rows(result_dir)
    out_path = plot_structural_ablation(
        structural_rows,
        out_dir,
        args.dpi,
        filename="fig_E2_structural_ablation.pdf",
    )
    if out_path is None:
        raise RuntimeError("未找到包含 M-R / B5_RegularLayered_LPA 的结构性消融结果。")
    print("[done] E2 结构性消融图已生成：")
    print(f"  - {out_path}")


if __name__ == "__main__":
    main()
