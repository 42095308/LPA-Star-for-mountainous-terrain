"""
Benchmark 能力的包内入口。
"""

from __future__ import annotations

from benchmark import parse_args, run_benchmark, run_benchmark_matrix_via_subprocess
from benchmark_matrix import run_benchmark_matrix

__all__ = [
    "parse_args",
    "run_benchmark",
    "run_benchmark_matrix",
    "run_benchmark_matrix_via_subprocess",
]
