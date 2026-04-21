"""
多场景流水线和历史脚本命令封装。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

from run_multi_scene import main as run_multi_scene_main


SINGLE_SCENE_FLOW = [
    "init_graph",
    "human_risk_osm",
    "safe_corridor",
    "communication_risk",
    "layered_graph",
    "task_generator",
    "benchmark",
]


def script_command(script_name: str, scenario_config: str | Path, workdir: str | Path = ".") -> List[str]:
    """生成兼容脚本命令。"""
    root = Path(workdir).resolve()
    return [
        sys.executable,
        str(root / script_name),
        "--scenario-config",
        str(scenario_config),
        "--workdir",
        str(root),
    ]


def run_script_command(command: Iterable[str], workdir: str | Path = ".") -> subprocess.CompletedProcess[str]:
    """执行兼容脚本命令并返回结果。"""
    return subprocess.run(
        list(command),
        cwd=str(Path(workdir).resolve()),
        text=True,
        check=False,
    )


__all__ = [
    "SINGLE_SCENE_FLOW",
    "run_multi_scene_main",
    "run_script_command",
    "script_command",
]
