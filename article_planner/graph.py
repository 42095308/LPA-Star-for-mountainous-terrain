"""
分层图构建命令封装。

`layered_graph.py` 仍保留历史的导入即执行结构，因此这里提供无副作用的命令生成函数。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .pipeline import script_command


def layered_graph_command(
    scenario_config: str | Path,
    workdir: str | Path = ".",
    skip_plot: bool = False,
) -> List[str]:
    """生成分层图构建命令。"""
    command = script_command("layered_graph.py", scenario_config, workdir)
    if skip_plot:
        command.append("--skip-plot")
    return command


__all__ = ["layered_graph_command"]
