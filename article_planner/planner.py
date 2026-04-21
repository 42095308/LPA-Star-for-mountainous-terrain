"""
LPA* 单次规划命令封装。

`lpa_star.py` 仍保留历史的导入即执行结构，因此这里提供无副作用的命令生成函数。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .pipeline import script_command


def lpa_star_command(
    scenario_config: str | Path,
    workdir: str | Path = ".",
    disable_seed_sweep: bool = True,
) -> List[str]:
    """生成 LPA* 单次演示命令。"""
    command = script_command("lpa_star.py", scenario_config, workdir)
    if disable_seed_sweep:
        command.append("--disable-seed-sweep")
    return command


__all__ = ["lpa_star_command"]
