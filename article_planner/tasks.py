"""
虚拟配送站和任务生成能力的包内入口。
"""

from __future__ import annotations

from task_generator import (
    auto_target_candidates,
    configured_targets,
    load_or_generate_depots,
    main,
    stratified_pairs,
)
from virtual_depots import generate_virtual_depots

__all__ = [
    "auto_target_candidates",
    "configured_targets",
    "generate_virtual_depots",
    "load_or_generate_depots",
    "main",
    "stratified_pairs",
]
