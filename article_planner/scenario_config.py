"""
场景配置读取工具。

本模块把不同 DEM 场景的路径、裁剪参数、目标点、风险词典和实验参数整理成
统一字典，所有执行入口都应通过这里读取场景，避免继续在流程代码中写死华山参数。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_SCENE_CONFIG = Path("scenarios") / "huashan.json"


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def default_config() -> Dict[str, Any]:
    return {
        "scene_name": "default",
        "dem_path": "data/raw/huashan/AP_19438_FBD_F0680_RT1.dem.tif",
        "crop": {
            "center_lon": 110.0798,
            "center_lat": 34.4829,
            "crop_size_m": 10000.0,
            "resolution_m": 12.5,
        },
        "targets": {},
        "default_start": "",
        "default_goal": "",
        "display_names": {},
        "output_dir": "outputs/{scene_name}",
        "virtual_depots": {
            "count": 2,
            "slope_max_deg": 12.0,
            "elevation_percentile_max": 35.0,
            "foot_elevation_percentile": 20.0,
            "edge_buffer_km": 1.0,
            "min_target_distance_km": 1.2,
            "min_depot_spacing_km": 1.5,
            "risk_max": 0.2,
            "name_prefix": "虚拟配送站",
        },
        "terrain_sampling": {
            "branch_node_budget": 3364,
            "backbone_node_budget": 3364,
            "terrain_ratio": 0.70,
            "supplement_grid_ratio": 0.30,
            "min_spacing_m": 130.0,
            "branch_terminal_radius_km": 1.4,
            "branch_corridor_width_km": 0.55,
            "low_risk_threshold": 0.30,
        },
        "adaptive_corridor": {
            "base_floor_offset_m": 30.0,
            "base_ceiling_offset_m": 120.0,
            "slope_threshold_deg": 18.0,
            "slope_high_deg": 42.0,
            "slope_floor_extra_m": 35.0,
            "ridge_floor_extra_m": 20.0,
            "open_ceiling_extra_m": 35.0,
            "terminal_radius_km": 0.75,
            "terminal_thickness_m": 72.0,
            "min_thickness_m": 60.0,
            "high_risk_threshold": 0.65,
            "layer_positions": [0.22, 0.52, 0.82],
        },
        "communication": {
            "enabled": True,
            "source_height_agl_m": 35.0,
            "edge_source_count": 2,
            "max_range_km": 5.0,
            "los_samples": 18,
            "coarse_stride": 4,
            "risk_threshold": 0.55,
            "weights": {
                "terrain": 0.45,
                "human": 0.35,
                "communication": 0.20,
            },
            "base_stations": [],
        },
        "task_generation": {
            "depot_count": 2,
            "target_count": 8,
            "pair_count": 12,
            "min_pair_distance_km": 1.5,
            "target_min_spacing_km": 0.8,
            "target_elevation_percentile_min": 65.0,
            "max_target_slope_deg": 38.0,
            "risk_max": 0.65,
            "distance_bins_km": [2.5, 5.0],
            "elevation_bins_m": [300.0, 900.0],
            "ridge_prominence_m": 120.0,
            "random_seed": 20260420,
        },
        "osm_file": "data/raw/huashan/map.osm",
        "osm_risk_keywords": {
            "L1_DANGEROUS_NAMES": [],
            "L2_PEAK_NAMES": [],
            "L2_HIGH_NAMES": [],
            "L3_MEDIUM_NAMES": [],
            "L4_LOW_ROAD_NAMES": [],
        },
    }


def load_scenario_config(config_path: str | Path | None = None, workdir: str | Path = ".") -> Dict[str, Any]:
    """读取场景 JSON；未显式传入时优先读取默认华山配置。"""
    root = Path(workdir).resolve()
    cfg = default_config()

    if config_path is None or str(config_path).strip() == "":
        default_path = root / DEFAULT_SCENE_CONFIG
        if default_path.exists():
            config_path = default_path
        else:
            return cfg

    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = _deep_update(cfg, data)
    cfg["_config_path"] = str(path)
    return cfg


def scenario_output_dir(config: Dict[str, Any], workdir: str | Path = ".") -> Path:
    """返回当前场景主输出目录。"""
    root = Path(workdir).resolve()
    scene_name = str(config.get("scene_name") or "default")
    raw = str(config.get("output_dir") or "outputs/{scene_name}")
    raw = raw.format(scene_name=scene_name)
    out = Path(raw)
    if not out.is_absolute():
        out = root / out
    return out


def resolve_path(path_value: str | Path, workdir: str | Path = ".") -> Path:
    """把配置中的相对路径解析到工作目录下。"""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(workdir).resolve() / path


def target_specs(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """读取目标点配置，保持 JSON 中的声明顺序。"""
    targets = config.get("targets") or {}
    return {str(k): dict(v) for k, v in targets.items()}


def display_names(config: Dict[str, Any]) -> Dict[str, str]:
    """合并目标点和显式展示名配置。"""
    names: Dict[str, str] = {}
    for name, spec in target_specs(config).items():
        if "display_name" in spec:
            names[name] = str(spec["display_name"])
    for key, value in (config.get("display_names") or {}).items():
        names[str(key)] = str(value)
    return names


def depot_params(config: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(default_config()["virtual_depots"])
    params.update(config.get("virtual_depots") or {})
    return params


def terrain_sampling_params(config: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(default_config()["terrain_sampling"])
    params.update(config.get("terrain_sampling") or {})
    return params


def adaptive_corridor_params(config: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(default_config()["adaptive_corridor"])
    params.update(config.get("adaptive_corridor") or {})
    return params


def communication_params(config: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(default_config()["communication"])
    params.update(config.get("communication") or {})
    weights = dict(default_config()["communication"]["weights"])
    weights.update((config.get("communication") or {}).get("weights") or {})
    params["weights"] = weights
    return params


def task_generation_params(config: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(default_config()["task_generation"])
    params.update(config.get("task_generation") or {})
    return params
