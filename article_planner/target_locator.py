"""
通用目标点定位工具。

目标点来自场景 JSON 的 targets，不再假设“华山五峰”。如果目标声明了 elev，
工具会在经纬度附近搜索最接近目标高程的像元；否则直接使用最近经纬度像元。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .geo import dem_rc_to_lonlat, lonlat_to_dem_rc, nearest_rc_from_lonlat, read_tiff_profile
from .scenario_config import load_scenario_config, resolve_path, scenario_output_dir, target_specs


def _best_by_elevation(
    dem: np.ndarray,
    row: int,
    col: int,
    expected_elev: float | None,
    search_radius_px: int,
) -> Tuple[int, int, float, float | None]:
    if expected_elev is None or search_radius_px <= 0:
        elev = float(dem[int(row), int(col)])
        return int(row), int(col), elev, None

    r0 = max(0, int(row) - int(search_radius_px))
    r1 = min(dem.shape[0], int(row) + int(search_radius_px) + 1)
    c0 = max(0, int(col) - int(search_radius_px))
    c1 = min(dem.shape[1], int(col) + int(search_radius_px) + 1)
    patch = dem[r0:r1, c0:c1]
    diff = np.abs(patch - float(expected_elev))
    best_idx = int(np.nanargmin(diff))
    lr, lc = np.unravel_index(best_idx, patch.shape)
    best_row = r0 + int(lr)
    best_col = c0 + int(lc)
    best_elev = float(dem[best_row, best_col])
    return best_row, best_col, best_elev, abs(best_elev - float(expected_elev))


def locate_targets_from_crop(
    cfg: Dict[str, Any],
    workdir: str | Path = ".",
    search_radius_px: int = 30,
) -> List[Dict[str, Any]]:
    """在场景输出目录的裁剪缓存中定位目标点。"""
    out_dir = scenario_output_dir(cfg, workdir)
    z = np.asarray(np.load(out_dir / "Z_crop.npy"), dtype=float)
    geo = np.load(out_dir / "Z_crop_geo.npz")
    lon_grid = np.asarray(geo["lon_grid"], dtype=float)
    lat_grid = np.asarray(geo["lat_grid"], dtype=float)

    rows: List[Dict[str, Any]] = []
    for name, spec in target_specs(cfg).items():
        rough_row, rough_col = nearest_rc_from_lonlat(lon_grid, lat_grid, float(spec["lon"]), float(spec["lat"]))
        expected = float(spec["elev"]) if "elev" in spec else None
        row, col, elev, elev_error = _best_by_elevation(z, rough_row, rough_col, expected, search_radius_px)
        rows.append(
            {
                "name": name,
                "source": "crop_cache",
                "row": int(row),
                "col": int(col),
                "lon": float(lon_grid[row, col]),
                "lat": float(lat_grid[row, col]),
                "elevation_m": elev,
                "expected_elevation_m": expected,
                "elevation_error_m": elev_error,
                "rough_row": int(rough_row),
                "rough_col": int(rough_col),
            }
        )
    return rows


def locate_targets_from_dem(
    cfg: Dict[str, Any],
    workdir: str | Path = ".",
    search_radius_px: int = 30,
) -> List[Dict[str, Any]]:
    """直接在源 DEM 中定位目标点，不依赖裁剪缓存。"""
    tif_path = resolve_path(str(cfg.get("dem_path")), workdir)
    profile = read_tiff_profile(tif_path, fallback_crs=cfg.get("source_crs"))
    dem = profile.dem
    dem[dem < -9000] = np.nan

    rows: List[Dict[str, Any]] = []
    for name, spec in target_specs(cfg).items():
        rough_row, rough_col = lonlat_to_dem_rc(
            float(spec["lon"]),
            float(spec["lat"]),
            profile.x0,
            profile.y0,
            profile.sx,
            profile.sy,
            profile.source_crs,
        )
        rough_row = int(np.clip(rough_row, 0, dem.shape[0] - 1))
        rough_col = int(np.clip(rough_col, 0, dem.shape[1] - 1))
        expected = float(spec["elev"]) if "elev" in spec else None
        row, col, elev, elev_error = _best_by_elevation(dem, rough_row, rough_col, expected, search_radius_px)
        lon, lat = dem_rc_to_lonlat(row, col, profile.x0, profile.y0, profile.sx, profile.sy, profile.source_crs)
        rows.append(
            {
                "name": name,
                "source": "source_dem",
                "row": int(row),
                "col": int(col),
                "lon": lon,
                "lat": lat,
                "elevation_m": elev,
                "expected_elevation_m": expected,
                "elevation_error_m": elev_error,
                "rough_row": int(rough_row),
                "rough_col": int(rough_col),
            }
        )
    return rows


def locate_targets(
    scenario_config: str | Path | None,
    workdir: str | Path = ".",
    source: str = "auto",
    search_radius_px: int = 30,
) -> Dict[str, Any]:
    """按指定来源定位目标点并返回可序列化结果。"""
    cfg = load_scenario_config(scenario_config, workdir)
    out_dir = scenario_output_dir(cfg, workdir)
    use_crop = source == "crop" or (
        source == "auto"
        and (out_dir / "Z_crop.npy").exists()
        and (out_dir / "Z_crop_geo.npz").exists()
    )
    rows = locate_targets_from_crop(cfg, workdir, search_radius_px) if use_crop else locate_targets_from_dem(
        cfg,
        workdir,
        search_radius_px,
    )
    return {
        "scene_name": cfg.get("scene_name", "default"),
        "config_path": cfg.get("_config_path", ""),
        "source": "crop_cache" if use_crop else "source_dem",
        "search_radius_px": int(search_radius_px),
        "targets": rows,
    }


def write_target_locations(payload: Dict[str, Any], output_path: str | Path) -> Path:
    """写出目标定位结果 JSON。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
