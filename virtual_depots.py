"""
虚拟配送站自动生成。

输入 DEM、经纬度网格、目标点和可选风险场，输出满足低坡度、低海拔、
低风险、远离目标点并靠近研究区边缘或山脚过渡区的配送站候选点。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from article_planner.scenario_config import depot_params, load_scenario_config, scenario_output_dir, target_specs


RESOLUTION_M = 12.5


def nearest_rc_by_lonlat(lon_grid: np.ndarray, lat_grid: np.ndarray, lon: float, lat: float) -> Tuple[int, int]:
    d2 = (lon_grid - lon) ** 2 + (lat_grid - lat) ** 2
    idx = int(np.argmin(d2))
    r, c = np.unravel_index(idx, lon_grid.shape)
    return int(r), int(c)


def _normalise(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=float)
    vmin = float(np.nanmin(values[finite]))
    vmax = float(np.nanmax(values[finite]))
    if vmax <= vmin + 1e-12:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)


def _target_rcs(
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    targets: Dict[str, Dict[str, Any]],
) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for spec in targets.values():
        if "row" in spec and "col" in spec:
            out.append((int(spec["row"]), int(spec["col"])))
        elif "lon" in spec and "lat" in spec:
            out.append(nearest_rc_by_lonlat(lon_grid, lat_grid, float(spec["lon"]), float(spec["lat"])))
    return out


def generate_virtual_depots(
    z: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    targets: Dict[str, Dict[str, Any]],
    params: Dict[str, Any],
    risk_human: Optional[np.ndarray] = None,
    resolution_m: float = RESOLUTION_M,
) -> List[Dict[str, Any]]:
    rows, cols = z.shape
    count = max(0, int(params.get("count", 2)))
    if count == 0:
        return []

    zf = np.asarray(z, dtype=float)
    gy, gx = np.gradient(zf, float(resolution_m), float(resolution_m))
    slope_deg = np.degrees(np.arctan(np.sqrt(gx * gx + gy * gy)))

    slope_max = float(params.get("slope_max_deg", 12.0))
    elev_pct = float(params.get("elevation_percentile_max", 35.0))
    foot_pct = float(params.get("foot_elevation_percentile", 20.0))
    edge_buffer_km = float(params.get("edge_buffer_km", 1.0))
    min_target_km = float(params.get("min_target_distance_km", 1.2))
    min_depot_km = float(params.get("min_depot_spacing_km", 1.5))
    risk_max = float(params.get("risk_max", 0.2))
    name_prefix = str(params.get("name_prefix", "虚拟配送站"))

    finite = np.isfinite(zf)
    elev_limit = float(np.nanpercentile(zf[finite], elev_pct))
    foot_limit = float(np.nanpercentile(zf[finite], foot_pct))

    rr, cc = np.indices(zf.shape)
    edge_dist_px = np.minimum.reduce([rr, cc, rows - 1 - rr, cols - 1 - cc]).astype(float)
    edge_dist_km = edge_dist_px * float(resolution_m) / 1000.0
    edge_or_foot = (edge_dist_km <= edge_buffer_km) | (zf <= foot_limit)

    target_rcs = _target_rcs(lon_grid, lat_grid, targets)
    if target_rcs:
        min_target_dist_km = np.full(zf.shape, np.inf, dtype=float)
        for tr, tc in target_rcs:
            dist = np.sqrt((rr - tr) ** 2 + (cc - tc) ** 2) * float(resolution_m) / 1000.0
            min_target_dist_km = np.minimum(min_target_dist_km, dist)
    else:
        min_target_dist_km = np.full(zf.shape, np.inf, dtype=float)

    risk = np.zeros_like(zf, dtype=float)
    if risk_human is not None and risk_human.shape == zf.shape:
        risk = np.clip(np.asarray(risk_human, dtype=float), 0.0, 1.0)

    mask = (
        finite
        & (slope_deg <= slope_max)
        & (zf <= elev_limit)
        & edge_or_foot
        & (min_target_dist_km >= min_target_km)
        & (risk <= risk_max)
    )

    cand = np.argwhere(mask)
    if cand.size == 0:
        return []

    slope_score = _normalise(slope_deg)
    elev_score = _normalise(zf)
    edge_score = np.clip(edge_dist_km / max(edge_buffer_km, 1e-9), 0.0, 1.0)
    target_bonus = 1.0 - np.clip(min_target_dist_km / max(min_target_km * 3.0, 1e-9), 0.0, 1.0)
    score = 0.35 * slope_score + 0.30 * elev_score + 0.20 * risk + 0.10 * edge_score + 0.05 * target_bonus

    order = sorted(cand.tolist(), key=lambda rc: float(score[int(rc[0]), int(rc[1])]))
    selected: List[Tuple[int, int]] = []
    for r, c in order:
        ok = True
        for sr, sc in selected:
            d_km = np.sqrt((r - sr) ** 2 + (c - sc) ** 2) * float(resolution_m) / 1000.0
            if d_km < min_depot_km:
                ok = False
                break
        if ok:
            selected.append((int(r), int(c)))
        if len(selected) >= count:
            break

    # 候选过少时降低站点间距要求，但不放宽安全、坡度和风险规则。
    if len(selected) < count:
        used = set(selected)
        for r, c in order:
            key = (int(r), int(c))
            if key in used:
                continue
            selected.append(key)
            used.add(key)
            if len(selected) >= count:
                break

    depots: List[Dict[str, Any]] = []
    for idx, (r, c) in enumerate(selected, start=1):
        depots.append(
            {
                "name": f"{name_prefix}{idx}",
                "row": int(r),
                "col": int(c),
                "lon": float(lon_grid[r, c]),
                "lat": float(lat_grid[r, c]),
                "elev": float(zf[r, c]),
                "slope_deg": float(slope_deg[r, c]),
                "risk": float(risk[r, c]),
                "edge_distance_km": float(edge_dist_km[r, c]),
                "min_target_distance_km": float(min_target_dist_km[r, c]),
                "rule": "low_slope_low_elevation_low_risk_edge_or_foothill",
            }
        )
    return depots


def main() -> None:
    parser = argparse.ArgumentParser(description="根据场景 DEM 自动生成虚拟配送站。")
    parser.add_argument("--scenario-config", type=str, default="")
    parser.add_argument("--workdir", type=str, default=".")
    parser.add_argument("--z-file", type=str, default="Z_crop.npy")
    parser.add_argument("--geo-file", type=str, default="Z_crop_geo.npz")
    parser.add_argument("--risk-file", type=str, default="risk_human.npy")
    parser.add_argument("--out-file", type=str, default="generated_depots.json")
    args = parser.parse_args()

    cfg = load_scenario_config(args.scenario_config or None, args.workdir)
    out_dir = scenario_output_dir(cfg, args.workdir)
    z_path = out_dir / args.z_file
    geo_path = out_dir / args.geo_file
    risk_path = out_dir / args.risk_file

    z = np.asarray(np.load(z_path), dtype=float)
    geo = np.load(geo_path)
    lon_grid = np.asarray(geo["lon_grid"], dtype=float)
    lat_grid = np.asarray(geo["lat_grid"], dtype=float)
    risk = np.asarray(np.load(risk_path), dtype=float) if risk_path.exists() else None

    depots = generate_virtual_depots(
        z,
        lon_grid,
        lat_grid,
        target_specs(cfg),
        depot_params(cfg),
        risk_human=risk,
        resolution_m=float((cfg.get("crop") or {}).get("resolution_m", RESOLUTION_M)),
    )
    payload = {"scene_name": cfg.get("scene_name", "default"), "depots": depots}
    (out_dir / args.out_file).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] 已生成 {len(depots)} 个虚拟配送站: {out_dir / args.out_file}")


if __name__ == "__main__":
    main()
