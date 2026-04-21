"""
自动任务生成器。

输入 DEM、场景目标点和虚拟配送站规则，输出可复用的配送任务集合。
该模块用于多 DEM 泛化实验，避免每个山体场景都手工指定起终点组合。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from article_planner.scenario_config import (
    depot_params,
    load_scenario_config,
    scenario_output_dir,
    target_specs,
    task_generation_params,
)
from virtual_depots import generate_virtual_depots


RESOLUTION_M = 12.5


def pixel_to_km(r: int, c: int, rows: int, resolution_m: float) -> Tuple[float, float]:
    return c * resolution_m / 1000.0, (rows - 1 - r) * resolution_m / 1000.0


def nearest_rc_by_lonlat(lon_grid: np.ndarray, lat_grid: np.ndarray, lon: float, lat: float) -> Tuple[int, int]:
    d2 = (lon_grid - lon) ** 2 + (lat_grid - lat) ** 2
    idx = int(np.argmin(d2))
    r, c = np.unravel_index(idx, lon_grid.shape)
    return int(r), int(c)


def slope_degrees(z: np.ndarray, resolution_m: float) -> np.ndarray:
    z_smooth = gaussian_filter(np.asarray(z, dtype=float), sigma=2.0)
    gy, gx = np.gradient(z_smooth, float(resolution_m), float(resolution_m))
    return np.degrees(np.arctan(np.sqrt(gx * gx + gy * gy)))


def load_or_generate_depots(
    z: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    out_dir: Path,
    cfg: Dict[str, Any],
    risk_human: np.ndarray | None,
    resolution_m: float,
    count: int,
) -> List[Dict[str, Any]]:
    depot_path = out_dir / "generated_depots.json"
    if depot_path.exists():
        depots = json.loads(depot_path.read_text(encoding="utf-8")).get("depots", [])
        if depots:
            return [dict(v) for v in depots[:count]]

    params = depot_params(cfg)
    params["count"] = int(count)
    depots = generate_virtual_depots(
        z,
        lon_grid,
        lat_grid,
        target_specs(cfg),
        params,
        risk_human=risk_human,
        resolution_m=resolution_m,
    )
    payload = {
        "scene_name": cfg.get("scene_name", "default"),
        "rule": "低坡度、低海拔、低风险、远离目标点且位于边缘或山脚过渡区",
        "depots": depots,
    }
    depot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return depots


def configured_targets(
    z: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    cfg: Dict[str, Any],
    risk_human: np.ndarray,
    slope: np.ndarray,
    resolution_m: float,
) -> List[Dict[str, Any]]:
    rows, _cols = z.shape
    out: List[Dict[str, Any]] = []
    for name, spec in target_specs(cfg).items():
        r, c = nearest_rc_by_lonlat(lon_grid, lat_grid, float(spec["lon"]), float(spec["lat"]))
        x, y = pixel_to_km(r, c, rows, resolution_m)
        out.append(
            {
                "name": str(name),
                "source": "configured_target",
                "row": int(r),
                "col": int(c),
                "x_km": float(x),
                "y_km": float(y),
                "lon": float(lon_grid[r, c]),
                "lat": float(lat_grid[r, c]),
                "elevation_m": float(z[r, c]),
                "slope_deg": float(slope[r, c]),
                "risk_human": float(risk_human[r, c]),
            }
        )
    return out


def auto_target_candidates(
    z: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    risk_human: np.ndarray,
    slope: np.ndarray,
    params: Dict[str, Any],
    resolution_m: float,
    existing: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows, cols = z.shape
    target_count = int(params.get("target_count", 8))
    need = max(0, target_count - len(existing))
    if need <= 0:
        return []

    elev_min = float(np.percentile(z, float(params.get("target_elevation_percentile_min", 65.0))))
    max_slope = float(params.get("max_target_slope_deg", 38.0))
    risk_max = float(params.get("risk_max", 0.65))
    min_spacing_px = max(1, int(round(float(params.get("target_min_spacing_km", 0.8)) * 1000.0 / resolution_m)))
    z_smooth = gaussian_filter(np.asarray(z, dtype=float), sigma=2.0)
    local_peak = z_smooth == maximum_filter(z_smooth, size=25)
    mask = (z >= elev_min) & (slope <= max_slope) & (risk_human <= risk_max) & local_peak
    if not np.any(mask):
        mask = (z >= elev_min) & (slope <= max_slope) & (risk_human <= risk_max)

    score = (z - np.nanmin(z)) / max(float(np.nanmax(z) - np.nanmin(z)), 1e-9)
    score = score + 0.25 * (1.0 - np.clip(risk_human, 0.0, 1.0)) - 0.15 * np.clip(slope / 60.0, 0.0, 1.0)
    flats = np.argwhere(mask)
    order = np.argsort(score[mask])[::-1]

    used = [(int(v["row"]), int(v["col"])) for v in existing]
    out: List[Dict[str, Any]] = []
    for idx in order:
        r, c = int(flats[idx, 0]), int(flats[idx, 1])
        ok = True
        for ur, uc in used:
            if (r - ur) * (r - ur) + (c - uc) * (c - uc) < min_spacing_px * min_spacing_px:
                ok = False
                break
        if not ok:
            continue
        x, y = pixel_to_km(r, c, rows, resolution_m)
        out.append(
            {
                "name": f"自动目标{len(out) + 1}",
                "source": "auto_dem_peak",
                "row": int(r),
                "col": int(c),
                "x_km": float(x),
                "y_km": float(y),
                "lon": float(lon_grid[r, c]),
                "lat": float(lat_grid[r, c]),
                "elevation_m": float(z[r, c]),
                "slope_deg": float(slope[r, c]),
                "risk_human": float(risk_human[r, c]),
            }
        )
        used.append((r, c))
        if len(out) >= need:
            break
    return out


def terrain_profile(z: np.ndarray, a: Dict[str, Any], b: Dict[str, Any], samples: int = 80) -> np.ndarray:
    rr = np.linspace(float(a["row"]), float(b["row"]), samples)
    cc = np.linspace(float(a["col"]), float(b["col"]), samples)
    rr = np.clip(np.rint(rr).astype(int), 0, z.shape[0] - 1)
    cc = np.clip(np.rint(cc).astype(int), 0, z.shape[1] - 1)
    return z[rr, cc].astype(float)


def classify_pair(
    z: np.ndarray,
    depot: Dict[str, Any],
    target: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    dx = float(target["x_km"]) - float(depot["x_km"])
    dy = float(target["y_km"]) - float(depot["y_km"])
    distance_km = float(np.sqrt(dx * dx + dy * dy))
    elev_gain = float(target["elevation_m"]) - float(depot.get("elevation_m", z[int(depot["row"]), int(depot["col"])]))
    profile = terrain_profile(z, depot, target)
    inner = profile[1:-1] if profile.size > 2 else profile
    ridge_relief = float(np.max(inner) - min(float(profile[0]), float(profile[-1]))) if inner.size else 0.0
    inner_prom = float(np.max(inner) - max(float(profile[0]), float(profile[-1]))) if inner.size else 0.0

    d_bins = [float(v) for v in params.get("distance_bins_km", [2.5, 5.0])]
    e_bins = [float(v) for v in params.get("elevation_bins_m", [300.0, 900.0])]
    ridge_thr = float(params.get("ridge_prominence_m", 120.0))
    if distance_km <= d_bins[0]:
        d_class = "short"
    elif distance_km <= d_bins[1]:
        d_class = "medium"
    else:
        d_class = "long"

    if abs(elev_gain) <= e_bins[0]:
        e_class = "low_delta"
    elif abs(elev_gain) <= e_bins[1]:
        e_class = "medium_delta"
    else:
        e_class = "high_delta"

    if inner_prom > ridge_thr:
        ridge_class = "cross_ridge"
    elif ridge_relief > max(ridge_thr, abs(elev_gain) * 0.45):
        ridge_class = "ridge_approach"
    else:
        ridge_class = "direct_slope"

    return {
        "distance_km": distance_km,
        "elevation_gain_m": elev_gain,
        "ridge_relief_m": ridge_relief,
        "inner_ridge_prominence_m": inner_prom,
        "distance_class": d_class,
        "elevation_class": e_class,
        "ridge_class": ridge_class,
    }


def stratified_pairs(
    z: np.ndarray,
    depots: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    min_dist = float(params.get("min_pair_distance_km", 1.5))
    pair_count = int(params.get("pair_count", 12))
    all_pairs: List[Dict[str, Any]] = []
    for depot in depots:
        d = dict(depot)
        if "elevation_m" not in d:
            d["elevation_m"] = float(z[int(d["row"]), int(d["col"])])
        for target in targets:
            metrics = classify_pair(z, d, target, params)
            if metrics["distance_km"] < min_dist:
                continue
            all_pairs.append(
                {
                    "depot": str(d["name"]),
                    "target": str(target["name"]),
                    "depot_source": str(d.get("source", "virtual_depot")),
                    "target_source": str(target.get("source", "target")),
                    **metrics,
                }
            )

    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for pair in all_pairs:
        key = (pair["distance_class"], pair["elevation_class"], pair["ridge_class"])
        groups.setdefault(key, []).append(pair)
    for items in groups.values():
        items.sort(key=lambda x: (-abs(float(x["elevation_gain_m"])), float(x["distance_km"])))

    selected: List[Dict[str, Any]] = []
    while len(selected) < pair_count and groups:
        progressed = False
        for key in sorted(list(groups.keys())):
            if not groups[key]:
                groups.pop(key, None)
                continue
            item = groups[key].pop(0)
            item["task_id"] = f"T{len(selected) + 1:03d}"
            item["stratum"] = "/".join(key)
            selected.append(item)
            progressed = True
            if len(selected) >= pair_count:
                break
        if not progressed:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="根据 DEM 自动生成配送任务候选集。")
    parser.add_argument("--scenario-config", type=str, default="")
    parser.add_argument("--workdir", type=str, default=".")
    parser.add_argument("--target-count", type=int, default=-1)
    parser.add_argument("--pair-count", type=int, default=-1)
    parser.add_argument("--output", type=str, default="generated_tasks.json")
    args = parser.parse_args()

    root = Path(args.workdir).resolve()
    cfg = load_scenario_config(args.scenario_config or None, root)
    out_dir = scenario_output_dir(cfg, root)
    params = task_generation_params(cfg)
    if int(args.target_count) > 0:
        params["target_count"] = int(args.target_count)
    if int(args.pair_count) > 0:
        params["pair_count"] = int(args.pair_count)

    z = np.asarray(np.load(out_dir / "Z_crop.npy"), dtype=float)
    geo = np.load(out_dir / "Z_crop_geo.npz")
    lon_grid = np.asarray(geo["lon_grid"], dtype=float)
    lat_grid = np.asarray(geo["lat_grid"], dtype=float)
    resolution_m = float((cfg.get("crop") or {}).get("resolution_m", RESOLUTION_M))
    risk_human = np.zeros_like(z, dtype=float)
    risk_path = out_dir / "risk_human.npy"
    if risk_path.exists():
        arr = np.asarray(np.load(risk_path), dtype=float)
        if arr.shape == z.shape:
            risk_human = np.clip(arr, 0.0, 1.0)

    slope = slope_degrees(z, resolution_m)
    depots = load_or_generate_depots(
        z,
        lon_grid,
        lat_grid,
        out_dir,
        cfg,
        risk_human,
        resolution_m,
        int(params.get("depot_count", 2)),
    )
    for d in depots:
        d["elevation_m"] = float(z[int(d["row"]), int(d["col"])])
        d["x_km"], d["y_km"] = pixel_to_km(int(d["row"]), int(d["col"]), z.shape[0], resolution_m)

    targets = configured_targets(z, lon_grid, lat_grid, cfg, risk_human, slope, resolution_m)
    targets.extend(auto_target_candidates(z, lon_grid, lat_grid, risk_human, slope, params, resolution_m, targets))
    tasks = stratified_pairs(z, depots, targets, params)

    payload = {
        "scene_name": cfg.get("scene_name", "default"),
        "params": params,
        "depots": depots,
        "targets": targets,
        "tasks": tasks,
        "summary": {
            "depot_count": len(depots),
            "target_count": len(targets),
            "task_count": len(tasks),
            "strata": sorted({t["stratum"] for t in tasks}),
        },
    }
    output_path = out_dir / args.output
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[任务生成] 已保存 {output_path}")
    print(
        f"[任务生成] depots={len(depots)}, targets={len(targets)}, "
        f"tasks={len(tasks)}, strata={len(payload['summary']['strata'])}"
    )


if __name__ == "__main__":
    main()
