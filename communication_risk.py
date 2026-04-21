"""
基于 DEM 视距的通信风险建模。

第一版采用工程可解释的软约束：地面通信源到飞行层节点无视距或距离较远时，
提高通信风险；规划阶段把该风险并入边权，同时输出路径覆盖和失联指标。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.ndimage import zoom

from article_planner.scenario_config import communication_params, load_scenario_config, scenario_output_dir


RESOLUTION_M = 12.5


def km_to_rc(x_km: float, y_km: float, rows: int, cols: int, resolution_m: float) -> Tuple[int, int]:
    c = int(np.clip(x_km * 1000.0 / resolution_m, 0, cols - 1))
    r = int(np.clip((rows - 1) - y_km * 1000.0 / resolution_m, 0, rows - 1))
    return r, c


def pixel_to_km(r: int, c: int, rows: int, resolution_m: float) -> Tuple[float, float]:
    return c * resolution_m / 1000.0, (rows - 1 - r) * resolution_m / 1000.0


def _nearest_rc_by_lonlat(lon_grid: np.ndarray, lat_grid: np.ndarray, lon: float, lat: float) -> Tuple[int, int]:
    d2 = (lon_grid - lon) ** 2 + (lat_grid - lat) ** 2
    idx = int(np.argmin(d2))
    r, c = np.unravel_index(idx, lon_grid.shape)
    return int(r), int(c)


def _edge_lowland_sources(z: np.ndarray, count: int, height_agl_m: float, resolution_m: float) -> List[Dict[str, float]]:
    rows, cols = z.shape
    if count <= 0:
        return []
    rr, cc = np.indices(z.shape)
    edge_dist = np.minimum.reduce([rr, cc, rows - 1 - rr, cols - 1 - cc])
    edge_mask = edge_dist <= max(2, int(round(600.0 / resolution_m)))
    score = np.where(edge_mask, z, np.inf)
    order = np.argsort(score.ravel())
    out: List[Dict[str, float]] = []
    used: List[Tuple[int, int]] = []
    min_spacing_px = int(round(1500.0 / resolution_m))
    for flat in order:
        r, c = np.unravel_index(int(flat), z.shape)
        if not np.isfinite(score[r, c]):
            continue
        ok = True
        for ur, uc in used:
            if (r - ur) * (r - ur) + (c - uc) * (c - uc) < min_spacing_px * min_spacing_px:
                ok = False
                break
        if not ok:
            continue
        x, y = pixel_to_km(int(r), int(c), rows, resolution_m)
        out.append(
            {
                "name": f"边缘通信源{len(out) + 1}",
                "row": int(r),
                "col": int(c),
                "x_km": float(x),
                "y_km": float(y),
                "height_m": float(z[r, c] + height_agl_m),
                "source": "lowland_edge",
            }
        )
        used.append((int(r), int(c)))
        if len(out) >= count:
            break
    return out


def load_sources(
    z: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    out_dir: Path,
    params: Dict[str, Any],
    resolution_m: float,
) -> List[Dict[str, float]]:
    rows, cols = z.shape
    height_agl = float(params.get("source_height_agl_m", 35.0))
    sources: List[Dict[str, float]] = []

    for item in params.get("base_stations", []) or []:
        if "row" in item and "col" in item:
            r, c = int(item["row"]), int(item["col"])
        elif "lon" in item and "lat" in item:
            r, c = _nearest_rc_by_lonlat(lon_grid, lat_grid, float(item["lon"]), float(item["lat"]))
        else:
            continue
        x, y = pixel_to_km(r, c, rows, resolution_m)
        height = float(item.get("height_m", z[r, c] + float(item.get("height_agl_m", height_agl))))
        sources.append(
            {
                "name": str(item.get("name", f"配置通信源{len(sources) + 1}")),
                "row": int(r),
                "col": int(c),
                "x_km": float(x),
                "y_km": float(y),
                "height_m": float(height),
                "source": "configured",
            }
        )

    depot_path = out_dir / "generated_depots.json"
    if depot_path.exists():
        depots = json.loads(depot_path.read_text(encoding="utf-8")).get("depots", [])
        for depot in depots:
            if "row" not in depot or "col" not in depot:
                continue
            r, c = int(depot["row"]), int(depot["col"])
            x, y = pixel_to_km(r, c, rows, resolution_m)
            sources.append(
                {
                    "name": str(depot.get("name", f"配送站通信源{len(sources) + 1}")),
                    "row": int(r),
                    "col": int(c),
                    "x_km": float(x),
                    "y_km": float(y),
                    "height_m": float(z[r, c] + height_agl),
                    "source": "virtual_depot",
                }
            )

    sources.extend(_edge_lowland_sources(z, int(params.get("edge_source_count", 2)), height_agl, resolution_m))
    # 去重，防止配送站正好也是边缘低地候选点。
    dedup: List[Dict[str, float]] = []
    seen = set()
    for src in sources:
        key = (int(src["row"]), int(src["col"]))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(src)
    return dedup


def line_of_sight(
    z: np.ndarray,
    src: Dict[str, float],
    dst_r: int,
    dst_c: int,
    dst_z: float,
    samples: int,
) -> bool:
    sr, sc = int(src["row"]), int(src["col"])
    sz = float(src["height_m"])
    for t in np.linspace(0.0, 1.0, max(2, int(samples))):
        r = int(round(sr + t * (dst_r - sr)))
        c = int(round(sc + t * (dst_c - sc)))
        r = int(np.clip(r, 0, z.shape[0] - 1))
        c = int(np.clip(c, 0, z.shape[1] - 1))
        line_z = sz + t * (float(dst_z) - sz)
        if float(z[r, c]) > line_z - 2.0:
            return False
    return True


def build_comm_risk(
    z: np.ndarray,
    layer_mid: np.ndarray,
    sources: List[Dict[str, float]],
    params: Dict[str, Any],
    resolution_m: float,
) -> np.ndarray:
    rows, cols = z.shape
    stride = max(1, int(params.get("coarse_stride", 4)))
    los_samples = max(4, int(params.get("los_samples", 18)))
    max_range_km = max(0.1, float(params.get("max_range_km", 5.0)))
    rr = np.arange(0, rows, stride, dtype=int)
    cc = np.arange(0, cols, stride, dtype=int)
    risk_small = np.ones((3, len(rr), len(cc)), dtype=float)

    if not sources:
        return risk_small.repeat(stride, axis=1)[:, :rows, :].repeat(stride, axis=2)[:, :, :cols].astype(np.float32)

    for lid in range(3):
        for ir, r in enumerate(rr):
            y_km = (rows - 1 - int(r)) * resolution_m / 1000.0
            for ic, c in enumerate(cc):
                x_km = int(c) * resolution_m / 1000.0
                dst_z = float(layer_mid[lid, r, c])
                best = 1.0
                for src in sources:
                    d = float(np.sqrt((x_km - src["x_km"]) ** 2 + (y_km - src["y_km"]) ** 2))
                    dist_risk = min(1.0, d / max_range_km)
                    has_los = line_of_sight(z, src, int(r), int(c), dst_z, los_samples)
                    cur = 0.25 * dist_risk if has_los else min(1.0, 0.75 + 0.25 * dist_risk)
                    if cur < best:
                        best = cur
                risk_small[lid, ir, ic] = best

    zoom_factors = (1.0, rows / risk_small.shape[1], cols / risk_small.shape[2])
    risk = zoom(risk_small, zoom_factors, order=1)
    risk = np.clip(risk[:, :rows, :cols], 0.0, 1.0).astype(np.float32)
    return risk


def main() -> None:
    parser = argparse.ArgumentParser(description="基于 DEM 视距生成三层通信风险栅格。")
    parser.add_argument("--scenario-config", type=str, default="")
    parser.add_argument("--workdir", type=str, default=".")
    args = parser.parse_args()

    root = Path(args.workdir).resolve()
    cfg = load_scenario_config(args.scenario_config or None, root)
    out_dir = scenario_output_dir(cfg, root)
    params = communication_params(cfg)
    if not bool(params.get("enabled", True)):
        print("[通信] 配置中已关闭通信风险建模。")
        return

    z = np.asarray(np.load(out_dir / "Z_crop.npy"), dtype=float)
    layer_mid = np.asarray(np.load(out_dir / "layer_mid.npy"), dtype=float)
    geo = np.load(out_dir / "Z_crop_geo.npz")
    lon_grid = np.asarray(geo["lon_grid"], dtype=float)
    lat_grid = np.asarray(geo["lat_grid"], dtype=float)
    resolution_m = float((cfg.get("crop") or {}).get("resolution_m", RESOLUTION_M))

    sources = load_sources(z, lon_grid, lat_grid, out_dir, params, resolution_m)
    risk = build_comm_risk(z, layer_mid, sources, params, resolution_m)
    np.save(out_dir / "risk_comm.npy", risk.astype(np.float32))

    summary = {
        "scene_name": cfg.get("scene_name", "default"),
        "sources": sources,
        "risk_shape": list(risk.shape),
        "risk_min": float(np.min(risk)),
        "risk_max": float(np.max(risk)),
        "risk_mean_by_layer": [float(np.mean(risk[i])) for i in range(risk.shape[0])],
        "coverage_ratio_by_layer": [
            float(np.mean(risk[i] <= float(params.get("risk_threshold", 0.55)))) for i in range(risk.shape[0])
        ],
        "params": params,
    }
    (out_dir / "communication_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[通信] 已生成 risk_comm.npy，shape={risk.shape}, source_n={len(sources)}")
    print(f"[通信] 摘要: {out_dir / 'communication_summary.json'}")


if __name__ == "__main__":
    main()
