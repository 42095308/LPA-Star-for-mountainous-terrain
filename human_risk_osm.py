"""
Build human exposure risk raster from OpenStreetMap for the Huashan study area.

Outputs:
    risk_trail.npy
    risk_hotspot.npy
    risk_human.npy
    osm_feature_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


RESOLUTION_M = 12.5
OVERPASS_URLS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]


def overpass_query(query: str, timeout_s: int = 120) -> dict:
    last_err = None
    for url in OVERPASS_URLS:
        try:
            resp = requests.post(url, data={"data": query}, timeout=timeout_s)
            if resp.status_code != 200:
                last_err = RuntimeError(f"{url} status={resp.status_code}")
                continue
            return resp.json()
        except Exception as exc:  # pragma: no cover
            last_err = exc
    raise RuntimeError(f"Overpass request failed: {last_err}")


def make_query(lat_min: float, lon_min: float, lat_max: float, lon_max: float) -> str:
    bbox = f"{lat_min:.8f},{lon_min:.8f},{lat_max:.8f},{lon_max:.8f}"
    return f"""
[out:json][timeout:120];
(
  way["highway"~"path|footway|track|steps|pedestrian"]({bbox});
  relation["route"="hiking"]({bbox});
  way(r);

  way["aerialway"]({bbox});
  node["aerialway"="station"]({bbox});
  way["aerialway"="station"]({bbox});

  node["tourism"="viewpoint"]({bbox});
  way["tourism"="viewpoint"]({bbox});
  node["natural"="peak"]({bbox});
  node["tourism"="attraction"]({bbox});
  way["tourism"="attraction"]({bbox});
);
out tags geom;
"""


def make_fallback_query(lat_min: float, lon_min: float, lat_max: float, lon_max: float) -> str:
    """Broader synonym query used when some target classes are missing."""
    bbox = f"{lat_min:.8f},{lon_min:.8f},{lat_max:.8f},{lon_max:.8f}"
    return f"""
[out:json][timeout:120];
(
  way["highway"~"path|footway|track|steps|pedestrian|bridleway"]({bbox});
  relation["route"~"hiking|foot"]({bbox});
  way["sac_scale"]({bbox});
  way(r);

  way["aerialway"]({bbox});
  node["aerialway"="station"]({bbox});
  way["aerialway"="station"]({bbox});

  node["tourism"~"information|attraction|viewpoint"]({bbox});
  way["tourism"~"information|attraction|viewpoint"]({bbox});
  node["natural"="peak"]({bbox});
  way["natural"="peak"]({bbox});
  node["man_made"="observation_tower"]({bbox});
  way["man_made"="observation_tower"]({bbox});
);
out tags geom;
"""


def _tag(tags: Dict[str, str], key: str) -> str:
    return str(tags.get(key, "")).strip().lower()


def is_trail(tags: Dict[str, str]) -> bool:
    highway = _tag(tags, "highway")
    route = _tag(tags, "route")
    if highway in {"path", "footway", "track", "steps", "pedestrian"}:
        return True
    if route in {"hiking", "foot"}:
        return True
    return False


def is_cable_line(tags: Dict[str, str]) -> bool:
    aerialway = _tag(tags, "aerialway")
    if not aerialway:
        return False
    return aerialway not in {"station", "pylon", "yes"}


def is_cable_station(tags: Dict[str, str]) -> bool:
    return _tag(tags, "aerialway") == "station"


def is_viewpoint(tags: Dict[str, str]) -> bool:
    tourism = _tag(tags, "tourism")
    natural = _tag(tags, "natural")
    if tourism in {"viewpoint", "attraction"}:
        return True
    if natural == "peak":
        return True
    return False


def extract_features(osm_json: dict) -> Dict[str, List]:
    trails: List[List[Tuple[float, float]]] = []
    cable_lines: List[List[Tuple[float, float]]] = []
    cable_stations: List[Tuple[float, float]] = []
    viewpoints: List[Tuple[float, float]] = []

    for e in osm_json.get("elements", []):
        tags = e.get("tags", {}) or {}
        etype = e.get("type")

        if etype == "way" and e.get("geometry"):
            geom = [(float(p["lon"]), float(p["lat"])) for p in e["geometry"]]
            if len(geom) >= 2:
                if is_trail(tags):
                    trails.append(geom)
                if is_cable_line(tags):
                    cable_lines.append(geom)
            # way centroid for hotspot-like classes
            lon = float(np.mean([p[0] for p in geom]))
            lat = float(np.mean([p[1] for p in geom]))
            if is_cable_station(tags):
                cable_stations.append((lon, lat))
            if is_viewpoint(tags):
                viewpoints.append((lon, lat))

        elif etype == "node" and ("lon" in e and "lat" in e):
            lon = float(e["lon"])
            lat = float(e["lat"])
            if is_cable_station(tags):
                cable_stations.append((lon, lat))
            if is_viewpoint(tags):
                viewpoints.append((lon, lat))

    return {
        "trails": trails,
        "cable_lines": cable_lines,
        "cable_stations": cable_stations,
        "viewpoints": viewpoints,
    }


def dedup_lonlat_points(points: Iterable[Tuple[float, float]], ndigits: int = 7) -> List[Tuple[float, float]]:
    seen = set()
    out: List[Tuple[float, float]] = []
    for lon, lat in points:
        key = (round(float(lon), ndigits), round(float(lat), ndigits))
        if key in seen:
            continue
        seen.add(key)
        out.append((float(lon), float(lat)))
    return out


def dedup_lonlat_lines(
    lines: Iterable[Sequence[Tuple[float, float]]], ndigits: int = 7
) -> List[List[Tuple[float, float]]]:
    seen = set()
    out: List[List[Tuple[float, float]]] = []
    for line in lines:
        if len(line) < 2:
            continue
        key = tuple((round(float(lon), ndigits), round(float(lat), ndigits)) for lon, lat in line)
        if key in seen:
            continue
        seen.add(key)
        out.append([(float(lon), float(lat)) for lon, lat in line])
    return out


def merge_features(a: Dict[str, List], b: Dict[str, List]) -> Dict[str, List]:
    out = {
        "trails": dedup_lonlat_lines(list(a.get("trails", [])) + list(b.get("trails", []))),
        "cable_lines": dedup_lonlat_lines(
            list(a.get("cable_lines", [])) + list(b.get("cable_lines", []))
        ),
        "cable_stations": dedup_lonlat_points(
            list(a.get("cable_stations", [])) + list(b.get("cable_stations", []))
        ),
        "viewpoints": dedup_lonlat_points(list(a.get("viewpoints", [])) + list(b.get("viewpoints", []))),
    }
    return out


def build_lonlat_tree(lon_grid: np.ndarray, lat_grid: np.ndarray) -> Tuple[cKDTree, int, int]:
    rows, cols = lon_grid.shape
    pts = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    tree = cKDTree(pts)
    return tree, rows, cols


def lonlat_to_rc(
    lon_lat: Sequence[Tuple[float, float]],
    tree: cKDTree,
    rows: int,
    cols: int,
) -> List[Tuple[int, int]]:
    if not lon_lat:
        return []
    q = np.asarray(lon_lat, dtype=float)
    _, idx = tree.query(q, k=1)
    idx = np.asarray(idx, dtype=int)
    rr = (idx // cols).astype(int)
    cc = (idx % cols).astype(int)
    return [(int(r), int(c)) for r, c in zip(rr, cc)]


def draw_line_mask(mask: np.ndarray, rc_line: Sequence[Tuple[int, int]]) -> None:
    if len(rc_line) < 2:
        return
    rows, cols = mask.shape
    for i in range(len(rc_line) - 1):
        r0, c0 = rc_line[i]
        r1, c1 = rc_line[i + 1]
        n = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
        rr = np.linspace(r0, r1, n)
        cc = np.linspace(c0, c1, n)
        rr = np.clip(np.rint(rr).astype(int), 0, rows - 1)
        cc = np.clip(np.rint(cc).astype(int), 0, cols - 1)
        mask[rr, cc] = True


def risk_from_distance_trail(d_m: np.ndarray) -> np.ndarray:
    out = np.zeros_like(d_m, dtype=float)
    m1 = d_m <= 30.0
    out[m1] = 1.0

    m2 = (d_m > 30.0) & (d_m <= 80.0)
    # 30m -> 1.0, 80m -> 0.3
    out[m2] = 1.0 - 0.7 * (d_m[m2] - 30.0) / 50.0

    m3 = d_m > 80.0
    out[m3] = 0.3 * np.exp(-(d_m[m3] - 80.0) / 120.0)
    return np.clip(out, 0.0, 1.0)


def risk_from_distance_hotspot(d_m: np.ndarray, sigma_m: float = 120.0) -> np.ndarray:
    out = np.exp(-(d_m ** 2) / (2.0 * sigma_m * sigma_m))
    return np.clip(out, 0.0, 1.0)


def dedup_points(points: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    seen = set()
    out = []
    for r, c in points:
        key = (int(r), int(c))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OSM-based human risk raster aligned with Z_crop.npy")
    parser.add_argument("--workdir", type=str, default=".")
    parser.add_argument("--z-file", type=str, default="Z_crop.npy")
    parser.add_argument("--geo-file", type=str, default="Z_crop_geo.npz")
    parser.add_argument("--out-trail", type=str, default="risk_trail.npy")
    parser.add_argument("--out-hotspot", type=str, default="risk_hotspot.npy")
    parser.add_argument("--out-human", type=str, default="risk_human.npy")
    parser.add_argument("--summary-json", type=str, default="osm_feature_summary.json")
    parser.add_argument("--out-preview", type=str, default="osm_human_risk_preview.png")
    parser.add_argument("--w-trail", type=float, default=0.6)
    parser.add_argument("--w-hotspot", type=float, default=0.4)
    parser.add_argument("--request-timeout", type=int, default=120)
    args = parser.parse_args()

    root = Path(args.workdir).resolve()
    z = np.load(root / args.z_file)
    geo = np.load(root / args.geo_file)
    lon_grid = np.asarray(geo["lon_grid"], dtype=float)
    lat_grid = np.asarray(geo["lat_grid"], dtype=float)

    if z.shape != lon_grid.shape or z.shape != lat_grid.shape:
        raise RuntimeError(f"Shape mismatch: Z={z.shape}, lon={lon_grid.shape}, lat={lat_grid.shape}")

    lat_min = float(np.min(lat_grid))
    lat_max = float(np.max(lat_grid))
    lon_min = float(np.min(lon_grid))
    lon_max = float(np.max(lon_grid))

    print("[1/5] study area bbox (WGS84):")
    print(f"      lat: {lat_min:.6f} .. {lat_max:.6f}")
    print(f"      lon: {lon_min:.6f} .. {lon_max:.6f}")

    print("[2/5] fetching OSM elements from Overpass...")
    query = make_query(lat_min, lon_min, lat_max, lon_max)
    t0 = time.perf_counter()
    osm = overpass_query(query=query, timeout_s=args.request_timeout)
    t1 = time.perf_counter()
    print(f"      fetched {len(osm.get('elements', []))} elements in {(t1 - t0):.2f}s")

    feats = extract_features(osm)
    missing_before = [
        k
        for k in ["trails", "cable_lines", "viewpoints"]
        if len(feats[k]) == 0
    ]
    used_fallback_query = False
    if missing_before:
        used_fallback_query = True
        print(f"[2b/5] missing classes in primary query: {missing_before}")
        print("       trying fallback synonym query...")
        q2 = make_fallback_query(lat_min, lon_min, lat_max, lon_max)
        t2 = time.perf_counter()
        osm2 = overpass_query(query=q2, timeout_s=args.request_timeout)
        t3 = time.perf_counter()
        print(f"       fallback fetched {len(osm2.get('elements', []))} elements in {(t3 - t2):.2f}s")
        feats = merge_features(feats, extract_features(osm2))

    missing_after = [
        k
        for k in ["trails", "cable_lines", "viewpoints"]
        if len(feats[k]) == 0
    ]
    if missing_after:
        print(f"[2c/5] still missing after fallback (abandoned): {missing_after}")

    print("[3/5] extracted feature counts:")
    for k in ["trails", "cable_lines", "cable_stations", "viewpoints"]:
        print(f"      {k}: {len(feats[k])}")

    print("[4/5] rasterizing OSM vectors to DEM grid...")
    tree, rows, cols = build_lonlat_tree(lon_grid, lat_grid)
    trail_mask = np.zeros((rows, cols), dtype=bool)
    hotspot_mask = np.zeros((rows, cols), dtype=bool)

    # trail lines: footpath + cable line
    for line in feats["trails"] + feats["cable_lines"]:
        rc = lonlat_to_rc(line, tree, rows, cols)
        draw_line_mask(trail_mask, rc)

    # hotspot points: viewpoint + cable station
    hotspot_lonlat = feats["viewpoints"] + feats["cable_stations"]
    hotspot_rc = dedup_points(lonlat_to_rc(hotspot_lonlat, tree, rows, cols))
    for r, c in hotspot_rc:
        hotspot_mask[r, c] = True

    # distance-to-feature risk fields
    if np.any(trail_mask):
        d_trail_m = distance_transform_edt(~trail_mask) * RESOLUTION_M
        risk_trail = risk_from_distance_trail(d_trail_m)
    else:
        risk_trail = np.zeros((rows, cols), dtype=float)

    if np.any(hotspot_mask):
        d_hot_m = distance_transform_edt(~hotspot_mask) * RESOLUTION_M
        risk_hotspot = risk_from_distance_hotspot(d_hot_m, sigma_m=120.0)
    else:
        risk_hotspot = np.zeros((rows, cols), dtype=float)

    w_trail = float(args.w_trail)
    w_hot = float(args.w_hotspot)
    w_sum = max(1e-9, w_trail + w_hot)
    risk_human = (w_trail / w_sum) * risk_trail + (w_hot / w_sum) * risk_hotspot
    risk_human = np.clip(risk_human, 0.0, 1.0)

    print("[5/5] saving risk rasters...")
    np.save(root / args.out_trail, risk_trail.astype(np.float32))
    np.save(root / args.out_hotspot, risk_hotspot.astype(np.float32))
    np.save(root / args.out_human, risk_human.astype(np.float32))

    # Quick QA figure to visually verify cableway/trail/viewpoint fusion.
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), dpi=150)
    axs[0].imshow(z, cmap="terrain")
    axs[0].set_title("DEM")
    axs[1].imshow(risk_trail, cmap="magma", vmin=0.0, vmax=1.0)
    axs[1].set_title("Trail + Cable Risk")
    axs[2].imshow(risk_human, cmap="inferno", vmin=0.0, vmax=1.0)
    axs[2].set_title("Human Risk (Combined)")
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(root / args.out_preview, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "bbox_wgs84": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        },
        "shape": [int(rows), int(cols)],
        "resolution_m": RESOLUTION_M,
        "feature_counts": {
            "trails": len(feats["trails"]),
            "cable_lines": len(feats["cable_lines"]),
            "cable_stations": len(feats["cable_stations"]),
            "viewpoints": len(feats["viewpoints"]),
            "hotspot_points_rasterized": len(hotspot_rc),
        },
        "query_fallback": {
            "used_fallback_query": used_fallback_query,
            "missing_before_fallback": missing_before,
            "missing_after_fallback": missing_after,
        },
        "weights": {
            "w_trail": w_trail / w_sum,
            "w_hotspot": w_hot / w_sum,
        },
        "risk_stats": {
            "trail_min": float(np.min(risk_trail)),
            "trail_max": float(np.max(risk_trail)),
            "hotspot_min": float(np.min(risk_hotspot)),
            "hotspot_max": float(np.max(risk_hotspot)),
            "human_min": float(np.min(risk_human)),
            "human_max": float(np.max(risk_human)),
            "human_mean": float(np.mean(risk_human)),
        },
    }
    (root / args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("done:")
    print(f"  {root / args.out_trail}")
    print(f"  {root / args.out_hotspot}")
    print(f"  {root / args.out_human}")
    print(f"  {root / args.summary_json}")
    print(f"  {root / args.out_preview}")


if __name__ == "__main__":
    main()
