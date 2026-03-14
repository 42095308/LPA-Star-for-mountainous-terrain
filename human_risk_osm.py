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
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
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


def configure_geo_axes(ax: plt.Axes, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> None:
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}°E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}°N"))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.6)


def add_scalebar(ax: plt.Axes, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> None:
    mid_lat = 0.5 * (lat_min + lat_max)
    meters_per_deg_lon = max(1.0, 111320.0 * math.cos(math.radians(mid_lat)))
    width_m = (lon_max - lon_min) * meters_per_deg_lon

    candidates_km = [0.5, 1.0, 2.0, 5.0]
    target_m = max(400.0, 0.18 * width_m)
    chosen_km = candidates_km[0]
    for c in candidates_km:
        if c * 1000.0 <= target_m:
            chosen_km = c

    dlon = (chosen_km * 1000.0) / meters_per_deg_lon
    x0 = lon_min + 0.05 * (lon_max - lon_min)
    y0 = lat_min + 0.06 * (lat_max - lat_min)

    ax.plot([x0, x0 + dlon], [y0, y0], color="white", lw=4.2, solid_capstyle="butt", zorder=60)
    ax.plot([x0, x0 + dlon], [y0, y0], color="black", lw=1.2, solid_capstyle="butt", zorder=61)
    ax.text(
        x0 + 0.5 * dlon,
        y0 + 0.012 * (lat_max - lat_min),
        f"{chosen_km:g} km",
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black", alpha=0.78),
        zorder=62,
    )


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

    # Paper-grade preview with geodetic axes, scale bar and OSM overlays.
    extent = [lon_min, lon_max, lat_min, lat_max]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.2), dpi=260)

    # Panel A: OSM trails/cable/features over DEM.
    ax0 = axs[0]
    ax0.imshow(z, cmap="terrain", extent=extent, origin="lower", aspect="auto")

    for line in feats["trails"]:
        lon = [p[0] for p in line]
        lat = [p[1] for p in line]
        ax0.plot(lon, lat, color="black", lw=2.2, alpha=0.65, zorder=10)
        ax0.plot(lon, lat, color="#D9FF66", lw=1.15, alpha=0.95, zorder=11)

    for line in feats["cable_lines"]:
        lon = [p[0] for p in line]
        lat = [p[1] for p in line]
        ax0.plot(lon, lat, color="black", lw=2.1, alpha=0.65, zorder=12)
        ax0.plot(lon, lat, color="#00D7FF", lw=1.25, alpha=0.95, ls="--", dashes=(4, 2), zorder=13)

    if feats["viewpoints"]:
        vv = np.asarray(feats["viewpoints"], dtype=float)
        ax0.scatter(vv[:, 0], vv[:, 1], s=26, marker="^", c="#FFD600", edgecolors="black", linewidths=0.4, zorder=20)
    if feats["cable_stations"]:
        cs = np.asarray(feats["cable_stations"], dtype=float)
        ax0.scatter(cs[:, 0], cs[:, 1], s=24, marker="s", c="#FF4081", edgecolors="black", linewidths=0.4, zorder=20)

    configure_geo_axes(ax0, lon_min, lon_max, lat_min, lat_max)
    add_scalebar(ax0, lon_min, lon_max, lat_min, lat_max)
    ax0.set_title("OSM Trails/Cableways/Hotspots over DEM", fontsize=11)

    legend_items = [
        Line2D([0], [0], color="#D9FF66", lw=2.0, label="Trails"),
        Line2D([0], [0], color="#00D7FF", lw=2.0, ls="--", label="Cableway lines"),
        Line2D([0], [0], marker="^", markersize=7, color="w", markerfacecolor="#FFD600", markeredgecolor="black", lw=0, label="Viewpoints"),
        Line2D([0], [0], marker="s", markersize=6.5, color="w", markerfacecolor="#FF4081", markeredgecolor="black", lw=0, label="Cableway stations"),
    ]
    ax0.legend(handles=legend_items, loc="upper right", fontsize=8, framealpha=0.88)

    # Panel B: Human risk intensity map.
    ax1 = axs[1]
    ax1.imshow(z, cmap="gray", extent=extent, origin="lower", alpha=0.30, aspect="auto")
    im = ax1.imshow(
        risk_human,
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
        origin="lower",
        alpha=0.90,
        aspect="auto",
    )
    for line in feats["trails"]:
        lon = [p[0] for p in line]
        lat = [p[1] for p in line]
        ax1.plot(lon, lat, color="white", lw=0.75, alpha=0.45, zorder=22)
    configure_geo_axes(ax1, lon_min, lon_max, lat_min, lat_max)
    add_scalebar(ax1, lon_min, lon_max, lat_min, lat_max)
    ax1.set_title("Human Exposure Risk Field", fontsize=11)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.02)
    cbar.set_label("Risk score (0-1)")

    fig.suptitle("Huashan OSM-Derived Human Risk (DEM-Aligned)", fontsize=13, y=0.995)
    plt.tight_layout(rect=[0, 0.0, 1, 0.97])
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
