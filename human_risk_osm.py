"""
Build Huashan human-exposure risk rasters from a local OSM file.

Risk design (user-specified):
    L1 (1.0): dangerous crowded routes, full risk within 50 m
    L2 (0.8): peak area + regular hiking paths, full risk within 30 m
    L3 (0.5): cableway lines/stations + popular facilities, Gaussian (sigma=120 m)
    L4 (0.2): scenic roads + foothill amenities, full risk within 30 m

Outputs:
    risk_trail.npy
    risk_hotspot.npy
    risk_human.npy
    osm_feature_summary.json
    osm_human_risk_preview.png
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


RESOLUTION_M = 12.5
LEVELS = (1, 2, 3, 4)

# ---- Name keywords (ASCII with Unicode escapes, to avoid source encoding issues) ----
L1_DANGEROUS_NAMES = [
    "\u5929\u68af",            # 天梯
    "\u957f\u7a7a\u6808\u9053",  # 长空栈道
    "\u667a\u53d6\u534e\u5c71\u8def",  # 智取华山路
    "\u82cd\u9f99\u5cad",      # 苍龙岭
    "\u767e\u5c3a\u5ce1",      # 百尺峡
    "\u8001\u541b\u7281\u6c9f",  # 老君犁沟
    "\u5343\u5c3a\u5e62",      # 千尺幢
    "\u9e5e\u5b50\u7ffb\u8eab",  # 鹞子翻身
    "\u56de\u5fc3\u77f3",      # 回心石
]

L2_PEAK_NAMES = [
    "\u534e\u5c71\u5357\u5cf0",  # 华山南峰
    "\u534e\u5c71\u4e1c\u5cf0",  # 华山东峰
    "\u534e\u5c71\u897f\u5cf0",  # 华山西峰
    "\u534e\u5c71\u5317\u5cf0",  # 华山北峰
    "\u534e\u5c71\u4e2d\u5cf0",  # 华山中峰
]
L2_HIGH_NAMES = [
    "\u91d1\u9501\u5173",                  # 金锁关
    "\u5357\u5929\u95e8",                  # 南天门
    "\u4e0b\u68cb\u4ead",                  # 下棋亭
    "\u81ea\u53e4\u534e\u5c71\u4e00\u6761\u8def",  # 自古华山一条路
]

L3_MEDIUM_NAMES = [
    "\u4e1c\u5cf0\u9152\u5e97",  # 东峰酒店
    "\u601d\u8fc7\u5d16",        # 思过崖
    "\u6bdb\u4ed9\u5e99",        # 毛仙庙
    "\u5c71\u795e\u5e99",        # 山神庙
    "\u7fa4\u4ed9\u89c2",        # 群仙观
]

L4_LOW_ROAD_NAMES = [
    "\u74ee\u5cea\u65c5\u6e38\u516c\u8def",  # 瓮峪旅游公路
    "\u9ec4\u752b\u5cea\u516c\u8def",        # 黄甫峪公路
]

# Label verification list for summary output.
EXPECTED_LABELS = {
    "L1": L1_DANGEROUS_NAMES,
    "L2": L2_PEAK_NAMES + L2_HIGH_NAMES,
    "L3": L3_MEDIUM_NAMES,
    "L4": L4_LOW_ROAD_NAMES,
}


@dataclass
class WayRecord:
    way_id: int
    refs: List[int]
    tags: Dict[str, str]


def _norm(s: Optional[str]) -> str:
    return str(s or "").strip()


def _tag_l(tags: Dict[str, str], key: str) -> str:
    return _norm(tags.get(key)).lower()


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    if not text:
        return False
    return any(k in text for k in keywords)


def parse_osm(osm_path: Path) -> Tuple[Dict[int, Tuple[float, float]], List[Tuple[int, Dict[str, str]]], List[WayRecord], List[str]]:
    tree = ET.parse(osm_path)
    root = tree.getroot()

    nodes: Dict[int, Tuple[float, float]] = {}
    tagged_nodes: List[Tuple[int, Dict[str, str]]] = []
    ways: List[WayRecord] = []
    all_names: List[str] = []

    for e in root:
        if e.tag == "node":
            if "id" not in e.attrib or "lon" not in e.attrib or "lat" not in e.attrib:
                continue
            node_id = int(e.attrib["id"])
            lon = float(e.attrib["lon"])
            lat = float(e.attrib["lat"])
            nodes[node_id] = (lon, lat)

            tags: Dict[str, str] = {}
            for t in e.findall("tag"):
                k = t.attrib.get("k")
                v = t.attrib.get("v")
                if k:
                    tags[k] = _norm(v)
            if tags:
                tagged_nodes.append((node_id, tags))
                if "name" in tags and tags["name"]:
                    all_names.append(tags["name"])

        elif e.tag == "way":
            if "id" not in e.attrib:
                continue
            way_id = int(e.attrib["id"])
            refs = []
            for nd in e.findall("nd"):
                ref_s = nd.attrib.get("ref")
                if ref_s:
                    refs.append(int(ref_s))
            tags: Dict[str, str] = {}
            for t in e.findall("tag"):
                k = t.attrib.get("k")
                v = t.attrib.get("v")
                if k:
                    tags[k] = _norm(v)
            ways.append(WayRecord(way_id=way_id, refs=refs, tags=tags))
            if "name" in tags and tags["name"]:
                all_names.append(tags["name"])

    return nodes, tagged_nodes, ways, all_names


def classify_level(tags: Dict[str, str]) -> Optional[int]:
    name = _norm(tags.get("name"))
    highway = _tag_l(tags, "highway")
    aerialway = _tag_l(tags, "aerialway")
    tourism = _tag_l(tags, "tourism")
    natural = _tag_l(tags, "natural")
    amenity = _tag_l(tags, "amenity")
    building = _tag_l(tags, "building")

    # ---- L1: extreme risk ----
    if _contains_any(name, L1_DANGEROUS_NAMES):
        return 1

    # ---- L3: medium risk (explicit cableway tags have higher priority than name-based peak matching) ----
    if aerialway in {"cable_car", "gondola", "station"}:
        return 3

    # ---- L2: high risk ----
    if _contains_any(name, L2_PEAK_NAMES):
        return 2
    if _contains_any(name, L2_HIGH_NAMES):
        return 2
    if highway in {"steps", "path", "footway"}:
        return 2
    if natural == "peak":
        return 2
    if tourism in {"attraction", "viewpoint"} and _contains_any(name, L2_HIGH_NAMES):
        return 2

    # ---- L3: medium risk ----
    if tourism == "guest_house":
        return 3
    if amenity == "place_of_worship":
        return 3
    if _contains_any(name, L3_MEDIUM_NAMES):
        return 3
    if building == "yes" and _contains_any(name, ["\u7fa4\u4ed9\u89c2"]):  # 群仙观
        return 3

    # ---- L4: low risk ----
    if highway in {"tertiary", "service", "unclassified"}:
        return 4
    if amenity in {"toilets", "parking", "bus_station"}:
        return 4
    if _contains_any(name, L4_LOW_ROAD_NAMES):
        return 4

    return None


def is_line_way(tags: Dict[str, str], level: int) -> bool:
    highway = _tag_l(tags, "highway")
    aerialway = _tag_l(tags, "aerialway")
    natural = _tag_l(tags, "natural")
    area = _tag_l(tags, "area")
    building = _tag_l(tags, "building")
    tourism = _tag_l(tags, "tourism")

    if area == "yes":
        return False
    if building:
        return False
    if level == 3 and aerialway in {"cable_car", "gondola"}:
        return True
    if highway:
        return True
    if natural in {"ridge"}:
        return True
    if level == 1 and tourism == "attraction":
        return True
    return False


def point_in_bbox(
    lon: float,
    lat: float,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    pad: float = 0.0,
) -> bool:
    return (lon_min - pad) <= lon <= (lon_max + pad) and (lat_min - pad) <= lat <= (lat_max + pad)


def line_bbox_intersects(
    coords: Sequence[Tuple[float, float]],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    pad: float = 0.0,
) -> bool:
    if not coords:
        return False
    lons = [p[0] for p in coords]
    lats = [p[1] for p in coords]
    c_lon_min, c_lon_max = min(lons), max(lons)
    c_lat_min, c_lat_max = min(lats), max(lats)
    return not (
        c_lon_max < (lon_min - pad)
        or c_lon_min > (lon_max + pad)
        or c_lat_max < (lat_min - pad)
        or c_lat_min > (lat_max + pad)
    )


def dedup_points(points: Iterable[Tuple[float, float]], ndigits: int = 7) -> List[Tuple[float, float]]:
    seen = set()
    out: List[Tuple[float, float]] = []
    for lon, lat in points:
        key = (round(float(lon), ndigits), round(float(lat), ndigits))
        if key in seen:
            continue
        seen.add(key)
        out.append((float(lon), float(lat)))
    return out


def dedup_lines(
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


def build_lonlat_tree(lon_grid: np.ndarray, lat_grid: np.ndarray) -> Tuple[cKDTree, int, int]:
    rows, cols = lon_grid.shape
    pts = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    return cKDTree(pts), rows, cols


def lonlat_to_rc(
    lonlat: Sequence[Tuple[float, float]],
    tree: cKDTree,
    rows: int,
    cols: int,
) -> List[Tuple[int, int]]:
    if not lonlat:
        return []
    q = np.asarray(lonlat, dtype=float)
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


def risk_from_buffer(mask: np.ndarray, risk_value: float, radius_m: float) -> np.ndarray:
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=float)
    d_m = distance_transform_edt(~mask) * RESOLUTION_M
    return np.where(d_m <= radius_m, risk_value, 0.0).astype(float)


def risk_from_gaussian(mask: np.ndarray, risk_peak: float, sigma_m: float) -> np.ndarray:
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=float)
    d_m = distance_transform_edt(~mask) * RESOLUTION_M
    return (risk_peak * np.exp(-(d_m ** 2) / (2.0 * sigma_m * sigma_m))).astype(float)


def configure_geo_axes(ax: plt.Axes, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> None:
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}N"))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.24, linestyle="--", linewidth=0.6)


def add_scalebar(ax: plt.Axes, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> None:
    mid_lat = 0.5 * (lat_min + lat_max)
    meters_per_deg_lon = max(1.0, 111320.0 * math.cos(math.radians(mid_lat)))
    width_m = (lon_max - lon_min) * meters_per_deg_lon
    target_m = max(500.0, 0.2 * width_m)
    candidates_km = [0.5, 1.0, 2.0, 5.0]
    chosen_km = candidates_km[0]
    for c in candidates_km:
        if c * 1000.0 <= target_m:
            chosen_km = c

    dlon = (chosen_km * 1000.0) / meters_per_deg_lon
    x0 = lon_min + 0.05 * (lon_max - lon_min)
    y0 = lat_min + 0.06 * (lat_max - lat_min)
    ax.plot([x0, x0 + dlon], [y0, y0], color="white", lw=4.2, zorder=60, solid_capstyle="butt")
    ax.plot([x0, x0 + dlon], [y0, y0], color="black", lw=1.0, zorder=61, solid_capstyle="butt")
    ax.text(
        x0 + 0.5 * dlon,
        y0 + 0.012 * (lat_max - lat_min),
        f"{chosen_km:g} km",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="black",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black", alpha=0.8),
        zorder=62,
    )


def check_expected_labels(all_names: Sequence[str]) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}
    for level_name, expect in EXPECTED_LABELS.items():
        found: List[str] = []
        missing: List[str] = []
        for target in expect:
            ok = any(target in n for n in all_names)
            if ok:
                found.append(target)
            else:
                missing.append(target)
        out[level_name] = {"found": found, "missing": missing}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Huashan OSM human-risk raster aligned with Z_crop.npy")
    parser.add_argument("--workdir", type=str, default=".")
    parser.add_argument("--osm-file", type=str, default="map.osm")
    parser.add_argument("--z-file", type=str, default="Z_crop.npy")
    parser.add_argument("--geo-file", type=str, default="Z_crop_geo.npz")
    parser.add_argument("--out-trail", type=str, default="risk_trail.npy")
    parser.add_argument("--out-hotspot", type=str, default="risk_hotspot.npy")
    parser.add_argument("--out-human", type=str, default="risk_human.npy")
    parser.add_argument("--summary-json", type=str, default="osm_feature_summary.json")
    parser.add_argument("--out-preview", type=str, default="osm_human_risk_preview.png")
    args = parser.parse_args()

    root = Path(args.workdir).resolve()
    osm_path = Path(args.osm_file)
    if not osm_path.is_absolute():
        osm_path = root / osm_path
    if not osm_path.exists():
        raise FileNotFoundError(f"OSM file not found: {osm_path}")

    z = np.asarray(np.load(root / args.z_file), dtype=float)
    geo = np.load(root / args.geo_file)
    lon_grid = np.asarray(geo["lon_grid"], dtype=float)
    lat_grid = np.asarray(geo["lat_grid"], dtype=float)
    if z.shape != lon_grid.shape or z.shape != lat_grid.shape:
        raise RuntimeError(f"Shape mismatch: Z={z.shape}, lon={lon_grid.shape}, lat={lat_grid.shape}")

    lat_min = float(np.min(lat_grid))
    lat_max = float(np.max(lat_grid))
    lon_min = float(np.min(lon_grid))
    lon_max = float(np.max(lon_grid))
    print("[1/5] study area bbox (WGS84)")
    print(f"      lat: {lat_min:.6f} .. {lat_max:.6f}")
    print(f"      lon: {lon_min:.6f} .. {lon_max:.6f}")
    print(f"[2/5] parsing local OSM: {osm_path}")

    nodes, tagged_nodes, ways, all_names = parse_osm(osm_path)
    print(f"      nodes={len(nodes)}, tagged_nodes={len(tagged_nodes)}, ways={len(ways)}")

    lines: Dict[int, List[List[Tuple[float, float]]]] = {lv: [] for lv in LEVELS}
    points: Dict[int, List[Tuple[float, float]]] = {lv: [] for lv in LEVELS}
    matched_names: Dict[int, set] = {lv: set() for lv in LEVELS}

    # Tagged nodes
    for node_id, tags in tagged_nodes:
        lv = classify_level(tags)
        if lv is None:
            continue
        lonlat = nodes.get(node_id)
        if lonlat is None:
            continue
        lon, lat = lonlat
        if not point_in_bbox(lon, lat, lon_min, lon_max, lat_min, lat_max, pad=0.01):
            continue
        points[lv].append((lon, lat))
        name = _norm(tags.get("name"))
        if name:
            matched_names[lv].add(name)

    # Ways
    for w in ways:
        lv = classify_level(w.tags)
        if lv is None:
            continue
        coords = [nodes[r] for r in w.refs if r in nodes]
        if not coords:
            continue
        if not line_bbox_intersects(coords, lon_min, lon_max, lat_min, lat_max, pad=0.01):
            continue
        name = _norm(w.tags.get("name"))
        if name:
            matched_names[lv].add(name)

        if is_line_way(w.tags, lv) and len(coords) >= 2:
            clipped = [p for p in coords if point_in_bbox(p[0], p[1], lon_min, lon_max, lat_min, lat_max, pad=0.01)]
            if len(clipped) >= 2:
                lines[lv].append(clipped)
            else:
                # Keep intersecting line as a fallback so cableway-line risk is not lost.
                lines[lv].append(coords)
        else:
            lon = float(np.mean([p[0] for p in coords]))
            lat = float(np.mean([p[1] for p in coords]))
            if point_in_bbox(lon, lat, lon_min, lon_max, lat_min, lat_max, pad=0.01):
                points[lv].append((lon, lat))

    # Deduplicate to avoid redundant rasterization.
    for lv in LEVELS:
        lines[lv] = dedup_lines(lines[lv])
        points[lv] = dedup_points(points[lv])

    print("[3/5] classified OSM feature counts")
    for lv in LEVELS:
        print(f"      L{lv}: lines={len(lines[lv])}, points={len(points[lv])}")

    print("[4/5] rasterizing and building risk fields")
    tree, rows, cols = build_lonlat_tree(lon_grid, lat_grid)
    masks: Dict[int, np.ndarray] = {lv: np.zeros((rows, cols), dtype=bool) for lv in LEVELS}

    for lv in LEVELS:
        for line in lines[lv]:
            rc = lonlat_to_rc(line, tree, rows, cols)
            draw_line_mask(masks[lv], rc)
        if points[lv]:
            rc_pts = lonlat_to_rc(points[lv], tree, rows, cols)
            for r, c in rc_pts:
                masks[lv][r, c] = True

    risk_l1 = risk_from_buffer(masks[1], risk_value=1.0, radius_m=50.0)
    risk_l2 = risk_from_buffer(masks[2], risk_value=0.8, radius_m=30.0)
    risk_l3 = risk_from_gaussian(masks[3], risk_peak=0.5, sigma_m=120.0)
    risk_l4 = risk_from_buffer(masks[4], risk_value=0.2, radius_m=30.0)

    # Keep split outputs for compatibility with existing pipeline.
    risk_trail = np.maximum(risk_l1, risk_l2)
    risk_hotspot = np.maximum(risk_l3, risk_l4)
    risk_human = np.maximum.reduce([risk_l1, risk_l2, risk_l3, risk_l4])
    risk_trail = np.clip(risk_trail, 0.0, 1.0)
    risk_hotspot = np.clip(risk_hotspot, 0.0, 1.0)
    risk_human = np.clip(risk_human, 0.0, 1.0)

    print("[5/5] saving risk rasters")
    np.save(root / args.out_trail, risk_trail.astype(np.float32))
    np.save(root / args.out_hotspot, risk_hotspot.astype(np.float32))
    np.save(root / args.out_human, risk_human.astype(np.float32))

    # ---- Paper preview ----
    level_style = {
        1: {"color": "#D32F2F", "marker": "X", "label": "L1 Extreme (1.0, 50m buffer)"},
        2: {"color": "#F57C00", "marker": "^", "label": "L2 High (0.8, 30m buffer)"},
        3: {"color": "#FBC02D", "marker": "s", "label": "L3 Medium (0.5, Gaussian)"},
        4: {"color": "#388E3C", "marker": "o", "label": "L4 Low (0.2, 30m buffer)"},
    }

    extent = [lon_min, lon_max, lat_min, lat_max]
    fig = plt.figure(figsize=(18.0, 6.6), dpi=260)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.95], wspace=0.18)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Panel A: DEM + color-coded OSM labels (vector only).
    ax0.imshow(z, cmap="terrain", extent=extent, origin="lower", aspect="auto")
    for lv in LEVELS:
        style = level_style[lv]
        for line in lines[lv]:
            lons = [p[0] for p in line]
            lats = [p[1] for p in line]
            ax0.plot(lons, lats, color="black", lw=2.0, alpha=0.45, zorder=10 + lv)
            ax0.plot(lons, lats, color=style["color"], lw=1.1, alpha=0.95, zorder=11 + lv)
        if points[lv]:
            arr = np.asarray(points[lv], dtype=float)
            ax0.scatter(
                arr[:, 0],
                arr[:, 1],
                s=20,
                marker=style["marker"],
                c=style["color"],
                edgecolors="black",
                linewidths=0.35,
                alpha=0.95,
                zorder=20 + lv,
            )
    configure_geo_axes(ax0, lon_min, lon_max, lat_min, lat_max)
    add_scalebar(ax0, lon_min, lon_max, lat_min, lat_max)
    ax0.set_title("Panel A: OSM Labels by Risk Level", fontsize=11)
    legend_items = [Line2D([0], [0], color=level_style[lv]["color"], lw=2.2, label=f"L{lv}") for lv in LEVELS]
    ax0.legend(handles=legend_items, loc="upper right", fontsize=8.0, framealpha=0.9, title="Risk Level")

    # Panel B: combined risk heatmap only (no label clutter).
    ax1.imshow(z, cmap="gray", extent=extent, origin="lower", aspect="auto", alpha=0.28)
    im = ax1.imshow(
        risk_human,
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
        origin="lower",
        aspect="auto",
        alpha=0.92,
    )
    configure_geo_axes(ax1, lon_min, lon_max, lat_min, lat_max)
    add_scalebar(ax1, lon_min, lon_max, lat_min, lat_max)
    ax1.set_title("Panel B: Combined Human Exposure Risk", fontsize=11)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.02)
    cbar.set_label("Risk score (0-1)")

    # Panel C: legend/label mapping table.
    ax2.axis("off")
    ax2.set_title("Panel C: Color-to-Label Mapping", fontsize=11, loc="left", pad=10)

    legend_rows = [
        (
            1,
            "L1 (risk=1.0, 50m buffer)",
            "Dangerous routes: Tian Ti, Changkong Plank, Zhiqu Huashan Rd,\n"
            "Canglong Ridge, Baichi Gorge, Laojun Ligou, Qianchi Chuang,\n"
            "Yaozi Fanshen, Huixin Stone",
        ),
        (
            2,
            "L2 (risk=0.8, 30m buffer)",
            "Peak/major hiking area: natural=peak, named attractions\n"
            "(Jinsuo Pass, Nantianmen, Xiaqi Pavilion), highway=steps/path/footway",
        ),
        (
            3,
            "L3 (risk=0.5, sigma=120m)",
            "Cableway + hotspots: aerialway=cable_car/gondola/station,\n"
            "tourism=guest_house, amenity=place_of_worship, specific hotspots",
        ),
        (
            4,
            "L4 (risk=0.2, 30m buffer)",
            "Scenic roads + foothill facilities: highway=tertiary/service/unclassified,\n"
            "amenity=toilets/parking/bus_station",
        ),
    ]

    y = 0.94
    for lv, title_txt, desc in legend_rows:
        style = level_style[lv]
        ax2.plot([0.03, 0.14], [y, y], transform=ax2.transAxes, color=style["color"], lw=4.0, solid_capstyle="round")
        ax2.scatter(
            [0.085],
            [y],
            transform=ax2.transAxes,
            s=42,
            marker=style["marker"],
            c=style["color"],
            edgecolors="black",
            linewidths=0.45,
            zorder=5,
        )
        ax2.text(0.17, y + 0.012, title_txt, transform=ax2.transAxes, fontsize=9.2, fontweight="bold", va="top")
        ax2.text(0.17, y - 0.020, desc, transform=ax2.transAxes, fontsize=8.1, va="top")
        y -= 0.235

    ax2.text(
        0.03,
        0.03,
        f"Detected features: L1({len(lines[1])} lines/{len(points[1])} points), "
        f"L2({len(lines[2])}/{len(points[2])}), "
        f"L3({len(lines[3])}/{len(points[3])}), "
        f"L4({len(lines[4])}/{len(points[4])})",
        transform=ax2.transAxes,
        fontsize=8.2,
        color="dimgray",
    )

    fig.suptitle("Huashan Four-Level Human Risk Modeling from OSM", fontsize=13, y=0.995)
    fig.subplots_adjust(left=0.04, right=0.985, bottom=0.06, top=0.93, wspace=0.18)
    plt.savefig(root / args.out_preview, bbox_inches="tight")
    plt.close(fig)

    label_check = check_expected_labels(all_names)
    summary = {
        "osm_file": str(osm_path),
        "bbox_wgs84": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        },
        "shape": [int(rows), int(cols)],
        "resolution_m": RESOLUTION_M,
        "risk_model": {
            "L1": {"risk": 1.0, "buffer_m": 50.0},
            "L2": {"risk": 0.8, "buffer_m": 30.0},
            "L3": {"risk": 0.5, "gaussian_sigma_m": 120.0},
            "L4": {"risk": 0.2, "buffer_m": 30.0},
        },
        "feature_counts_by_level": {
            f"L{lv}": {
                "lines": len(lines[lv]),
                "points": len(points[lv]),
                "named_features": sorted(list(matched_names[lv]))[:120],
            }
            for lv in LEVELS
        },
        "requested_label_check": label_check,
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
