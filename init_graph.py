"""
Crop Huashan DEM to the study area and export aligned geo grids.

This version supports center correction using a WGS84 coordinate:
default center (from user): lon=110.0798, lat=34.4829.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pyproj import Transformer


TIF_FILE = "AP_19438_FBD_F0680_RT1.dem.tif"
CACHE_FILE = "Z_crop.npy"
CACHE_GEO = "Z_crop_geo.npz"
CACHE_META = "Z_crop_meta.json"

EPSG_SRC = "EPSG:32649"  # from GeoKeyDirectoryTag (WGS84 / UTM zone 49N)
EPSG_WGS84 = "EPSG:4326"

DEFAULT_CENTER_LON = 110.0798
DEFAULT_CENTER_LAT = 34.4829
DEFAULT_CROP_SIZE_M = 10_000.0
RESOLUTION_M = 12.5

# Legacy pixel peaks used only for debugging current-center offset.
LEGACY_PEAK_PIXELS = {
    "South": {"row": 4609, "col": 1938},
    "East": {"row": 4642, "col": 1985},
    "West": {"row": 4600, "col": 1949},
    "North": {"row": 4468, "col": 2004},
    "Central": {"row": 4594, "col": 1951},
}

# Peak labels for plotting (WGS84).
PEAKS_WGS84 = {
    "South Peak": {"lon": 110.0781, "lat": 34.4778, "elev": 2150.0},
    "East Peak": {"lon": 110.0820, "lat": 34.4811, "elev": 2100.0},
    "West Peak": {"lon": 110.0768, "lat": 34.4816, "elev": 2038.0},
    "North Peak": {"lon": 110.0813, "lat": 34.4934, "elev": 1615.0},
    "Central Peak": {"lon": 110.0808, "lat": 34.4806, "elev": 2043.0},
}


def read_tiff_with_georef(tif_path: Path) -> Tuple[np.ndarray, float, float, float, float]:
    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]
        dem = page.asarray().astype(float)
        scale = page.tags["ModelPixelScaleTag"].value
        tie = page.tags["ModelTiepointTag"].value

    sx = float(scale[0])
    sy = float(scale[1])
    x0 = float(tie[3])  # upper-left corner x
    y0 = float(tie[4])  # upper-left corner y
    return dem, x0, y0, sx, sy


def pixel_to_xy(row: float, col: float, x0: float, y0: float, sx: float, sy: float) -> Tuple[float, float]:
    x = x0 + col * sx
    y = y0 - row * sy
    return x, y


def xy_to_pixel(x: float, y: float, x0: float, y0: float, sx: float, sy: float) -> Tuple[int, int]:
    col = int(round((x - x0) / sx))
    row = int(round((y0 - y) / sy))
    return row, col


def bounded_crop_window(
    row_center: int,
    col_center: int,
    half: int,
    n_rows: int,
    n_cols: int,
) -> Tuple[int, int, int, int]:
    row_min = row_center - half
    row_max = row_center + half
    col_min = col_center - half
    col_max = col_center + half

    if row_min < 0:
        row_max -= row_min
        row_min = 0
    if col_min < 0:
        col_max -= col_min
        col_min = 0
    if row_max > n_rows:
        shift = row_max - n_rows
        row_min -= shift
        row_max = n_rows
    if col_max > n_cols:
        shift = col_max - n_cols
        col_min -= shift
        col_max = n_cols

    row_min = max(0, row_min)
    col_min = max(0, col_min)
    row_max = min(n_rows, row_max)
    col_max = min(n_cols, col_max)
    return row_min, row_max, col_min, col_max


def build_lonlat_grids(
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    x0: float,
    y0: float,
    sx: float,
    sy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    rows = row_max - row_min
    cols = col_max - col_min
    rr = np.arange(row_min, row_max, dtype=float)
    cc = np.arange(col_min, col_max, dtype=float)
    rr2, cc2 = np.meshgrid(rr, cc, indexing="ij")
    x2 = x0 + cc2 * sx
    y2 = y0 - rr2 * sy

    tf_to_wgs = Transformer.from_crs(EPSG_SRC, EPSG_WGS84, always_xy=True)
    lon_flat, lat_flat = tf_to_wgs.transform(x2.ravel(), y2.ravel())
    lon_grid = lon_flat.reshape(rows, cols)
    lat_grid = lat_flat.reshape(rows, cols)
    return lon_grid, lat_grid


def nearest_rc_from_lonlat(lon_grid: np.ndarray, lat_grid: np.ndarray, lon: float, lat: float) -> Tuple[int, int]:
    d2 = (lon_grid - lon) ** 2 + (lat_grid - lat) ** 2
    idx = int(np.argmin(d2))
    r, c = np.unravel_index(idx, lon_grid.shape)
    return int(r), int(c)


def cache_matches(meta_path: Path, center_lon: float, center_lat: float, crop_size_m: float) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        abs(float(meta.get("center_lon", 0.0)) - center_lon) < 1e-9
        and abs(float(meta.get("center_lat", 0.0)) - center_lat) < 1e-9
        and abs(float(meta.get("crop_size_m", 0.0)) - crop_size_m) < 1e-9
        and meta.get("row0_orientation", "") == "north"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop Huashan DEM and export geo-aligned cache.")
    parser.add_argument("--center-lon", type=float, default=DEFAULT_CENTER_LON)
    parser.add_argument("--center-lat", type=float, default=DEFAULT_CENTER_LAT)
    parser.add_argument("--crop-size-m", type=float, default=DEFAULT_CROP_SIZE_M)
    parser.add_argument("--force-recrop", action="store_true")
    args = parser.parse_args()

    center_lon = float(args.center_lon)
    center_lat = float(args.center_lat)
    crop_size_m = float(args.crop_size_m)

    tif_path = Path(TIF_FILE)
    if not tif_path.exists():
        raise FileNotFoundError(f"Missing DEM file: {tif_path.resolve()}")

    use_cache = (
        Path(CACHE_FILE).exists()
        and Path(CACHE_GEO).exists()
        and cache_matches(Path(CACHE_META), center_lon, center_lat, crop_size_m)
        and (not args.force_recrop)
    )

    if use_cache:
        print("[cache] using existing crop cache (center and size match).")
        z_crop = np.asarray(np.load(CACHE_FILE), dtype=float)
        geo = np.load(CACHE_GEO)
        lon_grid = np.asarray(geo["lon_grid"], dtype=float)
        lat_grid = np.asarray(geo["lat_grid"], dtype=float)
        meta = json.loads(Path(CACHE_META).read_text(encoding="utf-8"))
        row_min = int(meta["row_min"])
        row_max = int(meta["row_max"])
        col_min = int(meta["col_min"])
        col_max = int(meta["col_max"])
    else:
        print("[crop] reading source DEM and recropping...")
        dem, x0, y0, sx, sy = read_tiff_with_georef(tif_path)
        dem[dem < -9000] = np.nan
        n_rows, n_cols = dem.shape
        half = int(round((crop_size_m / 2.0) / RESOLUTION_M))

        # Print legacy center and corresponding WGS84 (for diagnosis).
        legacy_row = int(np.mean([v["row"] for v in LEGACY_PEAK_PIXELS.values()]))
        legacy_col = int(np.mean([v["col"] for v in LEGACY_PEAK_PIXELS.values()]))
        tf_to_wgs = Transformer.from_crs(EPSG_SRC, EPSG_WGS84, always_xy=True)
        tf_to_utm = Transformer.from_crs(EPSG_WGS84, EPSG_SRC, always_xy=True)
        x_old, y_old = pixel_to_xy(legacy_row, legacy_col, x0, y0, sx, sy)
        lon_old, lat_old = tf_to_wgs.transform(x_old, y_old)

        x_new, y_new = tf_to_utm.transform(center_lon, center_lat)
        center_row, center_col = xy_to_pixel(x_new, y_new, x0, y0, sx, sy)
        x_new_chk, y_new_chk = pixel_to_xy(center_row, center_col, x0, y0, sx, sy)
        lon_new_chk, lat_new_chk = tf_to_wgs.transform(x_new_chk, y_new_chk)

        d_lon = lon_new_chk - lon_old
        d_lat = lat_new_chk - lat_old
        print(f"[debug] legacy center pixel: row={legacy_row}, col={legacy_col}")
        print(f"[debug] legacy center lon/lat: lon={lon_old:.6f}, lat={lat_old:.6f}")
        print(f"[debug] target center lon/lat: lon={center_lon:.6f}, lat={center_lat:.6f}")
        print(f"[debug] target center pixel: row={center_row}, col={center_col}")
        print(f"[debug] snapped target lon/lat: lon={lon_new_chk:.6f}, lat={lat_new_chk:.6f}")
        print(f"[debug] offset from legacy to target: dlon={d_lon:+.6f}, dlat={d_lat:+.6f}")

        row_min, row_max, col_min, col_max = bounded_crop_window(center_row, center_col, half, n_rows, n_cols)
        z_crop = dem[row_min:row_max, col_min:col_max]
        lon_grid, lat_grid = build_lonlat_grids(row_min, row_max, col_min, col_max, x0, y0, sx, sy)

        np.save(CACHE_FILE, z_crop.astype(np.float32))
        np.savez(CACHE_GEO, lon_grid=lon_grid.astype(np.float64), lat_grid=lat_grid.astype(np.float64))
        meta = {
            "center_lon": center_lon,
            "center_lat": center_lat,
            "crop_size_m": crop_size_m,
            "row_min": row_min,
            "row_max": row_max,
            "col_min": col_min,
            "col_max": col_max,
            "row0_orientation": "north",
        }
        Path(CACHE_META).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print("[crop] cache saved: Z_crop.npy / Z_crop_geo.npz / Z_crop_meta.json")

    rows, cols = z_crop.shape
    print(f"[info] DEM shape: {rows} x {cols}")
    print(f"[info] elevation range: {np.nanmin(z_crop):.1f} m .. {np.nanmax(z_crop):.1f} m")
    print(f"[info] lon range: {lon_grid.min():.6f} .. {lon_grid.max():.6f}")
    print(f"[info] lat range: {lat_grid.min():.6f} .. {lat_grid.max():.6f}")
    print(f"[info] center (grid mean): lon={np.mean(lon_grid):.6f}, lat={np.mean(lat_grid):.6f}")

    # Peak marker coordinates in current crop.
    peak_plot: Dict[str, Dict[str, float]] = {}
    for name, p in PEAKS_WGS84.items():
        r, c = nearest_rc_from_lonlat(lon_grid, lat_grid, p["lon"], p["lat"])
        x_km = c * RESOLUTION_M / 1000.0
        y_km = (rows - 1 - r) * RESOLUTION_M / 1000.0
        peak_plot[name] = {
            "x": x_km,
            "y": y_km,
            "lon": float(lon_grid[r, c]),
            "lat": float(lat_grid[r, c]),
            "elev": float(p["elev"]),
        }

    # Plot
    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(121)
    extent = [0.0, cols * RESOLUTION_M / 1000.0, 0.0, rows * RESOLUTION_M / 1000.0]
    im = ax1.imshow(z_crop, cmap="terrain", extent=extent, origin="upper", aspect="equal")
    plt.colorbar(im, ax=ax1, label="Elevation (m)", shrink=0.8)

    for name, p in peak_plot.items():
        ax1.plot(p["x"], p["y"], "r^", markersize=10, zorder=5)
        ax1.annotate(
            f"{name} {p['elev']:.0f}m\n{p['lon']:.5f}E\n{p['lat']:.5f}N",
            xy=(p["x"], p["y"]),
            xytext=(p["x"] + 0.30, p["y"] + 0.30),
            fontsize=8,
            color="darkred",
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="red", alpha=0.85),
        )

    ax1.set_xlabel("East-West (km)")
    ax1.set_ylabel("South-North (km)")
    ax1.set_title("Huashan Core DEM (Top View)")
    ax1.grid(True, alpha=0.3, linestyle="--")

    annot = ax1.annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.9),
        arrowprops=dict(arrowstyle="->"),
        fontsize=9,
    )
    annot.set_visible(False)

    def on_hover(event):
        if event.inaxes != ax1:
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return
        x_km = event.xdata
        y_km = event.ydata
        if x_km is None or y_km is None:
            return
        c = int(np.clip(x_km * 1000.0 / RESOLUTION_M, 0, cols - 1))
        r = int(np.clip((rows - 1) - y_km * 1000.0 / RESOLUTION_M, 0, rows - 1))
        elev = z_crop[r, c]
        if np.isnan(elev):
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return
        lon = float(lon_grid[r, c])
        lat = float(lat_grid[r, c])
        annot.xy = (x_km, y_km)
        annot.set_text(f"Lon: {lon:.5f}E\nLat: {lat:.5f}N\nElevation: {elev:.1f} m")
        annot.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    ax2 = fig.add_subplot(122, projection="3d")
    step = 6
    r_idx = np.arange(0, rows, step, dtype=int)
    c_idx = np.arange(0, cols, step, dtype=int)
    z_s = z_crop[np.ix_(r_idx, c_idx)]
    x_vals = c_idx * RESOLUTION_M / 1000.0
    y_vals = (rows - 1 - r_idx) * RESOLUTION_M / 1000.0
    xg, yg = np.meshgrid(x_vals, y_vals)
    ax2.plot_surface(xg, yg, z_s, cmap="terrain", alpha=0.85, linewidth=0)
    for name, p in peak_plot.items():
        ax2.scatter(p["x"], p["y"], p["elev"] + 80.0, color="red", s=55, zorder=5)
        ax2.text(p["x"], p["y"], p["elev"] + 190.0, name, fontsize=8, color="red", ha="center")

    ax2.view_init(elev=30, azim=225)
    ax2.set_xlabel("East-West (km)", labelpad=8)
    ax2.set_ylabel("South-North (km)", labelpad=8)
    ax2.set_zlabel("Elevation (m)", labelpad=8)
    ax2.set_title("Huashan Core DEM (3D View)")

    plt.tight_layout()
    plt.savefig("huashan_final.png", dpi=300, bbox_inches="tight")
    print("[done] huashan_final.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
