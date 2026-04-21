"""
DEM 地理配准和像元坐标工具。
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile
from pyproj import Transformer


EPSG_SRC = "EPSG:32649"
EPSG_WGS84 = "EPSG:4326"


def read_tiff_with_georef(tif_path: str | Path) -> Tuple[np.ndarray, float, float, float, float]:
    """读取 GeoTIFF 的 DEM 数组和仿射配准基础参数。"""
    with tifffile.TiffFile(Path(tif_path)) as tif:
        page = tif.pages[0]
        dem = page.asarray().astype(float)
        scale = page.tags["ModelPixelScaleTag"].value
        tie = page.tags["ModelTiepointTag"].value

    sx = float(scale[0])
    sy = float(scale[1])
    x0 = float(tie[3])
    y0 = float(tie[4])
    return dem, x0, y0, sx, sy


def pixel_to_xy(row: float, col: float, x0: float, y0: float, sx: float, sy: float) -> Tuple[float, float]:
    """把 DEM 像元坐标转换为投影坐标。"""
    x = x0 + col * sx
    y = y0 - row * sy
    return x, y


def xy_to_pixel(x: float, y: float, x0: float, y0: float, sx: float, sy: float) -> Tuple[int, int]:
    """把投影坐标转换为 DEM 像元坐标。"""
    col = int(round((x - x0) / sx))
    row = int(round((y0 - y) / sy))
    return row, col


def nearest_rc_from_lonlat(lon_grid: np.ndarray, lat_grid: np.ndarray, lon: float, lat: float) -> Tuple[int, int]:
    """在经纬度网格中查找最接近给定经纬度的像元。"""
    d2 = (lon_grid - float(lon)) ** 2 + (lat_grid - float(lat)) ** 2
    idx = int(np.nanargmin(d2))
    r, c = np.unravel_index(idx, lon_grid.shape)
    return int(r), int(c)


def lonlat_to_dem_rc(
    lon: float,
    lat: float,
    x0: float,
    y0: float,
    sx: float,
    sy: float,
    epsg_src: str = EPSG_SRC,
) -> Tuple[int, int]:
    """把 WGS84 经纬度转换到源 DEM 像元坐标。"""
    tf_to_utm = Transformer.from_crs(EPSG_WGS84, epsg_src, always_xy=True)
    x, y = tf_to_utm.transform(float(lon), float(lat))
    return xy_to_pixel(x, y, x0, y0, sx, sy)


def dem_rc_to_lonlat(
    row: int,
    col: int,
    x0: float,
    y0: float,
    sx: float,
    sy: float,
    epsg_src: str = EPSG_SRC,
) -> Tuple[float, float]:
    """把源 DEM 像元坐标转换为 WGS84 经纬度。"""
    tf_to_wgs = Transformer.from_crs(epsg_src, EPSG_WGS84, always_xy=True)
    x, y = pixel_to_xy(float(row), float(col), x0, y0, sx, sy)
    lon, lat = tf_to_wgs.transform(x, y)
    return float(lon), float(lat)
