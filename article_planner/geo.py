"""
DEM 地理配准和像元坐标工具。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile
from pyproj import CRS, Transformer


WGS84_CRS = "EPSG:4326"


@dataclass(frozen=True)
class GeoTiffProfile:
    """GeoTIFF 的最小地理配准信息。"""

    dem: np.ndarray
    x0: float
    y0: float
    sx: float
    sy: float
    source_crs: str

    @property
    def resolution_m(self) -> float:
        """返回像元尺度的米制代表值；当前工程假设输入 DEM 是米制投影坐标。"""
        crs = CRS.from_user_input(self.source_crs)
        units = {str(axis.unit_name).lower() for axis in crs.axis_info}
        if not any(unit in {"metre", "meter", "m"} for unit in units):
            raise ValueError(
                f"DEM 源 CRS={self.source_crs} 不是米制投影坐标，无法从像元尺度直接得到米制分辨率。"
                "请先把 DEM 重投影到米制投影坐标系。"
            )
        return float((abs(self.sx) + abs(self.sy)) / 2.0)


def source_crs_from_geokeys(page: tifffile.TiffPage, fallback: str | None = None) -> str:
    """从 GeoTIFF GeoKeyDirectoryTag 中解析源 CRS。"""
    tag = page.tags.get("GeoKeyDirectoryTag")
    if tag is not None:
        values = tuple(int(v) for v in tag.value)
        if len(values) >= 4:
            key_count = int(values[3])
            for i in range(key_count):
                start = 4 + i * 4
                if start + 3 >= len(values):
                    break
                key_id, tiff_tag_location, count, value_offset = values[start : start + 4]
                if tiff_tag_location == 0 and count == 1 and key_id in {3072, 2048}:
                    code = int(value_offset)
                    if code not in {0, 32767}:
                        return CRS.from_epsg(code).to_string()

    if fallback:
        return CRS.from_user_input(fallback).to_string()
    raise ValueError("GeoTIFF 缺少可解析 CRS；请检查 DEM 元数据，或在场景配置中提供 source_crs。")


def read_tiff_profile(tif_path: str | Path, fallback_crs: str | None = None) -> GeoTiffProfile:
    """读取 GeoTIFF 的 DEM 数组、仿射配准参数和源 CRS。"""
    with tifffile.TiffFile(Path(tif_path)) as tif:
        page = tif.pages[0]
        dem = page.asarray().astype(float)
        scale = page.tags["ModelPixelScaleTag"].value
        tie = page.tags["ModelTiepointTag"].value
        source_crs = source_crs_from_geokeys(page, fallback=fallback_crs)

    sx = float(scale[0])
    sy = float(scale[1])
    x0 = float(tie[3])
    y0 = float(tie[4])
    return GeoTiffProfile(dem=dem, x0=x0, y0=y0, sx=sx, sy=sy, source_crs=source_crs)


def read_tiff_with_georef(tif_path: str | Path) -> Tuple[np.ndarray, float, float, float, float]:
    """读取 GeoTIFF 的 DEM 数组和仿射配准基础参数。"""
    profile = read_tiff_profile(tif_path)
    return profile.dem, profile.x0, profile.y0, profile.sx, profile.sy


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
    source_crs: str,
) -> Tuple[int, int]:
    """把 WGS84 经纬度转换到源 DEM 像元坐标。"""
    tf_to_utm = Transformer.from_crs(WGS84_CRS, source_crs, always_xy=True)
    x, y = tf_to_utm.transform(float(lon), float(lat))
    return xy_to_pixel(x, y, x0, y0, sx, sy)


def dem_rc_to_lonlat(
    row: int,
    col: int,
    x0: float,
    y0: float,
    sx: float,
    sy: float,
    source_crs: str,
) -> Tuple[float, float]:
    """把源 DEM 像元坐标转换为 WGS84 经纬度。"""
    tf_to_wgs = Transformer.from_crs(source_crs, WGS84_CRS, always_xy=True)
    x, y = pixel_to_xy(float(row), float(col), x0, y0, sx, sy)
    lon, lat = tf_to_wgs.transform(x, y)
    return float(lon), float(lat)
