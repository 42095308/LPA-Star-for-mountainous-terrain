# import rasterio

# tif_file = "AP_19438_FBD_F0680_RT1.dem.tif"

# with rasterio.open(tif_file) as src:
#     print("CRS:", src.crs)
#     print("Bounds:", src.bounds)
#     print("Resolution:", src.res)

# import os
# print("当前工作目录:", os.getcwd())
# print("文件是否存在:", os.path.exists("huashan_crop.tif"))

# from pyproj import Transformer

# peak_x = 400357.625
# peak_y = 3804566.5

# transformer = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)
# lon, lat = transformer.transform(peak_x, peak_y)

# print("最高点经纬度:", lon, lat)


# import rasterio
# from pyproj import Transformer

# tif_file = "AP_19438_FBD_F0680_RT1.dem.tif"

# # 华山主峰（建议先用南峰/主峰区域）
# huashan_lon = 110.080
# huashan_lat = 34.478

# with rasterio.open(tif_file) as src:
#     print("DEM bounds:", src.bounds)
#     print("DEM CRS:", src.crs)

#     # 经纬度 -> DEM投影坐标
#     transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
#     x, y = transformer.transform(huashan_lon, huashan_lat)

#     print("\n华山主峰转换后的UTM坐标:")
#     print("x =", x)
#     print("y =", y)

#     # 判断是否在 DEM 范围内
#     in_bounds = (
#         src.bounds.left <= x <= src.bounds.right and
#         src.bounds.bottom <= y <= src.bounds.top
#     )
#     print("\n华山主峰是否在当前DEM范围内:", in_bounds)

#     if in_bounds:
#         row, col = src.index(x, y)
#         dem = src.read(1)
#         elev = dem[row, col]

#         print("对应像素位置:", row, col)
#         print("该像素高程:", elev)


# import rasterio
# from pyproj import Transformer
# from rasterio.transform import xy
# import numpy as np

# tif_file = "AP_19438_FBD_F0680_RT1.dem.tif"

# # 华山主峰近似经纬度
# huashan_lon = 110.080
# huashan_lat = 34.478

# # 搜索半径（单位：像元）
# search_radius = 80

# with rasterio.open(tif_file) as src:
#     dem = src.read(1).astype(float)
#     nodata = src.nodata
#     if nodata is not None:
#         dem[dem == nodata] = np.nan

#     # 初始化坐标逆向转换器 (DEM坐标系 -> WGS84经纬度)
#     transformer_back = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

#     print("========================================")
#     print("1. 全局最高点搜索 (验证整张DEM的数据极限)")
#     print("========================================")
    
#     # 找全局最高点
#     global_max_idx = np.nanargmax(dem)
#     global_r, global_c = np.unravel_index(global_max_idx, dem.shape)
#     global_elev = dem[global_r, global_c]
    
#     # 转换全局最高点坐标
#     global_x, global_y = xy(src.transform, global_r, global_c)
#     global_lon, global_lat = transformer_back.transform(global_x, global_y)
    
#     print("全局最高点像元位置:", global_r, global_c)
#     print("全局最高点UTM坐标:", global_x, global_y)
#     print("全局最高点经纬度:", global_lon, global_lat)
#     print(f"全局最高点高程: {global_elev} 米\n")


#     print("========================================")
#     print("2. 局部最高点搜索 (原代码逻辑)")
#     print("========================================")
    
#     # 经纬度 -> DEM坐标系
#     transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
#     x, y = transformer.transform(huashan_lon, huashan_lat)

#     # 找到主峰近似位置对应像元
#     center_row, center_col = src.index(x, y)

#     print("华山主峰近似像元位置:", center_row, center_col)
#     print("该像元高程:", dem[center_row, center_col])

#     # 取附近窗口
#     r0 = max(0, center_row - search_radius)
#     r1 = min(dem.shape[0], center_row + search_radius + 1)
#     c0 = max(0, center_col - search_radius)
#     c1 = min(dem.shape[1], center_col + search_radius + 1)

#     patch = dem[r0:r1, c0:c1]

#     # 找局部最高点
#     local_max_idx = np.nanargmax(patch)
#     local_r, local_c = np.unravel_index(local_max_idx, patch.shape)

#     peak_row = r0 + local_r
#     peak_col = c0 + local_c
#     peak_elev = dem[peak_row, peak_col]

#     peak_x, peak_y = xy(src.transform, peak_row, peak_col)

#     # 转回经纬度
#     peak_lon, peak_lat = transformer_back.transform(peak_x, peak_y)

#     print("局部最高点像元位置:", peak_row, peak_col)
#     print("局部最高点UTM坐标:", peak_x, peak_y)
#     print("局部最高点经纬度:", peak_lon, peak_lat)
#     print(f"局部最高点高程: {peak_elev} 米")


# import rasterio
# from pyproj import Transformer
# from rasterio.transform import xy
# import numpy as np

# tif_file = "AP_19438_FBD_F0680_RT1.dem.tif"

# # 南峰坐标，多试几个常见来源
# candidates = {
#     "南峰_标准":  (110.083, 34.483),
#     "南峰_游记":  (110.080, 34.478),
#     "游客中心":   (110.071, 34.493),
# }

# # 大幅扩大搜索半径到 5km（400个像元）
# search_radius = 400

# with rasterio.open(tif_file) as src:
#     dem = src.read(1).astype(float)
#     nodata = src.nodata
#     if nodata is not None:
#         dem[dem == nodata] = np.nan
#     else:
#         dem[dem < -100] = np.nan

#     transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
#     transformer_back = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

#     print(f"DEM 总体高程范围: {np.nanmin(dem):.1f}m ~ {np.nanmax(dem):.1f}m")
#     print(f"DEM 全局最高点高程: {np.nanmax(dem):.1f}m")

#     # 找全局最高点
#     global_max_idx = np.nanargmax(dem)
#     gr, gc = np.unravel_index(global_max_idx, dem.shape)
#     gx, gy = xy(src.transform, gr, gc)
#     glon, glat = transformer_back.transform(gx, gy)
#     print(f"\n全局最高点: 像元({gr}, {gc}), 经纬度({glon:.5f}, {glat:.5f}), 高程: {dem[gr,gc]:.1f}m")

#     print("\n--- 各候选坐标附近最高点 ---")
#     for name, (lon, lat) in candidates.items():
#         x, y = transformer.transform(lon, lat)
#         crow, ccol = src.index(x, y)

#         r0 = max(0, crow - search_radius)
#         r1 = min(dem.shape[0], crow + search_radius + 1)
#         c0 = max(0, ccol - search_radius)
#         c1 = min(dem.shape[1], ccol + search_radius + 1)
#         patch = dem[r0:r1, c0:c1]

#         local_max_idx = np.nanargmax(patch)
#         local_r, local_c = np.unravel_index(local_max_idx, patch.shape)
#         peak_row = r0 + local_r
#         peak_col = c0 + local_c
#         peak_elev = dem[peak_row, peak_col]
#         px, py = xy(src.transform, peak_row, peak_col)
#         plon, plat = transformer_back.transform(px, py)

#         print(f"[{name}] 初始像元({crow},{ccol}) -> 局部最高: ({peak_row},{peak_col}), "
#               f"经纬度({plon:.5f},{plat:.5f}), 高程:{peak_elev:.1f}m")


# import rasterio
# from pyproj import Transformer
# from rasterio.transform import xy
# import numpy as np

# tif_file = "AP_19438_FBD_F0680_RT1.dem.tif"

# # 系统偏移量（DEM坐标 = 真实坐标 + offset）
# LON_OFFSET = +0.0485
# LAT_OFFSET = -0.0170

# # 华山五峰真实经纬度 + 真实高程
# peaks = {
#     "南峰": {"lon": 110.0781, "lat": 34.4778, "elev": 2154.9},
#     "东峰": {"lon": 110.0880, "lat": 34.4786, "elev": 2096.2},
#     "西峰": {"lon": 110.0820, "lat": 34.4831, "elev": 2082.0},
#     "北峰": {"lon": 110.0869, "lat": 34.4924, "elev": 1614.9},
#     "中峰": {"lon": 110.0808, "lat": 34.4806, "elev": 2037.8},
# }

# # 搜索窗口（校准后精确搜，只需 30 像元 = 375m）
# search_radius = 30

# with rasterio.open(tif_file) as src:
#     dem = src.read(1).astype(float)
#     nodata = src.nodata
#     if nodata is not None:
#         dem[dem == nodata] = np.nan
#     else:
#         dem[dem < -100] = np.nan

#     transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
#     transformer_back = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

#     print(f"{'峰名':<6} {'像元(row,col)':<18} {'DEM高程':>10} {'真实高程':>10} {'误差':>8} {'校准后经度':>12} {'校准后纬度':>12}")
#     print("-" * 85)

#     peak_pixels = {}  # 保存像元坐标供后续路径规划使用

#     for name, info in peaks.items():
#         # 用校准后的坐标去 DEM 里找
#         adj_lon = info["lon"] + LON_OFFSET
#         adj_lat = info["lat"] + LAT_OFFSET

#         x, y = transformer.transform(adj_lon, adj_lat)
#         crow, ccol = src.index(x, y)

#         r0 = max(0, crow - search_radius)
#         r1 = min(dem.shape[0], crow + search_radius + 1)
#         c0 = max(0, ccol - search_radius)
#         c1 = min(dem.shape[1], ccol + search_radius + 1)
#         patch = dem[r0:r1, c0:c1]

#         local_max_idx = np.nanargmax(patch)
#         lr, lc = np.unravel_index(local_max_idx, patch.shape)
#         peak_row = r0 + lr
#         peak_col = c0 + lc
#         peak_elev = dem[peak_row, peak_col]

#         px, py = xy(src.transform, peak_row, peak_col)
#         found_lon, found_lat = transformer_back.transform(px, py)
#         # 反校准回真实经纬度
#         real_lon = found_lon - LON_OFFSET
#         real_lat = found_lat - LAT_OFFSET

#         err = abs(peak_elev - info["elev"])
#         peak_pixels[name] = (peak_row, peak_col)

#         print(f"{name:<6} ({peak_row},{peak_col}){'':<6} {peak_elev:>10.1f} {info['elev']:>10.1f} {err:>8.1f} {real_lon:>12.5f} {real_lat:>12.5f}")

#     print("\n像元坐标汇总（用于路径规划）：")
#     for name, (r, c) in peak_pixels.items():
#         print(f"  {name}: row={r}, col={c}")



"""
用于找出华山5峰的坐标位置。
参考数据为：
    1. 东峰（朝阳峰）：34.4786°N，110.0880°E（34°28′43″N，110°05′17″E），海拔2096.2米

    2. 南峰（落雁峰，华山极顶）：34.4778°N，110.0781°E（34°28′40″N，110°04′41″E），海拔2154.9米；另一个常用实测点34°28′37″N，110°05′05″E

    3. 西峰（莲花峰）：34.4831°N，110.0820°E（34°28′59″N，110°04′55″E）；山峰库标准34.4816°N，110.0768°E（34°28′54″N，110°04′36″E），海拔2082.6米

    4. 北峰（云台峰）：34.4924°N，110.0869°E（34°29′33″N，110°05′13″E），海拔1614.9米

    5. 中峰（玉女峰）：34.4806°N，110.0808°E（34°28′50″N，110°04′51″E），海拔2037.8/2042.5米（不同数据源的高程取整差异）


以下代码的结果：建议保存为常量

TIF_FILE = "AP_19438_FBD_F0680_RT1.dem.tif"
RESOLUTION = 12.5   # 米/像元
LON_OFFSET = +0.0485
LAT_OFFSET = -0.0170

PEAKS = {
    "南峰": {"row": 4609, "col": 1938, "elev": 2154.0},
    "东峰": {"row": 4642, "col": 1985, "elev": 2096.0},
    "西峰": {"row": 4600, "col": 1949, "elev": 2082.0},
    "北峰": {"row": 4468, "col": 2004, "elev": 1615.0},
    "中峰": {"row": 4594, "col": 1951, "elev": 2038.0},
}    
"""

import rasterio
from pyproj import Transformer
from rasterio.transform import xy
import numpy as np

tif_file = "AP_19438_FBD_F0680_RT1.dem.tif"

LON_OFFSET = +0.0485
LAT_OFFSET = -0.0170

peaks = {
    "南峰": {"lon": 110.0781, "lat": 34.4778, "elev": 2154.9},
    "东峰": {"lon": 110.0880, "lat": 34.4786, "elev": 2096.2},
    "西峰": {"lon": 110.0820, "lat": 34.4831, "elev": 2082.0},
    "北峰": {"lon": 110.0869, "lat": 34.4924, "elev": 1614.9},
    "中峰": {"lon": 110.0808, "lat": 34.4806, "elev": 2037.8},
}

search_radius = 30

with rasterio.open(tif_file) as src:
    dem = src.read(1).astype(float)
    nodata = src.nodata
    if nodata is not None:
        dem[dem == nodata] = np.nan
    else:
        dem[dem < -100] = np.nan

    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    transformer_back = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

    print(f"{'峰名':<6} {'像元(row,col)':<18} {'DEM高程':>10} {'真实高程':>10} {'误差':>8} {'校准后经度':>12} {'校准后纬度':>12}")
    print("-" * 85)

    peak_pixels = {}

    for name, info in peaks.items():
        adj_lon = info["lon"] + LON_OFFSET
        adj_lat = info["lat"] + LAT_OFFSET

        x, y = transformer.transform(adj_lon, adj_lat)
        crow, ccol = src.index(x, y)

        r0 = max(0, crow - search_radius)
        r1 = min(dem.shape[0], crow + search_radius + 1)
        c0 = max(0, ccol - search_radius)
        c1 = min(dem.shape[1], ccol + search_radius + 1)
        patch = dem[r0:r1, c0:c1]

        # ✅ 关键改动：找最接近目标高程的点，而不是最高点
        diff = np.abs(patch - info["elev"])
        best_idx = np.nanargmin(diff)
        lr, lc = np.unravel_index(best_idx, patch.shape)
        peak_row = r0 + lr
        peak_col = c0 + lc
        peak_elev = dem[peak_row, peak_col]

        px, py = xy(src.transform, peak_row, peak_col)
        found_lon, found_lat = transformer_back.transform(px, py)
        real_lon = found_lon - LON_OFFSET
        real_lat = found_lat - LAT_OFFSET

        err = abs(peak_elev - info["elev"])
        peak_pixels[name] = (peak_row, peak_col)

        print(f"{name:<6} ({peak_row},{peak_col}){'':<6} {peak_elev:>10.1f} "
              f"{info['elev']:>10.1f} {err:>8.1f} {real_lon:>12.5f} {real_lat:>12.5f}")

    print("\n像元坐标汇总（用于路径规划）：")
    for name, (r, c) in peak_pixels.items():
        print(f"  {name}: row={r}, col={c}")