"""
================================================================================
文件名：huashan_dem.py
用    途：华山景区 DEM 数据裁剪、坐标转换与可视化
================================================================================

【功能说明】
    1. 从原始 ALOS PALSAR DEM（.tif）中，以华山五峰为中心裁剪 10km x 10km 区域
    2. 将 UTM 投影坐标系转换为 WGS84 经纬度坐标系
    3. 生成并排可视化图（左：俯视热力图，右：3D 地形图）
    4. 左图支持鼠标悬停，实时显示当前位置的经纬度和高程

【缓存机制】
    首次运行：从 .tif 裁剪数据，生成经纬度查找表，耗时约 10~30 秒
    后续运行：直接读取缓存文件，秒级启动
    重置缓存：手动删除以下两个缓存文件即可强制重新裁剪

【输入文件】
    AP_19438_FBD_F0680_RT1.dem.tif
        原始 ALOS PALSAR DEM 数据，分辨率 12.5m/像元
        坐标系：UTM 投影（需转换为 WGS84）
        下载来源：NASA EarthData（https://earthdata.nasa.gov）

【输出文件】
    huashan_final.png
        可视化结果图（俯视热力图 + 3D 地形图），用于论文插图

    Z_crop.npy
        裁剪后的高程矩阵，shape=(800, 800)，单位：米
        flipud 处理后行方向为南到北，与地图方向一致
        供后续步骤（可飞空间生成、分层建图）直接读取

    Z_crop_geo.npz
        经纬度查找表，包含两个数组：
            lon_grid[row, col]：每个像素的经度（单位：度E）
            lat_grid[row, col]：每个像素的纬度（单位：度N）
        与 Z_crop.npy 的行列索引一一对应

【依赖库】
    pip install numpy rasterio pyproj matplotlib

【后续步骤】
    Step 2：生成可飞空间（safe_corridor.py）
        输入：Z_crop.npy
        输出：floor_height.npy（飞行下限面）
               ceiling_height.npy（飞行上限面）
    Step 3：分层拓扑路网构建（layered_graph.py）
    Step 4：LPA* 动态增量重规划（lpa_star.py）
================================================================================
"""

import numpy as np
import os
import rasterio
from rasterio.transform import xy as rio_xy
from pyproj import Transformer
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
font = FontProperties(family='SimHei')

# ===== 配置参数 =====
TIF_FILE   = "AP_19438_FBD_F0680_RT1.dem.tif"
CACHE_FILE = "Z_crop.npy"
CACHE_GEO  = "Z_crop_geo.npz"
RESOLUTION = 12.5

PEAKS = {
    "南峰": {"row": 4609, "col": 1938, "elev": 2154.0},
    "东峰": {"row": 4642, "col": 1985, "elev": 2096.0},
    "西峰": {"row": 4600, "col": 1949, "elev": 2082.0},
    "北峰": {"row": 4468, "col": 2004, "elev": 1615.0},
    "中峰": {"row": 4594, "col": 1951, "elev": 2038.0},
}

# ===== 裁剪范围计算 =====
center_row = int(np.mean([p["row"] for p in PEAKS.values()]))
center_col = int(np.mean([p["col"] for p in PEAKS.values()]))
half       = int(10000 / 2 / RESOLUTION)
row_min    = center_row - half
row_max    = center_row + half
col_min    = center_col - half
col_max    = center_col + half
total_rows = row_max - row_min
total_cols = col_max - col_min

# ===== 缓存逻辑 =====
if os.path.exists(CACHE_FILE) and os.path.exists(CACHE_GEO):
    print("[缓存] 检测到缓存文件，直接读取...")
    Z_crop = np.load(CACHE_FILE)
    geo = np.load(CACHE_GEO)
    # 每个像素的经纬度查找表（flipud后的）
    lon_grid = geo["lon_grid"]
    lat_grid = geo["lat_grid"]
    print(f"[缓存] 读取完成，shape={Z_crop.shape}")

else:
    print(f"[裁剪] 未找到缓存，从 {TIF_FILE} 裁剪...")

    with rasterio.open(TIF_FILE) as src:
        Z_full = src.read(1).astype(float)
        Z_full[Z_full < -9000] = np.nan
        transform = src.transform
        src_crs   = src.crs

    print(f"[信息] 原始坐标系: {src_crs}")

    # 建立 UTM → WGS84 转换器
    transformer = Transformer.from_crs(
        src_crs, "EPSG:4326", always_xy=True
    )

    # 裁剪
    Z_crop = Z_full[row_min:row_max, col_min:col_max]
    Z_crop = np.flipud(Z_crop)

    # ===== 为每个像素计算经纬度查找表 =====
    print("[计算] 正在生成经纬度查找表（约需10秒）...")
    rows_idx = np.arange(total_rows)
    cols_idx = np.arange(total_cols)

    # 每个像素在原始tif中的行列号
    orig_rows = (row_min + (total_rows - 1 - rows_idx)).astype(int)  # flipud
    orig_cols = (col_min + cols_idx).astype(int)

    # 向量化计算所有像素的 UTM 坐标
    orig_rows_2d, orig_cols_2d = np.meshgrid(orig_rows, orig_cols, indexing='ij')
    utm_x_2d = transform.c + orig_cols_2d * transform.a + orig_rows_2d * transform.b
    utm_y_2d = transform.f + orig_cols_2d * transform.d + orig_rows_2d * transform.e

    # 批量转换为经纬度
    lon_flat, lat_flat = transformer.transform(
        utm_x_2d.ravel(), utm_y_2d.ravel()
    )
    lon_grid = lon_flat.reshape(total_rows, total_cols)
    lat_grid = lat_flat.reshape(total_rows, total_cols)

    # 保存缓存
    np.save(CACHE_FILE, Z_crop)
    np.savez(CACHE_GEO, lon_grid=lon_grid, lat_grid=lat_grid)
    print(f"[裁剪] 完成，缓存已保存")

print(f"高程范围: {np.nanmin(Z_crop):.0f}m ~ {np.nanmax(Z_crop):.0f}m")
print(f"经度范围: {lon_grid.min():.4f}°E ~ {lon_grid.max():.4f}°E")
print(f"纬度范围: {lat_grid.min():.4f}°N ~ {lat_grid.max():.4f}°N")

# ===== 计算峰值 km 坐标 =====
peak_coords = {}
for name, p in PEAKS.items():
    r_in_crop = p["row"] - row_min
    c_in_crop = p["col"] - col_min
    r_flipped = total_rows - 1 - r_in_crop
    x_km = c_in_crop * RESOLUTION / 1000
    y_km = r_flipped * RESOLUTION / 1000
    lon  = lon_grid[r_flipped, c_in_crop]
    lat  = lat_grid[r_flipped, c_in_crop]
    peak_coords[name] = {
        "x": x_km, "y": y_km,
        "elev": p["elev"],
        "lon": lon, "lat": lat
    }
    print(f"  {name}: {lon:.5f}°E, {lat:.5f}°N, 海拔={p['elev']}m")

# ===== 绘图 =====
fig = plt.figure(figsize=(20, 8))

# ---------- 左图：俯视热力图 ----------
ax1 = fig.add_subplot(121)
extent = [0, total_cols * RESOLUTION / 1000,
          0, total_rows * RESOLUTION / 1000]
im = ax1.imshow(Z_crop, cmap='terrain',
                extent=extent, origin='upper', aspect='equal')
plt.colorbar(im, ax=ax1, label='高程 (m)', shrink=0.8)

# 标注峰值
for name, c in peak_coords.items():
    ax1.plot(c["x"], c["y"], 'r^', markersize=10, zorder=5)
    ax1.annotate(
        f'{name}  {c["elev"]:.0f}m\n'
        f'{c["lon"]:.5f}°E\n'
        f'{c["lat"]:.5f}°N',
        xy=(c["x"], c["y"]),
        xytext=(c["x"] + 0.5, c["y"] + 0.5),
        fontproperties=font, fontsize=8, color='darkred',
        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
        bbox=dict(boxstyle='round,pad=0.3',
                  facecolor='white', edgecolor='red', alpha=0.85)
    )

ax1.set_xlabel('东西方向 (km)', fontproperties=font)
ax1.set_ylabel('南北方向 (km)', fontproperties=font)
ax1.set_title('华山核心区域 俯视热力图（鼠标悬停查看经纬度和高程）',
              fontproperties=font, fontsize=11)
ax1.grid(True, alpha=0.3, linestyle='--')

# ===== 鼠标悬停交互 =====
annot = ax1.annotate(
    "", xy=(0, 0), xytext=(15, 15),
    textcoords="offset points",
    bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.9),
    arrowprops=dict(arrowstyle="->"),
    fontsize=9
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

    col_idx = int(x_km * 1000 / RESOLUTION)
    row_idx = int(y_km * 1000 / RESOLUTION)
    col_idx = np.clip(col_idx, 0, total_cols - 1)
    row_idx = np.clip(row_idx, 0, total_rows - 1)

    elev = Z_crop[row_idx, col_idx]
    if np.isnan(elev):
        annot.set_visible(False)
        fig.canvas.draw_idle()
        return

    # 直接从查找表取经纬度
    lon = lon_grid[row_idx, col_idx]
    lat = lat_grid[row_idx, col_idx]

    text = (f"经度: {lon:.5f}°E\n"
            f"纬度: {lat:.5f}°N\n"
            f"高程: {elev:.1f} m")

    annot.xy = (x_km, y_km)
    annot.set_text(text)
    annot.set_visible(True)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", on_hover)

# ---------- 右图：3D 视图 ----------
ax2 = fig.add_subplot(122, projection='3d')
step   = 5
Z_show = Z_crop[::step, ::step]
rows_s, cols_s = Z_show.shape
xg = np.arange(cols_s) * step * RESOLUTION / 1000
yg = np.arange(rows_s) * step * RESOLUTION / 1000
X_grid, Y_grid = np.meshgrid(xg, yg)

surf = ax2.plot_surface(X_grid, Y_grid, Z_show,
                        cmap='terrain', alpha=0.85, linewidth=0)

for name, c in peak_coords.items():
    ax2.scatter(c["x"], c["y"], c["elev"] + 100,
                color='red', s=60, zorder=5)
    ax2.text(c["x"], c["y"], c["elev"] + 260,
             name, fontsize=8, color='red',
             fontproperties=font, ha='center')

ax2.view_init(elev=30, azim=225)
ax2.set_xlabel('东西 (km)', fontproperties=font, labelpad=8)
ax2.set_ylabel('南北 (km)', fontproperties=font, labelpad=8)
ax2.set_zlabel('高度 (m)', fontproperties=font, labelpad=8)
ax2.set_title('华山核心区域 3D视图', fontproperties=font, fontsize=12)

plt.tight_layout()
plt.savefig('huashan_final.png', dpi=150, bbox_inches='tight')
print("\n静态图已保存为 huashan_final.png")
print("交互窗口已打开，鼠标悬停在左图查看经纬度和高程")
plt.show()