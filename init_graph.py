"""
================================================================================
鏂囦欢鍚嶏細huashan_dem.py
鐢?   閫旓細鍗庡北鏅尯 DEM 鏁版嵁瑁佸壀銆佸潗鏍囪浆鎹笌鍙鍖?
================================================================================

銆愬姛鑳借鏄庛€?
    1. 浠庡師濮?ALOS PALSAR DEM锛?tif锛変腑锛屼互鍗庡北浜斿嘲涓轰腑蹇冭鍓?10km x 10km 鍖哄煙
    2. 灏?UTM 鎶曞奖鍧愭爣绯昏浆鎹负 WGS84 缁忕含搴﹀潗鏍囩郴
    3. 鐢熸垚骞舵帓鍙鍖栧浘锛堝乏锛氫刊瑙嗙儹鍔涘浘锛屽彸锛?D 鍦板舰鍥撅級
    4. 宸﹀浘鏀寔榧犳爣鎮仠锛屽疄鏃舵樉绀哄綋鍓嶄綅缃殑缁忕含搴﹀拰楂樼▼

銆愮紦瀛樻満鍒躲€?
    棣栨杩愯锛氫粠 .tif 瑁佸壀鏁版嵁锛岀敓鎴愮粡绾害鏌ユ壘琛紝鑰楁椂绾?10~30 绉?
    鍚庣画杩愯锛氱洿鎺ヨ鍙栫紦瀛樻枃浠讹紝绉掔骇鍚姩
    閲嶇疆缂撳瓨锛氭墜鍔ㄥ垹闄や互涓嬩袱涓紦瀛樻枃浠跺嵆鍙己鍒堕噸鏂拌鍓?

銆愯緭鍏ユ枃浠躲€?
    AP_19438_FBD_F0680_RT1.dem.tif
        鍘熷 ALOS PALSAR DEM 鏁版嵁锛屽垎杈ㄧ巼 12.5m/鍍忓厓
        鍧愭爣绯伙細UTM 鎶曞奖锛堥渶杞崲涓?WGS84锛?
        涓嬭浇鏉ユ簮锛歂ASA EarthData锛坔ttps://earthdata.nasa.gov锛?

銆愯緭鍑烘枃浠躲€?
    huashan_final.png
        鍙鍖栫粨鏋滃浘锛堜刊瑙嗙儹鍔涘浘 + 3D 鍦板舰鍥撅級锛岀敤浜庤鏂囨彃鍥?

    Z_crop.npy
        瑁佸壀鍚庣殑楂樼▼鐭╅樀锛宻hape=(800, 800)锛屽崟浣嶏細绫?
        flipud 澶勭悊鍚庤鏂瑰悜涓哄崡鍒板寳锛屼笌鍦板浘鏂瑰悜涓€鑷?
        渚涘悗缁楠わ紙鍙绌洪棿鐢熸垚銆佸垎灞傚缓鍥撅級鐩存帴璇诲彇

    Z_crop_geo.npz
        缁忕含搴︽煡鎵捐〃锛屽寘鍚袱涓暟缁勶細
            lon_grid[row, col]锛氭瘡涓儚绱犵殑缁忓害锛堝崟浣嶏細搴锛?
            lat_grid[row, col]锛氭瘡涓儚绱犵殑绾害锛堝崟浣嶏細搴锛?
        涓?Z_crop.npy 鐨勮鍒楃储寮曚竴涓€瀵瑰簲

銆愪緷璧栧簱銆?
    pip install numpy rasterio pyproj matplotlib

銆愬悗缁楠ゃ€?
    Step 2锛氱敓鎴愬彲椋炵┖闂达紙safe_corridor.py锛?
        杈撳叆锛歓_crop.npy
        杈撳嚭锛歠loor_height.npy锛堥琛屼笅闄愰潰锛?
               ceiling_height.npy锛堥琛屼笂闄愰潰锛?
    Step 3锛氬垎灞傛嫇鎵戣矾缃戞瀯寤猴紙layered_graph.py锛?
    Step 4锛歀PA* 鍔ㄦ€佸閲忛噸瑙勫垝锛坙pa_star.py锛?
================================================================================
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

try:
    import rasterio
except Exception:
    rasterio = None

try:
    from pyproj import Transformer
except Exception:
    Transformer = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="backslashreplace")

matplotlib.rcParams['font.family'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ===== 閰嶇疆鍙傛暟 =====
TIF_FILE   = "AP_19438_FBD_F0680_RT1.dem.tif"
CACHE_FILE = "Z_crop.npy"
CACHE_GEO  = "Z_crop_geo.npz"
RESOLUTION = 12.5

PEAKS = {
    "South Peak": {"row": 4609, "col": 1938, "elev": 2154.0},
    "East Peak": {"row": 4642, "col": 1985, "elev": 2096.0},
    "West Peak": {"row": 4600, "col": 1949, "elev": 2082.0},
    "North Peak": {"row": 4468, "col": 2004, "elev": 1615.0},
    "Central Peak": {"row": 4594, "col": 1951, "elev": 2038.0},
}

# ===== 瑁佸壀鑼冨洿璁＄畻 =====
center_row = int(np.mean([p["row"] for p in PEAKS.values()]))
center_col = int(np.mean([p["col"] for p in PEAKS.values()]))
half       = int(10000 / 2 / RESOLUTION)
row_min    = center_row - half
row_max    = center_row + half
col_min    = center_col - half
col_max    = center_col + half
total_rows = row_max - row_min
total_cols = col_max - col_min

# ===== 缂撳瓨閫昏緫 =====
if os.path.exists(CACHE_FILE) and os.path.exists(CACHE_GEO):
    print("[缂撳瓨] 妫€娴嬪埌缂撳瓨鏂囦欢锛岀洿鎺ヨ鍙?..")
    Z_crop = np.load(CACHE_FILE)
    geo = np.load(CACHE_GEO)
    # 姣忎釜鍍忕礌鐨勭粡绾害鏌ユ壘琛紙flipud鍚庣殑锛?
    lon_grid = geo["lon_grid"]
    lat_grid = geo["lat_grid"]
    print(f"[缂撳瓨] 璇诲彇瀹屾垚锛宻hape={Z_crop.shape}")

else:
    if rasterio is None:
        raise RuntimeError('rasterio is not installed and cache files are missing. Install rasterio or prepare Z_crop.npy and Z_crop_geo.npz.')
    if Transformer is None:
        raise RuntimeError("pyproj is not installed and cache files are missing.")
    print(f"[瑁佸壀] 鏈壘鍒扮紦瀛橈紝浠?{TIF_FILE} 瑁佸壀...")

    with rasterio.open(TIF_FILE) as src:
        Z_full = src.read(1).astype(float)
        Z_full[Z_full < -9000] = np.nan
        transform = src.transform
        src_crs   = src.crs

    print(f"[淇℃伅] 鍘熷鍧愭爣绯? {src_crs}")

    # 寤虹珛 UTM 鈫?WGS84 杞崲鍣?
    transformer = Transformer.from_crs(
        src_crs, "EPSG:4326", always_xy=True
    )

    # 瑁佸壀
    Z_crop = Z_full[row_min:row_max, col_min:col_max]
    Z_crop = np.flipud(Z_crop)

    # ===== 涓烘瘡涓儚绱犺绠楃粡绾害鏌ユ壘琛?=====
    print("[璁＄畻] 姝ｅ湪鐢熸垚缁忕含搴︽煡鎵捐〃锛堢害闇€10绉掞級...")
    rows_idx = np.arange(total_rows)
    cols_idx = np.arange(total_cols)

    # 姣忎釜鍍忕礌鍦ㄥ師濮媡if涓殑琛屽垪鍙?
    orig_rows = (row_min + (total_rows - 1 - rows_idx)).astype(int)  # flipud
    orig_cols = (col_min + cols_idx).astype(int)

    # 鍚戦噺鍖栬绠楁墍鏈夊儚绱犵殑 UTM 鍧愭爣
    orig_rows_2d, orig_cols_2d = np.meshgrid(orig_rows, orig_cols, indexing='ij')
    utm_x_2d = transform.c + orig_cols_2d * transform.a + orig_rows_2d * transform.b
    utm_y_2d = transform.f + orig_cols_2d * transform.d + orig_rows_2d * transform.e

    # 鎵归噺杞崲涓虹粡绾害
    lon_flat, lat_flat = transformer.transform(
        utm_x_2d.ravel(), utm_y_2d.ravel()
    )
    lon_grid = lon_flat.reshape(total_rows, total_cols)
    lat_grid = lat_flat.reshape(total_rows, total_cols)

    # 淇濆瓨缂撳瓨
    np.save(CACHE_FILE, Z_crop)
    np.savez(CACHE_GEO, lon_grid=lon_grid, lat_grid=lat_grid)
    print(f"[瑁佸壀] 瀹屾垚锛岀紦瀛樺凡淇濆瓨")

print(f"楂樼▼鑼冨洿: {np.nanmin(Z_crop):.0f}m ~ {np.nanmax(Z_crop):.0f}m")
print(f"缁忓害鑼冨洿: {lon_grid.min():.4f}掳E ~ {lon_grid.max():.4f}掳E")
print(f"绾害鑼冨洿: {lat_grid.min():.4f}掳N ~ {lat_grid.max():.4f}掳N")

# ===== 璁＄畻宄板€?km 鍧愭爣 =====
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
    print(f"  {name}: {lon:.5f}掳E, {lat:.5f}掳N, 娴锋嫈={p['elev']}m")

# ===== 缁樺浘 =====
fig = plt.figure(figsize=(20, 8))

# ---------- 宸﹀浘锛氫刊瑙嗙儹鍔涘浘 ----------
ax1 = fig.add_subplot(121)
extent = [0, total_cols * RESOLUTION / 1000,
          0, total_rows * RESOLUTION / 1000]
im = ax1.imshow(Z_crop, cmap='terrain',
                extent=extent, origin='upper', aspect='equal')
plt.colorbar(im, ax=ax1, label='Elevation (m)', shrink=0.8)

# 鏍囨敞宄板€?
for name, c in peak_coords.items():
    ax1.plot(c["x"], c["y"], 'r^', markersize=10, zorder=5)
    ax1.annotate(
        f'{name}  {c["elev"]:.0f}m\n'
        f'{c["lon"]:.5f}°E\n'
        f'{c["lat"]:.5f}°N',
        xy=(c["x"], c["y"]),
        xytext=(c["x"] + 0.5, c["y"] + 0.5),
        fontsize=8, color='darkred',
        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
        bbox=dict(boxstyle='round,pad=0.3',
                  facecolor='white', edgecolor='red', alpha=0.85)
    )

ax1.set_xlabel('East-West (km)')
ax1.set_ylabel('South-North (km)')
ax1.set_title(
    "Huashan Core DEM (Top View with Peak Labels)",
    fontsize=11,
)
ax1.grid(True, alpha=0.3, linestyle='--')

# ===== 榧犳爣鎮仠浜や簰 =====
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

    # 鐩存帴浠庢煡鎵捐〃鍙栫粡绾害
    lon = lon_grid[row_idx, col_idx]
    lat = lat_grid[row_idx, col_idx]

    text = (
        f"Lon: {lon:.5f}°E\n"
        f"Lat: {lat:.5f}°N\n"
        f"Elevation: {elev:.1f} m"
    )

    annot.xy = (x_km, y_km)
    annot.set_text(text)
    annot.set_visible(True)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", on_hover)

# ---------- 鍙冲浘锛?D 瑙嗗浘 ----------
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
             ha='center')

ax2.view_init(elev=30, azim=225)
ax2.set_xlabel('East-West (km)', labelpad=8)
ax2.set_ylabel('South-North (km)', labelpad=8)
ax2.set_zlabel('Elevation (m)', labelpad=8)
ax2.set_title('Huashan Core DEM (3D View)', fontsize=12)

plt.tight_layout()
plt.savefig('huashan_final.png', dpi=150, bbox_inches='tight')
print("\n闈欐€佸浘宸蹭繚瀛樹负 huashan_final.png")
print("浜や簰绐楀彛宸叉墦寮€锛岄紶鏍囨偓鍋滃湪宸﹀浘鏌ョ湅缁忕含搴﹀拰楂樼▼")
plt.show()
