"""
================================================================================
文件名：safe_corridor.py
用    途：生成可飞走廊并分层，为后续路网构建提供输入
================================================================================

【功能说明】
    Step 2：基于 DEM 高程矩阵，生成每个(x,y)点的可飞高度走廊
    Step 3：将走廊切分为三层（末端进近 / 区域支路 / 骨干航路）

【飞行参数】
    下限偏移 H_MIN_OFFSET = 30m
        依据：避开华山地表茂密树冠层与潜在低空索道
    上限偏移 H_MAX_OFFSET = 120m
        依据：中国民航局《无人驾驶航空器飞行管理暂行条例》
              规定微轻小无人机真高上限为 120m

【三层定义（相对地面高度）】
    Layer 1 末端进近层 Terminal Layer  ：30 ~  60m
        用途：无人机起飞/降落爬升下降阶段，接驳华山各峰顶站点
    Layer 2 区域支路层 Regional Branch ：60 ~  90m
        用途：连接各山峰与主干道的接驳转运
    Layer 3 骨干航路层 Backbone Layer  ：90 ~ 120m
        用途：高速巡航层，视野开阔，通信受遮挡概率最小

【输入文件】
    Z_crop.npy          裁剪后的高程矩阵（由 huashan_dem.py 生成）

【输出文件】
    floor.npy           飞行下限绝对高度矩阵（Z + 50m），单位：米
    ceiling.npy         飞行上限绝对高度矩阵（Z + 120m），单位：米
    layer_mid.npy       三层中心高度矩阵，shape=(3, H, W)，单位：米
                            layer_mid[0]：末端进近层中心（Z + 60m）
                            layer_mid[1]：区域支路层中心（Z + 82.5m）
                            layer_mid[2]：骨干航路层中心（Z + 107.5m）
    corridor_vis.png    可视化结果图（走廊剖面 + 三层俯视图）

【后续步骤】
    Step 4：分层拓扑路网构建（layered_graph.py）
        输入：layer_mid.npy、Z_crop.npy
================================================================================
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
font = FontProperties(family='SimHei')

# ===== 配置参数（修改这里即可）=====
CACHE_DEM   = "Z_crop.npy"
RESOLUTION  = 12.5          # 米/像元

H_MIN_OFFSET = 30           # 距地面最低安全高度（米）：避开树冠层与低空索道
H_MAX_OFFSET = 120          # 距地面最高限制（米）

# 三层边界（相对地面高度，单位：米）
LAYERS = {
    "末端进近层": {"low": 30,  "high": 60,  "color": "#2196F3"},  # 蓝  30~60m  起降阶段
    "区域支路层": {"low": 60,  "high": 90,  "color": "#4CAF50"},  # 绿  60~90m  节点接驳
    "骨干航路层": {"low": 90,  "high": 120, "color": "#FF5722"},  # 红橙 90~120m 高速巡航
}

# ===== 读取 DEM =====
assert os.path.exists(CACHE_DEM), f"找不到 {CACHE_DEM}，请先运行 huashan_dem.py"
Z = np.load(CACHE_DEM)
rows, cols = Z.shape
print(f"[读取] DEM shape={Z.shape}，高程范围: {np.nanmin(Z):.0f}~{np.nanmax(Z):.0f}m")

# ===== Step 2：生成可飞走廊 =====
floor   = Z + H_MIN_OFFSET   # 飞行下限（绝对高度）
ceiling = Z + H_MAX_OFFSET   # 飞行上限（绝对高度）

np.save("floor.npy",   floor)
np.save("ceiling.npy", ceiling)
print(f"[Step2] 飞行下限范围: {floor.min():.0f}~{floor.max():.0f}m")
print(f"[Step2] 飞行上限范围: {ceiling.min():.0f}~{ceiling.max():.0f}m")

# ===== Step 3：生成三层中心高度矩阵 =====
layer_names = list(LAYERS.keys())
layer_mid_list = []

for name, cfg in LAYERS.items():
    mid = Z + (cfg["low"] + cfg["high"]) / 2.0
    layer_mid_list.append(mid)
    print(f"[Step3] {name}（{cfg['low']}~{cfg['high']}m）"
          f" 中心高度: {mid.min():.0f}~{mid.max():.0f}m")

layer_mid = np.stack(layer_mid_list, axis=0)   # shape=(3, H, W)
np.save("layer_mid.npy", layer_mid)
print(f"[Step3] layer_mid.npy 已保存，shape={layer_mid.shape}")

# ===== 可视化 =====
fig = plt.figure(figsize=(22, 14))
fig.suptitle('华山可飞走廊与三层分层结果', fontproperties=font, fontsize=14, y=0.98)

# ---------- 图1：走廊剖面图（取中间一行的东西向剖面）----------
ax1 = fig.add_subplot(231)
mid_row = rows // 2
x_km = np.arange(cols) * RESOLUTION / 1000

ax1.fill_between(x_km, Z[mid_row], floor[mid_row],
                 color='#8B4513', alpha=0.8, label='地形（不可飞）')
ax1.fill_between(x_km, floor[mid_row], ceiling[mid_row],
                 color='#87CEEB', alpha=0.5, label='可飞走廊')

# 三层边界线
colors_line = ['#2196F3', '#4CAF50', '#FF5722']
for i, (name, cfg) in enumerate(LAYERS.items()):
    ax1.plot(x_km, Z[mid_row] + cfg["high"],
             color=colors_line[i], lw=1.5, linestyle='--',
             label=f'{name}上界（+{cfg["high"]}m）')

ax1.set_xlabel('东西方向 (km)', fontproperties=font)
ax1.set_ylabel('绝对高度 (m)', fontproperties=font)
ax1.set_title('走廊剖面图（中间行东西向）', fontproperties=font)
ax1.legend(prop=font, fontsize=7, loc='upper right')
ax1.grid(True, alpha=0.3)

# ---------- 图2：可飞走廊厚度图（ceiling - floor = 常数70m）----------
ax2 = fig.add_subplot(232)
thickness = ceiling - floor
im2 = ax2.imshow(thickness, cmap='YlOrRd',
                 extent=[0, cols*RESOLUTION/1000, 0, rows*RESOLUTION/1000],
                 origin='upper', aspect='equal')
plt.colorbar(im2, ax=ax2, label='走廊厚度 (m)', shrink=0.8)
ax2.set_xlabel('东西方向 (km)', fontproperties=font)
ax2.set_ylabel('南北方向 (km)', fontproperties=font)
ax2.set_title(f'可飞走廊厚度（固定 {H_MAX_OFFSET-H_MIN_OFFSET}m）',
              fontproperties=font)
ax2.grid(True, alpha=0.3, linestyle='--')

# ---------- 图3/4/5：三层中心高度俯视图 ----------
cmaps = ['Blues', 'Greens', 'Oranges']
positions = [234, 235, 236]

for i, (name, cfg) in enumerate(LAYERS.items()):
    ax = fig.add_subplot(positions[i])
    im = ax.imshow(layer_mid[i], cmap=cmaps[i],
                   extent=[0, cols*RESOLUTION/1000,
                           0, rows*RESOLUTION/1000],
                   origin='upper', aspect='equal')
    plt.colorbar(im, ax=ax, label='绝对高度 (m)', shrink=0.8)
    ax.set_xlabel('东西方向 (km)', fontproperties=font)
    ax.set_ylabel('南北方向 (km)', fontproperties=font)
    ax.set_title(f'Layer {i+1}：{name}\n（地面 +{cfg["low"]}~+{cfg["high"]}m）',
                 fontproperties=font, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

# ---------- 图6：3D 走廊可视化（抽样）----------
ax6 = fig.add_subplot(233, projection='3d')
step = 10
Z_s = Z[::step, ::step]
f_s = floor[::step, ::step]
c_s = ceiling[::step, ::step]
rs, cs = Z_s.shape
xg = np.arange(cs) * step * RESOLUTION / 1000
yg = np.arange(rs) * step * RESOLUTION / 1000
Xg, Yg = np.meshgrid(xg, yg)

ax6.plot_surface(Xg, Yg, Z_s,   color='#8B4513', alpha=0.7)
ax6.plot_surface(Xg, Yg, f_s,   color='#2196F3', alpha=0.2)
ax6.plot_surface(Xg, Yg, c_s,   color='#FF5722', alpha=0.15)

ax6.view_init(elev=25, azim=225)
ax6.set_xlabel('东西 (km)', fontproperties=font, labelpad=6)
ax6.set_ylabel('南北 (km)', fontproperties=font, labelpad=6)
ax6.set_zlabel('高度 (m)', fontproperties=font, labelpad=6)
ax6.set_title('3D 可飞走廊示意\n（棕：地形，蓝：下限，橙：上限）',
              fontproperties=font, fontsize=9)

plt.tight_layout()
plt.savefig('corridor_vis.png', dpi=150, bbox_inches='tight')
print("\n[完成] corridor_vis.png 已保存")
print("[完成] 输出文件：floor.npy / ceiling.npy / layer_mid.npy")
print("[下一步] 运行 layered_graph.py 进行分层路网构建")
plt.show()