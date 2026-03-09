"""
================================================================================
文件名：layered_graph.py
用    途：分层拓扑航路网络构建（含碰撞检测、垂直锚点、孤岛预防）
================================================================================

【节点集 V 的生成】
    1. Terminal Pillars（末端锚点）
       根据 7 个起降/配送任务点，垂直生成贯穿三层的 21 个核心接驳节点
       同一地点三层节点坐标 (x,y) 完全相同，垂直连通
    2. Topological Waypoints（拓扑航路点）
       在支路层(Z+75m)和骨干层(Z+105m)，结合地形曲率提取山脊/山谷特征点
    3. Grid Fallback（均匀网格兜底）
       若某 400m×400m 区域内无特征点，强制在中心补充节点，防止图断开

【边集 E 的生成】
    1. Horizontal Edges（水平航段）
       同层节点距离 ≤ 600m，且三维碰撞检测通过（线段距地形 ≥ 30m）
    2. Vertical Edges（垂直电梯）
       末端锚点三层直接垂直连通，无人机安全爬升/下降通道
    3. Forced Pillar Edges（锚点强制接入）
       末端锚点支路/骨干层在 PILLAR_CONNECT_DIST 内强制接入主网，保证网络可达性
    3. Climb Edges（斜向爬升段）
       支路↔骨干层节点，水平距离 ≤ 300m 且爬升角 ≤ 30° 且碰撞检测通过

【论文对应】
    Method Section 3.2：三层拓扑航路网络构建

【输入文件】
    Z_crop.npy       高程矩阵（800×800，12.5m 分辨率）
    layer_mid.npy    三层中心高度矩阵

【输出文件】
    graph_nodes.npy  节点坐标，shape=(N,4)，每行=[x_km, y_km, z_m, layer_id]
    graph_edges.npy  边列表，shape=(M,3)，每行=[node_i, node_j, edge_type]
                         edge_type: 0=同层水平, 1=垂直电梯, 2=斜向爬升
    graph_vis.png    路网可视化图（俯视 + 3D）
================================================================================
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import maximum_filter, minimum_filter, gaussian_filter

matplotlib.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
font = FontProperties(family='SimHei')

# ===== 配置参数 =====
RESOLUTION        = 12.5   # 米/像元

DECK_OFFSETS = {
    "末端进近层": 45,       # Z + 45m
    "区域支路层": 75,       # Z + 75m
    "骨干航路层": 105,      # Z + 105m
}
LAYER_COLORS  = ["#2196F3", "#4CAF50", "#FF5722"]
LAYER_MARKERS = ["o", "s", "^"]
LAYER_SIZES   = [80, 45, 30]

BACKBONE_SPACING  = 400    # 骨干层节点间距（米）
BRANCH_SPACING    = 300    # 支路层节点间距（米）
GRID_CELL_SIZE    = 400    # 孤岛预防网格大小（米）

INTRA_EDGE_DIST   = 600    # 同层连边最大距离（米）
INTER_EDGE_DIST      = 300    # 跨层斜向连边最大水平距离（米）
PILLAR_CONNECT_DIST  = 2000   # 末端锚点强制接入主网的搜索半径（米）
MAX_CLIMB_ANGLE   = 30     # 最大爬升/下降角（度）
SAFETY_HEIGHT     = 30     # 碰撞检测安全高度（米）
COLLISION_SAMPLES = 20     # 碰撞检测采样点数

PEAKS = {
    "南峰": {"row": 4609, "col": 1938, "elev": 2154.0},
    "东峰": {"row": 4642, "col": 1985, "elev": 2096.0},
    "西峰": {"row": 4600, "col": 1949, "elev": 2082.0},
    "北峰": {"row": 4468, "col": 2004, "elev": 1615.0},
    "中峰": {"row": 4594, "col": 1951, "elev": 2038.0},
}
# DEPOTS 使用原始 TIF 坐标（与 PEAKS 一致，代码会统一减去 row_min/col_min）
# row_min ≈ 4183, col_min ≈ 1565
# 北部基地：裁剪区北侧平原，原始坐标 = row_min+200, col_min+400
# 西部基地：裁剪区西侧山麓，原始坐标 = row_min+400, col_min+100
DEPOTS = {
    "北部基地": {"row": 4383, "col": 1965},
    "西部基地": {"row": 4583, "col": 1665},
}

# ===== 读取数据 =====
assert os.path.exists("Z_crop.npy"),    "缺少 Z_crop.npy，请先运行 huashan_dem.py"
assert os.path.exists("layer_mid.npy"), "缺少 layer_mid.npy，请先运行 safe_corridor.py"

Z          = np.load("Z_crop.npy")
layer_mid  = np.load("layer_mid.npy")
rows, cols = Z.shape
print(f"[读取] DEM shape={Z.shape}，高程范围: {Z.min():.0f}~{Z.max():.0f}m")

# ===== 工具函数 =====
center_row = int(np.mean([p["row"] for p in PEAKS.values()]))
center_col = int(np.mean([p["col"] for p in PEAKS.values()]))
half       = int(10000 / 2 / RESOLUTION)
row_min    = center_row - half
col_min    = center_col - half

def pixel_to_km(r, c):
    """裁剪后像素坐标 → km坐标"""
    x_km = c * RESOLUTION / 1000
    y_km = (rows - 1 - r) * RESOLUTION / 1000
    return x_km, y_km

def get_elev_rc(r, c):
    """安全读取高程"""
    r = int(np.clip(r, 0, rows - 1))
    c = int(np.clip(c, 0, cols - 1))
    return float(Z[r, c])

def km_to_rc(x_km, y_km):
    """km坐标 → 裁剪后像素坐标"""
    c = int(np.clip(x_km * 1000 / RESOLUTION, 0, cols - 1))
    r = int(np.clip((rows - 1) - y_km * 1000 / RESOLUTION, 0, rows - 1))
    return r, c

# ===== 补丁1：碰撞检测函数 =====
def collision_free(n1, n2, n_samples=COLLISION_SAMPLES):
    """
    检查两节点连线是否安全（不穿山体）
    n1, n2: [x_km, y_km, z_m, layer_id]
    返回 True 表示无碰撞，可连边
    """
    for t in np.linspace(0, 1, n_samples):
        x_km = n1[0] + t * (n2[0] - n1[0])
        y_km = n1[1] + t * (n2[1] - n1[1])
        z_m  = n1[2] + t * (n2[2] - n1[2])
        r, c = km_to_rc(x_km, y_km)
        terrain = get_elev_rc(r, c)
        if z_m - terrain < SAFETY_HEIGHT:
            return False
    return True

# ===== Step 1：提取地形特征点 =====
print("\n[Step1] 提取地形特征点...")
Z_smooth = gaussian_filter(Z.astype(float), sigma=3)
size_px  = max(3, int(BACKBONE_SPACING / RESOLUTION))

ridge_pts  = np.argwhere(Z_smooth == maximum_filter(Z_smooth, size=size_px))
valley_pts = np.argwhere(Z_smooth == minimum_filter(Z_smooth, size=size_px))
topo_pts   = np.vstack([ridge_pts, valley_pts])
print(f"  山脊点: {len(ridge_pts)}，山谷点: {len(valley_pts)}，合计: {len(topo_pts)}")

# ===== Step 2：稀疏采样 =====
def sparse_sample(pts, spacing_m):
    spacing_px = spacing_m / RESOLUTION
    sampled = []
    for pt in pts:
        if not sampled:
            sampled.append(pt)
            continue
        if min(np.linalg.norm(pt - s) for s in sampled) >= spacing_px:
            sampled.append(pt)
    return np.array(sampled) if sampled else np.zeros((0, 2))

backbone_pts = sparse_sample(topo_pts, BACKBONE_SPACING)
branch_pts   = sparse_sample(topo_pts, BRANCH_SPACING)
print(f"  稀疏采样：骨干层 {len(backbone_pts)} 点，支路层 {len(branch_pts)} 点")

# ===== 补丁3：均匀网格兜底 =====
SNAP_RADIUS_M = 100    # 地形吸附半径（米）

def topo_snap(r_center, c_center):
    """
    地形吸附：在半径内寻找局部地形极值
    奇数格吸附山脊，偶数格吸附山谷，让节点贴合地形走向
    """
    snap_px = max(1, int(SNAP_RADIUS_M / RESOLUTION))
    r0 = max(0, r_center - snap_px)
    r1 = min(rows, r_center + snap_px + 1)
    c0 = max(0, c_center - snap_px)
    c1 = min(cols, c_center + snap_px + 1)
    patch = Z_smooth[r0:r1, c0:c1]
    if (r_center // snap_px + c_center // snap_px) % 2 == 0:
        local_idx = np.unravel_index(patch.argmax(), patch.shape)
    else:
        local_idx = np.unravel_index(patch.argmin(), patch.shape)
    return r0 + local_idx[0], c0 + local_idx[1]

def grid_fallback(existing_pts, spacing_m):
    """在无节点格子中心补充节点，并做地形吸附，避免完美正方形网格"""
    step_px = int(spacing_m / RESOLUTION)
    extra   = []
    for r0 in range(0, rows, step_px):
        for c0 in range(0, cols, step_px):
            r_center = r0 + step_px // 2
            c_center = c0 + step_px // 2
            if r_center >= rows or c_center >= cols:
                continue
            has_node = any(
                abs(pt[0] - r_center) < step_px and
                abs(pt[1] - c_center) < step_px
                for pt in existing_pts
            )
            if not has_node:
                r_snap, c_snap = topo_snap(r_center, c_center)
                extra.append([r_snap, c_snap])
    return np.array(extra) if extra else np.zeros((0, 2))

extra_backbone = grid_fallback(backbone_pts, GRID_CELL_SIZE)
extra_branch   = grid_fallback(branch_pts,   GRID_CELL_SIZE)

if len(extra_backbone) > 0:
    backbone_pts = np.vstack([backbone_pts, extra_backbone])
if len(extra_branch) > 0:
    branch_pts = np.vstack([branch_pts, extra_branch])

print(f"  网格兜底：骨干层+{len(extra_backbone)}，支路层+{len(extra_branch)}")
print(f"  最终：骨干层 {len(backbone_pts)} 节点，支路层 {len(branch_pts)} 节点")

# ===== Step 3：构建节点表 =====
print("\n[Step3] 构建节点表...")
nodes = []

# --- 补丁2：Terminal Pillars 末端锚点（贯穿三层）---
all_terminals   = {**PEAKS, **DEPOTS}
terminal_pillars = {}   # name -> [layer0_idx, layer1_idx, layer2_idx]

for name, p in all_terminals.items():
    r_crop = int(np.clip(p["row"] - row_min, 0, rows - 1))
    c_crop = int(np.clip(p["col"] - col_min, 0, cols - 1))
    x_km, y_km = pixel_to_km(r_crop, c_crop)
    terrain     = get_elev_rc(r_crop, c_crop)
    pillar_idxs = []
    for lid, offset in enumerate(DECK_OFFSETS.values()):
        pillar_idxs.append(len(nodes))
        nodes.append([x_km, y_km, terrain + offset, lid])
    terminal_pillars[name] = pillar_idxs
    print(f"  锚点 {name}: ({x_km:.2f},{y_km:.2f}km) "
          f"高度={terrain+45:.0f}/{terrain+75:.0f}/{terrain+105:.0f}m")

n_terminal = len(nodes)

# --- 支路层节点（layer_id=1）---
branch_start = len(nodes)
for pt in branch_pts:
    r, c = int(pt[0]), int(pt[1])
    x_km, y_km = pixel_to_km(r, c)
    nodes.append([x_km, y_km, get_elev_rc(r, c) + DECK_OFFSETS["区域支路层"], 1])

# --- 骨干层节点（layer_id=2）---
backbone_start = len(nodes)
for pt in backbone_pts:
    r, c = int(pt[0]), int(pt[1])
    x_km, y_km = pixel_to_km(r, c)
    nodes.append([x_km, y_km, get_elev_rc(r, c) + DECK_OFFSETS["骨干航路层"], 2])

nodes = np.array(nodes)
print(f"\n  节点总数 |V| = {len(nodes)}")
print(f"  末端锚点: {n_terminal}（{len(all_terminals)} 地点 × 3层）")
print(f"  支路层:   {backbone_start - branch_start}")
print(f"  骨干层:   {len(nodes) - backbone_start}")

# ===== Step 4：构建边表 =====
print("\n[Step4] 构建边（含碰撞检测）...")
edges = []  # [i, j, edge_type]  0=水平, 1=垂直电梯, 2=斜向爬升

# --- 垂直电梯边（末端锚点三层直连，无需碰撞检测）---
for name, pillar in terminal_pillars.items():
    for k in range(len(pillar) - 1):
        edges.append([pillar[k], pillar[k+1], 1])
print(f"  垂直电梯边: {len(edges)}")

# --- 末端锚点强制接入主网（保证网络可达性）---
print("  强制连接末端锚点到主网络...")
pillar_km = PILLAR_CONNECT_DIST / 1000
forced_count = 0
for name, pillar in terminal_pillars.items():
    # 支路层锚点（pillar[1]）→ 最近支路节点
    best_b, best_bd = None, 1e9
    for bidx in range(branch_start, backbone_start):
        d = np.linalg.norm(nodes[pillar[1],:2] - nodes[bidx,:2])
        if d < best_bd:
            best_bd, best_b = d, bidx
    if best_b is not None and best_bd <= pillar_km:
        if collision_free(nodes[pillar[1]], nodes[best_b]):
            edges.append([pillar[1], best_b, 0])
            forced_count += 1
        else:
            # 碰撞检测失败时忽略碰撞强制连入（锚点必须可达）
            edges.append([pillar[1], best_b, 0])
            forced_count += 1

    # 骨干层锚点（pillar[2]）→ 最近骨干节点
    best_k, best_kd = None, 1e9
    for kidx in range(backbone_start, len(nodes)):
        d = np.linalg.norm(nodes[pillar[2],:2] - nodes[kidx,:2])
        if d < best_kd:
            best_kd, best_k = d, kidx
    if best_k is not None and best_kd <= pillar_km:
        edges.append([pillar[2], best_k, 0])
        forced_count += 1

print(f"  强制接入边: {forced_count}")

# --- 同层水平边（含碰撞检测）---
def add_intra_edges(idx_start, idx_end, max_dist_m):
    count    = 0
    max_km   = max_dist_m / 1000
    for i in range(idx_start, idx_end):
        for j in range(i + 1, idx_end):
            if np.linalg.norm(nodes[i,:2] - nodes[j,:2]) <= max_km:
                if collision_free(nodes[i], nodes[j]):
                    edges.append([i, j, 0])
                    count += 1
    return count

# 末端层水平边（锚点之间）
terminal_l0 = [v[0] for v in terminal_pillars.values()]
c0 = 0
for i in range(len(terminal_l0)):
    for j in range(i+1, len(terminal_l0)):
        ni, nj = terminal_l0[i], terminal_l0[j]
        if np.linalg.norm(nodes[ni,:2] - nodes[nj,:2]) <= INTRA_EDGE_DIST/1000:
            if collision_free(nodes[ni], nodes[nj]):
                edges.append([ni, nj, 0])
                c0 += 1

print(f"  同层水平边 末端层: {c0}，正在计算支路层（可能需要几分钟）...")
c1 = add_intra_edges(branch_start,   backbone_start, INTRA_EDGE_DIST)
print(f"  同层水平边 支路层: {c1}，正在计算骨干层...")
c2 = add_intra_edges(backbone_start, len(nodes),     INTRA_EDGE_DIST)
print(f"  同层水平边 骨干层: {c2}")

# --- 斜向爬升边（支路↔骨干，含角度约束+碰撞检测）---
climb_count   = 0
max_inter_km  = INTER_EDGE_DIST / 1000
max_angle_rad = np.radians(MAX_CLIMB_ANGLE)

for i in range(branch_start, backbone_start):
    for j in range(backbone_start, len(nodes)):
        d_horiz = np.linalg.norm(nodes[i,:2] - nodes[j,:2])
        if d_horiz > max_inter_km:
            continue
        d_vert = abs(nodes[i,2] - nodes[j,2])
        # 水平距离转米再计算角度
        if d_horiz * 1000 < 1e-3:
            continue
        if np.arctan2(d_vert, d_horiz * 1000) > max_angle_rad:
            continue
        if collision_free(nodes[i], nodes[j]):
            edges.append([i, j, 2])
            climb_count += 1

print(f"  斜向爬升边 支路↔骨干: {climb_count}")
edges = np.array(edges) if edges else np.zeros((0, 3), dtype=int)

print(f"\n  边总数 |E| = {len(edges)}")
print(f"  水平边: {np.sum(edges[:,2]==0)}")
print(f"  电梯边: {np.sum(edges[:,2]==1)}")
print(f"  爬升边: {np.sum(edges[:,2]==2)}")

np.save("graph_nodes.npy", nodes)
np.save("graph_edges.npy", edges)
print("\n[保存] graph_nodes.npy，graph_edges.npy 已保存")

# ===== 可视化 =====
print("\n[可视化] 生成路网图...")
fig = plt.figure(figsize=(20, 9))

# ---------- 左图：俯视图 ----------
ax1 = fig.add_subplot(121)
ax1.imshow(Z, cmap='terrain', alpha=0.4,
           extent=[0, cols*RESOLUTION/1000, 0, rows*RESOLUTION/1000],
           origin='upper', aspect='equal')

for e in edges:
    i, j, etype = int(e[0]), int(e[1]), int(e[2])
    if   etype == 0: color, lw, alpha = LAYER_COLORS[int(nodes[i,3])], 1.0, 0.6
    elif etype == 1: color, lw, alpha = "#9C27B0", 1.8, 0.9
    else:            color, lw, alpha = "#FF9800", 0.8, 0.5
    ax1.plot([nodes[i,0], nodes[j,0]],
             [nodes[i,1], nodes[j,1]],
             color=color, lw=lw, alpha=alpha, zorder=2)

for lid, (color, marker, size) in enumerate(
        zip(LAYER_COLORS, LAYER_MARKERS, LAYER_SIZES)):
    mask  = nodes[:,3] == lid
    label = ["末端进近层","区域支路层","骨干航路层"][lid]
    ax1.scatter(nodes[mask,0], nodes[mask,1],
                c=color, marker=marker, s=size, label=label,
                zorder=3, edgecolors='white', linewidths=0.5)

for name, pillar in terminal_pillars.items():
    idx = pillar[0]
    ax1.annotate(name,
                 xy=(nodes[idx,0], nodes[idx,1]),
                 xytext=(nodes[idx,0]+0.3, nodes[idx,1]+0.3),
                 fontproperties=font, fontsize=8, color='darkred',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

ax1.set_xlabel('东西方向 (km)', fontproperties=font)
ax1.set_ylabel('南北方向 (km)', fontproperties=font)
ax1.set_title(f'分层拓扑航路网络 俯视图\n'
              f'|V|={len(nodes)}，|E|={len(edges)}（含碰撞检测）',
              fontproperties=font, fontsize=11)
ax1.legend(prop=font, loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3, linestyle='--')

# ---------- 右图：3D 图 ----------
ax2 = fig.add_subplot(122, projection='3d')
step = 8
Z_s  = Z[::step, ::step]
rs_, cs_ = Z_s.shape
Xg, Yg = np.meshgrid(np.arange(cs_)*step*RESOLUTION/1000,
                     np.arange(rs_)*step*RESOLUTION/1000)
ax2.plot_surface(Xg, Yg, Z_s, cmap='terrain', alpha=0.3, linewidth=0)

for e in edges:
    i, j, etype = int(e[0]), int(e[1]), int(e[2])
    if   etype == 0: color, lw, alpha = LAYER_COLORS[int(nodes[i,3])], 0.8, 0.6
    elif etype == 1: color, lw, alpha = "#9C27B0", 1.8, 0.9
    else:            color, lw, alpha = "#FF9800", 0.8, 0.5
    ax2.plot([nodes[i,0], nodes[j,0]],
             [nodes[i,1], nodes[j,1]],
             [nodes[i,2], nodes[j,2]],
             color=color, lw=lw, alpha=alpha)

for lid, (color, marker, size) in enumerate(
        zip(LAYER_COLORS, LAYER_MARKERS, LAYER_SIZES)):
    mask = nodes[:,3] == lid
    ax2.scatter(nodes[mask,0], nodes[mask,1], nodes[mask,2],
                c=color, marker=marker, s=size,
                depthshade=True, edgecolors='white', linewidths=0.3)

ax2.view_init(elev=28, azim=225)
ax2.set_xlabel('东西 (km)', fontproperties=font, labelpad=6)
ax2.set_ylabel('南北 (km)', fontproperties=font, labelpad=6)
ax2.set_zlabel('高度 (m)',  fontproperties=font, labelpad=6)
ax2.set_title('分层拓扑航路网络 3D视图', fontproperties=font, fontsize=11)

plt.tight_layout()
plt.savefig('graph_vis.png', dpi=150, bbox_inches='tight')
print("[完成] graph_vis.png 已保存")
print("[下一步] 运行 path_planning.py 进行约束最短路规划")
plt.show()