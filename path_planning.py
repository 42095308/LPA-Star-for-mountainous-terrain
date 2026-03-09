"""
================================================================================
文件名：path_planning.py
用    途：基于分层拓扑路网的约束最短路规划（Dijkstra）
================================================================================

【代价函数】
    cost(e) = α·t(e) + β·E(e) + γ·R(e)
        t(e)：飞行时间代价（边长 / 巡航速度，归一化）
        E(e)：能耗代价（与距离和爬升高度相关，归一化）
        R(e)：地面风险代价（路径下方地形高程越低风险越高，归一化）

【权重】
    α = 0.3（时间）
    β = 0.2（能耗）
    γ = 0.5（风险，最高优先级）

【硬约束】
    累计风险 R_total ≤ R_MAX（超出则路径不合法）

【起点→终点】
    北部基地 → 南峰

【输入文件】
    graph_nodes.npy    节点坐标，shape=(N,4)，[x_km, y_km, z_m, layer_id]
    graph_edges.npy    边列表，shape=(M,3)，[i, j, edge_type]
    Z_crop.npy         高程矩阵

【输出文件】
    path_result.npy    最优路径节点索引列表
    path_vis.png       路径可视化图（俯视 + 3D）

【论文对应】
    Method Section 3.3：多目标代价建模与约束最短路
================================================================================
"""

import numpy as np
import os
import heapq
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
font = FontProperties(family='SimHei')

# ===== 配置参数 =====
ALPHA      = 0.3    # 时间权重
BETA       = 0.2    # 能耗权重
GAMMA      = 0.5    # 风险权重
R_MAX      = 9999   # 先放开验证连通性；论文实验时设为实际路径风险均值的2~3倍

UAV_SPEED  = 15.0   # 无人机巡航速度（m/s）
UAV_POWER  = 500.0  # 巡航功率（W），用于能耗估算

RESOLUTION = 12.5   # 米/像元

# 起点和终点名称（必须和 layered_graph.py 中的 PEAKS/DEPOTS 一致）
START_NAME = "北部基地"
GOAL_NAME  = "南峰"

# ===== 读取数据 =====
assert os.path.exists("graph_nodes.npy"), "缺少 graph_nodes.npy，请先运行 layered_graph.py"
assert os.path.exists("graph_edges.npy"), "缺少 graph_edges.npy，请先运行 layered_graph.py"
assert os.path.exists("Z_crop.npy"),      "缺少 Z_crop.npy"

nodes = np.load("graph_nodes.npy")   # shape=(N,4)
edges = np.load("graph_edges.npy")   # shape=(M,3)
Z     = np.load("Z_crop.npy")
rows, cols = Z.shape
N = len(nodes)
print(f"[读取] |V|={N}，|E|={len(edges)}")

# ===== 工具函数 =====
def km_to_rc(x_km, y_km):
    c = int(np.clip(x_km * 1000 / RESOLUTION, 0, cols - 1))
    r = int(np.clip((rows - 1) - y_km * 1000 / RESOLUTION, 0, rows - 1))
    return r, c

def get_terrain(x_km, y_km):
    r, c = km_to_rc(x_km, y_km)
    return float(Z[r, c])

SAFETY_HEIGHT     = 30
COLLISION_SAMPLES = 20

def collision_free(n1, n2, n_samples=COLLISION_SAMPLES):
    """检查两节点连线是否安全（不穿山体），用于视线平滑"""
    for t in np.linspace(0, 1, n_samples):
        x_km = n1[0] + t * (n2[0] - n1[0])
        y_km = n1[1] + t * (n2[1] - n1[1])
        z_m  = n1[2] + t * (n2[2] - n1[2])
        r, c = km_to_rc(x_km, y_km)
        r = int(np.clip(r, 0, rows - 1))
        c = int(np.clip(c, 0, cols - 1))
        if z_m - float(Z[r, c]) < SAFETY_HEIGHT:
            return False
    return True

# ===== 边代价计算 =====
def compute_edge_costs(ni, nj):
    """
    计算边 (ni, nj) 的三个归一化代价
    返回 (t_norm, E_norm, R_norm)
    """
    xi, yi, zi = nodes[ni, 0], nodes[ni, 1], nodes[ni, 2]
    xj, yj, zj = nodes[nj, 0], nodes[nj, 1], nodes[nj, 2]

    # 三维边长（米）
    d_horiz = np.linalg.norm([xj - xi, yj - yi]) * 1000   # km→m
    d_vert  = abs(zj - zi)
    d_3d    = np.sqrt(d_horiz**2 + d_vert**2)

    # --- 时间代价 t(e) ---
    t_raw = d_3d / UAV_SPEED    # 秒

    # --- 能耗代价 E(e) ---
    # 水平飞行功率 + 爬升额外功率（简化模型）
    climb_power = max(0, (zj - zi)) * 9.8 * 5.0   # 假设无人机5kg，重力做功
    E_raw = (UAV_POWER * t_raw + climb_power) / 1000   # kJ

    # --- 风险代价 R(e) ---
    # 沿边采样10个点，取路径下方地形高程的均值
    # 地形高程越低（谷地/平原）→ 人口密度可能越高 → 风险越高
    # 用 "飞行高度相对地形的余量" 作为安全指标：余量越小风险越高
    risk_samples = []
    for t in np.linspace(0, 1, 10):
        x  = xi + t * (xj - xi)
        y  = yi + t * (yj - yi)
        z  = zi + t * (zj - zi)
        terrain = get_terrain(x, y)
        margin  = z - terrain    # 距地面高度余量
        # 余量越小 → 风险越高，用反比映射
        risk_samples.append(max(0, 1.0 - margin / 200.0))
    R_raw = float(np.mean(risk_samples))

    return t_raw, E_raw, R_raw

# ===== 预计算所有边代价并归一化 =====
print("[计算] 预计算边代价...")
edge_costs_raw = []
for e in edges:
    i, j = int(e[0]), int(e[1])
    t, E, R = compute_edge_costs(i, j)
    edge_costs_raw.append([t, E, R])

edge_costs_raw = np.array(edge_costs_raw)

# 归一化到 [0, 1]
t_max = edge_costs_raw[:, 0].max() + 1e-9
E_max = edge_costs_raw[:, 1].max() + 1e-9
R_max_val = edge_costs_raw[:, 2].max() + 1e-9

edge_costs_norm = edge_costs_raw / np.array([t_max, E_max, R_max_val])
edge_weights    = (ALPHA * edge_costs_norm[:, 0] +
                   BETA  * edge_costs_norm[:, 1] +
                   GAMMA * edge_costs_norm[:, 2])

print(f"  时间代价范围: {edge_costs_raw[:,0].min():.1f}~{edge_costs_raw[:,0].max():.1f}s")
print(f"  能耗代价范围: {edge_costs_raw[:,1].min():.2f}~{edge_costs_raw[:,1].max():.2f}kJ")
print(f"  风险代价范围: {edge_costs_raw[:,2].min():.3f}~{edge_costs_raw[:,2].max():.3f}")

# ===== 构建邻接表 =====
adj = [[] for _ in range(N)]
for k, e in enumerate(edges):
    i, j = int(e[0]), int(e[1])
    w = float(edge_weights[k])
    r = float(edge_costs_norm[k, 2])   # 风险分量（用于硬约束）
    adj[i].append((j, w, r))
    adj[j].append((i, w, r))           # 无向图

# ===== 找起点和终点节点索引 =====
# 末端锚点层（layer_id=0）的节点对应 PEAKS/DEPOTS
# 按照 layered_graph.py 中的顺序：先PEAKS（5个），再DEPOTS（2个）
PEAKS_ORDER  = ["南峰", "东峰", "西峰", "北峰", "中峰"]
DEPOTS_ORDER = ["北部基地", "西部基地"]
ALL_TERMINALS = PEAKS_ORDER + DEPOTS_ORDER   # 共7个，每个3层

def find_terminal_node(name, layer_id=0):
    """找到指定名称在指定层的节点索引"""
    idx_in_list = ALL_TERMINALS.index(name)
    node_idx    = idx_in_list * 3 + layer_id   # 每个锚点占3个节点
    return node_idx

# 起点/终点策略：
# 先尝试用支路层（layer_id=1），支路层直接连入主网
# 若支路层邻居数不足，自动降级到末端层（layer_id=0）
def find_best_terminal(name):
    """找到指定名称连通性最好的层的节点索引"""
    for lid in [1, 2, 0]:   # 优先支路层，其次骨干层，最后末端层
        idx = find_terminal_node(name, layer_id=lid)
        neighbors = [v for v, w, r in adj[idx]]
        if len(neighbors) >= 2:
            return idx, lid
    return find_terminal_node(name, layer_id=0), 0

start_idx, start_lid = find_best_terminal(START_NAME)
goal_idx,  goal_lid  = find_best_terminal(GOAL_NAME)

print(f"\n[规划] 起点: {START_NAME} → 节点#{start_idx}(layer={start_lid}) "
      f"({nodes[start_idx,0]:.2f}km, {nodes[start_idx,1]:.2f}km, {nodes[start_idx,2]:.0f}m) "
      f"邻居数={len([v for v,w,r in adj[start_idx]])}")
print(f"[规划] 终点: {GOAL_NAME}  → 节点#{goal_idx}(layer={goal_lid}) "
      f"({nodes[goal_idx,0]:.2f}km, {nodes[goal_idx,1]:.2f}km, {nodes[goal_idx,2]:.0f}m) "
      f"邻居数={len([v for v,w,r in adj[goal_idx]])}")

# ===== Dijkstra 约束最短路 =====
print("\n[Dijkstra] 开始搜索...")

dist     = np.full(N, np.inf)
risk_acc = np.full(N, np.inf)   # 累计风险
prev     = np.full(N, -1, dtype=int)
dist[start_idx]     = 0.0
risk_acc[start_idx] = 0.0

# 优先队列：(cost, risk_cumulative, node_idx)
pq = [(0.0, 0.0, start_idx)]
visited = set()

while pq:
    cost, risk, u = heapq.heappop(pq)
    if u in visited:
        continue
    visited.add(u)
    if u == goal_idx:
        break
    for v, w, r in adj[u]:
        if v in visited:
            continue
        new_cost = cost + w
        new_risk = risk + r
        # 硬约束：累计风险不超过上限
        if new_risk > R_MAX:
            continue
        if new_cost < dist[v]:
            dist[v]     = new_cost
            risk_acc[v] = new_risk
            prev[v]     = u
            heapq.heappush(pq, (new_cost, new_risk, v))

# ===== 回溯路径 =====
if dist[goal_idx] == np.inf:
    print("[警告] 未找到可行路径！请检查图的连通性或放宽 R_MAX 约束")
    path = []
else:
    path = []
    cur  = goal_idx
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    print(f"[结果] 找到路径！节点数: {len(path)}")
    print(f"  总代价:     {dist[goal_idx]:.4f}")
    print(f"  累计风险:   {risk_acc[goal_idx]:.4f}（上限 {R_MAX}）")

    # 计算实际物理指标
    total_dist = 0.0
    total_time = 0.0
    total_energy = 0.0
    for k in range(len(path) - 1):
        ni, nj = path[k], path[k+1]
        # 找对应边
        d3d = np.sqrt(
            ((nodes[nj,0]-nodes[ni,0])*1000)**2 +
            ((nodes[nj,1]-nodes[ni,1])*1000)**2 +
            (nodes[nj,2]-nodes[ni,2])**2
        )
        total_dist   += d3d
        total_time   += d3d / UAV_SPEED
        total_energy += UAV_POWER * (d3d / UAV_SPEED) / 1000

    print(f"  路径总长度: {total_dist/1000:.2f} km")
    print(f"  飞行时间:   {total_time/60:.1f} 分钟")
    print(f"  估算能耗:   {total_energy:.1f} kJ")

    np.save("path_result.npy", np.array(path))
    print("\n[保存] path_result.npy 已保存")

# ===== 视线平滑（Line-of-Sight Smoothing）=====
def los_smooth(raw_path):
    """
    视线剪枝：若路径点 P_i 和 P_j 之间无障碍（碰撞检测通过），
    则直接连 P_i→P_j，删除中间冗余折点。
    论文表述：消除因图搜索引入的直角折线，输出符合飞行运动学的平滑轨迹。
    """
    if len(raw_path) <= 2:
        return raw_path
    smoothed = [raw_path[0]]
    i = 0
    while i < len(raw_path) - 1:
        # 尽量跳到最远的可视点
        j = len(raw_path) - 1
        while j > i + 1:
            ni, nj = raw_path[i], raw_path[j]
            if collision_free(nodes[ni], nodes[nj]):
                break
            j -= 1
        smoothed.append(raw_path[j])
        i = j
    return smoothed

if path:
    path_smooth = los_smooth(path)
    print(f"\n[平滑] 视线剪枝：{len(path)} 个节点 → {len(path_smooth)} 个节点")

    # 重新计算平滑路径物理指标
    total_dist_s = total_time_s = total_energy_s = 0.0
    for k in range(len(path_smooth) - 1):
        ni, nj = path_smooth[k], path_smooth[k+1]
        d3d = np.sqrt(
            ((nodes[nj,0]-nodes[ni,0])*1000)**2 +
            ((nodes[nj,1]-nodes[ni,1])*1000)**2 +
            (nodes[nj,2]-nodes[ni,2])**2
        )
        total_dist_s   += d3d
        total_time_s   += d3d / UAV_SPEED
        total_energy_s += UAV_POWER * (d3d / UAV_SPEED) / 1000

    print(f"  平滑后路径长度: {total_dist_s/1000:.2f} km（原 {total_dist/1000:.2f} km）")
    print(f"  平滑后飞行时间: {total_time_s/60:.1f} min（原 {total_time/60:.1f} min）")
    np.save("path_smooth.npy", np.array(path_smooth))
else:
    path_smooth = []

# ===== 可视化 =====
print("\n[可视化] 生成路径图...")
fig = plt.figure(figsize=(20, 9))

LAYER_COLORS  = ["#2196F3", "#4CAF50", "#FF5722"]
LAYER_MARKERS = ["o", "s", "^"]
LAYER_SIZES   = [60, 35, 25]

# ---------- 左图：俯视路径图 ----------
ax1 = fig.add_subplot(121)
ax1.imshow(Z, cmap='terrain', alpha=0.45,
           extent=[0, cols*RESOLUTION/1000, 0, rows*RESOLUTION/1000],
           origin='upper', aspect='equal')

# 画所有边（灰色淡显）
for e in edges:
    i, j = int(e[0]), int(e[1])
    ax1.plot([nodes[i,0], nodes[j,0]],
             [nodes[i,1], nodes[j,1]],
             color='gray', lw=0.4, alpha=0.3, zorder=1)

# 画所有节点
for lid, (color, marker, size) in enumerate(
        zip(LAYER_COLORS, LAYER_MARKERS, LAYER_SIZES)):
    mask  = nodes[:,3] == lid
    label = ["末端进近层","区域支路层","骨干航路层"][lid]
    ax1.scatter(nodes[mask,0], nodes[mask,1],
                c=color, marker=marker, s=size, label=label,
                zorder=2, alpha=0.5, edgecolors='white', linewidths=0.3)

# 画最优路径（橙色虚线，原始折线）
if path:
    px = [nodes[n,0] for n in path]
    py = [nodes[n,1] for n in path]
    ax1.plot(px, py, color='orange', lw=1.5, zorder=4,
             linestyle='--', alpha=0.7, label='原始路径（折线）')

# 画平滑路径（红色实线）
if path_smooth:
    spx = [nodes[n,0] for n in path_smooth]
    spy = [nodes[n,1] for n in path_smooth]
    ax1.plot(spx, spy, color='red', lw=2.5, zorder=5, label='平滑路径（视线剪枝）')
    ax1.scatter(spx, spy, c='red', s=50, zorder=6)

# 标注起终点
ax1.scatter(nodes[start_idx,0], nodes[start_idx,1],
            c='lime', s=150, marker='*', zorder=7, label=f'起点（{START_NAME}）')
ax1.scatter(nodes[goal_idx,0],  nodes[goal_idx,1],
            c='yellow', s=150, marker='*', zorder=7, label=f'终点（{GOAL_NAME}）')

ax1.set_xlabel('东西方向 (km)', fontproperties=font)
ax1.set_ylabel('南北方向 (km)', fontproperties=font)
title_str = (f'{START_NAME} → {GOAL_NAME}  约束最短路 + 视线平滑\n'
             f'α={ALPHA} β={BETA} γ={GAMMA}  '
             f'平滑后: {total_dist_s/1000:.2f}km  {total_time_s/60:.1f}min  '
             f'节点数: {len(path)}→{len(path_smooth)}'
             if path else f'{START_NAME} → {GOAL_NAME}（未找到路径）')
ax1.set_title(title_str, fontproperties=font, fontsize=10)
ax1.legend(prop=font, loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3, linestyle='--')

# ---------- 右图：3D 路径图 ----------
ax2 = fig.add_subplot(122, projection='3d')
step = 8
Z_s  = Z[::step, ::step]
rs_, cs_ = Z_s.shape
Xg, Yg = np.meshgrid(np.arange(cs_)*step*RESOLUTION/1000,
                     np.arange(rs_)*step*RESOLUTION/1000)
ax2.plot_surface(Xg, Yg, Z_s, cmap='terrain', alpha=0.35, linewidth=0)

# 所有边淡显
for e in edges:
    i, j = int(e[0]), int(e[1])
    ax2.plot([nodes[i,0], nodes[j,0]],
             [nodes[i,1], nodes[j,1]],
             [nodes[i,2], nodes[j,2]],
             color='gray', lw=0.3, alpha=0.2)

# 原始路径（橙色虚线）
if path:
    px = [nodes[n,0] for n in path]
    py = [nodes[n,1] for n in path]
    pz = [nodes[n,2] for n in path]
    ax2.plot(px, py, pz, color='orange', lw=1.5,
             linestyle='--', alpha=0.6)

# 平滑路径（红色）
if path_smooth:
    spx = [nodes[n,0] for n in path_smooth]
    spy = [nodes[n,1] for n in path_smooth]
    spz = [nodes[n,2] for n in path_smooth]
    ax2.plot(spx, spy, spz, color='red', lw=3.0, zorder=5)
    ax2.scatter(spx, spy, spz, c='red', s=50, zorder=6)

    # --- 地面投影（Ground Shadow）---
    n_seg = max(len(path_smooth)-1, 1)
    n_per_seg = max(6, 60 // n_seg)
    interp_x, interp_y, interp_z_path, interp_z_ground = [], [], [], []
    for k in range(len(path_smooth) - 1):
        ni, nj = path_smooth[k], path_smooth[k+1]
        for t in np.linspace(0, 1, n_per_seg, endpoint=(k == len(path_smooth)-2)):
            x = nodes[ni,0] + t*(nodes[nj,0]-nodes[ni,0])
            y = nodes[ni,1] + t*(nodes[nj,1]-nodes[ni,1])
            z = nodes[ni,2] + t*(nodes[nj,2]-nodes[ni,2])
            r_g, c_g = km_to_rc(x, y)
            r_g = int(np.clip(r_g, 0, rows-1))
            c_g = int(np.clip(c_g, 0, cols-1))
            interp_x.append(x);  interp_y.append(y)
            interp_z_path.append(z)
            interp_z_ground.append(float(Z[r_g, c_g]))

    # 灰色虚线：地面投影
    ax2.plot(interp_x, interp_y, interp_z_ground,
             color='gray', lw=1.5, linestyle='--', alpha=0.7, zorder=3,
             label='地面投影')

    # 垂直连线：每隔一段连接路径与地面投影
    n_drops = 8
    drop_idx = np.linspace(0, len(interp_x)-1, n_drops, dtype=int)
    for di in drop_idx:
        ax2.plot([interp_x[di], interp_x[di]],
                 [interp_y[di], interp_y[di]],
                 [interp_z_ground[di], interp_z_path[di]],
                 color='gray', lw=0.8, linestyle=':', alpha=0.6)

# 起终点
ax2.scatter([nodes[start_idx,0]], [nodes[start_idx,1]], [nodes[start_idx,2]],
            c='lime', s=200, marker='*', zorder=7)
ax2.scatter([nodes[goal_idx,0]],  [nodes[goal_idx,1]],  [nodes[goal_idx,2]],
            c='yellow', s=200, marker='*', zorder=7)

ax2.view_init(elev=28, azim=225)
ax2.set_xlabel('东西 (km)', fontproperties=font, labelpad=6)
ax2.set_ylabel('南北 (km)', fontproperties=font, labelpad=6)
ax2.set_zlabel('高度 (m)',  fontproperties=font, labelpad=6)
ax2.set_title('约束最短路 3D视图', fontproperties=font, fontsize=11)

plt.tight_layout()
plt.savefig('path_vis.png', dpi=150, bbox_inches='tight')
print("[完成] path_vis.png 已保存")
print("[下一步] 运行 lpa_star.py 进行动态增量重规划")
plt.show()