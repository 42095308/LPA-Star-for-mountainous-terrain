"""
================================================================================
文件名：lpa_star.py
用    途：基于 LPA* 的动态增量路径规划
================================================================================

【算法说明】
    LPA*（Lifelong Planning A*）by Koenig & Likhachev, 2004
    核心思想：维护每个节点的 g 值和 rhs 值
        g(s)   : 当前已知的从起点到 s 的最短路径代价（类比 Dijkstra 的 dist）
        rhs(s) : 一步超前估计值，rhs(s) = min_{s'∈pred(s)} [g(s') + c(s',s)]
    局部一致性：g(s) == rhs(s) → 节点一致（consistent）
                g(s) != rhs(s) → 节点不一致（inconsistent），需要更新

【三阶段流程】
    阶段1 初始规划：compute_shortest_path()，等价于 A*，同时建立 g/rhs 状态表
    阶段2 触发事件：update_edge_cost(u, v, new_cost)，模拟阵风/禁飞区封锁
    阶段3 增量重规划：再次调用 compute_shortest_path()，只遍历受影响的局部节点

【实验设计】
    对比指标（用于论文 Table）：
        - 初始规划遍历节点数
        - 增量重规划遍历节点数
        - 重规划耗时（ms）
        - 路径质量变化

【输入文件】
    graph_nodes.npy    节点坐标，shape=(N,4)，[x_km, y_km, z_m, layer_id]
    graph_edges.npy    边列表，shape=(M,3)，[i, j, edge_type]
    Z_crop.npy         高程矩阵

【输出文件】
    lpa_result.png     三阶段对比可视化图
================================================================================
"""

import numpy as np
import heapq
import time
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull, QhullError

matplotlib.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
font = FontProperties(family='SimHei')

# ===== 配置参数 =====
ALPHA      = 0.3
BETA       = 0.2
GAMMA      = 0.5
UAV_SPEED  = 15.0
UAV_POWER  = 500.0
RESOLUTION = 12.5
SAFETY_HEIGHT     = 30
COLLISION_SAMPLES = 20

# Risk fusion weights.
RISK_W_TERRAIN = 0.50
RISK_W_TRAIL   = 0.30
RISK_W_HOTSPOT = 0.20
RISK_W_HUMAN_COMBINED = 0.50

START_NAME = "北部基地"
GOAL_NAME  = "南峰"

# 触发事件：随机封锁路径上的 N_BLOCK 条边
N_BLOCK    = 2

# ===== 读取数据 =====
assert os.path.exists("graph_nodes.npy"), "缺少 graph_nodes.npy"
assert os.path.exists("graph_edges.npy"), "缺少 graph_edges.npy"
assert os.path.exists("Z_crop.npy"),      "缺少 Z_crop.npy"

nodes = np.load("graph_nodes.npy")
edges = np.load("graph_edges.npy")
Z     = np.load("Z_crop.npy")
rows, cols = Z.shape
N = len(nodes)
print(f"[读取] |V|={N}，|E|={len(edges)}")

# ===== Optional OSM human risk rasters =====
risk_trail = np.zeros((rows, cols), dtype=float)
risk_hotspot = np.zeros((rows, cols), dtype=float)
risk_human = np.zeros((rows, cols), dtype=float)
risk_mode = "terrain_only"

if os.path.exists("risk_trail.npy"):
    arr = np.load("risk_trail.npy").astype(float)
    if arr.shape == (rows, cols):
        risk_trail = np.clip(arr, 0.0, 1.0)
if os.path.exists("risk_hotspot.npy"):
    arr = np.load("risk_hotspot.npy").astype(float)
    if arr.shape == (rows, cols):
        risk_hotspot = np.clip(arr, 0.0, 1.0)
if os.path.exists("risk_human.npy"):
    arr = np.load("risk_human.npy").astype(float)
    if arr.shape == (rows, cols):
        risk_human = np.clip(arr, 0.0, 1.0)

has_split = (np.max(risk_trail) > 0.0) or (np.max(risk_hotspot) > 0.0)
has_combined = np.max(risk_human) > 0.0
if has_split:
    risk_mode = "terrain_trail_hotspot"
elif has_combined:
    risk_mode = "terrain_human_combined"

print(f"[风险场] mode={risk_mode}")

# ===== 工具函数 =====
def km_to_rc(x_km, y_km):
    c = int(np.clip(x_km * 1000 / RESOLUTION, 0, cols - 1))
    r = int(np.clip((rows - 1) - y_km * 1000 / RESOLUTION, 0, rows - 1))
    return r, c

def collision_free(n1, n2):
    for t in np.linspace(0, 1, COLLISION_SAMPLES):
        x = n1[0] + t*(n2[0]-n1[0])
        y = n1[1] + t*(n2[1]-n1[1])
        z = n1[2] + t*(n2[2]-n1[2])
        r, c = km_to_rc(x, y)
        if z - float(Z[r, c]) < SAFETY_HEIGHT:
            return False
    return True

# ===== 预计算边代价 =====
def compute_raw_edge_costs():
    raw = []
    for e in edges:
        i, j = int(e[0]), int(e[1])
        xi, yi, zi = nodes[i, 0], nodes[i, 1], nodes[i, 2]
        xj, yj, zj = nodes[j, 0], nodes[j, 1], nodes[j, 2]
        d_h = np.linalg.norm([xj-xi, yj-yi]) * 1000
        d_v = abs(zj - zi)
        d3d = np.sqrt(d_h**2 + d_v**2)
        t_raw = d3d / UAV_SPEED
        E_raw = (UAV_POWER * t_raw + max(0, zj-zi)*9.8*5.0) / 1000
        risk_samples = []
        for t in np.linspace(0, 1, 10):
            x = xi + t*(xj-xi); y = yi + t*(yj-yi); z = zi + t*(zj-zi)
            r, c = km_to_rc(x, y)
            terrain = float(Z[r, c])
            r_terrain = max(0.0, 1.0 - (z - terrain) / 200.0)

            if risk_mode == "terrain_trail_hotspot":
                r_total = (
                    RISK_W_TERRAIN * r_terrain
                    + RISK_W_TRAIL * float(risk_trail[r, c])
                    + RISK_W_HOTSPOT * float(risk_hotspot[r, c])
                )
            elif risk_mode == "terrain_human_combined":
                r_total = (
                    (1.0 - RISK_W_HUMAN_COMBINED) * r_terrain
                    + RISK_W_HUMAN_COMBINED * float(risk_human[r, c])
                )
            else:
                r_total = r_terrain

            risk_samples.append(float(np.clip(r_total, 0.0, 1.0)))
        R_raw = float(np.mean(risk_samples))
        raw.append([t_raw, E_raw, R_raw])
    raw = np.array(raw)
    t_max = raw[:,0].max() + 1e-9
    E_max = raw[:,1].max() + 1e-9
    R_max = raw[:,2].max() + 1e-9
    norm  = raw / np.array([t_max, E_max, R_max])
    weights = ALPHA*norm[:,0] + BETA*norm[:,1] + GAMMA*norm[:,2]
    return weights, t_max, E_max

print("[预计算] 边代价...")
edge_weights_base, EDGE_T_MAX, EDGE_E_MAX = compute_raw_edge_costs()
print("[heuristic] key1=min(g,rhs)+h; h=LB(time)+LB(energy), risk LB=0")

# 构建邻接表：adj[u] = [(v, edge_key), ...]
# edge_cost 存在字典里方便动态修改
adj       = [[] for _ in range(N)]
edge_cost = {}   # (min(u,v), max(u,v)) -> cost

for k, e in enumerate(edges):
    i, j = int(e[0]), int(e[1])
    key = (min(i,j), max(i,j))
    edge_cost[key] = float(edge_weights_base[k])
    adj[i].append(j)
    adj[j].append(i)

def get_cost(u, v):
    return edge_cost.get((min(u,v), max(u,v)), np.inf)

# ===== 找起终点 =====
PEAKS_ORDER  = ["南峰","东峰","西峰","北峰","中峰"]
DEPOTS_ORDER = ["北部基地","西部基地"]
ALL_TERMINALS = PEAKS_ORDER + DEPOTS_ORDER

def find_best_terminal(name):
    # 临时构建邻接查询
    tmp_adj = [[] for _ in range(N)]
    for e in edges:
        i, j = int(e[0]), int(e[1])
        tmp_adj[i].append(j); tmp_adj[j].append(i)
    idx_base = ALL_TERMINALS.index(name) * 3
    for lid in [1, 2, 0]:
        idx = idx_base + lid
        if len(tmp_adj[idx]) >= 2:
            return idx
    return idx_base

start_node = find_best_terminal(START_NAME)
goal_node  = find_best_terminal(GOAL_NAME)
print(f"[规划] 起点: {START_NAME} → 节点#{start_node} "
      f"({nodes[start_node,0]:.2f}km, {nodes[start_node,1]:.2f}km)")
print(f"[规划] 终点: {GOAL_NAME}  → 节点#{goal_node} "
      f"({nodes[goal_node,0]:.2f}km, {nodes[goal_node,1]:.2f}km)")

# ===== LPA* 类 =====
class LPAStar:
    """
    LPA*（Lifelong Planning A*）增量路径规划器

    核心数据结构：
        g[s]   : 当前从起点到 s 的已知最短路径代价
        rhs[s] : 一步超前值，rhs[s] = min_{pred} [g(pred) + c(pred, s)]
        U      : 优先队列，存放不一致节点
    """

    def __init__(self, start, goal):
        self.start  = start
        self.goal   = goal
        self.INF    = float('inf')

        # g 和 rhs 初始化为 ∞
        self.g   = np.full(N, self.INF)
        self.rhs = np.full(N, self.INF)

        # 起点 rhs = 0（出发代价为0）
        self.rhs[start] = 0.0

        # 优先队列：(key, node)，key = (k1, k2)
        # 用 counter 处理相同 key 的节点排序
        self._counter = 0
        self._heap    = []
        self._in_heap = {}   # node -> key（用于判断是否在队列中）

        self._push(start, self._calc_key(start))

        # 统计
        self.nodes_expanded = 0
        self.expanded_nodes_order = []
        self.expanded_nodes_unique = []

    def _calc_key(self, s):
        """
        LPA* priority key:
            k1 = min(g(s), rhs(s)) + h(s)
            k2 = min(g(s), rhs(s))
        """
        base = min(self.g[s], self.rhs[s])
        return (base + self._heuristic(s), base)

    def _heuristic(self, s):
        """
        Admissible heuristic for mountain UAV routing:
        lower-bound flight time + lower-bound energy.
        Risk lower bound is set to 0 to avoid overestimation.
        """
        dx = (nodes[self.goal, 0] - nodes[s, 0]) * 1000.0
        dy = (nodes[self.goal, 1] - nodes[s, 1]) * 1000.0
        dz = nodes[self.goal, 2] - nodes[s, 2]
        d3d = np.sqrt(dx * dx + dy * dy + dz * dz)

        t_lb = d3d / UAV_SPEED
        climb_lb = max(0.0, dz) * 9.8 * 5.0
        E_lb = (UAV_POWER * t_lb + climb_lb) / 1000.0

        h_time = ALPHA * (t_lb / EDGE_T_MAX)
        h_energy = BETA * (E_lb / EDGE_E_MAX)
        return h_time + h_energy

    def _push(self, node, key):
        self._counter += 1
        entry = (key[0], key[1], self._counter, node)
        heapq.heappush(self._heap, entry)
        self._in_heap[node] = key

    def _pop(self):
        while self._heap:
            k1, k2, _, node = heapq.heappop(self._heap)
            cur_key = self._in_heap.get(node)
            if cur_key is not None and cur_key == (k1, k2):
                del self._in_heap[node]
                return node, (k1, k2)
        return None, None

    def _top_key(self):
        while self._heap:
            k1, k2, _, node = self._heap[0]
            cur_key = self._in_heap.get(node)
            if cur_key is not None and cur_key == (k1, k2):
                return (k1, k2)
            heapq.heappop(self._heap)
        return (self.INF, self.INF)

    def update_vertex(self, u):
        """
        重新计算 u 的 rhs 值（向前看一步的最优代价）
        若 u 不一致，加入优先队列；若 u 一致，从队列移除
        """
        if u != self.start:
            # rhs(u) = min over predecessors p: g(p) + c(p, u)
            best = self.INF
            for p in adj[u]:
                c = get_cost(p, u)
                if self.g[p] + c < best:
                    best = self.g[p] + c
            self.rhs[u] = best

        # 维护优先队列中的一致性
        if u in self._in_heap:
            del self._in_heap[u]

        if self.g[u] != self.rhs[u]:
            self._push(u, self._calc_key(u))

    def compute_shortest_path(self):
        """
        核心搜索循环：
        持续处理优先队列中的不一致节点，直到终点一致且队列为空
        """
        self.nodes_expanded = 0
        self.expanded_nodes_order = []
        self.expanded_nodes_unique = []

        while True:
            top_key = self._top_key()
            goal_key = self._calc_key(self.goal)

            # 终止条件：终点一致 且 队列中没有更优的节点
            if (top_key >= goal_key and
                    self.g[self.goal] == self.rhs[self.goal]):
                break
            if top_key[0] == self.INF:
                break

            u, k_old = self._pop()
            if u is None:
                break

            self.nodes_expanded += 1
            self.expanded_nodes_order.append(u)
            k_new = self._calc_key(u)

            if k_old < k_new:
                # key 过期，重新入队
                self._push(u, k_new)

            elif self.g[u] > self.rhs[u]:
                # 过一致（overconsistent）：g 值可以降低
                self.g[u] = self.rhs[u]
                for s in adj[u]:
                    self.update_vertex(s)

            else:
                # 欠一致（underconsistent）：g 值需要提升
                self.g[u] = self.INF
                self.update_vertex(u)
                for s in adj[u]:
                    self.update_vertex(s)

        # 保序去重：便于可视化“更新波纹面”的空间覆盖
        self.expanded_nodes_unique = list(dict.fromkeys(self.expanded_nodes_order))
        return self.g[self.goal] < self.INF

    def update_edge_cost(self, u, v, new_cost):
        """
        动态修改边代价（阵风/禁飞区触发时调用）
        只更新受影响节点的 rhs 值，不重新初始化
        """
        key = (min(u, v), max(u, v))
        edge_cost[key] = new_cost
        # 只有这两个节点的 rhs 可能受影响
        self.update_vertex(u)
        self.update_vertex(v)

    def extract_path(self):
        """
        从 goal 向 start 回溯路径（沿最小 g 值的前驱节点）
        """
        if self.g[self.goal] == self.INF:
            return []
        path = [self.goal]
        cur  = self.goal
        seen = set()
        while cur != self.start:
            seen.add(cur)
            best_pred, best_cost = None, self.INF
            for p in adj[cur]:
                c = get_cost(p, cur)
                total = self.g[p] + c
                if total < best_cost:
                    best_cost = total
                    best_pred = p
            if best_pred is None or best_pred in seen:
                break
            path.append(best_pred)
            cur = best_pred
        path.reverse()
        return path if path[0] == self.start else []

    def path_length_km(self, path):
        """计算路径实际飞行距离（km）"""
        total = 0.0
        for k in range(len(path) - 1):
            ni, nj = path[k], path[k+1]
            d = np.sqrt(
                ((nodes[nj,0]-nodes[ni,0])*1000)**2 +
                ((nodes[nj,1]-nodes[ni,1])*1000)**2 +
                (nodes[nj,2]-nodes[ni,2])**2
            )
            total += d
        return total / 1000

# ===== 视线平滑 =====
def los_smooth(raw_path):
    """
    第一步：视线剪枝
    若 P_i 和 P_j 之间无碰撞，直接连 P_i→P_j，删除中间折点
    """
    if len(raw_path) <= 2:
        return raw_path
    smoothed = [raw_path[0]]
    i = 0
    while i < len(raw_path) - 1:
        j = len(raw_path) - 1
        while j > i + 1:
            if collision_free(nodes[raw_path[i]], nodes[raw_path[j]]):
                break
            j -= 1
        smoothed.append(raw_path[j])
        i = j
    return smoothed


def bspline_smooth(node_path, n_points=120, smooth_factor=0.0):
    """
    第二步：B 样条曲线平滑
    输入：视线剪枝后的节点索引列表
    输出：密集插值后的 3D 坐标数组 shape=(n_points, 3)，单位 [km, km, m]
    论文表述：采用三次 B 样条插值对离散航路点进行平滑，
             消除因图搜索引入的折角，输出符合飞行运动学的连续曲线。
    """
    if len(node_path) < 3:
        # 节点太少，直接线性插值
        coords = np.array([[nodes[n,0], nodes[n,1], nodes[n,2]]
                           for n in node_path])
        t = np.linspace(0, 1, n_points)
        result = np.zeros((n_points, 3))
        for dim in range(3):
            result[:, dim] = np.interp(t, np.linspace(0,1,len(coords)), coords[:,dim])
        return result

    coords = np.array([[nodes[n,0], nodes[n,1], nodes[n,2]]
                       for n in node_path])
    x, y, z = coords[:,0], coords[:,1], coords[:,2]

    # 参数化：弦长参数化（比均匀参数化更贴合实际距离）
    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + (np.diff(z)/1000)**2)
    u = np.concatenate([[0], np.cumsum(dists)])
    u /= u[-1]

    # 拟合三次 B 样条（k=3）
    k = min(3, len(node_path) - 1)
    tck, _ = splprep([x, y, z], u=u, k=k, s=smooth_factor)

    # 密集插值
    u_fine = np.linspace(0, 1, n_points)
    x_s, y_s, z_s = splev(u_fine, tck)

    return np.column_stack([x_s, y_s, z_s])


def smooth_path(raw_path):
    """完整两步平滑：LOS 剪枝 + B 样条"""
    step1 = los_smooth(raw_path)
    curve = bspline_smooth(step1)
    return step1, curve   # 返回剪枝后节点列表 + 样条曲线坐标

# ===================================================================
# ======================== 三阶段实验流程 ===========================
# ===================================================================

planner = LPAStar(start_node, goal_node)

# ------------------------------------------------------------------
# 阶段1：初始规划
# ------------------------------------------------------------------
print("\n" + "="*55)
print("阶段1：LPA* 初始规划")
print("="*55)

t0 = time.perf_counter()
found = planner.compute_shortest_path()
t1 = time.perf_counter()

phase1_time_ms    = (t1 - t0) * 1000
phase1_expanded   = planner.nodes_expanded
path1_raw         = planner.extract_path()
path1, curve1     = smooth_path(path1_raw)   # path1=剪枝节点, curve1=B样条坐标
path1_len         = planner.path_length_km(path1)

print(f"  找到路径: {'OK' if found else 'FAIL'}")
print(f"  遍历节点数:   {phase1_expanded}")
print(f"  规划耗时:     {phase1_time_ms:.2f} ms")
print(f"  路径节点数:   {len(path1_raw)} → LOS剪枝 {len(path1)} → B样条 {len(curve1)} 点")
print(f"  路径长度:     {path1_len:.2f} km")
print(f"  路径代价:     {planner.g[goal_node]:.4f}")

# ------------------------------------------------------------------
# 阶段2：触发突发事件——封锁路径上的边
# ------------------------------------------------------------------
print("\n" + "="*55)
print("阶段2：触发突发事件（封锁路径上的边）")
print("="*55)

blocked_edges = []
if len(path1_raw) >= 3:
    # 只在路径前半段选封锁边（避免遮挡终点区域）
    n = len(path1_raw)
    lo = max(1,   n // 3)
    hi = max(lo+1, 2*n // 3)
    interior = list(range(lo, hi))
    if not interior:
        interior = [max(1, n//2)]
    np.random.seed(42)
    block_indices = np.random.choice(
        interior,
        size=min(N_BLOCK, len(interior)),
        replace=False
    )
    for bi in block_indices:
        u = path1_raw[bi]
        v = path1_raw[bi + 1]
        blocked_edges.append((u, v))
        print(f"  封锁边: ({u}, {v})  "
              f"节点({nodes[u,0]:.2f},{nodes[u,1]:.2f}km) → "
              f"({nodes[v,0]:.2f},{nodes[v,1]:.2f}km)")
        planner.update_edge_cost(u, v, np.inf)   # 代价设为无穷大
else:
    print("  路径节点数不足，随机封锁图中一条边")
    np.random.seed(42)
    k = np.random.randint(len(edges))
    u, v = int(edges[k,0]), int(edges[k,1])
    blocked_edges.append((u, v))
    planner.update_edge_cost(u, v, np.inf)
    print(f"  封锁边: ({u}, {v})")

# ------------------------------------------------------------------
# 阶段3：增量重规划
# ------------------------------------------------------------------
print("\n" + "="*55)
print("阶段3：LPA* 增量重规划（不重新初始化）")
print("="*55)

t2 = time.perf_counter()
found2 = planner.compute_shortest_path()
t3 = time.perf_counter()

phase3_time_ms  = (t3 - t2) * 1000
phase3_expanded = planner.nodes_expanded
phase3_expanded_nodes_order = planner.expanded_nodes_order.copy()
path3_raw       = planner.extract_path()
path3, curve3   = smooth_path(path3_raw)
path3_len       = planner.path_length_km(path3)

print(f"  找到路径: {'OK' if found2 else 'FAIL'}")
print(f"  遍历节点数:   {phase3_expanded}  （初始规划: {phase1_expanded}）")
print(f"  重规划耗时:   {phase3_time_ms:.2f} ms  （初始规划: {phase1_time_ms:.2f} ms）")
print(f"  路径节点数:   {len(path3_raw)} → 平滑后 {len(path3)}")
print(f"  路径长度:     {path3_len:.2f} km  （初始: {path1_len:.2f} km）")
print(f"  路径代价:     {planner.g[goal_node]:.4f}")

# 论文 Table 汇总
print("\n" + "="*55)
print("【论文 Table 数据汇总】")
print("="*55)
print(f"  {'指标':<20} {'初始规划':>12} {'增量重规划':>12} {'加速比':>10}")
print(f"  {'-'*56}")
ratio_nodes = phase1_expanded / max(phase3_expanded, 1)
ratio_time  = phase1_time_ms  / max(phase3_time_ms,  0.01)
print(f"  {'遍历节点数':<19} {phase1_expanded:>12} {phase3_expanded:>12} {ratio_nodes:>9.1f}×")
print(f"  {'规划耗时 (ms)':<18} {phase1_time_ms:>12.2f} {phase3_time_ms:>12.2f} {ratio_time:>9.1f}×")
print(f"  {'路径长度 (km)':<18} {path1_len:>12.2f} {path3_len:>12.2f} {'—':>10}")

# ===================================================================
# ============ 阶段1 详细路径图（俯视 + 3D地面投影）=================
# ===================================================================
print("\n[可视化] 生成阶段1路径详细图（path_vis.png）...")

fig_pv, (axA, axB) = plt.subplots(1, 2, figsize=(20, 8),
                                   gridspec_kw={'width_ratios': [1, 1.1]})
fig_pv.suptitle(f'{START_NAME} → {GOAL_NAME}  初始路径（LPA* 阶段1）\n'
                f'α={ALPHA} β={BETA} γ={GAMMA}  '
                f'路径长={path1_len:.2f}km  时间={(path1_len*1000/UAV_SPEED/60):.1f}min  '
                f'节点数: {len(path1_raw)}→{len(path1)}',
                fontproperties=font, fontsize=11)

# --- 左：俯视图 ---
axA.imshow(Z, cmap='terrain', alpha=0.45,
           extent=[0, cols*RESOLUTION/1000, 0, rows*RESOLUTION/1000],
           origin='upper', aspect='equal')
for e in edges:
    i, j = int(e[0]), int(e[1])
    axA.plot([nodes[i,0],nodes[j,0]], [nodes[i,1],nodes[j,1]],
             color='gray', lw=0.3, alpha=0.15, zorder=1)
for lid, (color, marker) in enumerate(zip(["#2196F3","#4CAF50","#FF5722"], ["o","s","^"])):
    mask = nodes[:,3] == lid
    label = ["末端进近层","区域支路层","骨干航路层"][lid]
    axA.scatter(nodes[mask,0], nodes[mask,1], c=color, marker=marker,
                s=18, alpha=0.35, zorder=2, edgecolors='none', label=label)
if path1:
    px_raw = [nodes[n,0] for n in path1_raw]
    py_raw = [nodes[n,1] for n in path1_raw]
    # 原始折线（亮青色点线）
    axA.plot(px_raw, py_raw, color='#00E5FF', lw=2.1, zorder=1.8,
             linestyle=':', alpha=0.9, label='raw polyline')
    # B 样条平滑曲线（红色虚线）
    axA.plot(curve1[:,0], curve1[:,1], color='red', lw=2.0, zorder=5,
             linestyle='--', dashes=(6,3), label='B样条平滑路径')
    # 剪枝后的锚点（小圆点）
    px = [nodes[n,0] for n in path1]
    py = [nodes[n,1] for n in path1]
    axA.scatter(px[1:-1], py[1:-1], c='red', s=25, zorder=6, alpha=0.7)
    # 方向箭头沿样条曲线放置
    n_arr = min(4, len(curve1)-1)
    for k in np.linspace(0, len(curve1)-2, n_arr, dtype=int):
        axA.annotate('', xy=(curve1[k+1,0], curve1[k+1,1]),
                     xytext=(curve1[k,0], curve1[k,1]),
                     arrowprops=dict(arrowstyle='->', color='darkred',
                                     lw=0.8, mutation_scale=7, alpha=0.55), zorder=7)
axA.scatter(nodes[start_node,0], nodes[start_node,1],
            c='lime', s=90, marker='o', zorder=9, edgecolors='darkgreen', linewidths=1.5)
axA.scatter(nodes[goal_node,0], nodes[goal_node,1],
            c='yellow', s=90, marker='o', zorder=9, edgecolors='goldenrod', linewidths=1.5)
axA.annotate('S', xy=(nodes[start_node,0], nodes[start_node,1]),
             fontsize=8, color='darkgreen', fontweight='bold',
             xytext=(4,4), textcoords='offset points', zorder=10)
axA.annotate('G', xy=(nodes[goal_node,0], nodes[goal_node,1]),
             fontsize=8, color='goldenrod', fontweight='bold',
             xytext=(4,4), textcoords='offset points', zorder=10)
axA.set_xlabel('东西方向 (km)', fontproperties=font)
axA.set_ylabel('南北方向 (km)', fontproperties=font)
axA.set_title('俯视航路图', fontproperties=font, fontsize=10)
axA.legend(prop=font, loc='upper right', fontsize=7)
axA.grid(True, alpha=0.3, linestyle='--')

# --- 右：3D地面投影图 ---
axB = fig_pv.add_subplot(122, projection='3d')
step3d = 8
# 与 nodes 的 (x_km, y_km) 坐标系严格对齐：y 轴采用 (rows-1-r)*res
r_idx = np.arange(0, rows, step3d, dtype=int)
c_idx = np.arange(0, cols, step3d, dtype=int)
Z_s = Z[np.ix_(r_idx, c_idx)]
x_vals = c_idx * RESOLUTION / 1000
y_vals = (rows - 1 - r_idx) * RESOLUTION / 1000
Xg, Yg = np.meshgrid(x_vals, y_vals)
axB.plot_surface(Xg, Yg, Z_s, cmap='terrain', alpha=0.35, linewidth=0)
for e in edges:
    i, j = int(e[0]), int(e[1])
    axB.plot([nodes[i,0],nodes[j,0]], [nodes[i,1],nodes[j,1]],
             [nodes[i,2],nodes[j,2]], color='gray', lw=0.3, alpha=0.15)
if path1:
    # B 样条曲线（3D）
    axB.plot(curve1[:,0], curve1[:,1], curve1[:,2],
             color='red', lw=2.5, zorder=5, linestyle='--', dashes=(6,3))
    spx = [nodes[n,0] for n in path1]
    spy = [nodes[n,1] for n in path1]
    spz = [nodes[n,2] for n in path1]
    axB.scatter(spx, spy, spz, c='red', s=30, zorder=6, alpha=0.7)
    # 地面投影（沿 B 样条曲线逐点采样地形高程）
    ix = curve1[:,0]; iy = curve1[:,1]; iz_path = curve1[:,2]
    iz_gnd = []
    for k in range(len(curve1)):
        rg, cg = km_to_rc(ix[k], iy[k])
        rg = int(np.clip(rg, 0, rows-1)); cg = int(np.clip(cg, 0, cols-1))
        iz_gnd.append(float(Z[rg, cg]))
    iz_gnd = np.array(iz_gnd)
    # 投影线z坐标严格取地形高程Z[r,c]，避免“悬浮”观感
    axB.plot(ix, iy, iz_gnd, color='gray', lw=1.6,
             linestyle='--', alpha=0.7)
    drop_idx = np.linspace(0, len(ix)-1, 8, dtype=int)
    for di in drop_idx:
        axB.plot([ix[di],ix[di]], [iy[di],iy[di]],
                 [iz_gnd[di], iz_path[di]],
                 color='gray', lw=0.8, linestyle=':', alpha=0.55)
axB.scatter([nodes[start_node,0]], [nodes[start_node,1]], [nodes[start_node,2]],
            c='lime', s=150, marker='o', zorder=7)
axB.scatter([nodes[goal_node,0]],  [nodes[goal_node,1]],  [nodes[goal_node,2]],
            c='yellow', s=150, marker='o', zorder=7)
axB.view_init(elev=28, azim=225)
axB.set_xlabel('东西 (km)', fontproperties=font, labelpad=6)
axB.set_ylabel('南北 (km)', fontproperties=font, labelpad=6)
axB.set_zlabel('高度 (m)',  fontproperties=font, labelpad=6)
axB.set_title('3D航路图（含地面投影）', fontproperties=font, fontsize=10)

fig_pv.tight_layout()
fig_pv.savefig('path_vis.png', dpi=150, bbox_inches='tight')
print("[完成] path_vis.png 已保存")
plt.close(fig_pv)

# ===================================================================
# ======================== 可视化 ===================================
# ===================================================================
print("\n[可视化] 生成三阶段对比图...")
LAYER_COLORS  = ["#2196F3", "#4CAF50", "#FF5722"]
LAYER_MARKERS = ["o", "s", "^"]

fig, axes = plt.subplots(1, 3, figsize=(22, 8))
fig.suptitle(f'LPA* 动态增量重规划  {START_NAME} → {GOAL_NAME}',
             fontproperties=font, fontsize=13)

def draw_base(ax, title):
    ax.imshow(Z, cmap='terrain', alpha=0.45,
              extent=[0, cols*RESOLUTION/1000, 0, rows*RESOLUTION/1000],
              origin='upper', aspect='equal')
    for e in edges:
        i, j = int(e[0]), int(e[1])
        ax.plot([nodes[i,0],nodes[j,0]], [nodes[i,1],nodes[j,1]],
                color='gray', lw=0.3, alpha=0.15, zorder=1)
    for lid, (color, marker) in enumerate(zip(LAYER_COLORS, LAYER_MARKERS)):
        mask = nodes[:,3] == lid
        ax.scatter(nodes[mask,0], nodes[mask,1],
                   c=color, marker=marker, s=18, alpha=0.35,
                   zorder=2, edgecolors='none')
    # 起终点：小圆圈+文字标注，不遮挡路径
    ax.scatter(nodes[start_node,0], nodes[start_node,1],
               c='lime', s=80, marker='o', zorder=9,
               edgecolors='darkgreen', linewidths=1.5)
    ax.scatter(nodes[goal_node,0], nodes[goal_node,1],
               c='yellow', s=80, marker='o', zorder=9,
               edgecolors='goldenrod', linewidths=1.5)
    ax.annotate('S', xy=(nodes[start_node,0], nodes[start_node,1]),
                fontsize=7, color='darkgreen', fontweight='bold',
                xytext=(4, 4), textcoords='offset points', zorder=10)
    ax.annotate('G', xy=(nodes[goal_node,0], nodes[goal_node,1]),
                fontsize=7, color='goldenrod', fontweight='bold',
                xytext=(4, 4), textcoords='offset points', zorder=10)
    ax.set_title(title, fontproperties=font, fontsize=9.5, pad=6)
    ax.set_xlabel('东西 (km)', fontproperties=font, fontsize=8)
    ax.set_ylabel('南北 (km)', fontproperties=font, fontsize=8)
    ax.grid(True, alpha=0.2, linestyle='--')

def draw_blocked_topology(ax, blocked_edges, color='#E53935'):
    """强调离散图中真正被封锁/修改的底层拓扑边。"""
    for u, v in blocked_edges:
        mx = (nodes[u,0] + nodes[v,0]) / 2
        my = (nodes[u,1] + nodes[v,1]) / 2
        ax.plot([nodes[u,0], nodes[v,0]], [nodes[u,1], nodes[v,1]],
                color=color, lw=3.2, zorder=8, alpha=0.92,
                solid_capstyle='round', label='_nolegend_')
        ax.scatter([mx], [my], c=color, s=120, marker='X',
                   zorder=9, edgecolors='white', linewidths=0.6,
                   alpha=0.95, label='_nolegend_')

def draw_raw_path_with_blocked_segments(ax, raw_path, blocked_edges,
                                        path_alpha=0.85, blocked_alpha=0.95):
    """
    画出原始离散拓扑路径，并把受阻边在该折线上高亮，消除“平滑轨迹与断边错位”疑问。
    """
    if len(raw_path) < 2:
        return

    px_raw = [nodes[n,0] for n in raw_path]
    py_raw = [nodes[n,1] for n in raw_path]
    ax.plot(px_raw, py_raw, color='#B0BEC5', lw=1.8, zorder=6.0,
            alpha=path_alpha, linestyle='-',
            label='原路径(离散)')

    blocked_set = {(min(u, v), max(u, v)) for u, v in blocked_edges}
    first_hit = True
    for i in range(len(raw_path) - 1):
        u, v = int(raw_path[i]), int(raw_path[i + 1])
        key = (min(u, v), max(u, v))
        if key not in blocked_set:
            continue
        label = '原路径受阻段' if first_hit else None
        ax.plot([nodes[u,0], nodes[v,0]], [nodes[u,1], nodes[v,1]],
                color='#D50000', lw=4.2, zorder=8.6, alpha=blocked_alpha,
                solid_capstyle='round', label=label)
        first_hit = False

def draw_update_ripple(ax, expanded_order, cmap='magma'):
    """
    将LPA*增量搜索展开节点渲染成“更新波纹面”：
    凸包区域 + 外层半透明大圆 + 内层按展开顺序着色、带描边的实点。
    """
    if len(expanded_order) == 0:
        return
    idx = np.asarray(expanded_order, dtype=int)
    x = nodes[idx, 0]
    y = nodes[idx, 1]
    order = np.arange(len(idx))

    # 凸包：直接给出局部更新影响区边界
    uniq_idx = np.asarray(list(dict.fromkeys(idx.tolist())), dtype=int)
    if len(uniq_idx) >= 3:
        pts = np.column_stack([nodes[uniq_idx, 0], nodes[uniq_idx, 1]])
        try:
            hull = ConvexHull(pts)
            poly = pts[hull.vertices]
            patch = Polygon(poly, closed=True, fill=True,
                            facecolor='#D9B6FF', edgecolor='#6A1B9A',
                            linewidth=1.4, alpha=0.18, zorder=4.05,
                            label='LPA Local Update Area')
            ax.add_patch(patch)
        except QhullError:
            pass

    # 波纹底面（halo）
    ax.scatter(x, y, c=order, cmap=cmap, s=165, alpha=0.14,
               zorder=4.2, edgecolors='none')
    # 核心节点（顺序着色 + 描边增强对比度）
    ax.scatter(x, y, c=order, cmap=cmap, s=30, alpha=0.93,
               zorder=5.8, edgecolors='black', linewidths=0.45,
               label='LPA*更新节点')

    # 起/止展开节点标记，帮助审稿人读取传播方向
    s0, s1 = int(idx[0]), int(idx[-1])
    ax.scatter([nodes[s0,0]], [nodes[s0,1]], c='#00BCD4', s=46, marker='o',
               zorder=6.4, edgecolors='black', linewidths=0.4, alpha=0.95,
               label='_nolegend_')
    ax.scatter([nodes[s1,0]], [nodes[s1,1]], c='#FF9800', s=56, marker='*',
               zorder=6.5, edgecolors='black', linewidths=0.4, alpha=0.95,
               label='_nolegend_')

def apply_compact_legend(ax, ordered_labels, loc='upper left'):
    """压缩图例项数量，适配IEEE单栏缩放显示。"""
    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for h, l in zip(handles, labels):
        if (not l) or l.startswith('_'):
            continue
        if l not in by_label:
            by_label[l] = h

    picked_labels = [l for l in ordered_labels if l in by_label]
    if not picked_labels:
        return
    picked_handles = [by_label[l] for l in picked_labels]
    ax.legend(picked_handles, picked_labels,
              prop=font, fontsize=6.2, loc=loc,
              frameon=True, framealpha=0.82,
              borderpad=0.25, labelspacing=0.22, handlelength=1.7)

# --- 子图1：初始路径 ---
ax1 = axes[0]
draw_base(ax1, f'阶段1：初始规划\n遍历 {phase1_expanded} 节点  {phase1_time_ms:.1f} ms  {path1_len:.2f} km')
if path1:
    px = [nodes[n,0] for n in path1]
    py = [nodes[n,1] for n in path1]
    ax1.plot(curve1[:,0], curve1[:,1], color='red', lw=2.0, zorder=5,
             linestyle='--', dashes=(6,3), label='初始路径（B样条）')
    ax1.scatter(px[1:-1], py[1:-1], c='red', s=20, zorder=6, alpha=0.6)
    # 方向箭头沿 B 样条曲线放置
    n_arr1 = min(4, len(curve1)-1)
    for k in np.linspace(0, len(curve1)-2, n_arr1, dtype=int):
        ax1.annotate('', xy=(curve1[k+1,0], curve1[k+1,1]),
                     xytext=(curve1[k,0], curve1[k,1]),
                     arrowprops=dict(arrowstyle='->', color='darkred',
                                     lw=0.8, mutation_scale=7,
                                     alpha=0.5), zorder=7)
ax1.legend(prop=font, fontsize=7, loc='lower right')

# --- 子图2：触发事件 ---
ax2 = axes[1]
draw_base(ax2, (f'阶段2：事件触发（离散拓扑层）与局部影响范围\n'
                f'封锁 {len(blocked_edges)} 条底层拓扑边  |  '
                f'后续重规划展开 {phase3_expanded} 节点'))
if path1:
    draw_raw_path_with_blocked_segments(ax2, path1_raw, blocked_edges,
                                        path_alpha=0.90, blocked_alpha=0.95)
    ax2.plot(curve1[:,0], curve1[:,1], color='red', lw=1.7, zorder=4.0, alpha=0.40,
             linestyle='--', label='原路径(平滑)')
draw_update_ripple(ax2, phase3_expanded_nodes_order)
draw_blocked_topology(ax2, blocked_edges)
apply_compact_legend(
    ax2,
    ['LPA Local Update Area', 'LPA*更新节点', '原路径受阻段', '原路径(离散)', '原路径(平滑)'],
    loc='upper left'
)

# --- 子图3：重规划路径 ---
ax3 = axes[2]
draw_base(ax3, (f'阶段3：增量重规划\n' 
                f'遍历 {phase3_expanded} 节点  {phase3_time_ms:.1f} ms  {path3_len:.2f} km\n'
                f'加速比  节点 {ratio_nodes:.1f}×    时间 {ratio_time:.1f}×'))
if path1:
    draw_raw_path_with_blocked_segments(ax3, path1_raw, blocked_edges,
                                        path_alpha=0.45, blocked_alpha=0.70)
    ax3.plot(curve1[:,0], curve1[:,1], color='red', lw=1.5, zorder=3.3, alpha=0.28,
             linestyle='--', dashes=(5,3), label='原路径(平滑受阻)')
draw_update_ripple(ax3, phase3_expanded_nodes_order)
draw_blocked_topology(ax3, blocked_edges)
if path3:
    spx = [nodes[n,0] for n in path3]
    spy = [nodes[n,1] for n in path3]
    ax3.plot(curve3[:,0], curve3[:,1], color='royalblue', lw=2.0, zorder=5,
             linestyle='--', dashes=(6,3), label='重规划路径（B样条）')
    ax3.scatter(spx[1:-1], spy[1:-1], c='royalblue', s=20, zorder=6, alpha=0.6)
    n_arr3 = min(4, len(curve3)-1)
    for k in np.linspace(0, len(curve3)-2, n_arr3, dtype=int):
        ax3.annotate('', xy=(curve3[k+1,0], curve3[k+1,1]),
                     xytext=(curve3[k,0], curve3[k,1]),
                     arrowprops=dict(arrowstyle='->', color='darkblue',
                                     lw=0.8, mutation_scale=7,
                                     alpha=0.5), zorder=7)
apply_compact_legend(
    ax3,
    ['LPA Local Update Area', 'LPA*更新节点', '原路径受阻段', '重规划路径（B样条）', '原路径(平滑受阻)'],
    loc='upper left'
)

plt.tight_layout()
plt.savefig('lpa_result.png', dpi=150, bbox_inches='tight')
print("[完成] lpa_result.png 已保存")
plt.show()
