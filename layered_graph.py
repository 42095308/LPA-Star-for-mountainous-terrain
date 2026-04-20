"""
================================================================================
文件名：layered_graph.py
用    途：分层拓扑航路网络构建（含碰撞检测、垂直锚点、孤岛预防）
================================================================================

【节点集 V 的生成】
    1. Terminal Pillars（末端锚点）
       根据 7 个起降/配送任务点，垂直生成贯穿三层的 21 个核心接驳节点
       同一地点三层节点坐标 (x,y) 完全相同，垂直连通
    2. Regular Grid Waypoints（规则稠密网格点）
       在支路层(Z+75m)和骨干层(Z+105m)，按约170m间距规则布点
       10km×10km 区域约 58×58 ≈ 3364 点/层

【边集 E 的生成】
    1. Horizontal Edges（水平航段）
       同层节点距离 ≤ 250m，且三维碰撞检测通过（线段距地形 ≥ 30m）
    2. Vertical Edges（垂直电梯）
       末端锚点三层直接垂直连通，无人机安全爬升/下降通道
    3. Safe Pillar Access Edges（锚点安全接入）
       末端锚点支路/骨干层在搜索半径内逐步寻找碰撞检测通过的主网接入点；
       若无安全接入点则标记锚点不可达，不强制造边
    3. Climb Edges（斜向爬升段）
       支路-骨干层节点，水平距离 ≤ 250m 且爬升角 ≤ 30° 且碰撞检测通过

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
import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.ndimage import maximum_filter, minimum_filter, gaussian_filter
from scipy.spatial import cKDTree

from scenario_config import (
    depot_params,
    display_names as load_display_names,
    load_scenario_config,
    scenario_output_dir,
    target_specs,
    terrain_sampling_params,
)
from terrain_sampling import build_terrain_samples
from virtual_depots import generate_virtual_depots

matplotlib.rcParams['font.family'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
font = FontProperties(family='DejaVu Sans')

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

NODE_SPACING_M    = 170    # 稠密布点间距（米）
BACKBONE_SPACING  = NODE_SPACING_M
BRANCH_SPACING    = NODE_SPACING_M
GRID_CELL_SIZE    = NODE_SPACING_M

INTRA_EDGE_DIST   = 250    # 同层连边最大距离（米）
INTER_EDGE_DIST   = 250    # 跨层斜向连边最大水平距离（米）
MAX_INTER_NEIGHBORS = 4    # 每个支路层节点最多连接到骨干层邻居数
PILLAR_CONNECT_DIST  = 2000   # 末端锚点强制接入主网的搜索半径（米）
PILLAR_CONNECT_RADII_M = [250, 500, 1000, 1500, 2000]  # 锚点安全接入逐步搜索半径
MAX_CLIMB_ANGLE   = 30     # 最大爬升/下降角（度）
SAFETY_HEIGHT     = 30     # 碰撞检测安全高度（米）
COLLISION_SAMPLES = 20     # 碰撞检测采样点数

PEAKS = {
    "南峰": {"lon": 110.0781, "lat": 34.4778, "elev": 2150.0},
    "东峰": {"lon": 110.0820, "lat": 34.4811, "elev": 2100.0},
    "西峰": {"lon": 110.0768, "lat": 34.4816, "elev": 2038.0},
    "北峰": {"lon": 110.0813, "lat": 34.4934, "elev": 1615.0},
    "中峰": {"lon": 110.0808, "lat": 34.4806, "elev": 2043.0},
}
DEPOTS = {
    # Assumed logistics depot anchors in WGS84 (replace proxy row/col fractions).
    # NOTE: these are project assumptions anchored to current study-area georegistration.
    "北部基地": {"lon": 110.079590, "lat": 34.505499},
    "西部基地": {"lon": 110.039004, "lat": 34.482642},
}

parser = argparse.ArgumentParser(description="构建分层拓扑航路网络。")
parser.add_argument("--scenario-config", type=str, default="")
parser.add_argument("--workdir", type=str, default=".")
parser.add_argument("--skip-plot", action="store_true", help="只生成图数据和安全状态，不生成可视化图片。")
args = parser.parse_args()

root = Path(args.workdir).resolve()
use_scene = bool(str(args.scenario_config).strip())
scene_cfg = load_scenario_config(args.scenario_config or None, root) if use_scene else {}
scene_name = str(scene_cfg.get("scene_name", "huashan")) if use_scene else "huashan"
data_dir = scenario_output_dir(scene_cfg, root) if use_scene else root
data_dir.mkdir(parents=True, exist_ok=True)
os.chdir(data_dir)

# ===== 读取数据 =====
assert os.path.exists("Z_crop.npy"),    "缺少 Z_crop.npy，请先运行 init_graph.py"
assert os.path.exists("layer_mid.npy"), "缺少 layer_mid.npy，请先运行 safe_corridor.py"
assert os.path.exists("Z_crop_geo.npz"), "缺少 Z_crop_geo.npz，请先运行 init_graph.py"

Z          = np.load("Z_crop.npy")
layer_mid  = np.load("layer_mid.npy")
floor_grid = np.load("floor.npy") if os.path.exists("floor.npy") else None
ceiling_grid = np.load("ceiling.npy") if os.path.exists("ceiling.npy") else None
layer_allowed = np.load("layer_allowed.npy") if os.path.exists("layer_allowed.npy") else None
geo        = np.load("Z_crop_geo.npz")
lon_grid   = np.asarray(geo["lon_grid"], dtype=float)
lat_grid   = np.asarray(geo["lat_grid"], dtype=float)
rows, cols = Z.shape
print(f"[读取] DEM shape={Z.shape}，高程范围: {Z.min():.0f}~{Z.max():.0f}m")

# ===== 工具函数 =====
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

def nearest_rc_by_lonlat(lon, lat):
    d2 = (lon_grid - lon) ** 2 + (lat_grid - lat) ** 2
    idx = int(np.argmin(d2))
    r, c = np.unravel_index(idx, lon_grid.shape)
    return int(r), int(c)

def get_layer_height(lid, r, c):
    """按自适应走廊层面读取节点高度。"""
    lid = int(np.clip(lid, 0, layer_mid.shape[0] - 1))
    r = int(np.clip(r, 0, rows - 1))
    c = int(np.clip(c, 0, cols - 1))
    return float(layer_mid[lid, r, c])

def layer_is_allowed(lid, r, c):
    if layer_allowed is None:
        return True
    lid = int(np.clip(lid, 0, layer_allowed.shape[0] - 1))
    r = int(np.clip(r, 0, rows - 1))
    c = int(np.clip(c, 0, cols - 1))
    return bool(layer_allowed[lid, r, c])

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
        if floor_grid is not None and z_m < float(floor_grid[r, c]) - 1e-6:
            return False
        if ceiling_grid is not None and z_m > float(ceiling_grid[r, c]) + 1e-6:
            return False
        lid = int(np.clip(round(n1[3] + t * (n2[3] - n1[3])), 0, 2))
        if not layer_is_allowed(lid, r, c):
            return False
    return True


if use_scene:
    PEAKS_ACTIVE = target_specs(scene_cfg)
    risk_human = None
    if os.path.exists("risk_human.npy"):
        risk_arr = np.load("risk_human.npy").astype(float)
        if risk_arr.shape == Z.shape:
            risk_human = np.clip(risk_arr, 0.0, 1.0)
    depot_list = generate_virtual_depots(
        Z,
        lon_grid,
        lat_grid,
        PEAKS_ACTIVE,
        depot_params(scene_cfg),
        risk_human=risk_human,
        resolution_m=RESOLUTION,
    )
    DEPOTS_ACTIVE = {d["name"]: d for d in depot_list}
    depot_payload = {
        "scene_name": scene_name,
        "rule": "低坡度、低海拔、低风险、远离目标点且位于边缘或山脚过渡区",
        "depots": depot_list,
    }
    Path("generated_depots.json").write_text(
        json.dumps(depot_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    DISPLAY_NAME = load_display_names(scene_cfg)
    for d in depot_list:
        DISPLAY_NAME.setdefault(d["name"], d["name"])
    print(f"[配送站] 已自动生成 {len(depot_list)} 个虚拟配送站，写入 generated_depots.json")
else:
    PEAKS_ACTIVE = PEAKS
    DEPOTS_ACTIVE = DEPOTS
    DISPLAY_NAME = {
        "南峰": "South Peak",
        "东峰": "East Peak",
        "西峰": "West Peak",
        "北峰": "North Peak",
        "中峰": "Central Peak",
        "北部基地": "North Depot",
        "西部基地": "West Depot",
    }

def build_terminal_specs():
    specs = {}
    sources = {}
    for name, p in PEAKS_ACTIVE.items():
        r, c = nearest_rc_by_lonlat(float(p["lon"]), float(p["lat"]))
        specs[name] = {"row": r, "col": c}
        sources[name] = "target"
    for name, p in DEPOTS_ACTIVE.items():
        if "row" in p and "col" in p:
            r, c = int(p["row"]), int(p["col"])
        else:
            r, c = nearest_rc_by_lonlat(float(p["lon"]), float(p["lat"]))
        specs[name] = {"row": r, "col": c}
        sources[name] = "virtual_depot" if use_scene else "depot"
    return specs, sources


terminal_specs, terminal_sources = build_terminal_specs()

# ===== Step 1：提取地形特征点 =====
print("\n[Step1] 提取地形特征点...")
Z_smooth = gaussian_filter(Z.astype(float), sigma=3)
size_px  = max(3, int(BACKBONE_SPACING / RESOLUTION))

ridge_pts  = np.argwhere(Z_smooth == maximum_filter(Z_smooth, size=size_px))
valley_pts = np.argwhere(Z_smooth == minimum_filter(Z_smooth, size=size_px))
topo_pts   = np.vstack([ridge_pts, valley_pts])
print(f"  山脊点: {len(ridge_pts)}，山谷点: {len(valley_pts)}，合计: {len(topo_pts)}")

# ===== Step 2：地形驱动节点采样（规则网格仅作补充） =====
terminal_rcs_for_sampling = [(int(v["row"]), int(v["col"])) for v in terminal_specs.values()]
risk_for_sampling = None
if os.path.exists("risk_human.npy"):
    arr = np.load("risk_human.npy").astype(float)
    if arr.shape == Z.shape:
        risk_for_sampling = np.clip(arr, 0.0, 1.0)

branch_pts, branch_roles, backbone_pts, backbone_roles, sampling_meta = build_terrain_samples(
    Z,
    risk_for_sampling,
    terminal_rcs_for_sampling,
    terrain_sampling_params(scene_cfg) if use_scene else terrain_sampling_params({}),
    resolution_m=RESOLUTION,
    layer_allowed=layer_allowed,
)
print(
    f"  地形驱动采样: 支路层 {len(branch_pts)} 点，骨干层 {len(backbone_pts)} 点；"
    f"补充网格占比 支路={sampling_meta['actual_branch_grid_ratio']:.2%}, "
    f"骨干={sampling_meta['actual_backbone_grid_ratio']:.2%}"
)

# ===== Step 3：构建节点表 =====
print("\n[Step3] 构建节点表...")
nodes = []
node_roles = []

# --- 补丁2：Terminal Pillars 末端锚点（贯穿三层）---
all_terminals   = terminal_specs
terminal_pillars = {}   # name -> [layer0_idx, layer1_idx, layer2_idx]
terminal_status = {
    "scene_name": scene_name,
    "terminal_order": list(all_terminals.keys()),
    "terminals": {},
    "sampling_meta": sampling_meta,
}

for name, p in all_terminals.items():
    r_crop = int(np.clip(p["row"], 0, rows - 1))
    c_crop = int(np.clip(p["col"], 0, cols - 1))
    x_km, y_km = pixel_to_km(r_crop, c_crop)
    terrain     = get_elev_rc(r_crop, c_crop)
    pillar_idxs = []
    for lid, _offset in enumerate(DECK_OFFSETS.values()):
        pillar_idxs.append(len(nodes))
        nodes.append([x_km, y_km, get_layer_height(lid, r_crop, c_crop), lid])
        node_roles.append({"role": "terminal_anchor", "name": name, "layer": int(lid)})
    terminal_pillars[name] = pillar_idxs
    terminal_status["terminals"][name] = {
        "source": terminal_sources.get(name, "unknown"),
        "row": int(r_crop),
        "col": int(c_crop),
        "indices": [int(v) for v in pillar_idxs],
        "branch_connected": False,
        "backbone_connected": False,
        "reachable": False,
        "connect_edges": [],
    }
    print(f"  锚点 {name}: ({x_km:.2f},{y_km:.2f}km) "
          f"高度={nodes[pillar_idxs[0]][2]:.0f}/{nodes[pillar_idxs[1]][2]:.0f}/{nodes[pillar_idxs[2]][2]:.0f}m")

n_terminal = len(nodes)

# --- 支路层节点（layer_id=1）---
branch_start = len(nodes)
for pt in branch_pts:
    r, c = int(pt[0]), int(pt[1])
    x_km, y_km = pixel_to_km(r, c)
    nodes.append([x_km, y_km, get_layer_height(1, r, c), 1])
    role = branch_roles[len(nodes) - branch_start - 1] if len(branch_roles) >= (len(nodes) - branch_start) else "branch"
    node_roles.append({"role": role, "layer": 1, "row": int(r), "col": int(c)})

# --- 骨干层节点（layer_id=2）---
backbone_start = len(nodes)
for pt in backbone_pts:
    r, c = int(pt[0]), int(pt[1])
    x_km, y_km = pixel_to_km(r, c)
    nodes.append([x_km, y_km, get_layer_height(2, r, c), 2])
    role = backbone_roles[len(nodes) - backbone_start - 1] if len(backbone_roles) >= (len(nodes) - backbone_start) else "backbone"
    node_roles.append({"role": role, "layer": 2, "row": int(r), "col": int(c)})

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

# --- 末端锚点安全接入主网（禁止碰撞失败后的强制连边）---
print("  安全搜索末端锚点到主网络的接入边...")
safe_anchor_count = 0

branch_idx_arr = np.arange(branch_start, backbone_start, dtype=int)
backbone_idx_arr = np.arange(backbone_start, len(nodes), dtype=int)
branch_tree_for_anchor = cKDTree(nodes[branch_idx_arr, :2])
backbone_tree_for_anchor = cKDTree(nodes[backbone_idx_arr, :2])


def connect_anchor_to_layer(anchor_idx, candidate_indices, tree, layer_label):
    """按半径逐步搜索安全接入节点；碰撞失败的候选绝不加边。"""
    anchor_xy = nodes[anchor_idx, :2]
    checked = 0
    for radius_m in PILLAR_CONNECT_RADII_M:
        radius_km = float(radius_m) / 1000.0
        cand_locals = tree.query_ball_point(anchor_xy, r=radius_km)
        cand_locals = sorted(
            cand_locals,
            key=lambda local: float(np.linalg.norm(anchor_xy - nodes[int(candidate_indices[local]), :2])),
        )
        for local in cand_locals:
            cand_idx = int(candidate_indices[local])
            checked += 1
            if collision_free(nodes[anchor_idx], nodes[cand_idx]):
                edges.append([int(anchor_idx), cand_idx, 0])
                return {
                    "connected": True,
                    "node": cand_idx,
                    "distance_km": float(np.linalg.norm(anchor_xy - nodes[cand_idx, :2])),
                    "radius_m": int(radius_m),
                    "checked_candidates": int(checked),
                    "layer": layer_label,
                }
    return {
        "connected": False,
        "node": None,
        "distance_km": None,
        "radius_m": int(PILLAR_CONNECT_RADII_M[-1]),
        "checked_candidates": int(checked),
        "layer": layer_label,
    }


for name, pillar in terminal_pillars.items():
    branch_result = connect_anchor_to_layer(
        pillar[1],
        branch_idx_arr,
        branch_tree_for_anchor,
        "区域支路层",
    )
    backbone_result = connect_anchor_to_layer(
        pillar[2],
        backbone_idx_arr,
        backbone_tree_for_anchor,
        "骨干航路层",
    )

    st = terminal_status["terminals"][name]
    st["branch_connected"] = bool(branch_result["connected"])
    st["backbone_connected"] = bool(backbone_result["connected"])
    st["reachable"] = bool(branch_result["connected"] or backbone_result["connected"])
    st["connect_attempts"] = {
        "branch": branch_result,
        "backbone": backbone_result,
    }
    if branch_result["connected"]:
        st["connect_edges"].append([int(pillar[1]), int(branch_result["node"]), 0])
        safe_anchor_count += 1
    if backbone_result["connected"]:
        st["connect_edges"].append([int(pillar[2]), int(backbone_result["node"]), 0])
        safe_anchor_count += 1

    if st["reachable"]:
        print(
            f"    {name}: 安全接入成功 "
            f"(支路={st['branch_connected']}, 骨干={st['backbone_connected']})"
        )
    else:
        print(f"    {name}: 未找到满足碰撞约束的接入边，标记为不可达")

print(f"  安全接入边: {safe_anchor_count}")

# --- 同层水平边（含碰撞检测）---
def add_intra_edges(idx_start, idx_end, max_dist_m):
    count  = 0
    max_km = max_dist_m / 1000
    idx = np.arange(idx_start, idx_end, dtype=int)
    if len(idx) < 2:
        return count

    coords = nodes[idx, :2]
    tree = cKDTree(coords)
    # 只遍历邻域候选边，避免 O(N^2) 全配对
    for li, lj in tree.query_pairs(r=max_km):
        i = int(idx[li])
        j = int(idx[lj])
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

# --- 斜向爬升边（支路-骨干，含角度约束+碰撞检测）---
climb_count   = 0
max_inter_km  = INTER_EDGE_DIST / 1000
max_angle_rad = np.radians(MAX_CLIMB_ANGLE)

branch_idx = np.arange(branch_start, backbone_start, dtype=int)
backbone_idx = np.arange(backbone_start, len(nodes), dtype=int)
branch_xy = nodes[branch_idx, :2]
backbone_xy = nodes[backbone_idx, :2]
backbone_tree = cKDTree(backbone_xy)
neighbor_lists = backbone_tree.query_ball_point(branch_xy, r=max_inter_km)

for b_local, cand_locals in enumerate(neighbor_lists):
    if len(cand_locals) == 0:
        continue

    i = int(branch_idx[b_local])
    # 限制跨层邻居上限，控制总边数
    if len(cand_locals) > MAX_INTER_NEIGHBORS:
        cand_locals = sorted(
            cand_locals,
            key=lambda c_local: np.linalg.norm(branch_xy[b_local] - backbone_xy[c_local])
        )[:MAX_INTER_NEIGHBORS]

    for c_local in cand_locals:
        j = int(backbone_idx[c_local])
        d_horiz = np.linalg.norm(nodes[i, :2] - nodes[j, :2])
        d_vert = abs(nodes[i, 2] - nodes[j, 2])
        # 水平距离转米再计算角度
        if d_horiz * 1000 < 1e-3:
            continue
        if np.arctan2(d_vert, d_horiz * 1000) > max_angle_rad:
            continue
        if collision_free(nodes[i], nodes[j]):
            edges.append([i, j, 2])
            climb_count += 1

print(f"  斜向爬升边 支路-骨干: {climb_count}")
edges = np.array(edges) if edges else np.zeros((0, 3), dtype=int)

print("  保存前校验全部非垂直边的安全间隙...")
unsafe_edges = []
for eid, e in enumerate(edges):
    i, j, etype = int(e[0]), int(e[1]), int(e[2])
    if etype == 1:
        continue
    if not collision_free(nodes[i], nodes[j]):
        unsafe_edges.append({"edge_id": int(eid), "u": i, "v": j, "edge_type": etype})
terminal_status["unsafe_edge_violations"] = len(unsafe_edges)
if unsafe_edges:
    terminal_status["unsafe_edges_preview"] = unsafe_edges[:20]
    Path("graph_terminal_status.json").write_text(
        json.dumps(terminal_status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    raise RuntimeError(
        f"发现 {len(unsafe_edges)} 条非垂直边未通过碰撞检测，已拒绝保存 graph_edges.npy。"
    )

print(f"\n  边总数 |E| = {len(edges)}")
print(f"  水平边: {np.sum(edges[:,2]==0)}")
print(f"  电梯边: {np.sum(edges[:,2]==1)}")
print(f"  爬升边: {np.sum(edges[:,2]==2)}")

np.save("graph_nodes.npy", nodes)
np.save("graph_edges.npy", edges)
Path("graph_terminal_status.json").write_text(
    json.dumps(terminal_status, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
role_payload = {
    "scene_name": scene_name,
    "sampling_meta": sampling_meta,
    "node_roles": node_roles,
}
Path("graph_node_roles.json").write_text(
    json.dumps(role_payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("\n[保存] graph_nodes.npy，graph_edges.npy 已保存")
print("[保存] graph_terminal_status.json 已保存")
print("[保存] graph_node_roles.json 已保存")

if args.skip_plot:
    print("[可视化] 已按 --skip-plot 跳过 graph_vis.png 生成")
    raise SystemExit(0)

# ===== 可视化 =====
print("\n[可视化] 生成路网图...")
fig = plt.figure(figsize=(20, 9))

# ---------- 左图：俯视图 ----------
ax1 = fig.add_subplot(121)
ax1.imshow(Z, cmap='terrain', alpha=0.4,
           extent=[0, cols*RESOLUTION/1000, 0, rows*RESOLUTION/1000],
           origin='upper', aspect='equal')

rng = np.random.default_rng(42)  # 固定随机种子，结果可复现
for e in edges:
    i, j, etype = int(e[0]), int(e[1]), int(e[2])
    if etype == 0:
        if rng.random() > 0.10:   # 90%的水平边跳过不画
            continue
        color, lw, alpha = LAYER_COLORS[int(nodes[i,3])], 0.8, 0.5
    elif etype == 1:
        color, lw, alpha = "#9C27B0", 2.0, 0.95  # 垂直电梯边全部显示
    else:
        color, lw, alpha = "#FF9800", 1.0, 0.6   # 斜向爬升边全部显示
    ax1.plot([nodes[i,0], nodes[j,0]],
             [nodes[i,1], nodes[j,1]],
             color=color, lw=lw, alpha=alpha, zorder=2)

for lid, (color, marker, size) in enumerate(
        zip(LAYER_COLORS, LAYER_MARKERS, [60, 8, 8])):
    mask  = np.where(nodes[:,3] == lid)[0]
    label = ["Terminal Layer", "Regional Layer", "Backbone Layer"][lid]
    if len(mask) == 0:
        continue
    if lid == 0:
        # 末端层=锚点，全部显示，放大
        show = mask
    else:
        # 支路/骨干网格节点，随机取20%
        show = rng.choice(mask, size=max(1, len(mask)//5), replace=False)
    ax1.scatter(nodes[show,0], nodes[show,1],
                c=color, marker=marker, s=size, label=label,
                alpha=0.5, zorder=3, edgecolors='none')

for name, pillar in terminal_pillars.items():
    idx = pillar[0]
    ax1.annotate(DISPLAY_NAME.get(name, name),
                 xy=(nodes[idx,0], nodes[idx,1]),
                 xytext=(nodes[idx,0]+0.3, nodes[idx,1]+0.3),
                 fontproperties=font, fontsize=8, color='darkred',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

ax1.set_xlabel('East-West (km)', fontproperties=font)
ax1.set_ylabel('South-North (km)', fontproperties=font)
ax1.set_title(f'Layered Airway Topology (Top View)\n'
              f'|V|={len(nodes)}, |E|={len(edges)} (collision-checked)',
              fontproperties=font, fontsize=11)
ax1.legend(prop=font, loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3, linestyle='--')

# ---------- 局部放大框：锚点柱 + 垂直电梯边 ----------
if terminal_pillars:
    focus_name = "虚拟配送站1" if "虚拟配送站1" in terminal_pillars else next(iter(terminal_pillars))
    focus_idx = terminal_pillars[focus_name][1]
    cx, cy = float(nodes[focus_idx, 0]), float(nodes[focus_idx, 1])
    span = 0.75
    xmin, xmax = cx - span, cx + span
    ymin, ymax = cy - span, cy + span

    axins = ax1.inset_axes([0.58, 0.05, 0.38, 0.38])
    axins.imshow(
        Z,
        cmap='terrain',
        alpha=0.35,
        extent=[0, cols * RESOLUTION / 1000, 0, rows * RESOLUTION / 1000],
        origin='upper',
        aspect='equal',
    )
    rng_zoom = np.random.default_rng(7)
    for e in edges:
        i, j, etype = int(e[0]), int(e[1]), int(e[2])
        x1, y1 = float(nodes[i, 0]), float(nodes[i, 1])
        x2, y2 = float(nodes[j, 0]), float(nodes[j, 1])
        in_box = (
            (xmin <= x1 <= xmax and ymin <= y1 <= ymax)
            or (xmin <= x2 <= xmax and ymin <= y2 <= ymax)
        )
        if not in_box:
            continue
        if etype == 0:
            if rng_zoom.random() > 0.25:
                continue
            color, lw, alpha = LAYER_COLORS[int(nodes[i, 3])], 0.9, 0.60
        elif etype == 1:
            color, lw, alpha = "#9C27B0", 2.2, 0.98
        else:
            color, lw, alpha = "#FF9800", 1.2, 0.75
        axins.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha, zorder=3)

    for lid, (color, marker) in enumerate(zip(LAYER_COLORS, LAYER_MARKERS)):
        idx = np.where(nodes[:, 3] == lid)[0]
        if len(idx) == 0:
            continue
        x = nodes[idx, 0]
        y = nodes[idx, 1]
        m = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        show = idx[m]
        if len(show) == 0:
            continue
        size = 42 if lid == 0 else 14
        axins.scatter(
            nodes[show, 0],
            nodes[show, 1],
            c=color,
            marker=marker,
            s=size,
            alpha=0.70,
            edgecolors='none',
            zorder=4,
        )

    axins.set_xlim(xmin, xmax)
    axins.set_ylim(ymin, ymax)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title(f"Zoom: pillar/elevator around {DISPLAY_NAME.get(focus_name, focus_name)}", fontsize=7.5)
    mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.45", lw=0.9)

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
ax2.set_xlabel('East-West (km)', fontproperties=font, labelpad=6)
ax2.set_ylabel('South-North (km)', fontproperties=font, labelpad=6)
ax2.set_zlabel('Elevation (m)',  fontproperties=font, labelpad=6)
ax2.set_title('Layered Airway Topology (3D View)', fontproperties=font, fontsize=11)

plt.tight_layout()
plt.savefig('graph_vis.png', dpi=300, bbox_inches='tight')
print("[完成] graph_vis.png 已保存")
print("[下一步] 先运行 human_risk_osm.py 生成游客风险场，再运行 lpa_star.py 进行动态规划")
plt.show()

