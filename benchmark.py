"""
面向场景配置的 Monte Carlo benchmark，比较四类规划基线。

Baselines
--------
B4: Proposed (Layered graph + multi-objective cost + LPA* incremental replanning)
B2: Ablation A (Layered graph + multi-objective cost + global A* recompute)
B3: Ablation B (Single-layer flattened graph + LPA* incremental replanning)
B1: Traditional (Coarse 3D voxel + Dijkstra from scratch)
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from article_planner.scenario_config import (
    DEFAULT_SCENE_CONFIG,
    communication_params,
    load_scenario_config,
    resolve_resolution_m,
    scenario_output_dir,
)

try:
    from scipy.stats import ttest_rel, wilcoxon
except Exception:  # pragma: no cover - optional
    ttest_rel = None
    wilcoxon = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional
    plt = None

from dynamic_events import AreaEvent, build_area_event_from_center, build_area_event_from_path, event_edge_cost


ALPHA = 0.3
BETA = 0.2
GAMMA = 0.5
UAV_SPEED = 15.0
UAV_POWER = 500.0
UAV_MASS = 5.0  # kg, reference UAV mass (conservative)
RESOLUTION: float | None = None
SAFETY_HEIGHT = 30.0
COLLISION_SAMPLES = 12
RISK_SAMPLES = 10
EPS = 1e-9

# OSM-human-risk fusion weights (same idea as lpa_star.py).
RISK_W_TERRAIN = 0.50
# Split human-risk weights for L1~L4 (sum=0.50).
RISK_W_L1 = 0.20
RISK_W_L2 = 0.10
RISK_W_L3 = 0.15
RISK_W_L4 = 0.05
RISK_W_HUMAN_COMBINED = 0.50

BASELINE_B4 = "B4_Proposed_LPA_Layered"
BASELINE_B2 = "B2_GlobalAstar_Layered"
BASELINE_B5 = "B5_RegularLayered_LPA"
# 兼容旧结果中曾使用的 B6 命名；新输出统一写 B5。
BASELINE_B6 = "B6_RegularLayered_LPA"
BASELINE_B3 = "B3_LPA_SingleLayer"
BASELINE_B1 = "B1_Voxel_Dijkstra"

STRUCTURAL_ABLATION_METHODS = [
    (BASELINE_B4, "Terrain-aware Layered LPA*", "proposed_full_method"),
    (BASELINE_B2, "Terrain-aware Layered A*", "incremental_replanning_ablation"),
    (BASELINE_B3, "Single-layer LPA*", "layered_network_ablation"),
    (BASELINE_B5, "Regular-layered LPA*", "terrain_aware_layering_ablation"),
    (BASELINE_B1, "Voxel Global Search", "classical_voxel_baseline"),
]

REGULAR_INTRA_EDGE_DIST_M = 250.0
REGULAR_INTER_EDGE_DIST_M = 250.0
REGULAR_MAX_INTER_NEIGHBORS = 4
REGULAR_MAX_CLIMB_ANGLE_DEG = 30.0
REGULAR_PILLAR_CONNECT_RADII_M = [250.0, 500.0, 1000.0, 1500.0, 2000.0]


def normalize_pair(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def km_to_rc(x_km: float, y_km: float, rows: int, cols: int) -> Tuple[int, int]:
    c = int(np.clip(x_km * 1000.0 / RESOLUTION, 0, cols - 1))
    r = int(np.clip((rows - 1) - y_km * 1000.0 / RESOLUTION, 0, rows - 1))
    return r, c


def terrain_at_xy(x_km: float, y_km: float, z_grid: np.ndarray) -> float:
    rows, cols = z_grid.shape
    r, c = km_to_rc(x_km, y_km, rows, cols)
    return float(z_grid[r, c])


def load_risk_fields(root: Path, shape: Tuple[int, int], config: Optional[dict] = None) -> Dict[str, object]:
    rows, cols = int(shape[0]), int(shape[1])
    risk_l1 = np.zeros((rows, cols), dtype=float)
    risk_l2 = np.zeros((rows, cols), dtype=float)
    risk_l3 = np.zeros((rows, cols), dtype=float)
    risk_l4 = np.zeros((rows, cols), dtype=float)
    risk_trail = np.zeros((rows, cols), dtype=float)
    risk_hotspot = np.zeros((rows, cols), dtype=float)
    risk_human = np.zeros((rows, cols), dtype=float)
    risk_comm = np.zeros((3, rows, cols), dtype=float)
    mode = "terrain_only"

    p = root / "risk_l1.npy"
    if p.exists():
        arr = np.asarray(np.load(p), dtype=float)
        if arr.shape == (rows, cols):
            risk_l1 = np.clip(arr, 0.0, 1.0)

    p = root / "risk_l2.npy"
    if p.exists():
        arr = np.asarray(np.load(p), dtype=float)
        if arr.shape == (rows, cols):
            risk_l2 = np.clip(arr, 0.0, 1.0)

    p = root / "risk_l3.npy"
    if p.exists():
        arr = np.asarray(np.load(p), dtype=float)
        if arr.shape == (rows, cols):
            risk_l3 = np.clip(arr, 0.0, 1.0)

    p = root / "risk_l4.npy"
    if p.exists():
        arr = np.asarray(np.load(p), dtype=float)
        if arr.shape == (rows, cols):
            risk_l4 = np.clip(arr, 0.0, 1.0)

    p = root / "risk_trail.npy"
    if p.exists():
        arr = np.asarray(np.load(p), dtype=float)
        if arr.shape == (rows, cols):
            risk_trail = np.clip(arr, 0.0, 1.0)

    p = root / "risk_hotspot.npy"
    if p.exists():
        arr = np.asarray(np.load(p), dtype=float)
        if arr.shape == (rows, cols):
            risk_hotspot = np.clip(arr, 0.0, 1.0)

    p = root / "risk_human.npy"
    if p.exists():
        arr = np.asarray(np.load(p), dtype=float)
        if arr.shape == (rows, cols):
            risk_human = np.clip(arr, 0.0, 1.0)

    p = root / "risk_comm.npy"
    if p.exists():
        arr = np.asarray(np.load(p), dtype=float)
        if arr.shape == (3, rows, cols):
            risk_comm = np.clip(arr, 0.0, 1.0)

    has_level_split = (
        (float(np.max(risk_l1)) > 0.0)
        or (float(np.max(risk_l2)) > 0.0)
        or (float(np.max(risk_l3)) > 0.0)
        or (float(np.max(risk_l4)) > 0.0)
    )
    has_legacy_split = (float(np.max(risk_trail)) > 0.0) or (float(np.max(risk_hotspot)) > 0.0)
    has_split = has_level_split or has_legacy_split
    has_combined = float(np.max(risk_human)) > 0.0
    # Keep split mode priority to avoid combined map masking L1~L4.
    if has_split:
        mode = "terrain_trail_hotspot"
    elif has_combined:
        mode = "terrain_human_combined"

    # Legacy fallback: map old split files to L1/L3.
    if (not has_level_split) and has_legacy_split:
        risk_l1 = risk_trail.copy()
        risk_l2 = np.zeros_like(risk_l1)
        risk_l3 = risk_hotspot.copy()
        risk_l4 = np.zeros_like(risk_l1)

    has_comm = float(np.max(risk_comm)) > 0.0
    params = communication_params(config or {})
    raw_weights = dict(params.get("weights", {}))
    terrain_w = float(raw_weights.get("terrain", 0.45))
    human_w = float(raw_weights.get("human", 0.35)) if has_split or has_combined else 0.0
    comm_w = float(raw_weights.get("communication", 0.20)) if has_comm else 0.0
    if terrain_w <= 0.0:
        terrain_w = 1.0
    w_sum = max(terrain_w + human_w + comm_w, EPS)
    risk_weights = {
        "terrain": float(terrain_w / w_sum),
        "human": float(human_w / w_sum),
        "communication": float(comm_w / w_sum),
    }

    return {
        "mode": mode,
        "risk_l1": risk_l1,
        "risk_l2": risk_l2,
        "risk_l3": risk_l3,
        "risk_l4": risk_l4,
        "risk_trail": risk_trail,
        "risk_hotspot": risk_hotspot,
        "risk_human": risk_human,
        "risk_comm": risk_comm,
        "has_comm": bool(has_comm),
        "risk_weights": risk_weights,
        "comm_risk_threshold": float(params.get("risk_threshold", 0.55)),
    }


def collision_free_segment(
    p1: np.ndarray,
    p2: np.ndarray,
    z_grid: np.ndarray,
    n_samples: int = COLLISION_SAMPLES,
) -> bool:
    rows, cols = z_grid.shape
    for t in np.linspace(0.0, 1.0, n_samples):
        x = float(p1[0] + t * (p2[0] - p1[0]))
        y = float(p1[1] + t * (p2[1] - p1[1]))
        z = float(p1[2] + t * (p2[2] - p1[2]))
        r, c = km_to_rc(x, y, rows, cols)
        if z - float(z_grid[r, c]) < SAFETY_HEIGHT:
            return False
    return True


def corridor_collision_free_segment(
    p1: np.ndarray,
    p2: np.ndarray,
    z_grid: np.ndarray,
    floor_grid: Optional[np.ndarray] = None,
    ceiling_grid: Optional[np.ndarray] = None,
    layer_allowed: Optional[np.ndarray] = None,
    n_samples: int = COLLISION_SAMPLES,
) -> bool:
    """检查线段是否同时满足地形净空、走廊边界与分层可用性约束。"""
    rows, cols = z_grid.shape
    for t in np.linspace(0.0, 1.0, n_samples):
        x = float(p1[0] + t * (p2[0] - p1[0]))
        y = float(p1[1] + t * (p2[1] - p1[1]))
        z = float(p1[2] + t * (p2[2] - p1[2]))
        r, c = km_to_rc(x, y, rows, cols)
        if z - float(z_grid[r, c]) < SAFETY_HEIGHT:
            return False
        if floor_grid is not None and z < float(floor_grid[r, c]) - 1e-6:
            return False
        if ceiling_grid is not None and z > float(ceiling_grid[r, c]) + 1e-6:
            return False
        if layer_allowed is not None:
            lid = int(np.clip(round(float(p1[3] + t * (p2[3] - p1[3]))), 0, layer_allowed.shape[0] - 1))
            if not bool(layer_allowed[lid, r, c]):
                return False
    return True


def select_regular_grid_points(
    allowed: np.ndarray,
    count: int,
    exclude: Optional[Sequence[Tuple[int, int]]] = None,
) -> np.ndarray:
    """在允许区域内均匀选取规则网格点，用于 B5 规则三层图。"""
    if count <= 0:
        return np.zeros((0, 2), dtype=int)

    rows, cols = allowed.shape
    exclude_set = {
        (int(np.clip(r, 0, rows - 1)), int(np.clip(c, 0, cols - 1)))
        for r, c in (exclude or [])
    }

    chosen: List[Tuple[int, int]] = []
    chosen_set = set()
    per_axis = max(2, int(math.ceil(math.sqrt(max(count, 1) * 1.8))))
    hard_limit = max(rows, cols) * 2 + 8

    while len(chosen) < count and per_axis <= hard_limit:
        rr = np.linspace(0, rows - 1, per_axis, dtype=int)
        cc = np.linspace(0, cols - 1, per_axis, dtype=int)
        grid: List[Tuple[int, int]] = []
        seen_local = set()
        for r in rr:
            for c in cc:
                key = (int(r), int(c))
                if key in seen_local or key in exclude_set or key in chosen_set:
                    continue
                seen_local.add(key)
                if bool(allowed[key[0], key[1]]):
                    grid.append(key)
        if grid:
            if len(chosen) + len(grid) >= count:
                need = count - len(chosen)
                take_idx = np.linspace(0, len(grid) - 1, need, dtype=int)
                for idx in take_idx:
                    key = grid[int(idx)]
                    if key in chosen_set:
                        continue
                    chosen.append(key)
                    chosen_set.add(key)
                break
            for key in grid:
                if key in chosen_set:
                    continue
                chosen.append(key)
                chosen_set.add(key)
        per_axis = int(math.ceil(per_axis * 1.35)) + 1

    if len(chosen) < count:
        allowed_pts = np.argwhere(allowed)
        if allowed_pts.size > 0:
            fill_candidates: List[Tuple[int, int]] = []
            for r, c in allowed_pts:
                key = (int(r), int(c))
                if key in exclude_set or key in chosen_set:
                    continue
                fill_candidates.append(key)
            if fill_candidates:
                need = min(count - len(chosen), len(fill_candidates))
                take_idx = np.linspace(0, len(fill_candidates) - 1, need, dtype=int)
                for idx in take_idx:
                    key = fill_candidates[int(idx)]
                    if key in chosen_set:
                        continue
                    chosen.append(key)
                    chosen_set.add(key)

    return np.asarray(chosen[:count], dtype=int)


def fused_risk_at_point(
    x_km: float,
    y_km: float,
    z_m: float,
    z_grid: np.ndarray,
    risk_fields: Optional[Dict[str, object]],
    fallback_lid: int = 1,
) -> Tuple[float, float]:
    """返回综合风险和通信风险，用于路径级暴露指标。"""
    rows, cols = z_grid.shape
    r, c = km_to_rc(x_km, y_km, rows, cols)
    terrain = float(z_grid[r, c])
    r_terrain = max(0.0, 1.0 - (z_m - terrain) / 200.0)
    if not risk_fields:
        return float(np.clip(r_terrain, 0.0, 1.0)), 0.0

    risk_mode = str(risk_fields.get("mode", "terrain_only"))
    risk_weights = dict(risk_fields.get("risk_weights", {"terrain": 1.0, "human": 0.0, "communication": 0.0}))
    r_human = 0.0
    if risk_mode == "terrain_trail_hotspot":
        risk_l1 = risk_fields.get("risk_l1")
        risk_l2 = risk_fields.get("risk_l2")
        risk_l3 = risk_fields.get("risk_l3")
        risk_l4 = risk_fields.get("risk_l4")
        if risk_l1 is not None and risk_l2 is not None and risk_l3 is not None and risk_l4 is not None:
            split_sum = max(RISK_W_L1 + RISK_W_L2 + RISK_W_L3 + RISK_W_L4, EPS)
            r_human = (
                (RISK_W_L1 / split_sum) * float(risk_l1[r, c])
                + (RISK_W_L2 / split_sum) * float(risk_l2[r, c])
                + (RISK_W_L3 / split_sum) * float(risk_l3[r, c])
                + (RISK_W_L4 / split_sum) * float(risk_l4[r, c])
            )
    elif risk_mode == "terrain_human_combined" and risk_fields.get("risk_human") is not None:
        r_human = float(risk_fields["risk_human"][r, c])

    r_comm = 0.0
    if bool(risk_fields.get("has_comm", False)) and risk_fields.get("risk_comm") is not None:
        lid = int(np.clip(fallback_lid, 0, 2))
        r_comm = float(risk_fields["risk_comm"][lid, r, c])
    r_total = (
        float(risk_weights.get("terrain", 1.0)) * r_terrain
        + float(risk_weights.get("human", 0.0)) * r_human
        + float(risk_weights.get("communication", 0.0)) * r_comm
    )
    return float(np.clip(r_total, 0.0, 1.0)), float(np.clip(r_comm, 0.0, 1.0))


def compute_node_path_extra_metrics(
    nodes: np.ndarray,
    path: Sequence[int],
    z_grid: Optional[np.ndarray],
    risk_fields: Optional[Dict[str, object]],
) -> Dict[str, float]:
    """统计山地安全和通信指标：最小净空、风险暴露积分、覆盖率和最大连续失联。"""
    defaults = {
        "min_clearance_m": float("nan"),
        "risk_exposure_integral": float("nan"),
        "comm_coverage_ratio": float("nan"),
        "max_comm_loss_time_s": float("nan"),
        "max_comm_loss_length_km": float("nan"),
    }
    if z_grid is None or not path or len(path) < 2:
        return defaults

    has_comm = bool(risk_fields and risk_fields.get("has_comm", False))
    comm_threshold = float((risk_fields or {}).get("comm_risk_threshold", 0.55))
    min_clearance = float("inf")
    exposure = 0.0
    total_len_km = 0.0
    covered_len_km = 0.0
    cur_lost_km = 0.0
    max_lost_km = 0.0

    for idx in range(len(path) - 1):
        u = int(path[idx])
        v = int(path[idx + 1])
        p1 = nodes[u]
        p2 = nodes[v]
        dx = (float(p2[0]) - float(p1[0])) * 1000.0
        dy = (float(p2[1]) - float(p1[1])) * 1000.0
        dz = float(p2[2]) - float(p1[2])
        seg_len_km = float(np.sqrt(dx * dx + dy * dy + dz * dz) / 1000.0)
        if seg_len_km <= EPS:
            continue
        seg_risk = 0.0
        seg_comm = 0.0
        sample_n = max(2, RISK_SAMPLES)
        for s in np.linspace(0.0, 1.0, sample_n):
            x = float(p1[0] + s * (p2[0] - p1[0]))
            y = float(p1[1] + s * (p2[1] - p1[1]))
            z = float(p1[2] + s * (p2[2] - p1[2]))
            r, c = km_to_rc(x, y, z_grid.shape[0], z_grid.shape[1])
            min_clearance = min(min_clearance, z - float(z_grid[r, c]))
            lid = int(round(float(p1[3]) + s * (float(p2[3]) - float(p1[3])))) if nodes.shape[1] >= 4 else 1
            risk, comm = fused_risk_at_point(x, y, z, z_grid, risk_fields, fallback_lid=lid)
            seg_risk += risk
            seg_comm += comm
        seg_risk /= sample_n
        seg_comm /= sample_n
        exposure += seg_risk * seg_len_km
        total_len_km += seg_len_km
        if (not has_comm) or seg_comm <= comm_threshold:
            covered_len_km += seg_len_km
            cur_lost_km = 0.0
        else:
            cur_lost_km += seg_len_km
            max_lost_km = max(max_lost_km, cur_lost_km)

    total_len_km = max(total_len_km, EPS)
    return {
        "min_clearance_m": float(min_clearance) if np.isfinite(min_clearance) else float("nan"),
        "risk_exposure_integral": float(exposure),
        "comm_coverage_ratio": float(covered_len_km / total_len_km),
        "max_comm_loss_time_s": float(max_lost_km * 1000.0 / UAV_SPEED),
        "max_comm_loss_length_km": float(max_lost_km),
    }


@dataclass
class WeightedGraph:
    name: str
    nodes: np.ndarray
    edge_pairs: np.ndarray
    edge_types: np.ndarray
    edge_weight: np.ndarray
    edge_time_raw: np.ndarray
    edge_energy_raw: np.ndarray
    edge_risk_raw: np.ndarray
    t_max: float
    e_max: float
    r_max: float
    adj: List[List[Tuple[int, int]]]
    pair_to_eid: Dict[Tuple[int, int], int]
    edge_midpoints: np.ndarray
    edge_mid_tree: cKDTree
    z_grid: Optional[np.ndarray] = None
    risk_fields: Optional[Dict[str, object]] = None

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_pairs.shape[0])

    def edge_id(self, u: int, v: int) -> Optional[int]:
        return self.pair_to_eid.get(normalize_pair(int(u), int(v)))

    def blocked_eids(self, blocked_pairs: Sequence[Tuple[int, int]]) -> set:
        out = set()
        for u, v in blocked_pairs:
            eid = self.edge_id(u, v)
            if eid is not None:
                out.add(eid)
        return out

    def path_metrics(self, path: Sequence[int]) -> Dict[str, float]:
        if not path or len(path) < 2:
            return {
                "cost": float("inf"),
                "time_s": float("inf"),
                "energy_kj": float("inf"),
                "risk": float("inf"),
                "length_km": float("inf"),
            }
        total_cost = 0.0
        total_t = 0.0
        total_e = 0.0
        total_r = 0.0
        total_len_km = 0.0
        for i in range(len(path) - 1):
            u = int(path[i])
            v = int(path[i + 1])
            eid = self.edge_id(u, v)
            if eid is None:
                return {
                    "cost": float("inf"),
                    "time_s": float("inf"),
                    "energy_kj": float("inf"),
                    "risk": float("inf"),
                    "length_km": float("inf"),
                }
            total_cost += float(self.edge_weight[eid])
            total_t += float(self.edge_time_raw[eid])
            total_e += float(self.edge_energy_raw[eid])
            total_r += float(self.edge_risk_raw[eid])
            dx = (self.nodes[v, 0] - self.nodes[u, 0]) * 1000.0
            dy = (self.nodes[v, 1] - self.nodes[u, 1]) * 1000.0
            dz = self.nodes[v, 2] - self.nodes[u, 2]
            total_len_km += float(np.sqrt(dx * dx + dy * dy + dz * dz) / 1000.0)
        out = {
            "cost": total_cost,
            "time_s": total_t,
            "energy_kj": total_e,
            "risk": total_r,
            "length_km": total_len_km,
        }
        out.update(compute_node_path_extra_metrics(self.nodes, path, self.z_grid, self.risk_fields))
        return out

    def heuristic(self, s: int, goal: int) -> float:
        dx = (self.nodes[goal, 0] - self.nodes[s, 0]) * 1000.0
        dy = (self.nodes[goal, 1] - self.nodes[s, 1]) * 1000.0
        dz = self.nodes[goal, 2] - self.nodes[s, 2]
        d3d = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        t_lb = d3d / UAV_SPEED
        climb_lb = max(0.0, dz) * 9.8 * UAV_MASS
        e_lb = (UAV_POWER * t_lb + climb_lb) / 1000.0
        return ALPHA * (t_lb / (self.t_max + EPS)) + BETA * (e_lb / (self.e_max + EPS))


def compute_edge_costs(
    nodes: np.ndarray,
    edge_pairs: np.ndarray,
    z_grid: np.ndarray,
    risk_fields: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    m = edge_pairs.shape[0]
    t_raw = np.zeros(m, dtype=float)
    e_raw = np.zeros(m, dtype=float)
    r_raw = np.zeros(m, dtype=float)
    rows, cols = z_grid.shape
    risk_mode = "terrain_only"
    risk_l1 = None
    risk_l2 = None
    risk_l3 = None
    risk_l4 = None
    risk_human = None
    risk_comm = None
    has_comm = False
    risk_weights = {"terrain": 1.0, "human": 0.0, "communication": 0.0}
    if risk_fields:
        risk_mode = str(risk_fields.get("mode", "terrain_only"))
        risk_l1 = risk_fields.get("risk_l1")
        risk_l2 = risk_fields.get("risk_l2")
        risk_l3 = risk_fields.get("risk_l3")
        risk_l4 = risk_fields.get("risk_l4")
        risk_human = risk_fields.get("risk_human")
        risk_comm = risk_fields.get("risk_comm")
        has_comm = bool(risk_fields.get("has_comm", False))
        risk_weights = dict(risk_fields.get("risk_weights", risk_weights))

    for k in range(m):
        i = int(edge_pairs[k, 0])
        j = int(edge_pairs[k, 1])
        xi, yi, zi = nodes[i, 0], nodes[i, 1], nodes[i, 2]
        xj, yj, zj = nodes[j, 0], nodes[j, 1], nodes[j, 2]
        dh = float(np.linalg.norm([xj - xi, yj - yi]) * 1000.0)
        dv = float(abs(zj - zi))
        d3d = float(np.sqrt(dh * dh + dv * dv))
        t = d3d / UAV_SPEED
        e = (UAV_POWER * t + max(0.0, zj - zi) * 9.8 * UAV_MASS) / 1000.0
        risk = 0.0
        for s in np.linspace(0.0, 1.0, RISK_SAMPLES):
            x = xi + s * (xj - xi)
            y = yi + s * (yj - yi)
            z = zi + s * (zj - zi)
            r, c = km_to_rc(float(x), float(y), rows, cols)
            terrain = float(z_grid[r, c])
            r_terrain = max(0.0, 1.0 - (z - terrain) / 200.0)
            r_human = 0.0
            if (
                risk_mode == "terrain_trail_hotspot"
                and risk_l1 is not None
                and risk_l2 is not None
                and risk_l3 is not None
                and risk_l4 is not None
            ):
                split_sum = max(RISK_W_L1 + RISK_W_L2 + RISK_W_L3 + RISK_W_L4, EPS)
                r_human = (
                    (RISK_W_L1 / split_sum) * float(risk_l1[r, c])
                    + (RISK_W_L2 / split_sum) * float(risk_l2[r, c])
                    + (RISK_W_L3 / split_sum) * float(risk_l3[r, c])
                    + (RISK_W_L4 / split_sum) * float(risk_l4[r, c])
                )
            elif risk_mode == "terrain_human_combined" and risk_human is not None:
                r_human = float(risk_human[r, c])
            r_comm = 0.0
            if has_comm and risk_comm is not None:
                fallback_lid = int(round(nodes[i, 3] + s * (nodes[j, 3] - nodes[i, 3]))) if nodes.shape[1] >= 4 else 1
                fallback_lid = int(np.clip(fallback_lid, 0, 2))
                r_comm = float(risk_comm[fallback_lid, r, c])
            r_total = (
                float(risk_weights.get("terrain", 1.0)) * r_terrain
                + float(risk_weights.get("human", 0.0)) * r_human
                + float(risk_weights.get("communication", 0.0)) * r_comm
            )
            risk += float(np.clip(r_total, 0.0, 1.0))
        risk /= RISK_SAMPLES
        t_raw[k] = t
        e_raw[k] = e
        r_raw[k] = risk

    t_max = float(np.max(t_raw) + EPS)
    e_max = float(np.max(e_raw) + EPS)
    r_max = float(np.max(r_raw) + EPS)
    w = ALPHA * (t_raw / t_max) + BETA * (e_raw / e_max) + GAMMA * (r_raw / r_max)
    return w, t_raw, e_raw, r_raw, t_max, e_max, r_max


def build_weighted_graph(
    name: str,
    nodes: np.ndarray,
    edges: np.ndarray,
    z_grid: np.ndarray,
    risk_fields: Optional[Dict[str, object]] = None,
) -> WeightedGraph:
    edge_pairs = np.asarray(edges[:, :2], dtype=int)
    if edges.shape[1] >= 3:
        edge_types = np.asarray(edges[:, 2], dtype=int)
    else:
        edge_types = np.zeros(len(edge_pairs), dtype=int)

    w, t_raw, e_raw, r_raw, t_max, e_max, r_max = compute_edge_costs(
        nodes, edge_pairs, z_grid, risk_fields=risk_fields
    )
    n = int(nodes.shape[0])
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    pair_to_eid: Dict[Tuple[int, int], int] = {}
    for eid, (u, v) in enumerate(edge_pairs):
        uu = int(u)
        vv = int(v)
        adj[uu].append((vv, eid))
        adj[vv].append((uu, eid))
        pair_to_eid[normalize_pair(uu, vv)] = eid

    mids = (nodes[edge_pairs[:, 0], :2] + nodes[edge_pairs[:, 1], :2]) * 0.5
    mid_tree = cKDTree(mids)

    return WeightedGraph(
        name=name,
        nodes=np.asarray(nodes, dtype=float),
        edge_pairs=edge_pairs,
        edge_types=edge_types,
        edge_weight=w,
        edge_time_raw=t_raw,
        edge_energy_raw=e_raw,
        edge_risk_raw=r_raw,
        t_max=t_max,
        e_max=e_max,
        r_max=r_max,
        adj=adj,
        pair_to_eid=pair_to_eid,
        edge_midpoints=mids,
        edge_mid_tree=mid_tree,
        z_grid=z_grid,
        risk_fields=risk_fields,
    )


def event_cost_for_eid(graph: WeightedGraph, eid: int, area_event: AreaEvent) -> float:
    """按动态区域事件类型计算某条边的新代价。"""
    return event_edge_cost(
        area_event.event_type,
        area_event.severity,
        (
            float(graph.edge_time_raw[eid]),
            float(graph.edge_energy_raw[eid]),
            float(graph.edge_risk_raw[eid]),
        ),
        (graph.t_max, graph.e_max, graph.r_max),
        (ALPHA, BETA, GAMMA),
    )


def area_event_cost_overrides(graph: WeightedGraph, area_events: Sequence[AreaEvent]) -> Dict[int, float]:
    """把一个或多个区域事件映射成 eid -> override cost。"""
    overrides: Dict[int, float] = {}
    for area_event in area_events:
        for u, v in area_event.affected_edges:
            eid = graph.edge_id(int(u), int(v))
            if eid is None:
                continue
            new_cost = event_cost_for_eid(graph, eid, area_event)
            old_cost = overrides.get(eid)
            if old_cost is None:
                overrides[eid] = float(new_cost)
            elif not np.isfinite(old_cost) or not np.isfinite(new_cost):
                overrides[eid] = float("inf")
            else:
                # 多个软扰动叠加时取更保守的代价。
                overrides[eid] = max(float(old_cost), float(new_cost))
    return overrides


def build_single_layer_graph(
    layered_graph: WeightedGraph,
    z_grid: np.ndarray,
    z_offset_m: float = 75.0,
    intra_dist_m: float = 250.0,
    collision_samples: int = 10,
    risk_fields: Optional[Dict[str, object]] = None,
) -> WeightedGraph:
    """构建单层展平图（B3 基线）。

    【归一化基准差异说明】
    B3 单层图的边代价通过 compute_edge_costs() 独立计算，其 t_max/e_max/r_max
    归一化基准与 B4（分层图）不同，因为图结构和边集不同。
    因此，跨基线的路径代价（无量纲加权总代价）数值不可直接比较。
    论文对比表中应当说明此点，或统一使用 B4 的归一化基准对所有基线做标准化。
    """
    nodes = layered_graph.nodes.copy()
    rows, cols = z_grid.shape
    for i in range(nodes.shape[0]):
        x_km = float(nodes[i, 0])
        y_km = float(nodes[i, 1])
        r, c = km_to_rc(x_km, y_km, rows, cols)
        nodes[i, 2] = float(z_grid[r, c]) + z_offset_m
        nodes[i, 3] = 1.0

    xy = nodes[:, :2]
    tree = cKDTree(xy)
    radius_km = intra_dist_m / 1000.0
    pairs = tree.query_pairs(r=radius_km)
    edges: List[Tuple[int, int, int]] = []

    for i, j in pairs:
        dxy = float(np.linalg.norm(nodes[i, :2] - nodes[j, :2]))
        if dxy < 1e-6:
            continue
        if collision_free_segment(nodes[i], nodes[j], z_grid, n_samples=collision_samples):
            edges.append((int(i), int(j), 0))

    if not edges:
        raise RuntimeError("Single-layer graph construction failed: no edges generated.")

    edge_arr = np.asarray(edges, dtype=int)
    return build_weighted_graph("B3_single_layer", nodes, edge_arr, z_grid, risk_fields=risk_fields)


def build_regular_layered_graph(
    z_grid: np.ndarray,
    layer_mid: np.ndarray,
    layer_allowed: np.ndarray,
    terminal_status: dict,
    resolution_m: float,
    branch_budget: int,
    backbone_budget: int,
    risk_fields: Optional[Dict[str, object]] = None,
    floor_grid: Optional[np.ndarray] = None,
    ceiling_grid: Optional[np.ndarray] = None,
) -> WeightedGraph:
    """构建 B5：规则三层图 + LPA*，仅把地形驱动采样替换为规则网格采样。"""
    rows, cols = z_grid.shape

    def pixel_to_km(r: int, c: int) -> Tuple[float, float]:
        return (float(c) * resolution_m / 1000.0, float(rows - 1 - r) * resolution_m / 1000.0)

    def layer_height(lid: int, r: int, c: int) -> float:
        rr = int(np.clip(r, 0, rows - 1))
        cc = int(np.clip(c, 0, cols - 1))
        return float(layer_mid[int(lid), rr, cc])

    terminals = terminal_status.get("terminals", {})
    if not terminals:
        raise RuntimeError("缺少 terminal_status.terminals，无法构建 B5 规则三层图。")

    terminal_rcs: List[Tuple[int, int]] = []
    terminal_items: List[Tuple[str, dict]] = []
    for name, meta in terminals.items():
        if "row" not in meta or "col" not in meta:
            continue
        rr = int(meta["row"])
        cc = int(meta["col"])
        terminal_rcs.append((rr, cc))
        terminal_items.append((str(name), dict(meta)))
    if not terminal_items:
        raise RuntimeError("terminal_status 中缺少有效的 row/col 终端锚点信息。")

    exclude_rc = terminal_rcs
    branch_pts = select_regular_grid_points(layer_allowed[1].astype(bool), int(branch_budget), exclude=exclude_rc)
    backbone_pts = select_regular_grid_points(layer_allowed[2].astype(bool), int(backbone_budget), exclude=exclude_rc)
    if len(branch_pts) == 0 or len(backbone_pts) == 0:
        raise RuntimeError("B5 规则三层图构建失败：规则网格采样为空。")

    nodes: List[List[float]] = []
    terminal_pillars: Dict[str, List[int]] = {}
    for name, meta in terminal_items:
        rr = int(meta["row"])
        cc = int(meta["col"])
        x_km, y_km = pixel_to_km(rr, cc)
        pillar_idxs: List[int] = []
        for lid in range(3):
            pillar_idxs.append(len(nodes))
            nodes.append([x_km, y_km, layer_height(lid, rr, cc), float(lid)])
        terminal_pillars[name] = pillar_idxs

    branch_start = len(nodes)
    for rr, cc in branch_pts:
        x_km, y_km = pixel_to_km(int(rr), int(cc))
        nodes.append([x_km, y_km, layer_height(1, int(rr), int(cc)), 1.0])

    backbone_start = len(nodes)
    for rr, cc in backbone_pts:
        x_km, y_km = pixel_to_km(int(rr), int(cc))
        nodes.append([x_km, y_km, layer_height(2, int(rr), int(cc)), 2.0])

    node_arr = np.asarray(nodes, dtype=float)
    branch_idx_arr = np.arange(branch_start, backbone_start, dtype=int)
    backbone_idx_arr = np.arange(backbone_start, len(node_arr), dtype=int)
    branch_tree = cKDTree(node_arr[branch_idx_arr, :2])
    backbone_tree = cKDTree(node_arr[backbone_idx_arr, :2])
    edges: List[Tuple[int, int, int]] = []

    for pillar in terminal_pillars.values():
        for k in range(len(pillar) - 1):
            edges.append((int(pillar[k]), int(pillar[k + 1]), 1))

    def connect_anchor_to_layer(anchor_idx: int, candidate_indices: np.ndarray, tree: cKDTree) -> None:
        anchor_xy = node_arr[int(anchor_idx), :2]
        for radius_m in REGULAR_PILLAR_CONNECT_RADII_M:
            radius_km = float(radius_m) / 1000.0
            cand_locals = tree.query_ball_point(anchor_xy, r=radius_km)
            cand_locals = sorted(
                cand_locals,
                key=lambda local: float(np.linalg.norm(anchor_xy - node_arr[int(candidate_indices[int(local)]), :2])),
            )
            for local in cand_locals:
                cand_idx = int(candidate_indices[int(local)])
                if corridor_collision_free_segment(
                    node_arr[int(anchor_idx)],
                    node_arr[cand_idx],
                    z_grid,
                    floor_grid=floor_grid,
                    ceiling_grid=ceiling_grid,
                    layer_allowed=layer_allowed,
                    n_samples=COLLISION_SAMPLES,
                ):
                    edges.append((int(anchor_idx), cand_idx, 0))
                    return

    for pillar in terminal_pillars.values():
        connect_anchor_to_layer(int(pillar[1]), branch_idx_arr, branch_tree)
        connect_anchor_to_layer(int(pillar[2]), backbone_idx_arr, backbone_tree)

    terminal_l0 = [pillar[0] for pillar in terminal_pillars.values()]
    for i in range(len(terminal_l0)):
        for j in range(i + 1, len(terminal_l0)):
            u = int(terminal_l0[i])
            v = int(terminal_l0[j])
            if np.linalg.norm(node_arr[u, :2] - node_arr[v, :2]) > REGULAR_INTRA_EDGE_DIST_M / 1000.0:
                continue
            if corridor_collision_free_segment(
                node_arr[u],
                node_arr[v],
                z_grid,
                floor_grid=floor_grid,
                ceiling_grid=ceiling_grid,
                layer_allowed=layer_allowed,
                n_samples=COLLISION_SAMPLES,
            ):
                edges.append((u, v, 0))

    def add_intra_edges(idx_start: int, idx_end: int, max_dist_m: float) -> None:
        idx = np.arange(int(idx_start), int(idx_end), dtype=int)
        if len(idx) < 2:
            return
        tree = cKDTree(node_arr[idx, :2])
        for local_i, local_j in tree.query_pairs(r=float(max_dist_m) / 1000.0):
            u = int(idx[int(local_i)])
            v = int(idx[int(local_j)])
            if corridor_collision_free_segment(
                node_arr[u],
                node_arr[v],
                z_grid,
                floor_grid=floor_grid,
                ceiling_grid=ceiling_grid,
                layer_allowed=layer_allowed,
                n_samples=COLLISION_SAMPLES,
            ):
                edges.append((u, v, 0))

    add_intra_edges(branch_start, backbone_start, REGULAR_INTRA_EDGE_DIST_M)
    add_intra_edges(backbone_start, len(node_arr), REGULAR_INTRA_EDGE_DIST_M)

    max_inter_km = REGULAR_INTER_EDGE_DIST_M / 1000.0
    max_angle_rad = np.radians(REGULAR_MAX_CLIMB_ANGLE_DEG)
    backbone_xy = node_arr[backbone_idx_arr, :2]
    neighbor_lists = cKDTree(backbone_xy).query_ball_point(node_arr[branch_idx_arr, :2], r=max_inter_km)
    for b_local, cand_locals in enumerate(neighbor_lists):
        if not cand_locals:
            continue
        u = int(branch_idx_arr[int(b_local)])
        if len(cand_locals) > REGULAR_MAX_INTER_NEIGHBORS:
            cand_locals = sorted(
                cand_locals,
                key=lambda c_local: float(np.linalg.norm(node_arr[u, :2] - backbone_xy[int(c_local)])),
            )[:REGULAR_MAX_INTER_NEIGHBORS]
        for c_local in cand_locals:
            v = int(backbone_idx_arr[int(c_local)])
            horiz_km = float(np.linalg.norm(node_arr[u, :2] - node_arr[v, :2]))
            if horiz_km <= 1e-9:
                continue
            vert_m = abs(float(node_arr[u, 2] - node_arr[v, 2]))
            if math.atan2(vert_m, horiz_km * 1000.0) > max_angle_rad:
                continue
            if corridor_collision_free_segment(
                node_arr[u],
                node_arr[v],
                z_grid,
                floor_grid=floor_grid,
                ceiling_grid=ceiling_grid,
                layer_allowed=layer_allowed,
                n_samples=COLLISION_SAMPLES,
            ):
                edges.append((u, v, 2))

    if not edges:
        raise RuntimeError("B5 规则三层图构建失败：未生成任何边。")

    edge_arr = np.asarray(edges, dtype=int)
    return build_weighted_graph("B5_regular_layered", node_arr, edge_arr, z_grid, risk_fields=risk_fields)


class LPAStarPlanner:
    def __init__(self, graph: WeightedGraph, start: int, goal: int):
        self.gp = graph
        self.start = int(start)
        self.goal = int(goal)
        self.n = self.gp.n_nodes
        self.inf = float("inf")

        self.g = np.full(self.n, self.inf, dtype=float)
        self.rhs = np.full(self.n, self.inf, dtype=float)
        self.rhs[self.start] = 0.0

        self._counter = 0
        self._heap: List[Tuple[float, float, int, int]] = []
        self._in_heap: Dict[int, Tuple[float, float]] = {}
        self.blocked_eids: set = set()
        self.override_cost_by_eid: Dict[int, float] = {}

        self.nodes_expanded = 0
        self.expanded_order: List[int] = []
        self.total_expanded = 0
        self.total_queue_pushes = 0
        self.total_queue_pops = 0
        self.total_queue_stale_pops = 0
        self.total_updated_vertices = 0
        self.total_reopened_states = 0
        self._push(self.start, self._calc_key(self.start))

    def _cost_by_eid(self, eid: int) -> float:
        if eid in self.override_cost_by_eid:
            return float(self.override_cost_by_eid[eid])
        if eid in self.blocked_eids:
            return self.inf
        return float(self.gp.edge_weight[eid])

    def _calc_key(self, s: int) -> Tuple[float, float]:
        base = min(self.g[s], self.rhs[s])
        return (base + self.gp.heuristic(s, self.goal), base)

    def _push(self, node: int, key: Tuple[float, float]) -> None:
        self._counter += 1
        heapq.heappush(self._heap, (key[0], key[1], self._counter, int(node)))
        self._in_heap[int(node)] = key
        self.total_queue_pushes += 1

    def _pop(self) -> Tuple[Optional[int], Optional[Tuple[float, float]]]:
        while self._heap:
            k1, k2, _, node = heapq.heappop(self._heap)
            self.total_queue_pops += 1
            cur = self._in_heap.get(node)
            if cur is not None and cur == (k1, k2):
                del self._in_heap[node]
                return node, (k1, k2)
            self.total_queue_stale_pops += 1
        return None, None

    def _top_key(self) -> Tuple[float, float]:
        while self._heap:
            k1, k2, _, node = self._heap[0]
            cur = self._in_heap.get(node)
            if cur is not None and cur == (k1, k2):
                return (k1, k2)
            heapq.heappop(self._heap)
        return (self.inf, self.inf)

    def update_vertex(self, u: int) -> None:
        self.total_updated_vertices += 1
        if u != self.start:
            best = self.inf
            for p, eid in self.gp.adj[u]:
                c = self._cost_by_eid(eid)
                cand = self.g[p] + c
                if cand < best:
                    best = cand
            self.rhs[u] = best

        if u in self._in_heap:
            del self._in_heap[u]
        if self.g[u] != self.rhs[u]:
            self._push(u, self._calc_key(u))

    def block_edges(self, blocked_pairs: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
        n_added = 0
        affected = set()
        for u, v in blocked_pairs:
            eid = self.gp.edge_id(u, v)
            if eid is None:
                continue
            if eid in self.blocked_eids:
                continue
            self.blocked_eids.add(eid)
            n_added += 1
            affected.add(int(u))
            affected.add(int(v))
            self.update_vertex(int(u))
            self.update_vertex(int(v))
        return n_added, len(affected)

    def apply_area_event(self, area_event: AreaEvent) -> Tuple[int, int]:
        """把区域事件应用到当前增量规划器。"""
        overrides = area_event_cost_overrides(self.gp, [area_event])
        affected = set()
        n_changed = 0
        for eid, new_cost in overrides.items():
            old_cost = self.override_cost_by_eid.get(eid, float(self.gp.edge_weight[eid]))
            if np.isclose(old_cost, new_cost, rtol=1e-12, atol=1e-12, equal_nan=True):
                continue
            self.override_cost_by_eid[int(eid)] = float(new_cost)
            u, v = int(self.gp.edge_pairs[eid, 0]), int(self.gp.edge_pairs[eid, 1])
            affected.add(u)
            affected.add(v)
            n_changed += 1
            self.update_vertex(u)
            self.update_vertex(v)
        return n_changed, len(affected)

    def compute_shortest_path(self) -> bool:
        self.nodes_expanded = 0
        self.expanded_order = []
        while True:
            top_key = self._top_key()
            goal_key = self._calc_key(self.goal)
            if top_key >= goal_key and self.g[self.goal] == self.rhs[self.goal]:
                break
            if top_key[0] == self.inf:
                break

            u, k_old = self._pop()
            if u is None:
                break
            self.nodes_expanded += 1
            self.total_expanded += 1
            self.expanded_order.append(u)
            k_new = self._calc_key(u)

            if k_old < k_new:
                self._push(u, k_new)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s, _ in self.gp.adj[u]:
                    self.update_vertex(s)
            else:
                self.g[u] = self.inf
                self.total_reopened_states += 1
                self.update_vertex(u)
                for s, _ in self.gp.adj[u]:
                    self.update_vertex(s)
        return self.g[self.goal] < self.inf

    def counter_snapshot(self) -> Dict[str, int]:
        return {
            "expanded": int(self.total_expanded),
            "queue_pushes": int(self.total_queue_pushes),
            "queue_pops": int(self.total_queue_pops),
            "queue_stale_pops": int(self.total_queue_stale_pops),
            "updated_vertices": int(self.total_updated_vertices),
            "reopened_states": int(self.total_reopened_states),
        }

    def extract_path(self) -> List[int]:
        if self.g[self.goal] == self.inf:
            return []
        path = [self.goal]
        cur = self.goal
        seen = set()
        while cur != self.start:
            seen.add(cur)
            best_pred = None
            best_val = self.inf
            for p, eid in self.gp.adj[cur]:
                c = self._cost_by_eid(eid)
                val = self.g[p] + c
                if val < best_val:
                    best_val = val
                    best_pred = p
            if best_pred is None or best_pred in seen:
                break
            path.append(int(best_pred))
            cur = int(best_pred)
        path.reverse()
        if path and path[0] == self.start:
            return path
        return []


def astar_global_replan(
    graph: WeightedGraph,
    start: int,
    goal: int,
    area_events: Optional[Sequence[AreaEvent]] = None,
    blocked_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[bool, List[int], Dict[str, int]]:
    blocked_eids = graph.blocked_eids(blocked_pairs or [])
    override_cost = area_event_cost_overrides(graph, area_events or [])
    n = graph.n_nodes
    start = int(start)
    goal = int(goal)
    dist = np.full(n, float("inf"), dtype=float)
    prev = np.full(n, -1, dtype=int)
    closed = np.zeros(n, dtype=bool)
    dist[start] = 0.0
    heap: List[Tuple[float, float, int]] = [(graph.heuristic(start, goal), 0.0, start)]
    expanded = 0
    queue_pushes = 1
    queue_pops = 0
    updated_vertices = 0
    reopened_states = 0

    while heap:
        queue_pops += 1
        f, g, u = heapq.heappop(heap)
        _ = f
        if g > dist[u] + EPS:
            continue
        if closed[u]:
            reopened_states += 1
            continue
        closed[u] = True
        expanded += 1
        if u == goal:
            break
        for v, eid in graph.adj[u]:
            if eid in override_cost:
                step_cost = float(override_cost[eid])
            elif eid in blocked_eids:
                step_cost = float("inf")
            else:
                step_cost = float(graph.edge_weight[eid])
            if not np.isfinite(step_cost):
                continue
            ng = g + step_cost
            if ng + EPS < dist[v]:
                if closed[v]:
                    reopened_states += 1
                updated_vertices += 1
                dist[v] = ng
                prev[v] = u
                heapq.heappush(heap, (ng + graph.heuristic(v, goal), ng, v))
                queue_pushes += 1

    stats = {
        "expanded": int(expanded),
        "queue_pushes": int(queue_pushes),
        "queue_pops": int(queue_pops),
        "queue_stale_pops": 0,
        "updated_vertices": int(updated_vertices),
        "reopened_states": int(reopened_states),
    }
    if not np.isfinite(dist[goal]):
        return False, [], stats

    path = [goal]
    cur = goal
    seen = set()
    while cur != start:
        seen.add(cur)
        p = int(prev[cur])
        if p < 0 or p in seen:
            return False, [], stats
        path.append(p)
        cur = p
    path.reverse()
    return True, path, stats


class TraditionalVoxelDijkstra:
    """
    Coarse voxel baseline (B1): solve from scratch with Dijkstra.
    """

    def __init__(
        self,
        z_grid: np.ndarray,
        xy_step_m: float = 125.0,
        agl_low_m: float = 30.0,
        agl_high_m: float = 120.0,
        agl_step_m: float = 5.0,
    ):
        self.Z = np.asarray(z_grid, dtype=float)
        self.rows, self.cols = self.Z.shape
        self.xy_step_km = float(xy_step_m / 1000.0)
        self.xy_step_m = float(xy_step_m)
        self.agl_levels = np.arange(agl_low_m, agl_high_m + 1e-9, agl_step_m, dtype=float)
        self.nz = int(len(self.agl_levels))
        self.moves = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

        x_max = (self.cols - 1) * RESOLUTION / 1000.0
        y_max = (self.rows - 1) * RESOLUTION / 1000.0
        self.x_coords = np.arange(0.0, x_max + 1e-12, self.xy_step_km)
        self.y_coords = np.arange(0.0, y_max + 1e-12, self.xy_step_km)
        self.nx = int(len(self.x_coords))
        self.ny = int(len(self.y_coords))

        col_idx = np.clip((self.x_coords * 1000.0 / RESOLUTION).astype(int), 0, self.cols - 1)
        row_idx = np.clip(((self.rows - 1) - self.y_coords * 1000.0 / RESOLUTION).astype(int), 0, self.rows - 1)
        self.terrain = self.Z[np.ix_(row_idx, col_idx)]  # shape=(ny, nx)
        self.total_states = int(self.nx * self.ny * self.nz)

        xx, yy = np.meshgrid(self.x_coords, self.y_coords)
        self.X = xx
        self.Y = yy

    def _sid(self, ix: int, iy: int, iz: int) -> int:
        return int((iz * self.ny + iy) * self.nx + ix)

    def _decode(self, sid: int) -> Tuple[int, int, int]:
        ix = int(sid % self.nx)
        tmp = int(sid // self.nx)
        iy = int(tmp % self.ny)
        iz = int(tmp // self.ny)
        return ix, iy, iz

    def _nearest_state(self, x_km: float, y_km: float, z_m: float) -> Tuple[int, int, int]:
        ix = int(np.argmin(np.abs(self.x_coords - x_km)))
        iy = int(np.argmin(np.abs(self.y_coords - y_km)))
        terrain = float(self.terrain[iy, ix])
        agl = z_m - terrain
        iz = int(np.argmin(np.abs(self.agl_levels - agl)))
        return ix, iy, iz

    def build_storm_mask(self, storm_midpoints_xy: Sequence[Tuple[float, float]], radius_m: float = 220.0) -> np.ndarray:
        mask = np.zeros((self.ny, self.nx), dtype=bool)
        if not storm_midpoints_xy:
            return mask
        r2 = (radius_m / 1000.0) ** 2
        for mx, my in storm_midpoints_xy:
            dx = self.X - float(mx)
            dy = self.Y - float(my)
            mask |= (dx * dx + dy * dy) <= r2
        return mask

    def _state_xyz(self, ix: int, iy: int, iz: int) -> Tuple[float, float, float]:
        """返回体素状态 (ix,iy,iz) 对应的 (x_km, y_km, z_m) 坐标。"""
        x_km = float(self.x_coords[ix])
        y_km = float(self.y_coords[iy])
        z_m = float(self.terrain[iy, ix] + self.agl_levels[iz])
        return x_km, y_km, z_m

    def _compute_path_metrics(
        self,
        path_sids: List[int],
        risk_fields: Optional[Dict[str, object]] = None,
    ) -> Dict[str, float]:
        """沿已回溯的路径逐段事后计算时间、能耗、风险和多目标加权代价。

        使用与 B2/B3/B4 完全相同的代价公式，确保跨基线可比。
        """
        if len(path_sids) < 2:
            return {
                "path_len_km": float("inf"),
                "path_time_s": float("inf"),
                "path_energy_kj": float("inf"),
                "path_risk": float("inf"),
                "path_multi_cost": float("inf"),
                "min_clearance_m": float("nan"),
                "risk_exposure_integral": float("nan"),
                "comm_coverage_ratio": float("nan"),
                "max_comm_loss_time_s": float("nan"),
                "max_comm_loss_length_km": float("nan"),
            }

        # 解析风险场
        risk_mode = "terrain_only"
        risk_l1 = risk_l2 = risk_l3 = risk_l4 = risk_human = None
        risk_comm = None
        has_comm = False
        risk_weights = {"terrain": 1.0, "human": 0.0, "communication": 0.0}
        if risk_fields:
            risk_mode = str(risk_fields.get("mode", "terrain_only"))
            risk_l1 = risk_fields.get("risk_l1")
            risk_l2 = risk_fields.get("risk_l2")
            risk_l3 = risk_fields.get("risk_l3")
            risk_l4 = risk_fields.get("risk_l4")
            risk_human = risk_fields.get("risk_human")
            risk_comm = risk_fields.get("risk_comm")
            has_comm = bool(risk_fields.get("has_comm", False))
            risk_weights = dict(risk_fields.get("risk_weights", risk_weights))

        total_len = 0.0
        total_time = 0.0
        total_energy = 0.0
        total_risk = 0.0
        min_clearance = float("inf")
        risk_exposure = 0.0
        covered_len_km = 0.0
        cur_lost_km = 0.0
        max_lost_km = 0.0
        comm_threshold = float((risk_fields or {}).get("comm_risk_threshold", 0.55))
        n_segments = len(path_sids) - 1

        for k in range(n_segments):
            ix1, iy1, iz1 = self._decode(path_sids[k])
            ix2, iy2, iz2 = self._decode(path_sids[k + 1])
            x1, y1, z1 = self._state_xyz(ix1, iy1, iz1)
            x2, y2, z2 = self._state_xyz(ix2, iy2, iz2)

            dh = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 1000.0  # m
            dv = abs(z2 - z1)  # m
            d3d = math.sqrt(dh * dh + dv * dv)

            seg_time = d3d / UAV_SPEED
            climb = max(0.0, z2 - z1)
            seg_energy = (UAV_POWER * seg_time + climb * 9.8 * UAV_MASS) / 1000.0  # kJ

            # 沿线段采样风险（与 compute_edge_costs 一致）
            seg_risk = 0.0
            seg_comm = 0.0
            for s in np.linspace(0.0, 1.0, RISK_SAMPLES):
                x = x1 + s * (x2 - x1)
                y = y1 + s * (y2 - y1)
                z = z1 + s * (z2 - z1)
                r, c = km_to_rc(float(x), float(y), self.rows, self.cols)
                terrain_z = float(self.Z[r, c])
                min_clearance = min(min_clearance, z - terrain_z)
                r_terrain = max(0.0, 1.0 - (z - terrain_z) / 200.0)
                r_human = 0.0
                if (
                    risk_mode == "terrain_trail_hotspot"
                    and risk_l1 is not None
                    and risk_l2 is not None
                    and risk_l3 is not None
                    and risk_l4 is not None
                ):
                    split_sum = max(RISK_W_L1 + RISK_W_L2 + RISK_W_L3 + RISK_W_L4, EPS)
                    r_human = (
                        (RISK_W_L1 / split_sum) * float(risk_l1[r, c])
                        + (RISK_W_L2 / split_sum) * float(risk_l2[r, c])
                        + (RISK_W_L3 / split_sum) * float(risk_l3[r, c])
                        + (RISK_W_L4 / split_sum) * float(risk_l4[r, c])
                    )
                elif risk_mode == "terrain_human_combined" and risk_human is not None:
                    r_human = float(risk_human[r, c])
                r_comm = 0.0
                if has_comm and risk_comm is not None:
                    agl = z - terrain_z
                    lid = 0 if agl < 60.0 else (1 if agl < 90.0 else 2)
                    r_comm = float(risk_comm[lid, r, c])
                r_total = (
                    float(risk_weights.get("terrain", 1.0)) * r_terrain
                    + float(risk_weights.get("human", 0.0)) * r_human
                    + float(risk_weights.get("communication", 0.0)) * r_comm
                )
                seg_risk += float(np.clip(r_total, 0.0, 1.0))
                seg_comm += float(np.clip(r_comm, 0.0, 1.0))
            seg_risk /= RISK_SAMPLES
            seg_comm /= RISK_SAMPLES

            total_len += d3d / 1000.0  # km
            total_time += seg_time
            total_energy += seg_energy
            total_risk += seg_risk
            seg_len_km = d3d / 1000.0
            risk_exposure += seg_risk * seg_len_km
            if (not has_comm) or seg_comm <= comm_threshold:
                covered_len_km += seg_len_km
                cur_lost_km = 0.0
            else:
                cur_lost_km += seg_len_km
                max_lost_km = max(max_lost_km, cur_lost_km)

        # 归一化时取自身路径的单段最大值（与 compute_edge_costs 思路一致）
        t_max = max(total_time / max(n_segments, 1), EPS)
        e_max = max(total_energy / max(n_segments, 1), EPS)
        r_max = max(total_risk / max(n_segments, 1), EPS)
        multi_cost = (
            ALPHA * (total_time / (t_max * n_segments + EPS))
            + BETA * (total_energy / (e_max * n_segments + EPS))
            + GAMMA * (total_risk / (r_max * n_segments + EPS))
        ) * n_segments

        return {
            "path_len_km": total_len,
            "path_time_s": total_time,
            "path_energy_kj": total_energy,
            "path_risk": total_risk,
            "path_multi_cost": multi_cost,
            "min_clearance_m": float(min_clearance) if np.isfinite(min_clearance) else float("nan"),
            "risk_exposure_integral": float(risk_exposure),
            "comm_coverage_ratio": float(covered_len_km / max(total_len, EPS)),
            "max_comm_loss_time_s": float(max_lost_km * 1000.0 / UAV_SPEED),
            "max_comm_loss_length_km": float(max_lost_km),
        }

    def search(
        self,
        start_xyz: Tuple[float, float, float],
        goal_xyz: Tuple[float, float, float],
        storm_mask_xy: np.ndarray,
        timeout_s: float,
        max_expansions: int = 2_000_000,
        risk_fields: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Dijkstra 搜索并事后计算多目标代价。

        返回字典包含：ok, path_len_km, expanded, path_time_s,
        path_energy_kj, path_risk, path_multi_cost。
        """
        sx, sy, sz = start_xyz
        gx, gy, gz = goal_xyz
        six, siy, siz = self._nearest_state(float(sx), float(sy), float(sz))
        gix, giy, giz = self._nearest_state(float(gx), float(gy), float(gz))
        start_sid = self._sid(six, siy, siz)
        goal_sid = self._sid(gix, giy, giz)

        # 确保起终点可通行
        storm_mask_xy = storm_mask_xy.copy()
        storm_mask_xy[siy, six] = False
        storm_mask_xy[giy, gix] = False

        dist = np.full(self.total_states, float("inf"), dtype=float)
        prev = np.full(self.total_states, -1, dtype=int)
        visited = np.zeros(self.total_states, dtype=bool)
        dist[start_sid] = 0.0
        heap: List[Tuple[float, int]] = [(0.0, start_sid)]

        expanded = 0
        t0 = time.perf_counter()
        fail = {"ok": False, "path_len_km": float("inf"), "expanded": 0,
                "path_time_s": float("nan"), "path_energy_kj": float("nan"),
                "path_risk": float("nan"), "path_multi_cost": float("nan"),
                "min_clearance_m": float("nan"), "risk_exposure_integral": float("nan"),
                "comm_coverage_ratio": float("nan"), "max_comm_loss_time_s": float("nan"),
                "max_comm_loss_length_km": float("nan"), "failure_reason": "voxel_unreachable"}

        while heap:
            if (time.perf_counter() - t0) > timeout_s:
                fail["expanded"] = expanded
                fail["failure_reason"] = "voxel_timeout"
                return fail
            d, sid = heapq.heappop(heap)
            if visited[sid]:
                continue
            if d > dist[sid] + EPS:
                continue
            visited[sid] = True

            if sid == goal_sid:
                # 回溯路径
                path_sids: List[int] = [goal_sid]
                cur = goal_sid
                while cur != start_sid:
                    cur = int(prev[cur])
                    if cur < 0:
                        break
                    path_sids.append(cur)
                path_sids.reverse()

                # 事后计算完整指标
                metrics = self._compute_path_metrics(path_sids, risk_fields=risk_fields)
                return {
                    "ok": True,
                    "path_len_km": float(d / 1000.0),
                    "expanded": expanded,
                    "path_time_s": metrics["path_time_s"],
                    "path_energy_kj": metrics["path_energy_kj"],
                    "path_risk": metrics["path_risk"],
                    "path_multi_cost": metrics["path_multi_cost"],
                    "min_clearance_m": metrics["min_clearance_m"],
                    "risk_exposure_integral": metrics["risk_exposure_integral"],
                    "comm_coverage_ratio": metrics["comm_coverage_ratio"],
                    "max_comm_loss_time_s": metrics["max_comm_loss_time_s"],
                    "max_comm_loss_length_km": metrics["max_comm_loss_length_km"],
                }

            expanded += 1
            if expanded >= max_expansions:
                fail["expanded"] = expanded
                fail["failure_reason"] = "voxel_max_expansions"
                return fail

            ix, iy, iz = self._decode(sid)
            z1 = float(self.terrain[iy, ix] + self.agl_levels[iz])
            for dx, dy, dz in self.moves:
                nix = ix + dx
                niy = iy + dy
                niz = iz + dz
                if nix < 0 or nix >= self.nx or niy < 0 or niy >= self.ny or niz < 0 or niz >= self.nz:
                    continue
                if storm_mask_xy[niy, nix] and (nix != gix or niy != giy):
                    continue

                nsid = self._sid(nix, niy, niz)
                if visited[nsid]:
                    continue

                z2 = float(self.terrain[niy, nix] + self.agl_levels[niz])
                dxy = math.hypot(dx, dy) * self.xy_step_m
                dv = abs(z2 - z1)
                step = math.sqrt(dxy * dxy + dv * dv)
                nd = d + step
                if nd + EPS < dist[nsid]:
                    dist[nsid] = nd
                    prev[nsid] = sid
                    heapq.heappush(heap, (nd, nsid))

        fail["expanded"] = expanded
        return fail


def load_logistics_task_bundle(root: Path) -> Dict[str, object]:
    """读取任务生成器输出的物流任务集。"""
    path = root / "generated_tasks.json"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 {path}。请先运行 task_generator.py，确保 benchmark 使用物流任务而不是随机节点对。"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("tasks"):
        raise RuntimeError(f"{path} 中没有可用任务对。")
    return payload


def resolve_benchmark_data_context(args: argparse.Namespace, root: Path) -> Tuple[Dict[str, object], Path, bool]:
    """解析 benchmark 数据目录，并在默认场景存在时和 task_generator.py 保持一致。"""
    requested_scene = bool(str(getattr(args, "scenario_config", "")).strip())
    if requested_scene:
        scene_cfg = load_scenario_config(args.scenario_config or None, root)
        return scene_cfg, scenario_output_dir(scene_cfg, root), True

    if (root / "generated_tasks.json").exists():
        return {}, root, False

    if (root / DEFAULT_SCENE_CONFIG).exists():
        scene_cfg = load_scenario_config(None, root)
        return scene_cfg, scenario_output_dir(scene_cfg, root), True

    return {}, root, False


def resolve_scene_out_dir(raw_out_dir: str, scene_out: Path, workdir: Path) -> Path:
    """统一解析场景实验输出目录，避免 outputs/<scene> 被重复拼接。"""
    p = Path(raw_out_dir)
    if p.is_absolute():
        return p
    if str(p).replace("\\", "/").startswith("outputs/"):
        return (workdir / p).resolve()
    return (scene_out / p).resolve()


def resolve_output_dir(root: Path, out_dir_arg: str) -> Path:
    """兼容旧调用：无场景上下文时，相对路径按工作区根目录解析。"""
    return resolve_scene_out_dir(out_dir_arg, root, root)


def build_task_node_locator(graph: WeightedGraph) -> Dict[str, object]:
    """为物流任务坐标建立最近可用图节点查询结构。"""
    usable = np.asarray([i for i, adj_i in enumerate(graph.adj) if len(adj_i) > 0], dtype=int)
    if usable.size == 0:
        raise RuntimeError("图中没有可用于任务映射的连通节点。")
    loc: Dict[str, object] = {
        "all_idx": usable,
        "all_tree": cKDTree(graph.nodes[usable, :2]),
        "layer": {},
    }
    for lid in [1, 2, 0]:
        idx = np.asarray([i for i in usable if int(round(graph.nodes[i, 3])) == lid], dtype=int)
        if idx.size > 0:
            loc["layer"][lid] = (idx, cKDTree(graph.nodes[idx, :2]))
    return loc


def nearest_task_node(graph: WeightedGraph, locator: Dict[str, object], item: Dict[str, object], prefer: Sequence[int]) -> int:
    xy = np.array([float(item["x_km"]), float(item["y_km"])], dtype=float)
    for lid in prefer:
        layer_map = locator.get("layer", {})
        if lid not in layer_map:
            continue
        idx, tree = layer_map[lid]
        _d, local = tree.query(xy, k=1)
        return int(idx[int(local)])
    idx = locator["all_idx"]
    tree = locator["all_tree"]
    _d, local = tree.query(xy, k=1)
    return int(idx[int(local)])


def sample_logistics_start_goal(
    rng: np.random.Generator,
    graph: WeightedGraph,
    task_bundle: Dict[str, object],
    locator: Dict[str, object],
    min_dist_km: float,
    trial_index: int,
) -> Tuple[int, int, Dict[str, object]]:
    """按任务生成器给出的配送站-目标点任务对抽样。"""
    depots = {str(v["name"]): dict(v) for v in task_bundle.get("depots", [])}
    targets = {str(v["name"]): dict(v) for v in task_bundle.get("targets", [])}
    tasks = [dict(v) for v in task_bundle.get("tasks", [])]
    if not tasks:
        raise RuntimeError("任务集中没有 tasks。")
    offset = int(rng.integers(0, len(tasks)))
    for k in range(len(tasks)):
        task = tasks[(int(trial_index) - 1 + offset + k) % len(tasks)]
        depot = depots.get(str(task.get("depot")))
        target = targets.get(str(task.get("target")))
        if depot is None or target is None:
            continue
        start = nearest_task_node(graph, locator, depot, prefer=(1, 2, 0))
        goal = nearest_task_node(graph, locator, target, prefer=(1, 2, 0))
        if start == goal:
            continue
        d = float(np.linalg.norm(graph.nodes[start, :2] - graph.nodes[goal, :2]))
        if d < float(min_dist_km):
            continue
        task_meta = {
            **task,
            "start_node": int(start),
            "goal_node": int(goal),
            "start_xy_km": [float(graph.nodes[start, 0]), float(graph.nodes[start, 1])],
            "goal_xy_km": [float(graph.nodes[goal, 0]), float(graph.nodes[goal, 1])],
        }
        return int(start), int(goal), task_meta
    raise RuntimeError("没有找到可映射到当前图的物流任务对。")


def ci95(arr: np.ndarray) -> float:
    if arr.size <= 1:
        return float("nan")
    return float(1.96 * np.std(arr, ddof=1) / math.sqrt(arr.size))


def summarise_baseline(records: List[dict], baseline: str) -> dict:
    subset = [r for r in records if r["baseline"] == baseline]
    ok = [r for r in subset if r["success"]]
    failed = [r for r in subset if not r["success"]]
    failure_counts = Counter(str(r.get("failure_reason", "") or str(r.get("note", "")) or "unknown_failure") for r in failed)
    out = {
        "baseline": baseline,
        "n_trials": len(subset),
        "n_success": len(ok),
        "n_failed": len(failed),
        "failure_reason_top": "" if not failure_counts else f"{failure_counts.most_common(1)[0][0]}:{failure_counts.most_common(1)[0][1]}",
        "success_rate": (len(ok) / max(1, len(subset))),
        "mean_replan_ms": float("nan"),
        "std_replan_ms": float("nan"),
        "p50_replan_ms": float("nan"),
        "p95_replan_ms": float("nan"),
        "ci95_replan_ms": float("nan"),
        "mean_expanded": float("nan"),
        "std_expanded": float("nan"),
        "mean_cost": float("nan"),
        "mean_energy_kj": float("nan"),
        "mean_length_km": float("nan"),
        "mean_min_clearance_m": float("nan"),
        "mean_risk_exposure_integral": float("nan"),
        "mean_comm_coverage_ratio": float("nan"),
        "mean_max_comm_loss_time_s": float("nan"),
    }
    if not ok:
        return out

    ms = np.array([r["replan_ms"] for r in ok], dtype=float)
    ex = np.array([r["expanded"] for r in ok], dtype=float)
    c = np.array([r["path_cost"] for r in ok], dtype=float)
    e = np.array([r["path_energy_kj"] for r in ok], dtype=float)
    l = np.array([r["path_len_km"] for r in ok], dtype=float)
    clr = np.array([r.get("min_clearance_m", float("nan")) for r in ok], dtype=float)
    exp = np.array([r.get("risk_exposure_integral", float("nan")) for r in ok], dtype=float)
    cov = np.array([r.get("comm_coverage_ratio", float("nan")) for r in ok], dtype=float)
    loss = np.array([r.get("max_comm_loss_time_s", float("nan")) for r in ok], dtype=float)

    out.update(
        {
            "mean_replan_ms": float(np.mean(ms)),
            "std_replan_ms": float(np.std(ms, ddof=1) if ms.size > 1 else 0.0),
            "p50_replan_ms": float(np.percentile(ms, 50)),
            "p95_replan_ms": float(np.percentile(ms, 95)),
            "ci95_replan_ms": ci95(ms),
            "mean_expanded": float(np.mean(ex)),
            "std_expanded": float(np.std(ex, ddof=1) if ex.size > 1 else 0.0),
            "mean_cost": float(np.mean(c)),
            "mean_energy_kj": float(np.mean(e)),
            "mean_length_km": float(np.mean(l)),
            "mean_min_clearance_m": float(np.nanmean(clr)),
            "mean_risk_exposure_integral": float(np.nanmean(exp)),
            "mean_comm_coverage_ratio": float(np.nanmean(cov)),
            "mean_max_comm_loss_time_s": float(np.nanmean(loss)),
        }
    )
    return out


def build_structural_ablation_rows(summary_rows: List[dict]) -> List[dict]:
    """生成论文结构性消融 CSV，只保留五类明确方法。"""
    row_by_baseline = {str(r.get("baseline", "")): dict(r) for r in summary_rows}
    if BASELINE_B6 in row_by_baseline and BASELINE_B5 not in row_by_baseline:
        legacy_row = dict(row_by_baseline[BASELINE_B6])
        legacy_row["baseline"] = BASELINE_B5
        row_by_baseline[BASELINE_B5] = legacy_row

    rows: List[dict] = []
    for order, (baseline, method, role) in enumerate(STRUCTURAL_ABLATION_METHODS, start=1):
        src = row_by_baseline.get(baseline)
        if src is None:
            continue
        row = dict(src)
        row["method_order"] = order
        row["method"] = method
        row["baseline_id"] = baseline
        row["ablation_role"] = role
        rows.append(row)
    return rows


def paired_arrays(records: List[dict], a: str, b: str, field: str) -> Tuple[np.ndarray, np.ndarray]:
    by_trial = {}
    for r in records:
        if not r["success"]:
            continue
        by_trial[(r["trial"], r["baseline"])] = r
    aa: List[float] = []
    bb: List[float] = []
    trials = sorted({r["trial"] for r in records})
    for t in trials:
        ra = by_trial.get((t, a))
        rb = by_trial.get((t, b))
        if ra is None or rb is None:
            continue
        va = float(ra[field])
        vb = float(rb[field])
        if np.isfinite(va) and np.isfinite(vb):
            aa.append(va)
            bb.append(vb)
    return np.asarray(aa, dtype=float), np.asarray(bb, dtype=float)


def paired_significance(x: np.ndarray, y: np.ndarray) -> Dict[str, float | str]:
    """优先使用 Wilcoxon 符号秩检验；不可用时退化为配对 t 检验。"""
    if x.size < 2 or y.size < 2:
        return {"test_name": "na", "p_value": float("nan")}

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    diff = y_arr - x_arr

    if wilcoxon is not None:
        if np.allclose(diff, 0.0, rtol=1e-12, atol=1e-12):
            return {"test_name": "wilcoxon", "p_value": 1.0}
        try:
            res = wilcoxon(x_arr, y_arr, zero_method="wilcox", alternative="two-sided", mode="auto")
            return {"test_name": "wilcoxon", "p_value": float(res.pvalue)}
        except Exception:
            pass

    if ttest_rel is not None:
        try:
            res = ttest_rel(x_arr, y_arr, nan_policy="omit")
            return {"test_name": "paired_t", "p_value": float(res.pvalue)}
        except Exception:
            pass

    return {"test_name": "na", "p_value": float("nan")}


def paired_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    return float(paired_significance(x, y).get("p_value", float("nan")))


def write_csv(path: Path, rows: List[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def render_markdown(summary_rows: List[dict], pair_rows: List[dict], args: argparse.Namespace) -> str:
    lines: List[str] = []
    lines.append("# Benchmark Table")
    lines.append("")
    lines.append(
        f"- Trials requested: `{args.trials}`, random seed: `{args.seed}`, "
        f"area event: `{args.event_type}`, radius `{args.event_radius_km:.2f} km`, severity `{args.event_severity:.2f}`"
    )
    lines.append(
        f"- B1 voxel config: `xy_step={args.b1_xy_step_m:.0f}m`, "
        f"`agl_step={args.b1_agl_step_m:.0f}m`, timeout `{args.b1_timeout_s:.1f}s`"
    )
    lines.append("")
    lines.append("## Per-baseline summary")
    lines.append("")
    lines.append(
        "| Baseline | Success | Replan ms (mean+/-std) | P50/P95 ms | Expanded (mean) | Cost (mean) | Energy kJ (mean) | Length km (mean) | Min clearance m | Comm coverage | Max loss s | Risk exposure |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary_rows:
        succ = f"{r['n_success']}/{r['n_trials']} ({100.0*r['success_rate']:.1f}%)"
        ms = f"{r['mean_replan_ms']:.2f}+/-{r['std_replan_ms']:.2f}"
        p = f"{r['p50_replan_ms']:.2f}/{r['p95_replan_ms']:.2f}"
        lines.append(
            f"| {r['baseline']} | {succ} | {ms} | {p} | {r['mean_expanded']:.1f} | "
            f"{r['mean_cost']:.4f} | {r['mean_energy_kj']:.2f} | {r['mean_length_km']:.3f} | "
            f"{r['mean_min_clearance_m']:.1f} | {100.0*r['mean_comm_coverage_ratio']:.1f}% | "
            f"{r['mean_max_comm_loss_time_s']:.1f} | {r['mean_risk_exposure_integral']:.3f} |"
        )
    lines.append("")
    lines.append("## Paired significance checks")
    lines.append("")
    lines.append("| Pair | Metric | N paired | Mean A | Mean B | Median(B/A) | p-value |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in pair_rows:
        lines.append(
            f"| {r['pair']} | {r['metric']} | {r['n']} | {r['mean_a']:.4f} | {r['mean_b']:.4f} | "
            f"{r['median_ratio_b_over_a']:.3f} | {r['p_value']:.3e} |"
        )
    return "\n".join(lines) + "\n"


def render_four_baseline_markdown(summary_rows: List[dict], args: argparse.Namespace) -> str:
    label_map = {
        "B1_Voxel_Dijkstra": "B1 体素 Dijkstra",
        "B2_GlobalAstar_Layered": "B2 分层全局 A*",
        "B3_LPA_SingleLayer": "B3 单层展平 LPA*",
        "B4_Proposed_LPA_Layered": "B4 地形驱动三层 LPA*",
    }
    ordered = [
        "B1_Voxel_Dijkstra",
        "B2_GlobalAstar_Layered",
        "B3_LPA_SingleLayer",
        "B4_Proposed_LPA_Layered",
    ]
    row_by_baseline = {r["baseline"]: r for r in summary_rows}

    lines: List[str] = []
    lines.append("# 四基线对比表")
    lines.append("")
    lines.append(
        f"- 试验次数：`{args.trials}`，随机种子：`{args.seed}`，"
        f"区域事件：`{args.event_type}` / `{args.event_radius_km:.2f} km`"
    )
    lines.append("")
    lines.append("| 方法 | 规划时间 (ms) | 路径长度 (km) | 路径代价 | 能耗 (kJ) | 最小净空 (m) | 通信覆盖率 | 最长失联 (s) | 风险暴露 | 成功率 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in ordered:
        r = row_by_baseline.get(key)
        if r is None:
            lines.append(f"| {label_map[key]} | nan | nan | nan | nan | nan | nan | nan | nan | 0.0% |")
            continue
        t = float(r.get("mean_replan_ms", float("nan")))
        l = float(r.get("mean_length_km", float("nan")))
        c = float(r.get("mean_cost", float("nan")))
        e = float(r.get("mean_energy_kj", float("nan")))
        clr = float(r.get("mean_min_clearance_m", float("nan")))
        cov = float(r.get("mean_comm_coverage_ratio", float("nan")))
        loss = float(r.get("mean_max_comm_loss_time_s", float("nan")))
        exp = float(r.get("mean_risk_exposure_integral", float("nan")))
        succ = 100.0 * float(r.get("success_rate", 0.0))
        e_str = f"{e:.2f}" if np.isfinite(e) else "nan"
        cov_str = f"{100.0 * cov:.1f}%" if np.isfinite(cov) else "nan"
        lines.append(
            f"| {label_map[key]} | {t:.2f} | {l:.3f} | {c:.4f} | {e_str} | "
            f"{clr:.1f} | {cov_str} | {loss:.1f} | {exp:.3f} | {succ:.1f}% |"
        )
    lines.append("")
    lines.append(
        "Note: All four baselines use the same multi-objective weighted cost "
        "(α·Time + β·Energy + γ·Risk). B1 cost is computed post-hoc along its Dijkstra path. "
        "Cross-baseline cost comparison is valid for the same graph scale."
    )
    return "\n".join(lines) + "\n"


def render_benchmark_markdown_v2(summary_rows: List[dict], pair_rows: List[dict], args: argparse.Namespace) -> str:
    """增强版单次 benchmark 汇总表，显式给出中位数、P95 和检验类型。"""
    lines: List[str] = []
    lines.append("# Benchmark Table")
    lines.append("")
    lines.append(
        f"- Trials requested: `{args.trials}`, random seed: `{args.seed}`, "
        f"area event: `{args.event_type}`, radius `{args.event_radius_km:.2f} km`, severity `{args.event_severity:.2f}`"
    )
    lines.append(
        f"- B1 voxel config: `xy_step={args.b1_xy_step_m:.0f}m`, "
        f"`agl_step={args.b1_agl_step_m:.0f}m`, timeout `{args.b1_timeout_s:.1f}s`"
    )
    lines.append("")
    lines.append("## Per-baseline summary")
    lines.append("")
    lines.append(
        "| Baseline | Success | Replan ms (mean+/-std) | P50/P95 ms | Expanded (mean) | Cost (mean) | Energy kJ (mean) | Length km (mean) | Min clearance m | Comm coverage | Max loss s | Risk exposure |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary_rows:
        succ = f"{r['n_success']}/{r['n_trials']} ({100.0*r['success_rate']:.1f}%)"
        ms = f"{r['mean_replan_ms']:.2f}+/-{r['std_replan_ms']:.2f}"
        p = f"{r['p50_replan_ms']:.2f}/{r['p95_replan_ms']:.2f}"
        lines.append(
            f"| {r['baseline']} | {succ} | {ms} | {p} | {r['mean_expanded']:.1f} | "
            f"{r['mean_cost']:.4f} | {r['mean_energy_kj']:.2f} | {r['mean_length_km']:.3f} | "
            f"{r['mean_min_clearance_m']:.1f} | {100.0*r['mean_comm_coverage_ratio']:.1f}% | "
            f"{r['mean_max_comm_loss_time_s']:.1f} | {r['mean_risk_exposure_integral']:.3f} |"
        )
    lines.append("")
    lines.append("## Paired significance checks")
    lines.append("")
    lines.append("| Pair | Metric | N paired | Mean A | Mean B | P50 A | P50 B | P95 A | P95 B | Median(B/A) | Test | p-value |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for r in pair_rows:
        lines.append(
            f"| {r['pair']} | {r['metric']} | {r['n']} | {r['mean_a']:.4f} | {r['mean_b']:.4f} | "
            f"{r['median_a']:.4f} | {r['median_b']:.4f} | {r['p95_a']:.4f} | {r['p95_b']:.4f} | "
            f"{r['median_ratio_b_over_a']:.3f} | {r['test_name']} | {r['p_value']:.3e} |"
        )
    return "\n".join(lines) + "\n"


def render_single_event_comparison_markdown(summary_rows: List[dict], args: argparse.Namespace) -> str:
    """增强版多基线单事件对比表，纳入 B5 结构性消融。"""
    label_map = {
        "B1_Voxel_Dijkstra": "B1 Voxel",
        "B2_GlobalAstar_Layered": "B2 GlobalA*",
        "B3_LPA_SingleLayer": "B3 FlatLPA*",
        BASELINE_B5: "B5 RegularLayered LPA*",
        BASELINE_B6: "B6 RegularLayered LPA* (legacy)",
        "B4_Proposed_LPA_Layered": "B4 Proposed",
    }
    ordered = [
        "B1_Voxel_Dijkstra",
        "B2_GlobalAstar_Layered",
        "B3_LPA_SingleLayer",
        BASELINE_B5,
        "B4_Proposed_LPA_Layered",
    ]
    row_by_baseline = {r["baseline"]: r for r in summary_rows}

    lines: List[str] = []
    lines.append("# Single-Event Multi-Baseline Comparison Table")
    lines.append("")
    lines.append(
        f"- Trials: `{args.trials}`, seed: `{args.seed}`, "
        f"area event: `{args.event_type}` / `{args.event_radius_km:.2f} km`"
    )
    lines.append("")
    lines.append("| Method | Planning Time (mean+/-std, ms) | P50/P95 (ms) | Path Length (km) | Path Cost | Energy (kJ) | Min Clearance (m) | Comm Coverage | Max Loss (s) | Risk Exposure | Success Rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in ordered:
        r = row_by_baseline.get(key)
        if r is None:
            lines.append(f"| {label_map[key]} | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.0% |")
            continue
        t = float(r.get("mean_replan_ms", float("nan")))
        t_std = float(r.get("std_replan_ms", float("nan")))
        t_p50 = float(r.get("p50_replan_ms", float("nan")))
        t_p95 = float(r.get("p95_replan_ms", float("nan")))
        l = float(r.get("mean_length_km", float("nan")))
        c = float(r.get("mean_cost", float("nan")))
        e = float(r.get("mean_energy_kj", float("nan")))
        clr = float(r.get("mean_min_clearance_m", float("nan")))
        cov = float(r.get("mean_comm_coverage_ratio", float("nan")))
        loss = float(r.get("mean_max_comm_loss_time_s", float("nan")))
        exp = float(r.get("mean_risk_exposure_integral", float("nan")))
        succ = 100.0 * float(r.get("success_rate", 0.0))
        e_str = f"{e:.2f}" if np.isfinite(e) else "nan"
        cov_str = f"{100.0 * cov:.1f}%" if np.isfinite(cov) else "nan"
        lines.append(
            f"| {label_map[key]} | {t:.2f}+/-{t_std:.2f} | {t_p50:.2f}/{t_p95:.2f} | {l:.3f} | {c:.4f} | {e_str} | "
            f"{clr:.1f} | {cov_str} | {loss:.1f} | {exp:.3f} | {succ:.1f}% |"
        )
    lines.append("")
    lines.append(
        "Note: B5 keeps the same three-layer semantic structure and the same LPA* replanner as B4, "
        "but replaces terrain-driven node sampling with regular grid sampling. "
        "This isolates the contribution of terrain-aware layered network construction."
    )
    return "\n".join(lines) + "\n"


def render_benchmark_markdown_cn(summary_rows: List[dict], pair_rows: List[dict], args: argparse.Namespace) -> str:
    """中文版单次 benchmark 汇总表，显式给出中位数、P95 和显著性检验信息。"""
    label_map = {
        "B1_Voxel_Dijkstra": "B1 体素 Dijkstra",
        "B2_GlobalAstar_Layered": "B2 分层全局 A*",
        "B3_LPA_SingleLayer": "B3 单层展平 LPA*",
        "B4_Proposed_LPA_Layered": "B4 地形驱动三层 LPA*",
        BASELINE_B5: "B5 规则三层 LPA*",
        BASELINE_B6: "B6 规则三层 LPA*（旧命名）",
    }
    test_label = {
        "wilcoxon": "Wilcoxon",
        "paired_t": "配对 t 检验",
        "na": "不适用",
    }
    lines: List[str] = []
    lines.append("# 基准实验汇总表")
    lines.append("")
    lines.append(
        f"- 请求试验次数：`{args.trials}`，随机种子：`{args.seed}`，"
        f"区域事件：`{args.event_type}`，半径 `{args.event_radius_km:.2f} km`，强度 `{args.event_severity:.2f}`"
    )
    lines.append(
        f"- B1 体素参数：`xy_step={args.b1_xy_step_m:.0f}m`，"
        f"`agl_step={args.b1_agl_step_m:.0f}m`，超时 `{args.b1_timeout_s:.1f}s`"
    )
    lines.append("")
    lines.append("## 各基线统计")
    lines.append("")
    lines.append(
        "| 基线 | 成功情况 | 重规划时间 (均值+/-标准差, ms) | P50/P95 (ms) | 扩展节点均值 | 路径代价均值 | 能耗均值 (kJ) | 路径长度均值 (km) | 最小净空 (m) | 通信覆盖率 | 最长失联 (s) | 风险暴露 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary_rows:
        succ = f"{r['n_success']}/{r['n_trials']} ({100.0 * r['success_rate']:.1f}%)"
        ms = f"{r['mean_replan_ms']:.2f}+/-{r['std_replan_ms']:.2f}"
        p = f"{r['p50_replan_ms']:.2f}/{r['p95_replan_ms']:.2f}"
        baseline_label = label_map.get(str(r["baseline"]), str(r["baseline"]))
        lines.append(
            f"| {baseline_label} | {succ} | {ms} | {p} | {r['mean_expanded']:.1f} | "
            f"{r['mean_cost']:.4f} | {r['mean_energy_kj']:.2f} | {r['mean_length_km']:.3f} | "
            f"{r['mean_min_clearance_m']:.1f} | {100.0 * r['mean_comm_coverage_ratio']:.1f}% | "
            f"{r['mean_max_comm_loss_time_s']:.1f} | {r['mean_risk_exposure_integral']:.3f} |"
        )
    lines.append("")
    lines.append("## 配对显著性检验")
    lines.append("")
    lines.append("| 对比对 | 指标 | 配对样本数 | A 均值 | B 均值 | A 的 P50 | B 的 P50 | A 的 P95 | B 的 P95 | 中位数比值 (B/A) | 检验 | p 值 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for r in pair_rows:
        test_name = test_label.get(str(r["test_name"]), str(r["test_name"]))
        lines.append(
            f"| {r['pair']} | {r['metric']} | {r['n']} | {r['mean_a']:.4f} | {r['mean_b']:.4f} | "
            f"{r['median_a']:.4f} | {r['median_b']:.4f} | {r['p95_a']:.4f} | {r['p95_b']:.4f} | "
            f"{r['median_ratio_b_over_a']:.3f} | {test_name} | {r['p_value']:.3e} |"
        )
    return "\n".join(lines) + "\n"


def render_single_event_comparison_markdown_cn(summary_rows: List[dict], args: argparse.Namespace) -> str:
    """中文版多基线单事件对比表，纳入 B5 结构性消融。"""
    label_map = {
        "B1_Voxel_Dijkstra": "B1 体素 Dijkstra",
        "B2_GlobalAstar_Layered": "B2 分层全局 A*",
        "B3_LPA_SingleLayer": "B3 单层展平 LPA*",
        BASELINE_B5: "B5 规则三层 LPA*",
        BASELINE_B6: "B6 规则三层 LPA*（旧命名）",
        "B4_Proposed_LPA_Layered": "B4 地形驱动三层 LPA*",
    }
    ordered = [
        "B1_Voxel_Dijkstra",
        "B2_GlobalAstar_Layered",
        "B3_LPA_SingleLayer",
        BASELINE_B5,
        "B4_Proposed_LPA_Layered",
    ]
    row_by_baseline = {r["baseline"]: r for r in summary_rows}

    lines: List[str] = []
    lines.append("# 单事件多基线对比表")
    lines.append("")
    lines.append(
        f"- 试验次数：`{args.trials}`，随机种子：`{args.seed}`，"
        f"区域事件：`{args.event_type}` / `{args.event_radius_km:.2f} km`"
    )
    lines.append("")
    lines.append("| 方法 | 规划时间 (均值+/-标准差, ms) | P50/P95 (ms) | 路径长度 (km) | 路径代价 | 能耗 (kJ) | 最小净空 (m) | 通信覆盖率 | 最长失联 (s) | 风险暴露 | 成功率 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in ordered:
        r = row_by_baseline.get(key)
        if r is None:
            lines.append(f"| {label_map[key]} | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.0% |")
            continue
        t = float(r.get("mean_replan_ms", float("nan")))
        t_std = float(r.get("std_replan_ms", float("nan")))
        t_p50 = float(r.get("p50_replan_ms", float("nan")))
        t_p95 = float(r.get("p95_replan_ms", float("nan")))
        l = float(r.get("mean_length_km", float("nan")))
        c = float(r.get("mean_cost", float("nan")))
        e = float(r.get("mean_energy_kj", float("nan")))
        clr = float(r.get("mean_min_clearance_m", float("nan")))
        cov = float(r.get("mean_comm_coverage_ratio", float("nan")))
        loss = float(r.get("mean_max_comm_loss_time_s", float("nan")))
        exp = float(r.get("mean_risk_exposure_integral", float("nan")))
        succ = 100.0 * float(r.get("success_rate", 0.0))
        e_str = f"{e:.2f}" if np.isfinite(e) else "nan"
        cov_str = f"{100.0 * cov:.1f}%" if np.isfinite(cov) else "nan"
        lines.append(
            f"| {label_map[key]} | {t:.2f}+/-{t_std:.2f} | {t_p50:.2f}/{t_p95:.2f} | {l:.3f} | {c:.4f} | {e_str} | "
            f"{clr:.1f} | {cov_str} | {loss:.1f} | {exp:.3f} | {succ:.1f}% |"
        )
    lines.append("")
    lines.append(
        "说明：B5 保持与 B4 相同的三层语义结构和同一套 LPA* 重规划器，"
        "仅将地形驱动采样替换为规则网格采样，用来隔离“地形驱动分层航路网络构造”的贡献。"
    )
    return "\n".join(lines) + "\n"


def run_benchmark(args: argparse.Namespace) -> None:
    global RESOLUTION
    root = Path(args.workdir).resolve()
    scene_cfg, data_root, use_scene = resolve_benchmark_data_context(args, root)
    if use_scene and not str(getattr(args, "scenario_config", "")).strip():
        print(f"[场景] 使用默认场景输出目录: {data_root}")
    os.chdir(data_root)
    RESOLUTION = resolve_resolution_m(scene_cfg, data_root)
    out_dir = resolve_scene_out_dir(args.out_dir, data_root, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    z_grid = np.load("Z_crop.npy")
    layered_nodes = np.load("graph_nodes.npy")
    layered_edges = np.load("graph_edges.npy")
    risk_fields = load_risk_fields(data_root, z_grid.shape, scene_cfg if use_scene else {})
    print(
        f"[risk] mode={risk_fields['mode']}, comm={risk_fields['has_comm']}, "
        f"weights={risk_fields['risk_weights']}"
    )
    task_bundle = load_logistics_task_bundle(data_root)
    depots_by_name = {str(v["name"]): dict(v) for v in task_bundle.get("depots", [])}
    targets_by_name = {str(v["name"]): dict(v) for v in task_bundle.get("targets", [])}
    print(
        f"[tasks] logistics tasks={len(task_bundle.get('tasks', []))}, "
        f"depots={len(task_bundle.get('depots', []))}, targets={len(task_bundle.get('targets', []))}"
    )

    print("[build] loading layered graph...")
    layered_graph = build_weighted_graph(
        "B4_layered",
        layered_nodes,
        layered_edges,
        z_grid,
        risk_fields=risk_fields,
    )
    print(f"[build] layered graph: |V|={layered_graph.n_nodes}, |E|={layered_graph.n_edges}")
    layered_task_locator = build_task_node_locator(layered_graph)

    print("[build] building flattened single-layer graph for B3...")
    b3_graph = build_single_layer_graph(
        layered_graph,
        z_grid,
        z_offset_m=args.b3_z_offset_m,
        intra_dist_m=args.b3_intra_dist_m,
        collision_samples=args.b3_collision_samples,
        risk_fields=risk_fields,
    )
    print(f"[build] B3 graph: |V|={b3_graph.n_nodes}, |E|={b3_graph.n_edges}")

    b6_graph = None
    b6_task_locator = None
    terminal_status_path = data_root / "graph_terminal_status.json"
    layer_mid_path = data_root / "layer_mid.npy"
    layer_allowed_path = data_root / "layer_allowed.npy"
    if terminal_status_path.exists() and layer_mid_path.exists() and layer_allowed_path.exists():
        layer_mid = np.asarray(np.load(layer_mid_path), dtype=float)
        layer_allowed = np.asarray(np.load(layer_allowed_path), dtype=bool)
        floor_grid = np.asarray(np.load(data_root / "floor.npy"), dtype=float) if (data_root / "floor.npy").exists() else None
        ceiling_grid = np.asarray(np.load(data_root / "ceiling.npy"), dtype=float) if (data_root / "ceiling.npy").exists() else None
        terminal_status = json.loads(terminal_status_path.read_text(encoding="utf-8"))
        terminal_site_count = int(np.sum(np.rint(layered_nodes[:, 3]).astype(int) == 0))
        branch_regular_count = int(np.sum(np.rint(layered_nodes[:, 3]).astype(int) == 1) - terminal_site_count)
        backbone_regular_count = int(np.sum(np.rint(layered_nodes[:, 3]).astype(int) == 2) - terminal_site_count)
        print("[build] building regular layered graph for B5...")
        b6_graph = build_regular_layered_graph(
            z_grid,
            layer_mid,
            layer_allowed,
            terminal_status,
            resolution_m=RESOLUTION,
            branch_budget=branch_regular_count,
            backbone_budget=backbone_regular_count,
            risk_fields=risk_fields,
            floor_grid=floor_grid,
            ceiling_grid=ceiling_grid,
        )
        b6_task_locator = build_task_node_locator(b6_graph)
        print(f"[build] B5 regular layered graph: |V|={b6_graph.n_nodes}, |E|={b6_graph.n_edges}")

    voxel_planner = None
    if not args.skip_b1:
        print("[build] building coarse voxel planner for B1...")
        voxel_planner = TraditionalVoxelDijkstra(
            z_grid,
            xy_step_m=args.b1_xy_step_m,
            agl_low_m=args.b1_agl_low_m,
            agl_high_m=args.b1_agl_high_m,
            agl_step_m=args.b1_agl_step_m,
        )
        approx_nodes = voxel_planner.total_states
        print(
            "[build] B1 voxel states: "
            f"{approx_nodes} ({voxel_planner.nx}x{voxel_planner.ny}x{voxel_planner.nz})"
        )

    rng = np.random.default_rng(args.seed)
    records: List[dict] = []

    accepted = 0
    attempts = 0
    max_attempts = max(args.trials * args.max_attempt_factor, args.trials + 10)

    while accepted < args.trials and attempts < max_attempts:
        attempts += 1
        trial = accepted + 1
        try:
            start, goal, task_meta = sample_logistics_start_goal(
                rng,
                layered_graph,
                task_bundle,
                layered_task_locator,
                args.min_start_goal_dist_km,
                trial,
            )
        except RuntimeError:
            continue

        start_b6 = None
        goal_b6 = None
        if b6_graph is not None and b6_task_locator is not None:
            depot_item = depots_by_name.get(str(task_meta.get("depot", "")))
            target_item = targets_by_name.get(str(task_meta.get("target", "")))
            if depot_item is not None and target_item is not None:
                start_b6 = nearest_task_node(b6_graph, b6_task_locator, depot_item, prefer=(1, 2, 0))
                goal_b6 = nearest_task_node(b6_graph, b6_task_locator, target_item, prefer=(1, 2, 0))

        # B4 初始规划，用路径中段附近采样区域扰动中心。
        b4 = LPAStarPlanner(layered_graph, start, goal)
        ok_init_b4 = b4.compute_shortest_path()
        path_init_b4 = b4.extract_path() if ok_init_b4 else []
        if not ok_init_b4 or len(path_init_b4) < 3:
            continue

        area_event = build_area_event_from_path(
            layered_graph.nodes,
            layered_graph.edge_pairs,
            path_init_b4,
            rng,
            event_type=args.event_type,
            radius_km=args.event_radius_km,
            severity=args.event_severity,
        )
        if not area_event.affected_edges:
            continue

        # ---------- B4 (Proposed incremental LPA*) ----------
        b4.apply_area_event(area_event)
        t0 = time.perf_counter()
        ok_b4 = b4.compute_shortest_path()
        t1 = time.perf_counter()
        replan_b4_ms = (t1 - t0) * 1000.0
        path_b4 = b4.extract_path() if ok_b4 else []
        m_b4 = layered_graph.path_metrics(path_b4) if ok_b4 else {}

        records.append(
            {
                "trial": trial,
                "baseline": "B4_Proposed_LPA_Layered",
                "start_node": start,
                "goal_node": goal,
                "task_id": task_meta.get("task_id", ""),
                "task_depot": task_meta.get("depot", ""),
                "task_target": task_meta.get("target", ""),
                "task_stratum": task_meta.get("stratum", ""),
                "event_type": area_event.event_type,
                "event_center_x_km": area_event.center_x_km,
                "event_center_y_km": area_event.center_y_km,
                "event_radius_km": area_event.radius_km,
                "event_severity": area_event.severity,
                "event_affected_edges": len(area_event.affected_edges),
                "area_event_edges": json.dumps(area_event.affected_edges),
                "success": bool(ok_b4),
                "failure_reason": "" if ok_b4 else "event_path_disconnected",
                "replan_ms": replan_b4_ms if ok_b4 else float("nan"),
                "expanded": int(b4.nodes_expanded) if ok_b4 else -1,
                "path_cost": m_b4.get("cost", float("nan")),
                "path_energy_kj": m_b4.get("energy_kj", float("nan")),
                "path_len_km": m_b4.get("length_km", float("nan")),
                "min_clearance_m": m_b4.get("min_clearance_m", float("nan")),
                "risk_exposure_integral": m_b4.get("risk_exposure_integral", float("nan")),
                "comm_coverage_ratio": m_b4.get("comm_coverage_ratio", float("nan")),
                "max_comm_loss_time_s": m_b4.get("max_comm_loss_time_s", float("nan")),
                "max_comm_loss_length_km": m_b4.get("max_comm_loss_length_km", float("nan")),
                "note": "",
            }
        )

        # ---------- B2 (Global A* recompute) ----------
        t0 = time.perf_counter()
        ok_b2, path_b2, st_b2 = astar_global_replan(layered_graph, start, goal, area_events=[area_event])
        t1 = time.perf_counter()
        replan_b2_ms = (t1 - t0) * 1000.0
        m_b2 = layered_graph.path_metrics(path_b2) if ok_b2 else {}
        records.append(
            {
                "trial": trial,
                "baseline": "B2_GlobalAstar_Layered",
                "start_node": start,
                "goal_node": goal,
                "task_id": task_meta.get("task_id", ""),
                "task_depot": task_meta.get("depot", ""),
                "task_target": task_meta.get("target", ""),
                "task_stratum": task_meta.get("stratum", ""),
                "event_type": area_event.event_type,
                "event_center_x_km": area_event.center_x_km,
                "event_center_y_km": area_event.center_y_km,
                "event_radius_km": area_event.radius_km,
                "event_severity": area_event.severity,
                "event_affected_edges": len(area_event.affected_edges),
                "area_event_edges": json.dumps(area_event.affected_edges),
                "success": bool(ok_b2),
                "failure_reason": "" if ok_b2 else "event_path_disconnected",
                "replan_ms": replan_b2_ms if ok_b2 else float("nan"),
                "expanded": int(st_b2.get("expanded", -1)) if ok_b2 else -1,
                "path_cost": m_b2.get("cost", float("nan")),
                "path_energy_kj": m_b2.get("energy_kj", float("nan")),
                "path_len_km": m_b2.get("length_km", float("nan")),
                "min_clearance_m": m_b2.get("min_clearance_m", float("nan")),
                "risk_exposure_integral": m_b2.get("risk_exposure_integral", float("nan")),
                "comm_coverage_ratio": m_b2.get("comm_coverage_ratio", float("nan")),
                "max_comm_loss_time_s": m_b2.get("max_comm_loss_time_s", float("nan")),
                "max_comm_loss_length_km": m_b2.get("max_comm_loss_length_km", float("nan")),
                "note": "",
            }
        )

        # ---------- B3 (Flattened single-layer + LPA*) ----------
        area_event_b3 = build_area_event_from_center(
            b3_graph.nodes,
            b3_graph.edge_pairs,
            area_event.center_x_km,
            area_event.center_y_km,
            event_type=area_event.event_type,
            radius_km=area_event.radius_km,
            severity=area_event.severity,
        )
        b3 = LPAStarPlanner(b3_graph, start, goal)
        ok_init_b3 = b3.compute_shortest_path()
        if ok_init_b3 and area_event_b3.affected_edges:
            b3.apply_area_event(area_event_b3)
            t0 = time.perf_counter()
            ok_b3 = b3.compute_shortest_path()
            t1 = time.perf_counter()
            path_b3 = b3.extract_path() if ok_b3 else []
            m_b3 = b3_graph.path_metrics(path_b3) if ok_b3 else {}
            rec_b3 = {
                "trial": trial,
                "baseline": "B3_LPA_SingleLayer",
                "start_node": start,
                "goal_node": goal,
                "task_id": task_meta.get("task_id", ""),
                "task_depot": task_meta.get("depot", ""),
                "task_target": task_meta.get("target", ""),
                "task_stratum": task_meta.get("stratum", ""),
                "event_type": area_event.event_type,
                "event_center_x_km": area_event.center_x_km,
                "event_center_y_km": area_event.center_y_km,
                "event_radius_km": area_event.radius_km,
                "event_severity": area_event.severity,
                "event_affected_edges": len(area_event_b3.affected_edges),
                "area_event_edges": json.dumps(area_event_b3.affected_edges),
                "success": bool(ok_b3),
                "failure_reason": "" if ok_b3 else "event_path_disconnected",
                "replan_ms": (t1 - t0) * 1000.0 if ok_b3 else float("nan"),
                "expanded": int(b3.nodes_expanded) if ok_b3 else -1,
                "path_cost": m_b3.get("cost", float("nan")),
                "path_energy_kj": m_b3.get("energy_kj", float("nan")),
                "path_len_km": m_b3.get("length_km", float("nan")),
                "min_clearance_m": m_b3.get("min_clearance_m", float("nan")),
                "risk_exposure_integral": m_b3.get("risk_exposure_integral", float("nan")),
                "comm_coverage_ratio": m_b3.get("comm_coverage_ratio", float("nan")),
                "max_comm_loss_time_s": m_b3.get("max_comm_loss_time_s", float("nan")),
                "max_comm_loss_length_km": m_b3.get("max_comm_loss_length_km", float("nan")),
                "note": "",
            }
        else:
            rec_b3 = {
                "trial": trial,
                "baseline": "B3_LPA_SingleLayer",
                "start_node": start,
                "goal_node": goal,
                "task_id": task_meta.get("task_id", ""),
                "task_depot": task_meta.get("depot", ""),
                "task_target": task_meta.get("target", ""),
                "task_stratum": task_meta.get("stratum", ""),
                "event_type": area_event.event_type,
                "event_center_x_km": area_event.center_x_km,
                "event_center_y_km": area_event.center_y_km,
                "event_radius_km": area_event.radius_km,
                "event_severity": area_event.severity,
                "event_affected_edges": len(area_event_b3.affected_edges),
                "area_event_edges": json.dumps(area_event_b3.affected_edges),
                "success": False,
                "failure_reason": "start_goal_not_connected" if not ok_init_b3 else "event_schedule_unavailable",
                "replan_ms": float("nan"),
                "expanded": -1,
                "path_cost": float("nan"),
                "path_energy_kj": float("nan"),
                "path_len_km": float("nan"),
                "min_clearance_m": float("nan"),
                "risk_exposure_integral": float("nan"),
                "comm_coverage_ratio": float("nan"),
                "max_comm_loss_time_s": float("nan"),
                "max_comm_loss_length_km": float("nan"),
                "note": "init_fail_or_no_area_event_mapping",
            }
        records.append(rec_b3)

        # ---------- B5（规则三层图 + LPA*） ----------
        if b6_graph is not None and start_b6 is not None and goal_b6 is not None:
            area_event_b6 = build_area_event_from_center(
                b6_graph.nodes,
                b6_graph.edge_pairs,
                area_event.center_x_km,
                area_event.center_y_km,
                event_type=area_event.event_type,
                radius_km=area_event.radius_km,
                severity=area_event.severity,
            )
            b6 = LPAStarPlanner(b6_graph, int(start_b6), int(goal_b6))
            ok_init_b6 = b6.compute_shortest_path()
            if ok_init_b6 and area_event_b6.affected_edges:
                b6.apply_area_event(area_event_b6)
                t0 = time.perf_counter()
                ok_b6 = b6.compute_shortest_path()
                t1 = time.perf_counter()
                path_b6 = b6.extract_path() if ok_b6 else []
                m_b6 = b6_graph.path_metrics(path_b6) if ok_b6 else {}
                rec_b6 = {
                    "trial": trial,
                    "baseline": BASELINE_B5,
                    "start_node": int(start_b6),
                    "goal_node": int(goal_b6),
                    "task_id": task_meta.get("task_id", ""),
                    "task_depot": task_meta.get("depot", ""),
                    "task_target": task_meta.get("target", ""),
                    "task_stratum": task_meta.get("stratum", ""),
                    "event_type": area_event.event_type,
                    "event_center_x_km": area_event.center_x_km,
                    "event_center_y_km": area_event.center_y_km,
                    "event_radius_km": area_event.radius_km,
                    "event_severity": area_event.severity,
                    "event_affected_edges": len(area_event_b6.affected_edges),
                    "area_event_edges": json.dumps(area_event_b6.affected_edges),
                    "success": bool(ok_b6),
                    "failure_reason": "" if ok_b6 else "event_path_disconnected",
                    "replan_ms": (t1 - t0) * 1000.0 if ok_b6 else float("nan"),
                    "expanded": int(b6.nodes_expanded) if ok_b6 else -1,
                    "path_cost": m_b6.get("cost", float("nan")),
                    "path_energy_kj": m_b6.get("energy_kj", float("nan")),
                    "path_len_km": m_b6.get("length_km", float("nan")),
                    "min_clearance_m": m_b6.get("min_clearance_m", float("nan")),
                    "risk_exposure_integral": m_b6.get("risk_exposure_integral", float("nan")),
                    "comm_coverage_ratio": m_b6.get("comm_coverage_ratio", float("nan")),
                    "max_comm_loss_time_s": m_b6.get("max_comm_loss_time_s", float("nan")),
                    "max_comm_loss_length_km": m_b6.get("max_comm_loss_length_km", float("nan")),
                    "note": "",
                }
            else:
                rec_b6 = {
                    "trial": trial,
                    "baseline": BASELINE_B5,
                    "start_node": int(start_b6),
                    "goal_node": int(goal_b6),
                    "task_id": task_meta.get("task_id", ""),
                    "task_depot": task_meta.get("depot", ""),
                    "task_target": task_meta.get("target", ""),
                    "task_stratum": task_meta.get("stratum", ""),
                    "event_type": area_event.event_type,
                    "event_center_x_km": area_event.center_x_km,
                    "event_center_y_km": area_event.center_y_km,
                    "event_radius_km": area_event.radius_km,
                    "event_severity": area_event.severity,
                    "event_affected_edges": len(area_event_b6.affected_edges),
                    "area_event_edges": json.dumps(area_event_b6.affected_edges),
                    "success": False,
                    "failure_reason": "start_goal_not_connected" if not ok_init_b6 else "event_schedule_unavailable",
                    "replan_ms": float("nan"),
                    "expanded": -1,
                    "path_cost": float("nan"),
                    "path_energy_kj": float("nan"),
                    "path_len_km": float("nan"),
                    "min_clearance_m": float("nan"),
                    "risk_exposure_integral": float("nan"),
                    "comm_coverage_ratio": float("nan"),
                    "max_comm_loss_time_s": float("nan"),
                    "max_comm_loss_length_km": float("nan"),
                    "note": "init_fail_or_no_area_event_mapping",
                }
            records.append(rec_b6)

        # ---------- B1 (Traditional voxel + Dijkstra) ----------
        if voxel_planner is not None:
            start_xyz = (
                float(layered_graph.nodes[start, 0]),
                float(layered_graph.nodes[start, 1]),
                float(layered_graph.nodes[start, 2]),
            )
            goal_xyz = (
                float(layered_graph.nodes[goal, 0]),
                float(layered_graph.nodes[goal, 1]),
                float(layered_graph.nodes[goal, 2]),
            )
            mask = voxel_planner.build_storm_mask(
                [(area_event.center_x_km, area_event.center_y_km)],
                radius_m=area_event.radius_km * 1000.0,
            )
            t0 = time.perf_counter()
            b1_result = voxel_planner.search(
                start_xyz,
                goal_xyz,
                mask,
                timeout_s=args.b1_timeout_s,
                max_expansions=args.b1_max_expansions,
                risk_fields=risk_fields,
            )
            t1 = time.perf_counter()
            ok_b1 = bool(b1_result["ok"])
            records.append(
                {
                    "trial": trial,
                    "baseline": "B1_Voxel_Dijkstra",
                    "start_node": start,
                    "goal_node": goal,
                    "task_id": task_meta.get("task_id", ""),
                    "task_depot": task_meta.get("depot", ""),
                    "task_target": task_meta.get("target", ""),
                    "task_stratum": task_meta.get("stratum", ""),
                    "event_type": area_event.event_type,
                    "event_center_x_km": area_event.center_x_km,
                    "event_center_y_km": area_event.center_y_km,
                    "event_radius_km": area_event.radius_km,
                    "event_severity": area_event.severity,
                    "event_affected_edges": len(area_event.affected_edges),
                    "area_event_edges": json.dumps(area_event.affected_edges),
                    "success": ok_b1,
                    "failure_reason": "" if ok_b1 else str(b1_result.get("failure_reason", "voxel_unreachable")),
                    "replan_ms": (t1 - t0) * 1000.0 if ok_b1 else float("nan"),
                    "expanded": int(b1_result["expanded"]) if ok_b1 else -1,
                    "path_cost": float(b1_result["path_multi_cost"]) if ok_b1 else float("nan"),
                    "path_energy_kj": float(b1_result["path_energy_kj"]) if ok_b1 else float("nan"),
                    "path_len_km": float(b1_result["path_len_km"]) if ok_b1 else float("nan"),
                    "min_clearance_m": float(b1_result["min_clearance_m"]) if ok_b1 else float("nan"),
                    "risk_exposure_integral": float(b1_result["risk_exposure_integral"]) if ok_b1 else float("nan"),
                    "comm_coverage_ratio": float(b1_result["comm_coverage_ratio"]) if ok_b1 else float("nan"),
                    "max_comm_loss_time_s": float(b1_result["max_comm_loss_time_s"]) if ok_b1 else float("nan"),
                    "max_comm_loss_length_km": float(b1_result["max_comm_loss_length_km"]) if ok_b1 else float("nan"),
                    "note": "" if ok_b1 else "timeout_or_unreachable",
                }
            )

        accepted += 1
        if accepted % 5 == 0 or accepted == args.trials:
            print(f"[progress] accepted {accepted}/{args.trials} trials (attempts={attempts})")

    if accepted < args.trials:
        print(
            f"[warn] only collected {accepted} valid trials out of requested {args.trials}. "
            f"Consider increasing --max-attempt-factor."
        )

    baselines = [
        "B4_Proposed_LPA_Layered",
        "B2_GlobalAstar_Layered",
        "B3_LPA_SingleLayer",
    ]
    if b6_graph is not None:
        baselines.append(BASELINE_B5)
    if voxel_planner is not None:
        baselines.append("B1_Voxel_Dijkstra")

    summary_rows = [summarise_baseline(records, b) for b in baselines]

    pair_rows: List[dict] = []
    pair_cfg = [
        ("B4_Proposed_LPA_Layered", "B2_GlobalAstar_Layered", "replan_ms"),
        ("B4_Proposed_LPA_Layered", "B3_LPA_SingleLayer", "path_cost"),
    ]
    if b6_graph is not None:
        pair_cfg.extend(
            [
                ("B4_Proposed_LPA_Layered", BASELINE_B5, "replan_ms"),
                ("B4_Proposed_LPA_Layered", BASELINE_B5, "path_cost"),
                ("B4_Proposed_LPA_Layered", BASELINE_B5, "path_len_km"),
            ]
        )
    if voxel_planner is not None:
        pair_cfg.append(("B4_Proposed_LPA_Layered", "B1_Voxel_Dijkstra", "replan_ms"))

    for a, b, metric in pair_cfg:
        xa, xb = paired_arrays(records, a, b, metric)
        sig = paired_significance(xa, xb)
        if xa.size == 0:
            pair_rows.append(
                {
                    "pair": f"{a} vs {b}",
                    "metric": metric,
                    "n": 0,
                    "mean_a": float("nan"),
                    "mean_b": float("nan"),
                    "median_a": float("nan"),
                    "median_b": float("nan"),
                    "p95_a": float("nan"),
                    "p95_b": float("nan"),
                    "median_ratio_b_over_a": float("nan"),
                    "test_name": str(sig.get("test_name", "na")),
                    "p_value": float(sig.get("p_value", float("nan"))),
                }
            )
            continue
        ratio = np.median(xb / np.maximum(xa, EPS))
        pair_rows.append(
            {
                "pair": f"{a} vs {b}",
                "metric": metric,
                "n": int(xa.size),
                "mean_a": float(np.mean(xa)),
                "mean_b": float(np.mean(xb)),
                "median_a": float(np.median(xa)),
                "median_b": float(np.median(xb)),
                "p95_a": float(np.percentile(xa, 95)),
                "p95_b": float(np.percentile(xb, 95)),
                "median_ratio_b_over_a": float(ratio),
                "test_name": str(sig.get("test_name", "na")),
                "p_value": float(sig.get("p_value", float("nan"))),
            }
        )

    trial_fields = [
        "trial",
        "baseline",
        "start_node",
        "goal_node",
        "task_id",
        "task_depot",
        "task_target",
        "task_stratum",
        "event_type",
        "event_center_x_km",
        "event_center_y_km",
        "event_radius_km",
        "event_severity",
        "event_affected_edges",
        "area_event_edges",
        "success",
        "failure_reason",
        "replan_ms",
        "expanded",
        "path_cost",
        "path_energy_kj",
        "path_len_km",
        "min_clearance_m",
        "risk_exposure_integral",
        "comm_coverage_ratio",
        "max_comm_loss_time_s",
        "max_comm_loss_length_km",
        "note",
    ]
    write_csv(out_dir / "benchmark_trials.csv", records, trial_fields)

    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    if summary_fields:
        write_csv(out_dir / "benchmark_summary.csv", summary_rows, summary_fields)
        structural_rows = build_structural_ablation_rows(summary_rows)
        if structural_rows:
            structural_fields = ["method_order", "method", "baseline_id", "ablation_role"] + [
                f for f in summary_fields if f not in {"method_order", "method", "baseline_id", "ablation_role"}
            ]
            write_csv(out_dir / "benchmark_structural_ablation.csv", structural_rows, structural_fields)

    pair_fields = list(pair_rows[0].keys()) if pair_rows else []
    if pair_fields:
        write_csv(out_dir / "benchmark_pairwise.csv", pair_rows, pair_fields)

    md = render_benchmark_markdown_cn(summary_rows, pair_rows, args)
    (out_dir / "benchmark_table.md").write_text(md, encoding="utf-8")
    md4 = render_single_event_comparison_markdown_cn(summary_rows, args)
    (out_dir / "benchmark_table_four_baselines.md").write_text(md4, encoding="utf-8")
    (out_dir / "benchmark_table_structural_ablation.md").write_text(md4, encoding="utf-8")

    config = vars(args).copy()
    config["accepted_trials"] = accepted
    config["attempts"] = attempts
    (out_dir / "benchmark_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[done] outputs:")
    print(f"  - {out_dir / 'benchmark_trials.csv'}")
    print(f"  - {out_dir / 'benchmark_summary.csv'}")
    print(f"  - {out_dir / 'benchmark_structural_ablation.csv'}")
    print(f"  - {out_dir / 'benchmark_pairwise.csv'}")
    print(f"  - {out_dir / 'benchmark_table.md'}")
    print(f"  - {out_dir / 'benchmark_table_four_baselines.md'}")
    print(f"  - {out_dir / 'benchmark_table_structural_ablation.md'}")
    print(f"  - {out_dir / 'benchmark_config.json'}")


def run_benchmark_matrix_via_subprocess(args: argparse.Namespace) -> None:
    """Run the matrix-style A/B/C/D benchmark into one output directory."""
    script = Path(__file__).with_name("benchmark_matrix.py")
    if not script.exists():
        raise RuntimeError("benchmark_matrix.py is missing; cannot run matrix benchmark mode.")

    root = Path(args.workdir).resolve()
    _scene_cfg, data_root, _use_scene = resolve_benchmark_data_context(args, root)
    args.out_dir = str(resolve_scene_out_dir(args.out_dir, data_root, root))

    cmd = [
        sys.executable,
        str(script),
        "--workdir",
        str(args.workdir),
        "--out-dir",
        str(args.out_dir),
        "--trials",
        str(args.trials),
        "--key-trials",
        str(args.matrix_key_trials),
        "--seed",
        str(args.seed),
        "--n-block-grid",
        str(args.matrix_n_block_grid),
        "--k-events-grid",
        str(args.matrix_k_events_grid),
        "--scales",
        str(args.matrix_scales),
        "--scale-fractions",
        str(args.matrix_scale_fractions),
        "--focus-scale",
        str(args.matrix_focus_scale),
        "--focus-k-intensity",
        str(args.matrix_focus_k_intensity),
        "--focus-n-block-cont",
        str(args.matrix_focus_n_block_cont),
        "--focus-k-scale",
        str(args.matrix_focus_k_scale),
        "--focus-n-block-scale",
        str(args.matrix_focus_n_block_scale),
        "--focus-k-distribution",
        str(args.matrix_focus_k_distribution),
        "--plot-scale",
        str(args.matrix_plot_scale),
        "--plot-k-intensity",
        str(args.matrix_plot_k_intensity),
        "--plot-n-block-cont",
        str(args.matrix_plot_n_block_cont),
        "--plot-k-distribution",
        str(args.matrix_plot_k_distribution),
        "--event-radius-km",
        str(args.event_radius_km),
        "--event-type",
        str(args.event_type),
        "--event-severity",
        str(args.event_severity),
        "--event-pool-factor",
        str(args.matrix_event_pool_factor),
        "--min-start-goal-dist-km",
        str(args.min_start_goal_dist_km),
        "--max-attempt-factor",
        str(args.max_attempt_factor),
        "--progress-every",
        str(args.progress_every),
    ]
    if str(getattr(args, "scenario_config", "")).strip():
        cmd.extend(["--scenario-config", str(args.scenario_config)])
    if args.disable_plots:
        cmd.append("--disable-plots")

    print("[matrix] dispatching benchmark_matrix.py with unified experiment config...")
    print("[matrix] command:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"matrix benchmark failed with exit code {proc.returncode}")

    if args.skip_four_baseline:
        return

    # 同时在相同实验根目录下补充完整的单事件多基线对比表（B1/B2/B3/B4/B5）。
    baseline_dir = Path(args.out_dir).resolve() / "four_baseline"
    single_args = argparse.Namespace(**vars(args))
    single_args.mode = "single"
    single_args.out_dir = str(baseline_dir)
    single_args.n_block = int(args.matrix_focus_n_block_scale)
    print(
        "[matrix] 正在补充单事件多基线汇总："
        f"n_block={single_args.n_block}, out={baseline_dir}"
    )
    run_benchmark(single_args)

    # 将关键的单事件多基线产物提升到矩阵根目录，便于论文直接引用。
    out_dir = Path(args.out_dir).resolve()
    promote = [
        ("benchmark_table_four_baselines.md", "benchmark_table_four_baselines.md"),
        ("benchmark_table_structural_ablation.md", "benchmark_table_structural_ablation.md"),
        ("benchmark_summary.csv", "benchmark_summary_four_baselines.csv"),
        ("benchmark_structural_ablation.csv", "benchmark_structural_ablation.csv"),
        ("benchmark_trials.csv", "benchmark_trials_four_baselines.csv"),
    ]
    for src_name, dst_name in promote:
        src = baseline_dir / src_name
        dst = out_dir / dst_name
        if src.exists():
            shutil.copyfile(src, dst)
            print(f"[matrix] promoted: {dst}")

    # 确保根目录下的 benchmark_trials.csv 显式包含单事件 B1/B2/B3/B4/B5 结果。
    matrix_trials = out_dir / "benchmark_trials.csv"
    baseline_trials = baseline_dir / "benchmark_trials.csv"
    if matrix_trials.exists() and baseline_trials.exists():
        shutil.copyfile(matrix_trials, out_dir / "benchmark_trials_matrix.csv")

        with matrix_trials.open("r", newline="", encoding="utf-8") as f:
            matrix_rows = list(csv.DictReader(f))
        with baseline_trials.open("r", newline="", encoding="utf-8") as f:
            four_rows = list(csv.DictReader(f))

        for r in matrix_rows:
            r["experiment_type"] = "matrix_event_stream"
        for r in four_rows:
            r["experiment_type"] = "single_event_multi_baseline"

        all_rows = matrix_rows + four_rows
        if all_rows:
            fields: List[str] = []
            seen = set()
            for row in all_rows:
                for k in row.keys():
                    if k not in seen:
                        seen.add(k)
                        fields.append(k)
            with matrix_trials.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for row in all_rows:
                    writer.writerow(row)
            print(f"[matrix] merged trials written: {matrix_trials} (includes B1/B2/B3/B4/B5)")

    # 在矩阵配置中补充单事件多基线运行元数据。
    cfg_path = out_dir / "benchmark_config.json"
    cfg_data: dict = {}
    if cfg_path.exists():
        try:
            cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg_data = {}
    cfg_data["four_baseline_enabled"] = True
    cfg_data["four_baseline_skip_b1"] = bool(args.skip_b1)
    cfg_data["four_baseline_output_dir"] = str(baseline_dir)
    cfg_data["four_baseline_baselines"] = [
        "B1_Voxel_Dijkstra",
        "B2_GlobalAstar_Layered",
        "B3_LPA_SingleLayer",
        BASELINE_B5,
        "B4_Proposed_LPA_Layered",
    ]
    cfg_data["benchmark_trials_merged"] = bool(matrix_trials.exists() and baseline_trials.exists())
    cfg_path.write_text(json.dumps(cfg_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[matrix] updated config metadata: {cfg_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo benchmark for B1/B2/B3/B4/B5 baselines.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "matrix"],
        default="matrix",
        help="single: one-shot benchmark; matrix: A/B/C/D final experiment set in one output dir.",
    )
    parser.add_argument("--workdir", type=str, default=".", help="Project root containing *.npy data files.")
    parser.add_argument("--scenario-config", type=str, default="", help="场景配置 JSON。")
    parser.add_argument("--trials", type=int, default=50, help="Required accepted Monte Carlo trials.")
    parser.add_argument("--seed", type=int, default=20260309, help="Random seed.")
    parser.add_argument("--n-block", type=int, default=2, help="旧兼容参数；单次 benchmark 已改用区域扰动。")
    parser.add_argument("--event-type", type=str, choices=["no_fly", "wind", "comm_risk"], default="no_fly")
    parser.add_argument("--event-radius-km", type=float, default=0.8)
    parser.add_argument("--event-severity", type=float, default=1.0)
    parser.add_argument("--min-start-goal-dist-km", type=float, default=1.5, help="物流任务起终点映射后的最小平面距离。")
    parser.add_argument("--max-attempt-factor", type=int, default=5, help="Max attempts multiplier to collect valid trials.")
    parser.add_argument("--progress-every", type=int, default=5, help="Progress print interval for matrix mode.")
    parser.add_argument("--out-dir", type=str, default="benchmark_out_final", help="Output directory.")

    parser.add_argument("--b3-z-offset-m", type=float, default=75.0, help="Flattened single-layer altitude offset.")
    parser.add_argument("--b3-intra-dist-m", type=float, default=250.0, help="B3 same-layer edge radius.")
    parser.add_argument("--b3-collision-samples", type=int, default=10, help="Collision samples for B3 edge generation.")

    parser.add_argument("--skip-b1", action="store_true", help="Skip B1 voxel baseline.")
    parser.add_argument("--b1-xy-step-m", type=float, default=125.0, help="B1 voxel XY step in meters.")
    parser.add_argument("--b1-agl-low-m", type=float, default=30.0, help="B1 min AGL level.")
    parser.add_argument("--b1-agl-high-m", type=float, default=120.0, help="B1 max AGL level.")
    parser.add_argument("--b1-agl-step-m", type=float, default=5.0, help="B1 AGL step in meters.")
    parser.add_argument("--b1-timeout-s", type=float, default=8.0, help="Per-trial timeout for B1 search.")
    parser.add_argument("--b1-max-expansions", type=int, default=2_000_000, help="Hard cap for B1 expansions.")
    parser.add_argument("--storm-radius-m", type=float, default=220.0, help="旧兼容参数；B1 当前使用区域事件半径。")

    # Matrix mode defaults aligned with final paper experiment settings.
    parser.add_argument("--matrix-n-block-grid", type=str, default="2,4,6,8")
    parser.add_argument("--matrix-k-events-grid", type=str, default="1,3,5,7,10")
    parser.add_argument("--matrix-key-trials", type=int, default=0)
    parser.add_argument("--matrix-scales", type=str, default="small,medium,large")
    parser.add_argument("--matrix-scale-fractions", type=str, default="small:0.55,medium:0.78,large:1.0")
    parser.add_argument("--matrix-focus-scale", type=str, default="large")
    parser.add_argument("--matrix-focus-k-intensity", type=int, default=5)
    parser.add_argument("--matrix-focus-n-block-cont", type=int, default=4)
    parser.add_argument("--matrix-focus-k-scale", type=int, default=5)
    parser.add_argument("--matrix-focus-n-block-scale", type=int, default=4)
    parser.add_argument("--matrix-focus-k-distribution", type=int, default=10)
    parser.add_argument("--matrix-plot-scale", type=str, default="large")
    parser.add_argument("--matrix-plot-k-intensity", type=int, default=5)
    parser.add_argument("--matrix-plot-n-block-cont", type=int, default=4)
    parser.add_argument("--matrix-plot-k-distribution", type=int, default=10)
    parser.add_argument("--matrix-event-pool-factor", type=int, default=6)
    parser.add_argument("--disable-plots", action="store_true", help="Disable matrix plots.")
    parser.add_argument(
        "--skip-four-baseline",
        action="store_true",
        help="Skip extra single-mode run that outputs the complete B1/B2/B3/B4/B5 table.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "matrix":
        run_benchmark_matrix_via_subprocess(args)
    else:
        run_benchmark(args)
