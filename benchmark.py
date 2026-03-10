"""
Monte Carlo benchmark for four baselines on the Huashan planning project.

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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

try:
    from scipy.stats import ttest_rel
except Exception:  # pragma: no cover - optional
    ttest_rel = None


ALPHA = 0.3
BETA = 0.2
GAMMA = 0.5
UAV_SPEED = 15.0
UAV_POWER = 500.0
RESOLUTION = 12.5
SAFETY_HEIGHT = 30.0
COLLISION_SAMPLES = 12
RISK_SAMPLES = 10
EPS = 1e-9


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
        return {
            "cost": total_cost,
            "time_s": total_t,
            "energy_kj": total_e,
            "risk": total_r,
            "length_km": total_len_km,
        }

    def heuristic(self, s: int, goal: int) -> float:
        dx = (self.nodes[goal, 0] - self.nodes[s, 0]) * 1000.0
        dy = (self.nodes[goal, 1] - self.nodes[s, 1]) * 1000.0
        dz = self.nodes[goal, 2] - self.nodes[s, 2]
        d3d = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        t_lb = d3d / UAV_SPEED
        climb_lb = max(0.0, dz) * 9.8 * 5.0
        e_lb = (UAV_POWER * t_lb + climb_lb) / 1000.0
        return ALPHA * (t_lb / (self.t_max + EPS)) + BETA * (e_lb / (self.e_max + EPS))


def compute_edge_costs(
    nodes: np.ndarray,
    edge_pairs: np.ndarray,
    z_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    m = edge_pairs.shape[0]
    t_raw = np.zeros(m, dtype=float)
    e_raw = np.zeros(m, dtype=float)
    r_raw = np.zeros(m, dtype=float)
    rows, cols = z_grid.shape

    for k in range(m):
        i = int(edge_pairs[k, 0])
        j = int(edge_pairs[k, 1])
        xi, yi, zi = nodes[i, 0], nodes[i, 1], nodes[i, 2]
        xj, yj, zj = nodes[j, 0], nodes[j, 1], nodes[j, 2]
        dh = float(np.linalg.norm([xj - xi, yj - yi]) * 1000.0)
        dv = float(abs(zj - zi))
        d3d = float(np.sqrt(dh * dh + dv * dv))
        t = d3d / UAV_SPEED
        e = (UAV_POWER * t + max(0.0, zj - zi) * 9.8 * 5.0) / 1000.0
        risk = 0.0
        for s in np.linspace(0.0, 1.0, RISK_SAMPLES):
            x = xi + s * (xj - xi)
            y = yi + s * (yj - yi)
            z = zi + s * (zj - zi)
            r, c = km_to_rc(float(x), float(y), rows, cols)
            terrain = float(z_grid[r, c])
            risk += max(0.0, 1.0 - (z - terrain) / 200.0)
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
) -> WeightedGraph:
    edge_pairs = np.asarray(edges[:, :2], dtype=int)
    if edges.shape[1] >= 3:
        edge_types = np.asarray(edges[:, 2], dtype=int)
    else:
        edge_types = np.zeros(len(edge_pairs), dtype=int)

    w, t_raw, e_raw, r_raw, t_max, e_max, r_max = compute_edge_costs(nodes, edge_pairs, z_grid)
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
    )


def build_single_layer_graph(
    layered_graph: WeightedGraph,
    z_grid: np.ndarray,
    z_offset_m: float = 75.0,
    intra_dist_m: float = 250.0,
    collision_samples: int = 10,
) -> WeightedGraph:
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
    return build_weighted_graph("B3_single_layer", nodes, edge_arr, z_grid)


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
    blocked_pairs: Sequence[Tuple[int, int]],
) -> Tuple[bool, List[int], Dict[str, int]]:
    blocked_eids = graph.blocked_eids(blocked_pairs)
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
            if eid in blocked_eids:
                continue
            ng = g + float(graph.edge_weight[eid])
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

    def search(
        self,
        start_xyz: Tuple[float, float, float],
        goal_xyz: Tuple[float, float, float],
        storm_mask_xy: np.ndarray,
        timeout_s: float,
        max_expansions: int = 2_000_000,
    ) -> Tuple[bool, float, int]:
        sx, sy, sz = start_xyz
        gx, gy, gz = goal_xyz
        six, siy, siz = self._nearest_state(float(sx), float(sy), float(sz))
        gix, giy, giz = self._nearest_state(float(gx), float(gy), float(gz))
        start_sid = self._sid(six, siy, siz)
        goal_sid = self._sid(gix, giy, giz)

        # Ensure start/goal cells stay traversable.
        storm_mask_xy = storm_mask_xy.copy()
        storm_mask_xy[siy, six] = False
        storm_mask_xy[giy, gix] = False

        dist = np.full(self.total_states, float("inf"), dtype=float)
        visited = np.zeros(self.total_states, dtype=bool)
        dist[start_sid] = 0.0
        heap: List[Tuple[float, int]] = [(0.0, start_sid)]

        expanded = 0
        t0 = time.perf_counter()

        while heap:
            if (time.perf_counter() - t0) > timeout_s:
                return False, float("inf"), expanded
            d, sid = heapq.heappop(heap)
            if visited[sid]:
                continue
            if d > dist[sid] + EPS:
                continue
            visited[sid] = True

            if sid == goal_sid:
                return True, float(d / 1000.0), expanded

            expanded += 1
            if expanded >= max_expansions:
                return False, float("inf"), expanded

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
                    heapq.heappush(heap, (nd, nsid))

        return False, float("inf"), expanded


def choose_blocked_edges_from_path(path: Sequence[int], rng: np.random.Generator, n_block: int) -> List[Tuple[int, int]]:
    if len(path) < 2:
        return []
    seq = [normalize_pair(int(path[i]), int(path[i + 1])) for i in range(len(path) - 1)]
    seq = list(dict.fromkeys(seq))
    if not seq:
        return []
    if len(seq) <= n_block:
        return seq

    interior = seq[1:-1] if len(seq) > 2 else seq
    if not interior:
        interior = seq

    k = min(n_block, len(interior))
    picked_idx = rng.choice(len(interior), size=k, replace=False)
    picked = [interior[int(i)] for i in np.atleast_1d(picked_idx)]
    return picked


def map_blocked_edges_to_graph(
    src_graph: WeightedGraph,
    dst_graph: WeightedGraph,
    blocked_pairs: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    mapped: List[Tuple[int, int]] = []
    used = set()
    for u, v in blocked_pairs:
        key = normalize_pair(int(u), int(v))
        if key in dst_graph.pair_to_eid:
            if key not in used:
                mapped.append(key)
                used.add(key)
            continue

        mid = (src_graph.nodes[key[0], :2] + src_graph.nodes[key[1], :2]) * 0.5
        dist, eid = dst_graph.edge_mid_tree.query(mid, k=8)
        _ = dist
        eids = np.atleast_1d(eid).astype(int)
        for candidate in eids:
            pair = normalize_pair(
                int(dst_graph.edge_pairs[candidate, 0]),
                int(dst_graph.edge_pairs[candidate, 1]),
            )
            if pair not in used:
                mapped.append(pair)
                used.add(pair)
                break
    return mapped


def sample_start_goal(
    rng: np.random.Generator,
    nodes: np.ndarray,
    min_dist_km: float,
) -> Tuple[int, int]:
    n = int(nodes.shape[0])
    for _ in range(500):
        s = int(rng.integers(0, n))
        g = int(rng.integers(0, n))
        if s == g:
            continue
        d = float(np.linalg.norm(nodes[s, :2] - nodes[g, :2]))
        if d >= min_dist_km:
            return s, g
    s = int(rng.integers(0, n))
    g = int((s + rng.integers(1, n)) % n)
    return s, g


def ci95(arr: np.ndarray) -> float:
    if arr.size <= 1:
        return float("nan")
    return float(1.96 * np.std(arr, ddof=1) / math.sqrt(arr.size))


def summarise_baseline(records: List[dict], baseline: str) -> dict:
    subset = [r for r in records if r["baseline"] == baseline]
    ok = [r for r in subset if r["success"]]
    out = {
        "baseline": baseline,
        "n_trials": len(subset),
        "n_success": len(ok),
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
    }
    if not ok:
        return out

    ms = np.array([r["replan_ms"] for r in ok], dtype=float)
    ex = np.array([r["expanded"] for r in ok], dtype=float)
    c = np.array([r["path_cost"] for r in ok], dtype=float)
    e = np.array([r["path_energy_kj"] for r in ok], dtype=float)
    l = np.array([r["path_len_km"] for r in ok], dtype=float)

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
        }
    )
    return out


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


def paired_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if ttest_rel is None:
        return float("nan")
    res = ttest_rel(x, y, nan_policy="omit")
    return float(res.pvalue)


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
        f"- Trials requested: `{args.trials}`, random seed: `{args.seed}`, blocked edges per trial: `{args.n_block}`"
    )
    lines.append(
        f"- B1 voxel config: `xy_step={args.b1_xy_step_m:.0f}m`, "
        f"`agl_step={args.b1_agl_step_m:.0f}m`, timeout `{args.b1_timeout_s:.1f}s`"
    )
    lines.append("")
    lines.append("## Per-baseline summary")
    lines.append("")
    lines.append(
        "| Baseline | Success | Replan ms (mean+/-std) | P50/P95 ms | Expanded (mean) | Cost (mean) | Energy kJ (mean) | Length km (mean) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary_rows:
        succ = f"{r['n_success']}/{r['n_trials']} ({100.0*r['success_rate']:.1f}%)"
        ms = f"{r['mean_replan_ms']:.2f}+/-{r['std_replan_ms']:.2f}"
        p = f"{r['p50_replan_ms']:.2f}/{r['p95_replan_ms']:.2f}"
        lines.append(
            f"| {r['baseline']} | {succ} | {ms} | {p} | {r['mean_expanded']:.1f} | "
            f"{r['mean_cost']:.4f} | {r['mean_energy_kj']:.2f} | {r['mean_length_km']:.3f} |"
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


def run_benchmark(args: argparse.Namespace) -> None:
    root = Path(args.workdir).resolve()
    os.chdir(root)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    z_grid = np.load("Z_crop.npy")
    layered_nodes = np.load("graph_nodes.npy")
    layered_edges = np.load("graph_edges.npy")

    print("[build] loading layered graph...")
    layered_graph = build_weighted_graph("B4_layered", layered_nodes, layered_edges, z_grid)
    print(f"[build] layered graph: |V|={layered_graph.n_nodes}, |E|={layered_graph.n_edges}")

    print("[build] building flattened single-layer graph for B3...")
    b3_graph = build_single_layer_graph(
        layered_graph,
        z_grid,
        z_offset_m=args.b3_z_offset_m,
        intra_dist_m=args.b3_intra_dist_m,
        collision_samples=args.b3_collision_samples,
    )
    print(f"[build] B3 graph: |V|={b3_graph.n_nodes}, |E|={b3_graph.n_edges}")

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
        start, goal = sample_start_goal(rng, layered_graph.nodes, args.min_start_goal_dist_km)

        # B4 initial planning (to define scenario and storm edges)
        b4 = LPAStarPlanner(layered_graph, start, goal)
        ok_init_b4 = b4.compute_shortest_path()
        path_init_b4 = b4.extract_path() if ok_init_b4 else []
        if not ok_init_b4 or len(path_init_b4) < 3:
            continue

        blocked_b4 = choose_blocked_edges_from_path(path_init_b4, rng, args.n_block)
        if not blocked_b4:
            continue

        storm_midpoints = [
            tuple((layered_graph.nodes[u, :2] + layered_graph.nodes[v, :2]) * 0.5)
            for u, v in blocked_b4
        ]

        # ---------- B4 (Proposed incremental LPA*) ----------
        b4.block_edges(blocked_b4)
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
                "blocked_edges": json.dumps(blocked_b4),
                "success": bool(ok_b4),
                "replan_ms": replan_b4_ms if ok_b4 else float("nan"),
                "expanded": int(b4.nodes_expanded) if ok_b4 else -1,
                "path_cost": m_b4.get("cost", float("nan")),
                "path_energy_kj": m_b4.get("energy_kj", float("nan")),
                "path_len_km": m_b4.get("length_km", float("nan")),
                "note": "",
            }
        )

        # ---------- B2 (Global A* recompute) ----------
        t0 = time.perf_counter()
        ok_b2, path_b2, st_b2 = astar_global_replan(layered_graph, start, goal, blocked_b4)
        t1 = time.perf_counter()
        replan_b2_ms = (t1 - t0) * 1000.0
        m_b2 = layered_graph.path_metrics(path_b2) if ok_b2 else {}
        records.append(
            {
                "trial": trial,
                "baseline": "B2_GlobalAstar_Layered",
                "start_node": start,
                "goal_node": goal,
                "blocked_edges": json.dumps(blocked_b4),
                "success": bool(ok_b2),
                "replan_ms": replan_b2_ms if ok_b2 else float("nan"),
                "expanded": int(st_b2.get("expanded", -1)) if ok_b2 else -1,
                "path_cost": m_b2.get("cost", float("nan")),
                "path_energy_kj": m_b2.get("energy_kj", float("nan")),
                "path_len_km": m_b2.get("length_km", float("nan")),
                "note": "",
            }
        )

        # ---------- B3 (Flattened single-layer + LPA*) ----------
        blocked_b3 = map_blocked_edges_to_graph(layered_graph, b3_graph, blocked_b4)
        b3 = LPAStarPlanner(b3_graph, start, goal)
        ok_init_b3 = b3.compute_shortest_path()
        if ok_init_b3 and blocked_b3:
            b3.block_edges(blocked_b3)
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
                "blocked_edges": json.dumps(blocked_b3),
                "success": bool(ok_b3),
                "replan_ms": (t1 - t0) * 1000.0 if ok_b3 else float("nan"),
                "expanded": int(b3.nodes_expanded) if ok_b3 else -1,
                "path_cost": m_b3.get("cost", float("nan")),
                "path_energy_kj": m_b3.get("energy_kj", float("nan")),
                "path_len_km": m_b3.get("length_km", float("nan")),
                "note": "",
            }
        else:
            rec_b3 = {
                "trial": trial,
                "baseline": "B3_LPA_SingleLayer",
                "start_node": start,
                "goal_node": goal,
                "blocked_edges": json.dumps(blocked_b3),
                "success": False,
                "replan_ms": float("nan"),
                "expanded": -1,
                "path_cost": float("nan"),
                "path_energy_kj": float("nan"),
                "path_len_km": float("nan"),
                "note": "init_fail_or_no_storm_mapping",
            }
        records.append(rec_b3)

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
            mask = voxel_planner.build_storm_mask(storm_midpoints, radius_m=args.storm_radius_m)
            t0 = time.perf_counter()
            ok_b1, len_b1_km, exp_b1 = voxel_planner.search(
                start_xyz,
                goal_xyz,
                mask,
                timeout_s=args.b1_timeout_s,
                max_expansions=args.b1_max_expansions,
            )
            t1 = time.perf_counter()
            records.append(
                {
                    "trial": trial,
                    "baseline": "B1_Voxel_Dijkstra",
                    "start_node": start,
                    "goal_node": goal,
                    "blocked_edges": json.dumps(blocked_b4),
                    "success": bool(ok_b1),
                    "replan_ms": (t1 - t0) * 1000.0 if ok_b1 else float("nan"),
                    "expanded": int(exp_b1) if ok_b1 else -1,
                    "path_cost": float(len_b1_km) if ok_b1 else float("nan"),
                    "path_energy_kj": float("nan"),
                    "path_len_km": float(len_b1_km) if ok_b1 else float("nan"),
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
    if voxel_planner is not None:
        baselines.append("B1_Voxel_Dijkstra")

    summary_rows = [summarise_baseline(records, b) for b in baselines]

    pair_rows: List[dict] = []
    pair_cfg = [
        ("B4_Proposed_LPA_Layered", "B2_GlobalAstar_Layered", "replan_ms"),
        ("B4_Proposed_LPA_Layered", "B3_LPA_SingleLayer", "path_cost"),
    ]
    if voxel_planner is not None:
        pair_cfg.append(("B4_Proposed_LPA_Layered", "B1_Voxel_Dijkstra", "replan_ms"))

    for a, b, metric in pair_cfg:
        xa, xb = paired_arrays(records, a, b, metric)
        if xa.size == 0:
            pair_rows.append(
                {
                    "pair": f"{a} vs {b}",
                    "metric": metric,
                    "n": 0,
                    "mean_a": float("nan"),
                    "mean_b": float("nan"),
                    "median_ratio_b_over_a": float("nan"),
                    "p_value": float("nan"),
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
                "median_ratio_b_over_a": float(ratio),
                "p_value": paired_pvalue(xa, xb),
            }
        )

    trial_fields = [
        "trial",
        "baseline",
        "start_node",
        "goal_node",
        "blocked_edges",
        "success",
        "replan_ms",
        "expanded",
        "path_cost",
        "path_energy_kj",
        "path_len_km",
        "note",
    ]
    write_csv(out_dir / "benchmark_trials.csv", records, trial_fields)

    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    if summary_fields:
        write_csv(out_dir / "benchmark_summary.csv", summary_rows, summary_fields)

    pair_fields = list(pair_rows[0].keys()) if pair_rows else []
    if pair_fields:
        write_csv(out_dir / "benchmark_pairwise.csv", pair_rows, pair_fields)

    md = render_markdown(summary_rows, pair_rows, args)
    (out_dir / "benchmark_table.md").write_text(md, encoding="utf-8")

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
    print(f"  - {out_dir / 'benchmark_pairwise.csv'}")
    print(f"  - {out_dir / 'benchmark_table.md'}")
    print(f"  - {out_dir / 'benchmark_config.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo benchmark for B1/B2/B3/B4 baselines.")
    parser.add_argument("--workdir", type=str, default=".", help="Project root containing *.npy data files.")
    parser.add_argument("--trials", type=int, default=50, help="Required accepted Monte Carlo trials.")
    parser.add_argument("--seed", type=int, default=20260309, help="Random seed.")
    parser.add_argument("--n-block", type=int, default=2, help="Number of storm-blocked edges per trial.")
    parser.add_argument("--min-start-goal-dist-km", type=float, default=1.5, help="Min XY distance for random start/goal.")
    parser.add_argument("--max-attempt-factor", type=int, default=5, help="Max attempts multiplier to collect valid trials.")
    parser.add_argument("--out-dir", type=str, default="benchmark_out", help="Output directory.")

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
    parser.add_argument("--storm-radius-m", type=float, default=220.0, help="Storm radius used in B1 voxel blocking.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
