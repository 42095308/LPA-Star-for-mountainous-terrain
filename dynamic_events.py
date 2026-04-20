"""
区域动态事件建模。

事件由中心点、半径、类型和强度定义，再映射为半径内受影响的图边。
这比从当前路径中随机挑边更接近禁飞区、强风区和通信风险区等真实扰动。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


EVENT_TYPES = ("no_fly", "wind", "comm_risk")


@dataclass
class AreaEvent:
    event_type: str
    center_x_km: float
    center_y_km: float
    radius_km: float
    severity: float
    affected_edges: List[Tuple[int, int]]
    source_edge: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "event_type": self.event_type,
            "center_x_km": self.center_x_km,
            "center_y_km": self.center_y_km,
            "radius_km": self.radius_km,
            "severity": self.severity,
            "affected_edge_count": len(self.affected_edges),
            "affected_edges": self.affected_edges,
            "source_edge": self.source_edge,
        }


def normalize_pair(u: int, v: int) -> Tuple[int, int]:
    u = int(u)
    v = int(v)
    return (u, v) if u <= v else (v, u)


def edge_midpoints(nodes: np.ndarray, edge_pairs: np.ndarray) -> np.ndarray:
    pairs = np.asarray(edge_pairs, dtype=int)
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=float)
    return (nodes[pairs[:, 0], :2] + nodes[pairs[:, 1], :2]) * 0.5


def _choose_source_edge(path: Sequence[int], rng: np.random.Generator) -> Optional[Tuple[int, int]]:
    if len(path) < 2:
        return None
    if len(path) >= 4:
        lo = max(0, len(path) // 3)
        hi = max(lo + 1, 2 * len(path) // 3)
        idx = int(rng.integers(lo, min(hi, len(path) - 1)))
    else:
        idx = max(0, len(path) // 2 - 1)
    return int(path[idx]), int(path[idx + 1])


def build_area_event_from_path(
    nodes: np.ndarray,
    edge_pairs: np.ndarray,
    path: Sequence[int],
    rng: np.random.Generator,
    event_type: str = "no_fly",
    radius_km: float = 0.8,
    severity: float = 1.0,
    max_affected_edges: Optional[int] = None,
    used_edges: Optional[set] = None,
) -> AreaEvent:
    if event_type not in EVENT_TYPES:
        raise ValueError(f"未知事件类型: {event_type}")

    pairs = np.asarray(edge_pairs, dtype=int)
    mids = edge_midpoints(nodes, pairs)
    used_edges = used_edges or set()

    source = _choose_source_edge(path, rng)
    if source is not None:
        u, v = source
        center = (nodes[u, :2] + nodes[v, :2]) * 0.5
    elif len(mids) > 0:
        center = mids[int(rng.integers(0, len(mids)))]
    else:
        center = np.zeros(2, dtype=float)

    # 小幅扰动事件中心，避免每个种子总是落在边的正中点。
    radius_km = max(float(radius_km), 1e-6)
    jitter_len = float(rng.uniform(0.0, 0.25 * radius_km))
    jitter_ang = float(rng.uniform(0.0, 2.0 * np.pi))
    center = center + np.array([np.cos(jitter_ang), np.sin(jitter_ang)]) * jitter_len

    if len(mids) == 0:
        affected: List[Tuple[int, int]] = []
    else:
        dist = np.linalg.norm(mids - center.reshape(1, 2), axis=1)
        order = np.argsort(dist)
        in_radius = [int(i) for i in order if float(dist[i]) <= radius_km]
        if not in_radius:
            in_radius = [int(order[0])]

        affected = []
        for eid in in_radius:
            pair = normalize_pair(int(pairs[eid, 0]), int(pairs[eid, 1]))
            if pair in used_edges:
                continue
            affected.append(pair)
            if max_affected_edges is not None and len(affected) >= int(max_affected_edges):
                break

        if not affected:
            for eid in order:
                pair = normalize_pair(int(pairs[int(eid), 0]), int(pairs[int(eid), 1]))
                if pair not in used_edges:
                    affected.append(pair)
                    break

    return AreaEvent(
        event_type=event_type,
        center_x_km=float(center[0]),
        center_y_km=float(center[1]),
        radius_km=radius_km,
        severity=max(0.0, float(severity)),
        affected_edges=affected,
        source_edge=normalize_pair(*source) if source is not None else None,
    )


def event_edge_cost(
    event_type: str,
    severity: float,
    raw_component: Sequence[float],
    maxima: Sequence[float],
    weights: Sequence[float],
) -> float:
    """把事件类型映射到新的边代价。"""
    if event_type == "no_fly":
        return float("inf")

    t_raw, e_raw, r_raw = [float(v) for v in raw_component]
    t_max, e_max, r_max = [max(float(v), 1e-9) for v in maxima]
    alpha, beta, gamma = [float(v) for v in weights]
    severity = max(0.0, float(severity))

    if event_type == "wind":
        t_raw *= 1.0 + 0.35 * severity
        e_raw *= 1.0 + 0.50 * severity
    elif event_type == "comm_risk":
        r_raw = min(1.0, r_raw + 0.35 * severity * (1.0 - r_raw))
    else:
        raise ValueError(f"未知事件类型: {event_type}")

    return alpha * (t_raw / t_max) + beta * (e_raw / e_max) + gamma * (r_raw / r_max)

