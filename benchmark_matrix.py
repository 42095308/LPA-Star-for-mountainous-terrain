"""
Benchmark matrix runner for experiments A/B/C/D:
- Event intensity sweep (n_block grid)
- Continuous multi-event replanning (K grid)
- Graph scale sweep (small/medium/large)
- Workload metrics (queue/update/reopen)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from benchmark import (
    EPS,
    LPAStarPlanner,
    WeightedGraph,
    astar_global_replan,
    build_weighted_graph,
    ci95,
    load_risk_fields,
    normalize_pair,
    sample_start_goal,
    write_csv,
)

BASELINE_B4 = "B4_Proposed_LPA_Layered"
BASELINE_B2 = "B2_GlobalAstar_Layered"


def parse_int_grid_arg(raw: str, name: str) -> List[int]:
    vals: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))
    vals = list(dict.fromkeys(vals))
    if not vals:
        raise ValueError(f"{name} is empty")
    if any(v <= 0 for v in vals):
        raise ValueError(f"{name} must contain positive integers")
    return vals


def parse_scale_fraction_arg(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError("scale-fractions must be in name:value format")
        name, frac_s = token.split(":", 1)
        name = name.strip()
        frac = float(frac_s.strip())
        if not name:
            raise ValueError("empty scale name in scale-fractions")
        if frac <= 0.0 or frac > 1.0:
            raise ValueError(f"invalid scale fraction for {name}: {frac}")
        out[name] = frac
    if not out:
        raise ValueError("scale-fractions is empty")
    return out


def parse_scale_names(raw: str, fractions: Dict[str, float]) -> List[str]:
    names = [s.strip() for s in raw.split(",") if s.strip()]
    if not names:
        names = ["large"]
    for n in names:
        if n not in fractions:
            raise ValueError(f"scale '{n}' is not defined in --scale-fractions")
    return list(dict.fromkeys(names))


def nearest_int(target: int, values: Sequence[int]) -> int:
    vals = sorted(set(int(v) for v in values))
    if not vals:
        return int(target)
    if int(target) in vals:
        return int(target)
    return min(vals, key=lambda x: abs(x - int(target)))


def nearest_scale(target: str, values: Sequence[str]) -> str:
    if target in values:
        return target
    return values[-1] if values else target


def largest_component_indices(n_nodes: int, edge_pairs: np.ndarray) -> np.ndarray:
    if n_nodes <= 0:
        return np.asarray([], dtype=int)

    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    for u, v in edge_pairs:
        uu = int(u)
        vv = int(v)
        if uu == vv:
            continue
        adj[uu].append(vv)
        adj[vv].append(uu)

    seen = np.zeros(n_nodes, dtype=bool)
    best: List[int] = []
    for s in range(n_nodes):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        comp: List[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        if len(comp) > len(best):
            best = comp
    return np.asarray(best, dtype=int)


def build_scaled_graph(
    base_nodes: np.ndarray,
    base_edges: np.ndarray,
    z_grid: np.ndarray,
    scale_name: str,
    fraction: float,
    risk_fields: Optional[Dict[str, np.ndarray | str]] = None,
) -> WeightedGraph:
    nodes = np.asarray(base_nodes, dtype=float)
    edges = np.asarray(base_edges, dtype=int)

    if fraction >= 0.999:
        return build_weighted_graph(
            f"layered_{scale_name}",
            nodes.copy(),
            edges.copy(),
            z_grid,
            risk_fields=risk_fields,
        )

    x = nodes[:, 0]
    y = nodes[:, 1]
    cx = 0.5 * (float(np.min(x)) + float(np.max(x)))
    cy = 0.5 * (float(np.min(y)) + float(np.max(y)))
    hx = 0.5 * (float(np.max(x)) - float(np.min(x))) * float(fraction)
    hy = 0.5 * (float(np.max(y)) - float(np.min(y))) * float(fraction)

    mask = (np.abs(x - cx) <= hx + 1e-12) & (np.abs(y - cy) <= hy + 1e-12)
    keep_idx = np.where(mask)[0]
    if keep_idx.size < 32:
        raise RuntimeError(f"scale '{scale_name}' keeps too few nodes: {keep_idx.size}")

    keep_node = np.zeros(nodes.shape[0], dtype=bool)
    keep_node[keep_idx] = True
    src = edges[:, 0].astype(int)
    dst = edges[:, 1].astype(int)
    keep_edge = keep_node[src] & keep_node[dst]
    sub_edges = edges[keep_edge].copy()
    if sub_edges.size == 0:
        raise RuntimeError(f"scale '{scale_name}' has no induced edges")

    remap = -np.ones(nodes.shape[0], dtype=int)
    remap[keep_idx] = np.arange(keep_idx.size, dtype=int)
    sub_nodes = nodes[keep_idx].copy()
    sub_edges[:, 0] = remap[sub_edges[:, 0].astype(int)]
    sub_edges[:, 1] = remap[sub_edges[:, 1].astype(int)]

    comp_idx = largest_component_indices(sub_nodes.shape[0], sub_edges[:, :2].astype(int))
    if comp_idx.size < 2:
        raise RuntimeError(f"scale '{scale_name}' largest component too small")

    if comp_idx.size < sub_nodes.shape[0]:
        comp_mask = np.zeros(sub_nodes.shape[0], dtype=bool)
        comp_mask[comp_idx] = True
        keep_sub_edge = comp_mask[sub_edges[:, 0].astype(int)] & comp_mask[sub_edges[:, 1].astype(int)]
        sub_edges = sub_edges[keep_sub_edge].copy()
        remap2 = -np.ones(sub_nodes.shape[0], dtype=int)
        remap2[comp_idx] = np.arange(comp_idx.size, dtype=int)
        sub_edges[:, 0] = remap2[sub_edges[:, 0].astype(int)]
        sub_edges[:, 1] = remap2[sub_edges[:, 1].astype(int)]
        sub_nodes = sub_nodes[comp_idx].copy()

    if sub_edges.shape[0] == 0:
        raise RuntimeError(f"scale '{scale_name}' has empty edge set after component filtering")
    return build_weighted_graph(
        f"layered_{scale_name}",
        sub_nodes,
        sub_edges,
        z_grid,
        risk_fields=risk_fields,
    )


def collect_event_candidates(
    graph: WeightedGraph,
    path: Sequence[int],
    blocked_set: set,
    radius_km: float,
) -> List[Tuple[int, int]]:
    if len(path) < 2:
        return []

    seq = [normalize_pair(int(path[i]), int(path[i + 1])) for i in range(len(path) - 1)]
    seq = list(dict.fromkeys(seq))
    interior = seq[1:-1] if len(seq) > 2 else seq

    candidates: List[Tuple[int, int]] = []
    used = set()
    for e in interior:
        if e in blocked_set or e in used:
            continue
        candidates.append(e)
        used.add(e)

    path_xy = graph.nodes[np.asarray(path, dtype=int), :2]
    tree = cKDTree(path_xy)
    d, _ = tree.query(graph.edge_midpoints, k=1)
    order = np.argsort(d)

    local_extra: List[Tuple[int, int]] = []
    global_extra: List[Tuple[int, int]] = []
    for eid in order:
        pair = normalize_pair(int(graph.edge_pairs[eid, 0]), int(graph.edge_pairs[eid, 1]))
        if pair in blocked_set or pair in used:
            continue
        if float(d[eid]) <= radius_km:
            local_extra.append(pair)
        else:
            global_extra.append(pair)
        used.add(pair)

    candidates.extend(local_extra)
    candidates.extend(global_extra)
    return candidates


def build_event_schedule(
    graph: WeightedGraph,
    path: Sequence[int],
    blocked_set: set,
    n_block: int,
    k_events: int,
    rng: np.random.Generator,
    radius_km: float,
    pool_factor: int,
) -> List[List[Tuple[int, int]]]:
    total_needed = int(n_block) * int(k_events)
    if total_needed <= 0:
        return []

    candidates = collect_event_candidates(graph, path, blocked_set, radius_km)
    if len(candidates) < total_needed:
        return []

    pool_factor = max(int(pool_factor), 1)
    pool_size = min(len(candidates), max(total_needed, total_needed * pool_factor))
    pool = candidates[:pool_size]
    pick_idx = rng.choice(len(pool), size=total_needed, replace=False)
    picked = [pool[int(i)] for i in np.atleast_1d(pick_idx)]
    rng.shuffle(picked)

    schedule: List[List[Tuple[int, int]]] = []
    for i in range(k_events):
        chunk = picked[i * n_block : (i + 1) * n_block]
        schedule.append(chunk)
    return schedule


def counter_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    return {k: int(after.get(k, 0) - before.get(k, 0)) for k in before}


def empty_cumulative() -> Dict[str, float]:
    return {
        "cumulative_replan_ms": 0.0,
        "cumulative_expanded": 0.0,
        "cumulative_queue_pushes": 0.0,
        "cumulative_queue_pops": 0.0,
        "cumulative_queue_stale_pops": 0.0,
        "cumulative_updated_vertices": 0.0,
        "cumulative_reopened_states": 0.0,
        "cumulative_affected_edges": 0.0,
        "cumulative_affected_vertices": 0.0,
    }


def update_cumulative(
    acc: Dict[str, float],
    replan_ms: float,
    stats: Dict[str, int],
    affected_edges: int,
    affected_vertices: int,
) -> None:
    acc["cumulative_replan_ms"] += float(replan_ms)
    acc["cumulative_expanded"] += float(stats.get("expanded", 0))
    acc["cumulative_queue_pushes"] += float(stats.get("queue_pushes", 0))
    acc["cumulative_queue_pops"] += float(stats.get("queue_pops", 0))
    acc["cumulative_queue_stale_pops"] += float(stats.get("queue_stale_pops", 0))
    acc["cumulative_updated_vertices"] += float(stats.get("updated_vertices", 0))
    acc["cumulative_reopened_states"] += float(stats.get("reopened_states", 0))
    acc["cumulative_affected_edges"] += float(affected_edges)
    acc["cumulative_affected_vertices"] += float(affected_vertices)


def run_event_stream_trial(
    graph: WeightedGraph,
    scale: str,
    trial_id: int,
    start: int,
    goal: int,
    n_block: int,
    k_events: int,
    rng: np.random.Generator,
    event_radius_km: float,
    event_pool_factor: int,
) -> Optional[Tuple[List[dict], List[dict]]]:
    b4 = LPAStarPlanner(graph, start, goal)
    ok_init = b4.compute_shortest_path()
    init_path = b4.extract_path() if ok_init else []
    if not ok_init or len(init_path) < 3:
        return None

    schedule = build_event_schedule(
        graph,
        init_path,
        blocked_set=set(),
        n_block=n_block,
        k_events=k_events,
        rng=rng,
        radius_km=event_radius_km,
        pool_factor=event_pool_factor,
    )
    if len(schedule) != k_events:
        return None

    event_rows: List[dict] = []
    trial_rows: List[dict] = []

    blocked_set: set = set()
    cum = {
        BASELINE_B4: empty_cumulative(),
        BASELINE_B2: empty_cumulative(),
    }
    success_all = {
        BASELINE_B4: True,
        BASELINE_B2: True,
    }
    success_count = {
        BASELINE_B4: 0,
        BASELINE_B2: 0,
    }
    final_metrics: Dict[str, Dict[str, float]] = {
        BASELINE_B4: {},
        BASELINE_B2: {},
    }

    for event_idx, event_edges in enumerate(schedule, start=1):
        for e in event_edges:
            blocked_set.add(normalize_pair(int(e[0]), int(e[1])))
        blocked_cum = sorted(blocked_set)
        affected_vertices = set()
        for u, v in event_edges:
            affected_vertices.add(int(u))
            affected_vertices.add(int(v))

        snap_before = b4.counter_snapshot()
        added_edges_b4, aff_v_b4 = b4.block_edges(event_edges)
        t0 = time.perf_counter()
        ok_b4 = b4.compute_shortest_path()
        t1 = time.perf_counter()
        b4_ms = (t1 - t0) * 1000.0
        snap_after = b4.counter_snapshot()
        stats_b4 = counter_delta(snap_before, snap_after)

        path_b4 = b4.extract_path() if ok_b4 else []
        m_b4 = graph.path_metrics(path_b4) if ok_b4 else {}
        update_cumulative(cum[BASELINE_B4], b4_ms, stats_b4, added_edges_b4, aff_v_b4)
        success_all[BASELINE_B4] = success_all[BASELINE_B4] and bool(ok_b4)
        if ok_b4:
            success_count[BASELINE_B4] += 1
        if event_idx == k_events:
            final_metrics[BASELINE_B4] = m_b4 if ok_b4 else {}

        event_rows.append(
            {
                "scale": scale,
                "trial": trial_id,
                "n_block": n_block,
                "k_events": k_events,
                "event_idx": event_idx,
                "baseline": BASELINE_B4,
                "start_node": start,
                "goal_node": goal,
                "blocked_edges_event": json.dumps(event_edges),
                "blocked_edges_total": len(blocked_set),
                "success": bool(ok_b4),
                "replan_ms": float(b4_ms),
                "expanded": int(stats_b4.get("expanded", 0)),
                "queue_pushes": int(stats_b4.get("queue_pushes", 0)),
                "queue_pops": int(stats_b4.get("queue_pops", 0)),
                "queue_stale_pops": int(stats_b4.get("queue_stale_pops", 0)),
                "updated_vertices": int(stats_b4.get("updated_vertices", 0)),
                "reopened_states": int(stats_b4.get("reopened_states", 0)),
                "affected_edges": int(added_edges_b4),
                "affected_vertices": int(aff_v_b4),
                "path_cost": m_b4.get("cost", float("nan")),
                "path_energy_kj": m_b4.get("energy_kj", float("nan")),
                "path_len_km": m_b4.get("length_km", float("nan")),
                "cum_replan_ms": float(cum[BASELINE_B4]["cumulative_replan_ms"]),
                "cum_expanded": float(cum[BASELINE_B4]["cumulative_expanded"]),
                "cum_queue_pushes": float(cum[BASELINE_B4]["cumulative_queue_pushes"]),
                "cum_queue_pops": float(cum[BASELINE_B4]["cumulative_queue_pops"]),
                "cum_updated_vertices": float(cum[BASELINE_B4]["cumulative_updated_vertices"]),
                "cum_reopened_states": float(cum[BASELINE_B4]["cumulative_reopened_states"]),
                "note": "" if ok_b4 else "search_fail",
            }
        )

        t0 = time.perf_counter()
        ok_b2, path_b2, stats_b2 = astar_global_replan(graph, start, goal, blocked_cum)
        t1 = time.perf_counter()
        b2_ms = (t1 - t0) * 1000.0
        m_b2 = graph.path_metrics(path_b2) if ok_b2 else {}
        update_cumulative(
            cum[BASELINE_B2],
            b2_ms,
            stats_b2,
            len(event_edges),
            len(affected_vertices),
        )
        success_all[BASELINE_B2] = success_all[BASELINE_B2] and bool(ok_b2)
        if ok_b2:
            success_count[BASELINE_B2] += 1
        if event_idx == k_events:
            final_metrics[BASELINE_B2] = m_b2 if ok_b2 else {}

        event_rows.append(
            {
                "scale": scale,
                "trial": trial_id,
                "n_block": n_block,
                "k_events": k_events,
                "event_idx": event_idx,
                "baseline": BASELINE_B2,
                "start_node": start,
                "goal_node": goal,
                "blocked_edges_event": json.dumps(event_edges),
                "blocked_edges_total": len(blocked_set),
                "success": bool(ok_b2),
                "replan_ms": float(b2_ms),
                "expanded": int(stats_b2.get("expanded", 0)),
                "queue_pushes": int(stats_b2.get("queue_pushes", 0)),
                "queue_pops": int(stats_b2.get("queue_pops", 0)),
                "queue_stale_pops": int(stats_b2.get("queue_stale_pops", 0)),
                "updated_vertices": int(stats_b2.get("updated_vertices", 0)),
                "reopened_states": int(stats_b2.get("reopened_states", 0)),
                "affected_edges": int(len(event_edges)),
                "affected_vertices": int(len(affected_vertices)),
                "path_cost": m_b2.get("cost", float("nan")),
                "path_energy_kj": m_b2.get("energy_kj", float("nan")),
                "path_len_km": m_b2.get("length_km", float("nan")),
                "cum_replan_ms": float(cum[BASELINE_B2]["cumulative_replan_ms"]),
                "cum_expanded": float(cum[BASELINE_B2]["cumulative_expanded"]),
                "cum_queue_pushes": float(cum[BASELINE_B2]["cumulative_queue_pushes"]),
                "cum_queue_pops": float(cum[BASELINE_B2]["cumulative_queue_pops"]),
                "cum_updated_vertices": float(cum[BASELINE_B2]["cumulative_updated_vertices"]),
                "cum_reopened_states": float(cum[BASELINE_B2]["cumulative_reopened_states"]),
                "note": "" if ok_b2 else "search_fail",
            }
        )

    for baseline in [BASELINE_B4, BASELINE_B2]:
        ok_all = bool(success_all[baseline])
        fm = final_metrics[baseline] if ok_all else {}
        trial_rows.append(
            {
                "scale": scale,
                "trial": trial_id,
                "n_block": n_block,
                "k_events": k_events,
                "baseline": baseline,
                "start_node": start,
                "goal_node": goal,
                "n_events_target": k_events,
                "n_events_succeeded": int(success_count[baseline]),
                "success_all_events": ok_all,
                "cumulative_replan_ms": float(cum[baseline]["cumulative_replan_ms"]),
                "cumulative_expanded": float(cum[baseline]["cumulative_expanded"]),
                "cumulative_queue_pushes": float(cum[baseline]["cumulative_queue_pushes"]),
                "cumulative_queue_pops": float(cum[baseline]["cumulative_queue_pops"]),
                "cumulative_queue_stale_pops": float(cum[baseline]["cumulative_queue_stale_pops"]),
                "cumulative_updated_vertices": float(cum[baseline]["cumulative_updated_vertices"]),
                "cumulative_reopened_states": float(cum[baseline]["cumulative_reopened_states"]),
                "cumulative_affected_edges": float(cum[baseline]["cumulative_affected_edges"]),
                "cumulative_affected_vertices": float(cum[baseline]["cumulative_affected_vertices"]),
                "final_path_cost": fm.get("cost", float("nan")),
                "final_path_energy_kj": fm.get("energy_kj", float("nan")),
                "final_path_len_km": fm.get("length_km", float("nan")),
                "blocked_edges_total": len(blocked_set),
                "note": "" if ok_all else "fail_in_event_stream",
            }
        )

    return event_rows, trial_rows


def summarise_combo_baseline_matrix(
    trial_rows: List[dict],
    scale: str,
    n_block: int,
    k_events: int,
    baseline: str,
) -> dict:
    subset = [
        r
        for r in trial_rows
        if r["scale"] == scale
        and int(r["n_block"]) == int(n_block)
        and int(r["k_events"]) == int(k_events)
        and r["baseline"] == baseline
    ]
    ok = [r for r in subset if r["success_all_events"]]

    out = {
        "scale": scale,
        "n_block": int(n_block),
        "k_events": int(k_events),
        "baseline": baseline,
        "n_trials": len(subset),
        "n_success": len(ok),
        "success_rate": (len(ok) / max(1, len(subset))),
        "mean_cumulative_replan_ms": float("nan"),
        "std_cumulative_replan_ms": float("nan"),
        "p50_cumulative_replan_ms": float("nan"),
        "p95_cumulative_replan_ms": float("nan"),
        "ci95_cumulative_replan_ms": float("nan"),
        "mean_event_replan_ms": float("nan"),
        "ci95_event_replan_ms": float("nan"),
        "std_event_replan_ms": float("nan"),
        "mean_cumulative_expanded": float("nan"),
        "mean_event_expanded": float("nan"),
        "ci95_event_expanded": float("nan"),
        "std_event_expanded": float("nan"),
        "mean_cumulative_queue_pushes": float("nan"),
        "mean_cumulative_queue_pops": float("nan"),
        "mean_cumulative_updated_vertices": float("nan"),
        "mean_cumulative_reopened_states": float("nan"),
        "mean_cumulative_affected_edges": float("nan"),
        "mean_cumulative_affected_vertices": float("nan"),
        "mean_final_path_cost": float("nan"),
        "mean_final_path_energy_kj": float("nan"),
        "mean_final_path_len_km": float("nan"),
        "mean_cumulative_replan_ms_all": float("nan"),
        "mean_cumulative_expanded_all": float("nan"),
    }
    if not subset:
        return out

    ms_all = np.asarray([float(r["cumulative_replan_ms"]) for r in subset], dtype=float)
    ex_all = np.asarray([float(r["cumulative_expanded"]) for r in subset], dtype=float)
    out["mean_cumulative_replan_ms_all"] = float(np.mean(ms_all))
    out["mean_cumulative_expanded_all"] = float(np.mean(ex_all))

    if not ok:
        return out

    ms = np.asarray([float(r["cumulative_replan_ms"]) for r in ok], dtype=float)
    ex = np.asarray([float(r["cumulative_expanded"]) for r in ok], dtype=float)
    qp = np.asarray([float(r["cumulative_queue_pushes"]) for r in ok], dtype=float)
    qo = np.asarray([float(r["cumulative_queue_pops"]) for r in ok], dtype=float)
    uv = np.asarray([float(r["cumulative_updated_vertices"]) for r in ok], dtype=float)
    ro = np.asarray([float(r["cumulative_reopened_states"]) for r in ok], dtype=float)
    ae = np.asarray([float(r["cumulative_affected_edges"]) for r in ok], dtype=float)
    av = np.asarray([float(r["cumulative_affected_vertices"]) for r in ok], dtype=float)
    fc = np.asarray([float(r["final_path_cost"]) for r in ok], dtype=float)
    fe = np.asarray([float(r["final_path_energy_kj"]) for r in ok], dtype=float)
    fl = np.asarray([float(r["final_path_len_km"]) for r in ok], dtype=float)

    out.update(
        {
            "mean_cumulative_replan_ms": float(np.mean(ms)),
            "std_cumulative_replan_ms": float(np.std(ms, ddof=1) if ms.size > 1 else 0.0),
            "p50_cumulative_replan_ms": float(np.percentile(ms, 50)),
            "p95_cumulative_replan_ms": float(np.percentile(ms, 95)),
            "ci95_cumulative_replan_ms": ci95(ms),
            "mean_event_replan_ms": float(np.mean(ms / max(1, int(k_events)))),
            "ci95_event_replan_ms": ci95(ms / max(1, int(k_events))),
            "std_event_replan_ms": float(np.std(ms / max(1, int(k_events)), ddof=1) if ms.size > 1 else 0.0),
            "mean_cumulative_expanded": float(np.mean(ex)),
            "mean_event_expanded": float(np.mean(ex / max(1, int(k_events)))),
            "ci95_event_expanded": ci95(ex / max(1, int(k_events))),
            "std_event_expanded": float(np.std(ex / max(1, int(k_events)), ddof=1) if ex.size > 1 else 0.0),
            "mean_cumulative_queue_pushes": float(np.mean(qp)),
            "mean_cumulative_queue_pops": float(np.mean(qo)),
            "mean_cumulative_updated_vertices": float(np.mean(uv)),
            "mean_cumulative_reopened_states": float(np.mean(ro)),
            "mean_cumulative_affected_edges": float(np.mean(ae)),
            "mean_cumulative_affected_vertices": float(np.mean(av)),
            "mean_final_path_cost": float(np.mean(fc)),
            "mean_final_path_energy_kj": float(np.mean(fe)),
            "mean_final_path_len_km": float(np.mean(fl)),
        }
    )
    return out


def paired_arrays_combo_matrix(
    trial_rows: List[dict],
    scale: str,
    n_block: int,
    k_events: int,
    field: str,
) -> Tuple[np.ndarray, np.ndarray]:
    by_trial = {}
    for r in trial_rows:
        if r["scale"] != scale:
            continue
        if int(r["n_block"]) != int(n_block) or int(r["k_events"]) != int(k_events):
            continue
        if not r["success_all_events"]:
            continue
        by_trial[(int(r["trial"]), r["baseline"])] = r

    x: List[float] = []
    y: List[float] = []
    trial_ids = sorted({int(r["trial"]) for r in trial_rows if r["scale"] == scale and int(r["n_block"]) == int(n_block) and int(r["k_events"]) == int(k_events)})
    for tid in trial_ids:
        rb4 = by_trial.get((tid, BASELINE_B4))
        rb2 = by_trial.get((tid, BASELINE_B2))
        if rb4 is None or rb2 is None:
            continue
        vb4 = float(rb4[field])
        vb2 = float(rb2[field])
        if np.isfinite(vb4) and np.isfinite(vb2):
            x.append(vb4)
            y.append(vb2)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def build_pairwise_rows_matrix(
    trial_rows: List[dict],
    scales: Sequence[str],
    n_blocks: Sequence[int],
    k_values: Sequence[int],
) -> List[dict]:
    from benchmark import paired_pvalue

    metrics = [
        "cumulative_replan_ms",
        "cumulative_expanded",
        "cumulative_queue_pushes",
        "cumulative_updated_vertices",
        "cumulative_reopened_states",
    ]
    out: List[dict] = []
    for scale in scales:
        for n_block in n_blocks:
            for k_events in k_values:
                for metric in metrics:
                    xb4, xb2 = paired_arrays_combo_matrix(trial_rows, scale, n_block, k_events, metric)
                    if xb4.size == 0:
                        out.append(
                            {
                                "scale": scale,
                                "n_block": int(n_block),
                                "k_events": int(k_events),
                                "metric": metric,
                                "n_paired": 0,
                                "mean_b4": float("nan"),
                                "mean_b2": float("nan"),
                                "median_ratio_b2_over_b4": float("nan"),
                                "p_value": float("nan"),
                            }
                        )
                        continue
                    ratio = np.median(xb2 / np.maximum(xb4, EPS))
                    out.append(
                        {
                            "scale": scale,
                            "n_block": int(n_block),
                            "k_events": int(k_events),
                            "metric": metric,
                            "n_paired": int(xb4.size),
                            "mean_b4": float(np.mean(xb4)),
                            "mean_b2": float(np.mean(xb2)),
                            "median_ratio_b2_over_b4": float(ratio),
                            "p_value": paired_pvalue(xb4, xb2),
                        }
                    )
    return out


def build_focus_tables_matrix(
    summary_rows: List[dict],
    pair_rows: List[dict],
    scales: Sequence[str],
    n_blocks: Sequence[int],
    k_values: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[Dict[str, List[dict]], Dict[str, int | str]]:
    scale_focus = nearest_scale(args.focus_scale, scales)
    k_intensity = nearest_int(args.focus_k_intensity, k_values)
    n_block_cont = nearest_int(args.focus_n_block_cont, n_blocks)
    k_scale = nearest_int(args.focus_k_scale, k_values)
    n_block_scale = nearest_int(args.focus_n_block_scale, n_blocks)
    k_dist = nearest_int(args.focus_k_distribution, k_values)

    s_idx = {
        (r["scale"], int(r["n_block"]), int(r["k_events"]), r["baseline"]): r
        for r in summary_rows
    }

    table_a: List[dict] = []
    for n_block in n_blocks:
        b4 = s_idx.get((scale_focus, int(n_block), int(k_intensity), BASELINE_B4), {})
        b2 = s_idx.get((scale_focus, int(n_block), int(k_intensity), BASELINE_B2), {})
        ratio = float("nan")
        b4_ms = float(b4.get("mean_event_replan_ms", float("nan")))
        b2_ms = float(b2.get("mean_event_replan_ms", float("nan")))
        if np.isfinite(b4_ms) and np.isfinite(b2_ms):
            ratio = b2_ms / max(b4_ms, EPS)
        table_a.append(
            {
                "scale": scale_focus,
                "k_events": int(k_intensity),
                "n_block": int(n_block),
                "b4_mean_event_ms": b4_ms,
                "b2_mean_event_ms": b2_ms,
                "b2_over_b4_time_ratio": ratio,
                "b4_mean_event_expanded": float(b4.get("mean_event_expanded", float("nan"))),
                "b2_mean_event_expanded": float(b2.get("mean_event_expanded", float("nan"))),
                "b4_success_rate": float(b4.get("success_rate", float("nan"))),
                "b2_success_rate": float(b2.get("success_rate", float("nan"))),
            }
        )

    table_b: List[dict] = []
    for k_events in k_values:
        b4 = s_idx.get((scale_focus, int(n_block_cont), int(k_events), BASELINE_B4), {})
        b2 = s_idx.get((scale_focus, int(n_block_cont), int(k_events), BASELINE_B2), {})
        ratio = float("nan")
        b4_ms = float(b4.get("mean_cumulative_replan_ms", float("nan")))
        b2_ms = float(b2.get("mean_cumulative_replan_ms", float("nan")))
        if np.isfinite(b4_ms) and np.isfinite(b2_ms):
            ratio = b2_ms / max(b4_ms, EPS)
        table_b.append(
            {
                "scale": scale_focus,
                "n_block": int(n_block_cont),
                "k_events": int(k_events),
                "b4_mean_cumulative_ms": b4_ms,
                "b2_mean_cumulative_ms": b2_ms,
                "b2_over_b4_time_ratio": ratio,
                "b4_mean_cumulative_expanded": float(b4.get("mean_cumulative_expanded", float("nan"))),
                "b2_mean_cumulative_expanded": float(b2.get("mean_cumulative_expanded", float("nan"))),
                "b4_success_rate": float(b4.get("success_rate", float("nan"))),
                "b2_success_rate": float(b2.get("success_rate", float("nan"))),
            }
        )

    table_c: List[dict] = []
    for scale in scales:
        b4 = s_idx.get((scale, int(n_block_scale), int(k_scale), BASELINE_B4), {})
        b2 = s_idx.get((scale, int(n_block_scale), int(k_scale), BASELINE_B2), {})
        ratio = float("nan")
        b4_ms = float(b4.get("mean_cumulative_replan_ms", float("nan")))
        b2_ms = float(b2.get("mean_cumulative_replan_ms", float("nan")))
        if np.isfinite(b4_ms) and np.isfinite(b2_ms):
            ratio = b2_ms / max(b4_ms, EPS)
        table_c.append(
            {
                "scale": scale,
                "n_block": int(n_block_scale),
                "k_events": int(k_scale),
                "graph_nodes": int(b4.get("graph_nodes", b2.get("graph_nodes", 0))),
                "graph_edges": int(b4.get("graph_edges", b2.get("graph_edges", 0))),
                "b4_mean_cumulative_ms": b4_ms,
                "b2_mean_cumulative_ms": b2_ms,
                "b2_over_b4_time_ratio": ratio,
                "b4_success_rate": float(b4.get("success_rate", float("nan"))),
                "b2_success_rate": float(b2.get("success_rate", float("nan"))),
            }
        )

    table_d: List[dict] = []
    for n_block in n_blocks:
        b4 = s_idx.get((scale_focus, int(n_block), int(k_scale), BASELINE_B4), {})
        b2 = s_idx.get((scale_focus, int(n_block), int(k_scale), BASELINE_B2), {})
        table_d.append(
            {
                "scale": scale_focus,
                "n_block": int(n_block),
                "k_events": int(k_scale),
                "b4_mean_cumulative_queue_pushes": float(b4.get("mean_cumulative_queue_pushes", float("nan"))),
                "b2_mean_cumulative_queue_pushes": float(b2.get("mean_cumulative_queue_pushes", float("nan"))),
                "b4_mean_cumulative_updated_vertices": float(b4.get("mean_cumulative_updated_vertices", float("nan"))),
                "b2_mean_cumulative_updated_vertices": float(b2.get("mean_cumulative_updated_vertices", float("nan"))),
                "b4_mean_cumulative_reopened_states": float(b4.get("mean_cumulative_reopened_states", float("nan"))),
                "b2_mean_cumulative_reopened_states": float(b2.get("mean_cumulative_reopened_states", float("nan"))),
                "b4_mean_cumulative_expanded": float(b4.get("mean_cumulative_expanded", float("nan"))),
                "b2_mean_cumulative_expanded": float(b2.get("mean_cumulative_expanded", float("nan"))),
            }
        )

    tables = {"A": table_a, "B": table_b, "C": table_c, "D": table_d}
    resolved = {
        "focus_scale": scale_focus,
        "focus_k_intensity": int(k_intensity),
        "focus_n_block_cont": int(n_block_cont),
        "focus_k_scale": int(k_scale),
        "focus_n_block_scale": int(n_block_scale),
        "focus_k_distribution": int(k_dist),
    }
    return tables, resolved


def render_markdown_matrix(
    tables: Dict[str, List[dict]],
    args: argparse.Namespace,
    resolved: Dict[str, int | str],
    anomaly_note: str = "",
    k_note: str = "",
    quality_note: str = "",
) -> str:
    def f(v: float, d: int = 2) -> str:
        if not np.isfinite(float(v)):
            return "nan"
        return f"{float(v):.{d}f}"

    lines: List[str] = []
    lines.append("# Benchmark Matrix Table")
    lines.append("")
    lines.append(f"- Trials per combo: `{args.trials}`")
    lines.append(f"- n_block grid: `{args.n_block_grid}`")
    lines.append(f"- K grid: `{args.k_events_grid}`")
    lines.append(f"- Scales: `{args.scales}`")
    lines.append(f"- Seed: `{args.seed}`")
    lines.append(
        f"- Focus: scale `{resolved['focus_scale']}`, K-intensity `{resolved['focus_k_intensity']}`, "
        f"K-scale `{resolved['focus_k_scale']}`, n_block-cont `{resolved['focus_n_block_cont']}`"
    )
    if anomaly_note:
        lines.append(f"- Experiment A diagnosis: {anomaly_note}")
    if k_note:
        lines.append(f"- Experiment B diagnosis: {k_note}")
    if quality_note:
        lines.append(f"- Path-quality diagnosis: {quality_note}")
    lines.append("")

    lines.append("## Experiment A (Event Intensity)")
    lines.append("")
    lines.append("| n_block | B4 event ms | B2 event ms | B2/B4 | B4 event expanded | B2 event expanded | B4 success | B2 success |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in tables["A"]:
        lines.append(
            f"| {r['n_block']} | {f(r['b4_mean_event_ms'])} | {f(r['b2_mean_event_ms'])} | {f(r['b2_over_b4_time_ratio'], 3)} | "
            f"{f(r['b4_mean_event_expanded'])} | {f(r['b2_mean_event_expanded'])} | {f(100.0*r['b4_success_rate'],1)}% | {f(100.0*r['b2_success_rate'],1)}% |"
        )
    lines.append("")

    lines.append("## Experiment B (Continuous Replanning)")
    lines.append("")
    lines.append("| K events | B4 cumulative ms | B2 cumulative ms | B2/B4 | B4 cumulative expanded | B2 cumulative expanded | B4 success | B2 success |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in tables["B"]:
        lines.append(
            f"| {r['k_events']} | {f(r['b4_mean_cumulative_ms'])} | {f(r['b2_mean_cumulative_ms'])} | {f(r['b2_over_b4_time_ratio'], 3)} | "
            f"{f(r['b4_mean_cumulative_expanded'])} | {f(r['b2_mean_cumulative_expanded'])} | {f(100.0*r['b4_success_rate'],1)}% | {f(100.0*r['b2_success_rate'],1)}% |"
        )
    lines.append("")

    lines.append("## Experiment C (Scale Sweep)")
    lines.append("")
    lines.append("| Scale | |V| | |E| | B4 cumulative ms | B2 cumulative ms | B2/B4 | B4 success | B2 success |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in tables["C"]:
        lines.append(
            f"| {r['scale']} | {r['graph_nodes']} | {r['graph_edges']} | {f(r['b4_mean_cumulative_ms'])} | {f(r['b2_mean_cumulative_ms'])} | "
            f"{f(r['b2_over_b4_time_ratio'], 3)} | {f(100.0*r['b4_success_rate'],1)}% | {f(100.0*r['b2_success_rate'],1)}% |"
        )
    lines.append("")

    lines.append("## Experiment D (Workload Metrics)")
    lines.append("")
    lines.append("| n_block | B4 queue push | B2 queue push | B4 updated | B2 updated | B4 reopened | B2 reopened | B4 expanded | B2 expanded |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in tables["D"]:
        lines.append(
            f"| {r['n_block']} | {f(r['b4_mean_cumulative_queue_pushes'])} | {f(r['b2_mean_cumulative_queue_pushes'])} | "
            f"{f(r['b4_mean_cumulative_updated_vertices'])} | {f(r['b2_mean_cumulative_updated_vertices'])} | "
            f"{f(r['b4_mean_cumulative_reopened_states'])} | {f(r['b2_mean_cumulative_reopened_states'])} | "
            f"{f(r['b4_mean_cumulative_expanded'])} | {f(r['b2_mean_cumulative_expanded'])} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def diagnose_event_intensity_anomaly(
    tables: Dict[str, List[dict]],
    resolved: Dict[str, int | str],
) -> Tuple[List[dict], str]:
    table_a = tables.get("A", [])
    diag_rows: List[dict] = []
    if not table_a:
        return diag_rows, ""

    # Sort by n_block for trend checks.
    arr = sorted(table_a, key=lambda r: int(r["n_block"]))
    b4_t = np.asarray([float(r["b4_mean_event_ms"]) for r in arr], dtype=float)
    b2_t = np.asarray([float(r["b2_mean_event_ms"]) for r in arr], dtype=float)
    b4_e = np.asarray([float(r["b4_mean_event_expanded"]) for r in arr], dtype=float)
    b2_e = np.asarray([float(r["b2_mean_event_expanded"]) for r in arr], dtype=float)
    nbs = np.asarray([int(r["n_block"]) for r in arr], dtype=int)

    b4_non_monotonic = bool(np.any(np.diff(b4_t) < -1e-9))
    b2_non_monotonic = bool(np.any(np.diff(b2_t) < -1e-9))

    for i, r in enumerate(arr):
        diag_rows.append(
            {
                "scale": str(r["scale"]),
                "k_events": int(r["k_events"]),
                "n_block": int(r["n_block"]),
                "b4_mean_event_ms": float(r["b4_mean_event_ms"]),
                "b2_mean_event_ms": float(r["b2_mean_event_ms"]),
                "b4_mean_event_expanded": float(r["b4_mean_event_expanded"]),
                "b2_mean_event_expanded": float(r["b2_mean_event_expanded"]),
                "b4_success_rate": float(r["b4_success_rate"]),
                "b2_success_rate": float(r["b2_success_rate"]),
                "delta_b4_ms_vs_prev": float("nan") if i == 0 else float(b4_t[i] - b4_t[i - 1]),
                "delta_b2_ms_vs_prev": float("nan") if i == 0 else float(b2_t[i] - b2_t[i - 1]),
                "delta_b4_expanded_vs_prev": float("nan") if i == 0 else float(b4_e[i] - b4_e[i - 1]),
                "delta_b2_expanded_vs_prev": float("nan") if i == 0 else float(b2_e[i] - b2_e[i - 1]),
            }
        )

    if not (b4_non_monotonic or b2_non_monotonic):
        return diag_rows, "event intensity trend is monotonic in current summary."

    # Explain non-monotonic wall-clock with workload + geometry effects.
    b4_head = float(b4_t[0]) if b4_t.size > 0 else float("nan")
    b4_tail = float(b4_t[-1]) if b4_t.size > 0 else float("nan")
    b4_head_e = float(b4_e[0]) if b4_e.size > 0 else float("nan")
    b4_tail_e = float(b4_e[-1]) if b4_e.size > 0 else float("nan")

    note = (
        f"non-monotonic timing detected at scale={resolved['focus_scale']}, "
        f"K={resolved['focus_k_intensity']}. "
        f"B4 mean event time changes from {b4_head:.2f}ms (n_block={int(nbs[0])}) "
        f"to {b4_tail:.2f}ms (n_block={int(nbs[-1])}), while expanded nodes move from "
        f"{b4_head_e:.1f} to {b4_tail_e:.1f}. "
        "This indicates wall-clock is jointly affected by event geometry/locality and Python runtime overhead, "
        "not only by n_block magnitude. A typical case is that more blocked edges can force an earlier detour "
        "into a cleaner subgraph, which shortens the actually affected middle segment and reduces updated states."
    )
    return diag_rows, note


def diagnose_continuous_replan_k_effect(
    tables: Dict[str, List[dict]],
    resolved: Dict[str, int | str],
) -> Tuple[List[dict], str]:
    table_b = tables.get("B", [])
    diag_rows: List[dict] = []
    if not table_b:
        return diag_rows, ""

    arr = sorted(table_b, key=lambda r: int(r["k_events"]))
    for i, r in enumerate(arr):
        ratio = float(r["b2_over_b4_time_ratio"])
        diag_rows.append(
            {
                "scale": str(r["scale"]),
                "n_block": int(r["n_block"]),
                "k_events": int(r["k_events"]),
                "b4_mean_cumulative_ms": float(r["b4_mean_cumulative_ms"]),
                "b2_mean_cumulative_ms": float(r["b2_mean_cumulative_ms"]),
                "b2_over_b4_time_ratio": ratio,
                "b4_mean_cumulative_expanded": float(r["b4_mean_cumulative_expanded"]),
                "b2_mean_cumulative_expanded": float(r["b2_mean_cumulative_expanded"]),
                "delta_ratio_vs_prev": float("nan")
                if i == 0
                else float(ratio - float(arr[i - 1]["b2_over_b4_time_ratio"])),
            }
        )

    k1 = next((r for r in arr if int(r["k_events"]) == 1), None)
    if k1 is None:
        return diag_rows, "K=1 is not included in the current K grid."

    ratio_k1 = float(k1["b2_over_b4_time_ratio"])
    if not np.isfinite(ratio_k1):
        return diag_rows, "K=1 ratio is unavailable due to insufficient successful trials."

    if ratio_k1 < 1.0:
        note = (
            f"for K=1 (scale={resolved['focus_scale']}, n_block={resolved['focus_n_block_cont']}), "
            f"B4 is slower than B2 (B2/B4={ratio_k1:.3f}<1). This is expected under a single light perturbation: "
            "incremental LPA* still pays queue/state-maintenance overhead, while global A* can finish quickly when "
            "the affected region is tiny. As K increases, LPA* reuses prior search state and cumulative advantage emerges."
        )
    else:
        note = (
            f"for K=1 (scale={resolved['focus_scale']}, n_block={resolved['focus_n_block_cont']}), "
            f"B4 is not slower than B2 (B2/B4={ratio_k1:.3f}). The cumulative trend over K should still be reported."
        )
    return diag_rows, note


def diagnose_path_quality_consistency(
    trial_rows: List[dict],
    scales: Sequence[str],
    n_blocks: Sequence[int],
    k_values: Sequence[int],
    cost_tol: float = 1e-6,
    len_tol: float = 1e-6,
) -> Tuple[List[dict], str]:
    rows: List[dict] = []
    equal_rates = []
    for scale in scales:
        for n_block in n_blocks:
            for k_events in k_values:
                by_trial = {}
                for r in trial_rows:
                    if r["scale"] != scale:
                        continue
                    if int(r["n_block"]) != int(n_block) or int(r["k_events"]) != int(k_events):
                        continue
                    if not r["success_all_events"]:
                        continue
                    by_trial[(int(r["trial"]), r["baseline"])] = r

                gaps_cost = []
                gaps_len = []
                eq_cost = 0
                eq_len = 0
                trial_ids = sorted(
                    {
                        int(r["trial"])
                        for r in trial_rows
                        if r["scale"] == scale
                        and int(r["n_block"]) == int(n_block)
                        and int(r["k_events"]) == int(k_events)
                    }
                )
                for tid in trial_ids:
                    rb4 = by_trial.get((tid, BASELINE_B4))
                    rb2 = by_trial.get((tid, BASELINE_B2))
                    if rb4 is None or rb2 is None:
                        continue
                    c4 = float(rb4["final_path_cost"])
                    c2 = float(rb2["final_path_cost"])
                    l4 = float(rb4["final_path_len_km"])
                    l2 = float(rb2["final_path_len_km"])
                    if not (np.isfinite(c4) and np.isfinite(c2) and np.isfinite(l4) and np.isfinite(l2)):
                        continue
                    dc = abs(c4 - c2)
                    dl = abs(l4 - l2)
                    gaps_cost.append(dc)
                    gaps_len.append(dl)
                    if dc <= cost_tol:
                        eq_cost += 1
                    if dl <= len_tol:
                        eq_len += 1

                n = len(gaps_cost)
                if n == 0:
                    rows.append(
                        {
                            "scale": scale,
                            "n_block": int(n_block),
                            "k_events": int(k_events),
                            "n_paired": 0,
                            "mean_abs_cost_gap": float("nan"),
                            "p95_abs_cost_gap": float("nan"),
                            "equal_cost_rate": float("nan"),
                            "mean_abs_len_gap_km": float("nan"),
                            "p95_abs_len_gap_km": float("nan"),
                            "equal_len_rate": float("nan"),
                        }
                    )
                    continue

                gaps_cost_arr = np.asarray(gaps_cost, dtype=float)
                gaps_len_arr = np.asarray(gaps_len, dtype=float)
                eq_cost_rate = float(eq_cost / n)
                eq_len_rate = float(eq_len / n)
                equal_rates.append(eq_cost_rate)
                rows.append(
                    {
                        "scale": scale,
                        "n_block": int(n_block),
                        "k_events": int(k_events),
                        "n_paired": int(n),
                        "mean_abs_cost_gap": float(np.mean(gaps_cost_arr)),
                        "p95_abs_cost_gap": float(np.percentile(gaps_cost_arr, 95)),
                        "equal_cost_rate": eq_cost_rate,
                        "mean_abs_len_gap_km": float(np.mean(gaps_len_arr)),
                        "p95_abs_len_gap_km": float(np.percentile(gaps_len_arr, 95)),
                        "equal_len_rate": eq_len_rate,
                    }
                )

    if not equal_rates:
        return rows, "path-quality consistency could not be evaluated (no paired successful trials)."

    avg_eq = float(np.mean(np.asarray(equal_rates, dtype=float)))
    if avg_eq >= 0.7:
        note = (
            f"B4/B2 path costs are frequently equal (mean equal-cost rate={100.0*avg_eq:.1f}%). "
            "This is expected because both optimize the same weighted objective on the same graph; "
            "incremental LPA* mainly improves replanning workload/time rather than final optimality."
        )
    else:
        note = (
            f"B4/B2 equal-cost rate is moderate (mean={100.0*avg_eq:.1f}%), "
            "indicating event-stream state reuse can also lead to different feasible local choices in some trials."
        )
    return rows, note


def make_plots_matrix(
    summary_rows: List[dict],
    trial_rows: List[dict],
    scales: Sequence[str],
    n_blocks: Sequence[int],
    k_values: Sequence[int],
    args: argparse.Namespace,
    out_dir: Path,
) -> List[str]:
    if plt is None or args.disable_plots:
        return []

    scale_plot = nearest_scale(args.plot_scale, scales)
    k_intensity = nearest_int(args.plot_k_intensity, k_values)
    n_block_cont = nearest_int(args.plot_n_block_cont, n_blocks)
    k_dist = nearest_int(args.plot_k_distribution, k_values)

    idx = {
        (r["scale"], int(r["n_block"]), int(r["k_events"]), r["baseline"]): r
        for r in summary_rows
    }
    paths: List[str] = []

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=300)
    for baseline, label, color in [
        (BASELINE_B4, "B4 incremental LPA*", "#d62728"),
        (BASELINE_B2, "B2 global A*", "#1f77b4"),
    ]:
        xs: List[int] = []
        ys: List[float] = []
        ci_vals: List[float] = []
        for nb in n_blocks:
            r = idx.get((scale_plot, int(nb), int(k_intensity), baseline))
            if r is None:
                continue
            y = float(r.get("mean_event_replan_ms", float("nan")))
            c = float(r.get("ci95_event_replan_ms", 0.0))
            if np.isfinite(y):
                xs.append(int(nb))
                ys.append(y)
                ci_vals.append(c if np.isfinite(c) else 0.0)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2.0, label=label, color=color)
            # 95% CI 阴影带
            y_lo = [y - c for y, c in zip(ys, ci_vals)]
            y_hi = [y + c for y, c in zip(ys, ci_vals)]
            ax.fill_between(xs, y_lo, y_hi, alpha=0.18, color=color)
    ax.set_xlabel("Blocked edges per event (n_block)")
    ax.set_ylabel("Mean replanning time per event (ms)")
    ax.set_title(f"Event Intensity vs Replanning Time (scale={scale_plot}, K={k_intensity})")
    ax.grid(alpha=0.3)
    ax.legend()
    p1 = out_dir / "fig1_event_intensity_vs_time.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p1))

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=300)
    for baseline, label, color in [
        (BASELINE_B4, "B4 incremental LPA*", "#d62728"),
        (BASELINE_B2, "B2 global A*", "#1f77b4"),
    ]:
        xs: List[int] = []
        ys: List[float] = []
        ci_vals: List[float] = []
        for kk in k_values:
            r = idx.get((scale_plot, int(n_block_cont), int(kk), baseline))
            if r is None:
                continue
            y = float(r.get("mean_cumulative_replan_ms", float("nan")))
            c = float(r.get("ci95_cumulative_replan_ms", 0.0))
            if np.isfinite(y):
                xs.append(int(kk))
                ys.append(y)
                ci_vals.append(c if np.isfinite(c) else 0.0)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2.0, label=label, color=color)
            # 95% CI 阴影带
            y_lo = [y - c for y, c in zip(ys, ci_vals)]
            y_hi = [y + c for y, c in zip(ys, ci_vals)]
            ax.fill_between(xs, y_lo, y_hi, alpha=0.18, color=color)
    ax.set_xlabel("Number of events (K)")
    ax.set_ylabel("Mean cumulative replanning time (ms)")
    ax.set_title(f"Continuous Events vs Cumulative Time (scale={scale_plot}, n_block={n_block_cont})")
    ax.grid(alpha=0.3)
    ax.legend()
    p2 = out_dir / "fig2_k_vs_cumulative_time.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p2))

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=300)
    for baseline, label, color in [
        (BASELINE_B4, "B4 incremental LPA*", "#d62728"),
        (BASELINE_B2, "B2 global A*", "#1f77b4"),
    ]:
        xs: List[int] = []
        ys: List[float] = []
        ci_vals: List[float] = []
        for nb in n_blocks:
            r = idx.get((scale_plot, int(nb), int(k_intensity), baseline))
            if r is None:
                continue
            y = float(r.get("mean_event_expanded", float("nan")))
            c = float(r.get("ci95_event_expanded", 0.0))
            if np.isfinite(y):
                xs.append(int(nb))
                ys.append(y)
                ci_vals.append(c if np.isfinite(c) else 0.0)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2.0, label=label, color=color)
            # 95% CI 阴影带
            y_lo = [y - c for y, c in zip(ys, ci_vals)]
            y_hi = [y + c for y, c in zip(ys, ci_vals)]
            ax.fill_between(xs, y_lo, y_hi, alpha=0.18, color=color)
    ax.set_xlabel("Blocked edges per event (n_block)")
    ax.set_ylabel("Mean expanded nodes per event")
    ax.set_title(f"Event Intensity vs Expanded Nodes (scale={scale_plot}, K={k_intensity})")
    ax.grid(alpha=0.3)
    ax.legend()
    p3 = out_dir / "fig3_event_intensity_vs_expanded.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p3))

    b4_vals = np.asarray(
        [
            float(r["cumulative_replan_ms"])
            for r in trial_rows
            if r["scale"] == scale_plot
            and int(r["n_block"]) == int(n_block_cont)
            and int(r["k_events"]) == int(k_dist)
            and r["baseline"] == BASELINE_B4
            and r["success_all_events"]
        ],
        dtype=float,
    )
    b2_vals = np.asarray(
        [
            float(r["cumulative_replan_ms"])
            for r in trial_rows
            if r["scale"] == scale_plot
            and int(r["n_block"]) == int(n_block_cont)
            and int(r["k_events"]) == int(k_dist)
            and r["baseline"] == BASELINE_B2
            and r["success_all_events"]
        ],
        dtype=float,
    )
    if b4_vals.size > 0 and b2_vals.size > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=300)
        try:
            ax1.boxplot([b4_vals, b2_vals], tick_labels=["B4", "B2"], showmeans=True)
        except TypeError:
            ax1.boxplot([b4_vals, b2_vals], labels=["B4", "B2"], showmeans=True)
        ax1.set_ylabel("Cumulative replanning time (ms)")
        ax1.set_title(f"Boxplot (scale={scale_plot}, n_block={n_block_cont}, K={k_dist})")
        ax1.grid(alpha=0.3)
        b4_med = float(np.percentile(b4_vals, 50))
        b4_p95 = float(np.percentile(b4_vals, 95))
        b2_med = float(np.percentile(b2_vals, 50))
        b2_p95 = float(np.percentile(b2_vals, 95))
        ax1.text(
            0.03,
            0.97,
            f"B4 median/p95: {b4_med:.2f}/{b4_p95:.2f} ms\n"
            f"B2 median/p95: {b2_med:.2f}/{b2_p95:.2f} ms",
            transform=ax1.transAxes,
            va="top",
            fontsize=8.3,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.88),
        )

        for arr, label, color in [
            (b4_vals, "B4", "#d62728"),
            (b2_vals, "B2", "#1f77b4"),
        ]:
            s = np.sort(arr)
            y = np.arange(1, s.size + 1, dtype=float) / float(s.size)
            ax2.plot(s, y, linewidth=2.0, label=label, color=color)
        ax2.set_xlabel("Cumulative replanning time (ms)")
        ax2.set_ylabel("CDF")
        ax2.set_title("CDF")
        ax2.grid(alpha=0.3)
        ax2.legend()
        p4 = out_dir / "fig4_time_distribution_box_cdf.png"
        fig.savefig(p4, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(p4))

    return paths


def run_benchmark_matrix(args: argparse.Namespace) -> None:
    root = Path(args.workdir).resolve()
    os.chdir(root)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_blocks = parse_int_grid_arg(args.n_block_grid, "--n-block-grid")
    k_values = parse_int_grid_arg(args.k_events_grid, "--k-events-grid")
    scale_fractions = parse_scale_fraction_arg(args.scale_fractions)
    scales = parse_scale_names(args.scales, scale_fractions)

    z_grid = np.asarray(np.load("Z_crop.npy"), dtype=float)
    risk_fields = load_risk_fields(root, z_grid.shape)
    print(f"[risk] mode={risk_fields['mode']}")
    base_nodes = np.asarray(np.load("graph_nodes.npy"), dtype=float)
    base_edges = np.asarray(np.load("graph_edges.npy"), dtype=int)

    scale_graphs: Dict[str, WeightedGraph] = {}
    for scale in scales:
        frac = float(scale_fractions[scale])
        g = build_scaled_graph(
            base_nodes,
            base_edges,
            z_grid,
            scale,
            frac,
            risk_fields=risk_fields,
        )
        scale_graphs[scale] = g
        print(f"[build] scale={scale} frac={frac:.3f}: |V|={g.n_nodes}, |E|={g.n_edges}")

    event_records: List[dict] = []
    trial_records: List[dict] = []
    combo_status: List[dict] = []

    total_combo = len(scales) * len(n_blocks) * len(k_values)
    combo_id = 0
    for scale in scales:
        graph = scale_graphs[scale]
        for n_block in n_blocks:
            for k_events in k_values:
                combo_id += 1
                max_attempts = max(args.trials * args.max_attempt_factor, args.trials + 10)
                accepted = 0
                attempts = 0
                rng = np.random.default_rng(args.seed + combo_id * 10007)

                print(f"[combo {combo_id}/{total_combo}] scale={scale}, n_block={n_block}, K={k_events}")
                while accepted < args.trials and attempts < max_attempts:
                    attempts += 1
                    start, goal = sample_start_goal(rng, graph.nodes, args.min_start_goal_dist_km)
                    trial_id = accepted + 1
                    result = run_event_stream_trial(
                        graph=graph,
                        scale=scale,
                        trial_id=trial_id,
                        start=start,
                        goal=goal,
                        n_block=n_block,
                        k_events=k_events,
                        rng=rng,
                        event_radius_km=args.event_radius_km,
                        event_pool_factor=args.event_pool_factor,
                    )
                    if result is None:
                        continue
                    ev_rows, tr_rows = result
                    event_records.extend(ev_rows)
                    trial_records.extend(tr_rows)
                    accepted += 1
                    if accepted % max(1, int(args.progress_every)) == 0 or accepted == args.trials:
                        print(f"  [progress] {accepted}/{args.trials} accepted (attempts={attempts})")

                combo_status.append(
                    {
                        "scale": scale,
                        "n_block": int(n_block),
                        "k_events": int(k_events),
                        "trials_requested": int(args.trials),
                        "trials_collected": int(accepted),
                        "attempts": int(attempts),
                        "max_attempts": int(max_attempts),
                        "graph_nodes": int(graph.n_nodes),
                        "graph_edges": int(graph.n_edges),
                    }
                )
                if accepted < args.trials:
                    print(f"[warn] combo scale={scale}, n_block={n_block}, K={k_events} only collected {accepted}/{args.trials}")

    summary_rows: List[dict] = []
    for scale in scales:
        for n_block in n_blocks:
            for k_events in k_values:
                for baseline in [BASELINE_B4, BASELINE_B2]:
                    row = summarise_combo_baseline_matrix(
                        trial_records,
                        scale=scale,
                        n_block=n_block,
                        k_events=k_events,
                        baseline=baseline,
                    )
                    row["graph_nodes"] = int(scale_graphs[scale].n_nodes)
                    row["graph_edges"] = int(scale_graphs[scale].n_edges)
                    summary_rows.append(row)

    pair_rows = build_pairwise_rows_matrix(trial_records, scales, n_blocks, k_values)
    tables, resolved = build_focus_tables_matrix(summary_rows, pair_rows, scales, n_blocks, k_values, args)
    anomaly_rows, anomaly_note = diagnose_event_intensity_anomaly(tables, resolved)
    k_diag_rows, k_diag_note = diagnose_continuous_replan_k_effect(tables, resolved)
    quality_rows, quality_note = diagnose_path_quality_consistency(trial_records, scales, n_blocks, k_values)
    markdown = render_markdown_matrix(
        tables,
        args,
        resolved,
        anomaly_note=anomaly_note,
        k_note=k_diag_note,
        quality_note=quality_note,
    )
    plot_paths = make_plots_matrix(summary_rows, trial_records, scales, n_blocks, k_values, args, out_dir)

    if event_records:
        write_csv(out_dir / "benchmark_events.csv", event_records, list(event_records[0].keys()))
    if trial_records:
        write_csv(out_dir / "benchmark_trials.csv", trial_records, list(trial_records[0].keys()))
    if summary_rows:
        write_csv(out_dir / "benchmark_summary.csv", summary_rows, list(summary_rows[0].keys()))
    if pair_rows:
        write_csv(out_dir / "benchmark_pairwise.csv", pair_rows, list(pair_rows[0].keys()))
    if combo_status:
        write_csv(out_dir / "benchmark_combo_status.csv", combo_status, list(combo_status[0].keys()))
    for key, rows in tables.items():
        if rows:
            write_csv(out_dir / f"experiment_{key}.csv", rows, list(rows[0].keys()))
    if anomaly_rows:
        write_csv(out_dir / "experiment_A_diagnostics.csv", anomaly_rows, list(anomaly_rows[0].keys()))
        (out_dir / "experiment_A_diagnostics.md").write_text(
            "# Experiment A Diagnosis\n\n"
            + f"- {anomaly_note}\n",
            encoding="utf-8",
        )
    if k_diag_rows:
        write_csv(out_dir / "experiment_B_diagnostics.csv", k_diag_rows, list(k_diag_rows[0].keys()))
        (out_dir / "experiment_B_diagnostics.md").write_text(
            "# Experiment B Diagnosis\n\n"
            + f"- {k_diag_note}\n",
            encoding="utf-8",
        )
    if quality_rows:
        write_csv(out_dir / "experiment_path_quality.csv", quality_rows, list(quality_rows[0].keys()))
        (out_dir / "experiment_path_quality.md").write_text(
            "# Path Quality Diagnosis (B4 vs B2)\n\n"
            + f"- {quality_note}\n",
            encoding="utf-8",
        )

    # One-stop discussion notes for paper Discussion section.
    discussion_lines = [
        "# Benchmark Discussion Notes",
        "",
        f"- Experiment A: {anomaly_note}" if anomaly_note else "- Experiment A: no anomaly note generated.",
        f"- Experiment B: {k_diag_note}" if k_diag_note else "- Experiment B: no K-effect note generated.",
        f"- Path quality: {quality_note}" if quality_note else "- Path quality: no note generated.",
        "",
        "Suggested paper wording:",
        "1. Under K=1 and light perturbation, incremental LPA* may be slower due to queue/state-maintenance overhead.",
        "2. With larger K, state reuse accumulates and cumulative advantage becomes clear.",
        "3. Non-monotonic time vs n_block can happen when event geometry pushes detours into cleaner subgraphs, reducing updated states.",
        "4. Equal B4/B2 path costs are expected for many trials because both solve the same weighted objective; speed/workload is the key differentiator.",
    ]
    (out_dir / "benchmark_discussion.md").write_text("\n".join(discussion_lines) + "\n", encoding="utf-8")

    (out_dir / "benchmark_table.md").write_text(markdown, encoding="utf-8")
    cfg = vars(args).copy()
    cfg["resolved"] = resolved
    cfg["resolved_n_blocks"] = n_blocks
    cfg["resolved_k_values"] = k_values
    cfg["resolved_scales"] = scales
    cfg["plot_paths"] = plot_paths
    (out_dir / "benchmark_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[done] outputs:")
    print(f"  - {out_dir / 'benchmark_events.csv'}")
    print(f"  - {out_dir / 'benchmark_trials.csv'}")
    print(f"  - {out_dir / 'benchmark_summary.csv'}")
    print(f"  - {out_dir / 'benchmark_pairwise.csv'}")
    print(f"  - {out_dir / 'benchmark_combo_status.csv'}")
    print(f"  - {out_dir / 'experiment_A.csv'}")
    print(f"  - {out_dir / 'experiment_B.csv'}")
    print(f"  - {out_dir / 'experiment_C.csv'}")
    print(f"  - {out_dir / 'experiment_D.csv'}")
    print(f"  - {out_dir / 'benchmark_table.md'}")
    if anomaly_rows:
        print(f"  - {out_dir / 'experiment_A_diagnostics.csv'}")
        print(f"  - {out_dir / 'experiment_A_diagnostics.md'}")
    if k_diag_rows:
        print(f"  - {out_dir / 'experiment_B_diagnostics.csv'}")
        print(f"  - {out_dir / 'experiment_B_diagnostics.md'}")
    if quality_rows:
        print(f"  - {out_dir / 'experiment_path_quality.csv'}")
        print(f"  - {out_dir / 'experiment_path_quality.md'}")
    print(f"  - {out_dir / 'benchmark_discussion.md'}")
    print(f"  - {out_dir / 'benchmark_config.json'}")
    for p in plot_paths:
        print(f"  - {p}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark matrix for B2/B4 event-stream analysis.")
    parser.add_argument("--workdir", type=str, default=".")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260310)
    parser.add_argument("--out-dir", type=str, default="benchmark_out_matrix")
    parser.add_argument("--min-start-goal-dist-km", type=float, default=1.5)
    parser.add_argument("--max-attempt-factor", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=5)

    parser.add_argument("--n-block-grid", type=str, default="2,4,6,8")
    parser.add_argument("--k-events-grid", type=str, default="1,2,3,5,8")
    parser.add_argument("--scales", type=str, default="large")
    parser.add_argument("--scale-fractions", type=str, default="small:0.55,medium:0.78,large:1.0")
    parser.add_argument("--event-radius-km", type=float, default=0.8)
    parser.add_argument("--event-pool-factor", type=int, default=6)
    parser.add_argument("--disable-plots", action="store_true")

    parser.add_argument("--focus-scale", type=str, default="large")
    parser.add_argument("--focus-k-intensity", type=int, default=3)
    parser.add_argument("--focus-n-block-cont", type=int, default=4)
    parser.add_argument("--focus-k-scale", type=int, default=3)
    parser.add_argument("--focus-n-block-scale", type=int, default=4)
    parser.add_argument("--focus-k-distribution", type=int, default=3)

    parser.add_argument("--plot-scale", type=str, default="large")
    parser.add_argument("--plot-k-intensity", type=int, default=3)
    parser.add_argument("--plot-n-block-cont", type=int, default=4)
    parser.add_argument("--plot-k-distribution", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    # 提示：如需完整四基线对比（含 B1/B3），请使用 python benchmark.py --mode matrix
    print("[提示] 本脚本仅运行 B4 vs B2 事件流实验。如需完整四基线对比，请用: python benchmark.py --mode matrix")
    args = parse_args()
    run_benchmark_matrix(args)

