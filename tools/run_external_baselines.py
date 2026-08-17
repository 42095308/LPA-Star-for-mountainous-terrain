from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmark as bm  # noqa: E402


SCENES = ["huashan", "huangshan", "emeishan"]
METHODS = ["BI_APF_RRT_STAR", "GWO_DE"]
METHOD_NAMES = {
    "BI_APF_RRT_STAR": "BI-APF-RRT*",
    "GWO_DE": "GWO-DE",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def ci95(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size <= 1:
        return float("nan")
    return float(1.96 * np.std(arr, ddof=1) / math.sqrt(arr.size))


@dataclass
class AreaEventSpec:
    event_type: str
    center_x_km: float
    center_y_km: float
    radius_km: float
    severity: float


class FeasibilityChecker:
    """连续空间路径可行性检查，使用与主实验一致的走廊、净空、爬升角和区域事件约束。"""

    def __init__(
        self,
        z_grid: np.ndarray,
        floor_grid: np.ndarray,
        ceiling_grid: np.ndarray,
        event: AreaEventSpec | None = None,
        min_clearance_m: float = 30.0,
        max_climb_angle_deg: float = 30.0,
        samples_per_segment: int = 16,
        resolution_m: float | None = None,
        segment_tolerance: float = 1e-9,
    ) -> None:
        self.z_grid = np.asarray(z_grid, dtype=float)
        self.floor_grid = np.asarray(floor_grid, dtype=float)
        self.ceiling_grid = np.asarray(ceiling_grid, dtype=float)
        self.event = event
        self.min_clearance_m = float(min_clearance_m)
        self.max_climb_angle_rad = math.radians(float(max_climb_angle_deg))
        self.samples_per_segment = max(3, int(samples_per_segment))
        # 主实验的图边已通过离散安全检查，连续重采样时允许很小的边级容差，
        # 但点级 floor、ceiling、净空和 no_fly 约束仍保持严格拒绝。
        self.segment_tolerance = float(segment_tolerance)
        self.rows, self.cols = self.z_grid.shape
        self.resolution_m = float(resolution_m if resolution_m is not None else (bm.RESOLUTION or 12.5))
        self.x_max_km = (self.cols - 1) * self.resolution_m / 1000.0
        self.y_max_km = (self.rows - 1) * self.resolution_m / 1000.0

    def row_col(self, x_km: float, y_km: float) -> tuple[int, int] | None:
        if x_km < 0.0 or y_km < 0.0 or x_km > self.x_max_km or y_km > self.y_max_km:
            return None
        col = int(np.clip(round(x_km * 1000.0 / self.resolution_m), 0, self.cols - 1))
        row = int(np.clip(round((self.y_max_km - y_km) * 1000.0 / self.resolution_m), 0, self.rows - 1))
        return row, col

    def project_point(self, x_km: float, y_km: float, z_hint_m: float | None = None, frac: float | None = None) -> np.ndarray:
        rc = self.row_col(float(x_km), float(y_km))
        if rc is None:
            x_km = float(np.clip(x_km, 0.0, self.x_max_km))
            y_km = float(np.clip(y_km, 0.0, self.y_max_km))
            rc = self.row_col(x_km, y_km)
        assert rc is not None
        r, c = rc
        lo = max(float(self.floor_grid[r, c]), float(self.z_grid[r, c]) + self.min_clearance_m)
        hi = float(self.ceiling_grid[r, c])
        if hi < lo:
            hi = lo
        if frac is not None:
            z = lo + float(np.clip(frac, 0.0, 1.0)) * (hi - lo)
        elif z_hint_m is not None:
            z = float(np.clip(z_hint_m, lo, hi))
        else:
            z = 0.5 * (lo + hi)
        return np.array([float(x_km), float(y_km), float(z)], dtype=float)

    def random_point(self, rng: np.random.Generator) -> np.ndarray:
        for _ in range(80):
            x = float(rng.uniform(0.0, self.x_max_km))
            y = float(rng.uniform(0.0, self.y_max_km))
            point = self.project_point(x, y, frac=float(rng.uniform(0.10, 0.90)))
            if self.point_is_valid(point):
                return point
        x = float(rng.uniform(0.0, self.x_max_km))
        y = float(rng.uniform(0.0, self.y_max_km))
        return self.project_point(x, y, frac=float(rng.uniform(0.10, 0.90)))

    def point_violation(self, point: Sequence[float]) -> float:
        x, y, z = [float(v) for v in point]
        rc = self.row_col(x, y)
        if rc is None:
            return 10.0
        r, c = rc
        violation = 0.0
        terrain = float(self.z_grid[r, c])
        floor = float(self.floor_grid[r, c])
        ceiling = float(self.ceiling_grid[r, c])
        if z < floor:
            violation += (floor - z) / 30.0
        if z > ceiling:
            violation += (z - ceiling) / 30.0
        if z - terrain < self.min_clearance_m:
            violation += (self.min_clearance_m - (z - terrain)) / 30.0
        if self.event and self.event.event_type == "no_fly":
            d = math.hypot(x - self.event.center_x_km, y - self.event.center_y_km)
            if d <= self.event.radius_km:
                violation += 1.0 + (self.event.radius_km - d) / max(self.event.radius_km, 1e-6)
        return float(max(0.0, violation))

    def point_is_valid(self, point: Sequence[float]) -> bool:
        return self.point_violation(point) <= 1e-9

    def segment_violation(self, p1: Sequence[float], p2: Sequence[float]) -> float:
        a = np.asarray(p1, dtype=float)
        b = np.asarray(p2, dtype=float)
        horiz_m = float(np.linalg.norm((b[:2] - a[:2]) * 1000.0))
        dz = abs(float(b[2] - a[2]))
        if horiz_m <= 1e-9:
            climb_violation = 0.0 if dz <= 1e-9 else 10.0
        else:
            climb_violation = max(0.0, math.atan2(dz, horiz_m) - self.max_climb_angle_rad) / self.max_climb_angle_rad
        violation = climb_violation
        for s in np.linspace(0.0, 1.0, self.samples_per_segment):
            point = a + s * (b - a)
            violation += self.point_violation(point)
        return float(violation)

    def segment_is_valid(self, p1: Sequence[float], p2: Sequence[float]) -> bool:
        return self.segment_violation(p1, p2) <= self.segment_tolerance

    def path_is_valid(self, path: Sequence[Sequence[float]]) -> bool:
        if len(path) < 2:
            return False
        for point in path:
            if not self.point_is_valid(point):
                return False
        return all(self.segment_is_valid(path[i], path[i + 1]) for i in range(len(path) - 1))

    def path_penalty(self, path: Sequence[Sequence[float]]) -> float:
        if len(path) < 2:
            return 1e6
        penalty = sum(self.point_violation(p) for p in path)
        penalty += sum(self.segment_violation(path[i], path[i + 1]) for i in range(len(path) - 1))
        return float(penalty)


class ContinuousPathEvaluator:
    """连续路径评价器，复用主实验的时间、能耗、风险和通信覆盖统计口径。"""

    def __init__(
        self,
        checker: FeasibilityChecker,
        risk_fields: dict[str, object] | None,
        cost_maxima: tuple[float, float, float] | None = None,
    ) -> None:
        self.checker = checker
        self.risk_fields = risk_fields
        self.cost_maxima = cost_maxima

    def _nodes_from_path(self, path: Sequence[Sequence[float]]) -> np.ndarray:
        rows = []
        for point in path:
            x, y, z = [float(v) for v in point]
            rc = self.checker.row_col(x, y)
            if rc is None:
                layer = 1.0
            else:
                r, c = rc
                agl = z - float(self.checker.z_grid[r, c])
                layer = 0.0 if agl < 60.0 else (1.0 if agl < 90.0 else 2.0)
            rows.append([x, y, z, layer])
        return np.asarray(rows, dtype=float)

    def evaluate(self, path: Sequence[Sequence[float]]) -> dict[str, float]:
        if len(path) < 2:
            return self.failed_metrics()
        nodes = self._nodes_from_path(path)
        edge_pairs = np.asarray([[i, i + 1] for i in range(len(nodes) - 1)], dtype=int)
        try:
            weights, t_raw, e_raw, r_raw, _tmax, _emax, _rmax = bm.compute_edge_costs(
                nodes, edge_pairs, self.checker.z_grid, risk_fields=self.risk_fields
            )
            if self.cost_maxima is not None:
                t_max, e_max, r_max = self.cost_maxima
                weights = (
                    bm.ALPHA * (t_raw / max(float(t_max), bm.EPS))
                    + bm.BETA * (e_raw / max(float(e_max), bm.EPS))
                    + bm.GAMMA * (r_raw / max(float(r_max), bm.EPS))
                )
            extras = bm.compute_node_path_extra_metrics(
                nodes, list(range(len(nodes))), self.checker.z_grid, self.risk_fields
            )
        except Exception:
            return self.failed_metrics()
        length_km = 0.0
        for i in range(len(nodes) - 1):
            diff = nodes[i + 1, :3] - nodes[i, :3]
            length_km += float(np.linalg.norm([diff[0] * 1000.0, diff[1] * 1000.0, diff[2]]) / 1000.0)
        return {
            "path_cost": float(np.sum(weights)),
            "path_time_s": float(np.sum(t_raw)),
            "path_energy_kj": float(np.sum(e_raw)),
            "path_risk": float(np.sum(r_raw)),
            "path_len_km": float(length_km),
            "min_clearance_m": float(extras.get("min_clearance_m", float("nan"))),
            "risk_exposure_integral": float(extras.get("risk_exposure_integral", float("nan"))),
            "comm_coverage_ratio": float(extras.get("comm_coverage_ratio", float("nan"))),
            "max_comm_loss_time_s": float(extras.get("max_comm_loss_time_s", float("nan"))),
            "max_comm_loss_length_km": float(extras.get("max_comm_loss_length_km", float("nan"))),
        }

    @staticmethod
    def failed_metrics() -> dict[str, float]:
        return {
            "path_cost": float("nan"),
            "path_time_s": float("nan"),
            "path_energy_kj": float("nan"),
            "path_risk": float("nan"),
            "path_len_km": float("nan"),
            "min_clearance_m": float("nan"),
            "risk_exposure_integral": float("nan"),
            "comm_coverage_ratio": float("nan"),
            "max_comm_loss_time_s": float("nan"),
            "max_comm_loss_length_km": float("nan"),
        }


def shortcut_path(path: list[np.ndarray], checker: FeasibilityChecker, rounds: int = 80) -> list[np.ndarray]:
    if len(path) <= 2:
        return path
    out = [np.asarray(p, dtype=float) for p in path]
    changed = True
    count = 0
    while changed and count < rounds:
        changed = False
        count += 1
        i = 0
        while i < len(out) - 2:
            j = len(out) - 1
            while j > i + 1:
                if checker.segment_is_valid(out[i], out[j]):
                    out = out[: i + 1] + out[j:]
                    changed = True
                    break
                j -= 1
            i += 1
    return out


def path_length_km(path: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for i in range(len(path) - 1):
        a = np.asarray(path[i], dtype=float)
        b = np.asarray(path[i + 1], dtype=float)
        total += float(np.linalg.norm([(b[0] - a[0]) * 1000.0, (b[1] - a[1]) * 1000.0, b[2] - a[2]]) / 1000.0)
    return total


def point_altitude_fraction(point: Sequence[float], checker: FeasibilityChecker) -> float:
    rc = checker.row_col(float(point[0]), float(point[1]))
    if rc is None:
        return 0.5
    r, c = rc
    lo_z = max(float(checker.floor_grid[r, c]), float(checker.z_grid[r, c]) + checker.min_clearance_m)
    hi_z = max(float(checker.ceiling_grid[r, c]), lo_z)
    return float(np.clip((float(point[2]) - lo_z) / max(hi_z - lo_z, 1e-9), 0.05, 0.95))


def path_to_individual(path: Sequence[Sequence[float]], checker: FeasibilityChecker, waypoints: int) -> np.ndarray:
    pts = [np.asarray(p, dtype=float) for p in path]
    if len(pts) < 2:
        return np.zeros(waypoints * 3, dtype=float)
    seg_lengths = [0.0]
    for i in range(len(pts) - 1):
        seg_lengths.append(seg_lengths[-1] + path_length_km([pts[i], pts[i + 1]]))
    total = max(seg_lengths[-1], 1e-9)
    genes: list[float] = []
    for w in range(waypoints):
        target = total * (w + 1) / (waypoints + 1)
        idx = 0
        while idx < len(seg_lengths) - 2 and seg_lengths[idx + 1] < target:
            idx += 1
        span = max(seg_lengths[idx + 1] - seg_lengths[idx], 1e-9)
        frac = float(np.clip((target - seg_lengths[idx]) / span, 0.0, 1.0))
        point = pts[idx] + frac * (pts[idx + 1] - pts[idx])
        point = checker.project_point(float(point[0]), float(point[1]), z_hint_m=float(point[2]))
        genes.extend([float(point[0]), float(point[1]), point_altitude_fraction(point, checker)])
    return np.asarray(genes, dtype=float)


def event_detour_paths(start: np.ndarray, goal: np.ndarray, checker: FeasibilityChecker) -> list[list[np.ndarray]]:
    event = checker.event
    if event is None or event.event_type != "no_fly":
        return []
    a = np.asarray(start[:2], dtype=float)
    b = np.asarray(goal[:2], dtype=float)
    ab = b - a
    ab_norm = float(np.linalg.norm(ab))
    if ab_norm <= 1e-9:
        return []
    center = np.array([event.center_x_km, event.center_y_km], dtype=float)
    u = ab / ab_norm
    perp = np.array([-u[1], u[0]], dtype=float)
    proj = float(np.clip(np.dot(center - a, u), 0.0, ab_norm))
    closest = a + proj * u
    clearance = event.radius_km + 0.35
    if float(np.linalg.norm(closest - center)) > clearance:
        return []

    before = center - u * clearance
    after = center + u * clearance
    paths: list[list[np.ndarray]] = []
    for side in [-1.0, 1.0]:
        around = center + side * perp * clearance
        p1 = checker.project_point(float(before[0]), float(before[1]), z_hint_m=float(start[2] * 0.67 + goal[2] * 0.33))
        p2 = checker.project_point(float(around[0]), float(around[1]), z_hint_m=float(start[2] * 0.50 + goal[2] * 0.50))
        p3 = checker.project_point(float(after[0]), float(after[1]), z_hint_m=float(start[2] * 0.33 + goal[2] * 0.67))
        paths.append([start.copy(), p1, p2, p3, goal.copy()])
    return paths


def densify_terrain_following_path(
    path: Sequence[Sequence[float]],
    checker: FeasibilityChecker,
    max_step_km: float = 0.05,
) -> list[np.ndarray]:
    pts = [np.asarray(p, dtype=float) for p in path]
    if len(pts) < 2:
        return pts
    out = [checker.project_point(float(pts[0][0]), float(pts[0][1]), z_hint_m=float(pts[0][2]))]
    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        frac_a = point_altitude_fraction(a, checker)
        frac_b = point_altitude_fraction(b, checker)
        dist_xy = float(np.linalg.norm(b[:2] - a[:2]))
        steps = max(1, int(math.ceil(dist_xy / max(max_step_km, 1e-6))))
        for k in range(1, steps + 1):
            frac = k / steps
            xy = a[:2] + frac * (b[:2] - a[:2])
            if i == len(pts) - 2 and k == steps:
                point = checker.project_point(float(xy[0]), float(xy[1]), z_hint_m=float(b[2]))
            else:
                corridor_frac = float(np.clip(frac_a + frac * (frac_b - frac_a), 0.35, 0.65))
                point = checker.project_point(float(xy[0]), float(xy[1]), frac=corridor_frac)
            if np.linalg.norm(point[:2] - out[-1][:2]) > 1e-9 or abs(float(point[2] - out[-1][2])) > 1e-9:
                out.append(point)
    return out


def local_connection_is_valid(p1: Sequence[float], p2: Sequence[float], checker: FeasibilityChecker) -> bool:
    local_path = densify_terrain_following_path([p1, p2], checker)
    return checker.path_is_valid(local_path)


def fast_path_penalty(path: Sequence[Sequence[float]], checker: FeasibilityChecker) -> float:
    dense = densify_terrain_following_path(path, checker, max_step_km=0.12)
    if len(dense) < 2:
        return 1e6
    penalty = sum(checker.point_violation(point) for point in dense)
    for i in range(len(dense) - 1):
        a = np.asarray(dense[i], dtype=float)
        b = np.asarray(dense[i + 1], dtype=float)
        horiz_m = float(np.linalg.norm((b[:2] - a[:2]) * 1000.0))
        dz = abs(float(b[2] - a[2]))
        if horiz_m <= 1e-9:
            penalty += 0.0 if dz <= 1e-9 else 10.0
        else:
            penalty += max(0.0, math.atan2(dz, horiz_m) - checker.max_climb_angle_rad) / max(checker.max_climb_angle_rad, 1e-9)
    return float(penalty)


def fast_objective_cost(path: Sequence[Sequence[float]], evaluator: ContinuousPathEvaluator) -> float:
    if len(path) < 2:
        return 1e6
    try:
        nodes = evaluator._nodes_from_path(path)
        edge_pairs = np.asarray([[i, i + 1] for i in range(len(nodes) - 1)], dtype=int)
        weights, t_raw, e_raw, r_raw, _tmax, _emax, _rmax = bm.compute_edge_costs(
            nodes, edge_pairs, evaluator.checker.z_grid, risk_fields=evaluator.risk_fields
        )
        if evaluator.cost_maxima is not None:
            t_max, e_max, r_max = evaluator.cost_maxima
            weights = (
                bm.ALPHA * (t_raw / max(float(t_max), bm.EPS))
                + bm.BETA * (e_raw / max(float(e_max), bm.EPS))
                + bm.GAMMA * (r_raw / max(float(r_max), bm.EPS))
            )
        return float(np.sum(weights))
    except Exception:
        return float(path_length_km(path))


def informed_sample(
    checker: FeasibilityChecker,
    rng: np.random.Generator,
    start: np.ndarray,
    goal: np.ndarray,
    detour_points: Sequence[np.ndarray],
) -> np.ndarray:
    draw = float(rng.random())
    if detour_points and draw < 0.28:
        base = np.asarray(detour_points[int(rng.integers(0, len(detour_points)))], dtype=float)
        xy = base[:2] + rng.normal(0.0, 0.22, size=2)
        return checker.project_point(float(xy[0]), float(xy[1]), z_hint_m=float(base[2]))
    if draw < 0.68:
        frac = float(rng.uniform(0.0, 1.0))
        xy = start[:2] + frac * (goal[:2] - start[:2])
        xy += rng.normal(0.0, 0.45, size=2)
        z_hint = float(start[2] + frac * (goal[2] - start[2]) + rng.normal(0.0, 18.0))
        return checker.project_point(float(xy[0]), float(xy[1]), z_hint_m=z_hint)
    return checker.random_point(rng)


def reconstruct_path(points: list[np.ndarray], parents: list[int], idx: int) -> list[np.ndarray]:
    path = []
    cur = int(idx)
    while cur >= 0:
        path.append(points[cur])
        cur = int(parents[cur])
    path.reverse()
    return path


def steer(checker: FeasibilityChecker, source: np.ndarray, target: np.ndarray, step_km: float) -> np.ndarray:
    diff_xy = target[:2] - source[:2]
    dist_xy = float(np.linalg.norm(diff_xy))
    if dist_xy <= step_km:
        x, y = float(target[0]), float(target[1])
        z_hint = float(target[2])
        horiz_m = max(dist_xy * 1000.0, 1e-9)
    else:
        ratio = step_km / max(dist_xy, 1e-9)
        x, y = (source[:2] + diff_xy * ratio).tolist()
        z_hint = float(source[2] + (target[2] - source[2]) * ratio)
        horiz_m = max(step_km * 1000.0, 1e-9)
    max_dz = math.tan(checker.max_climb_angle_rad) * horiz_m * 0.92
    z_hint = float(np.clip(z_hint, float(source[2] - max_dz), float(source[2] + max_dz)))
    return checker.project_point(float(x), float(y), z_hint_m=z_hint)


def nearest_index(points: list[np.ndarray], target: np.ndarray) -> int:
    coords = np.asarray(points, dtype=float)
    d = np.linalg.norm(coords[:, :3] - target.reshape(1, 3), axis=1)
    return int(np.argmin(d))


def apf_guided_sample(
    checker: FeasibilityChecker,
    rng: np.random.Generator,
    current: np.ndarray,
    target: np.ndarray,
    goal_bias: float,
    start: np.ndarray,
    goal: np.ndarray,
    detour_points: Sequence[np.ndarray],
) -> np.ndarray:
    if rng.random() < goal_bias:
        return target.copy()
    sample = informed_sample(checker, rng, start, goal, detour_points)
    direction = sample[:2] - current[:2]
    attract = target[:2] - current[:2]
    combined = direction + 0.45 * attract
    if checker.event is not None:
        center = np.array([checker.event.center_x_km, checker.event.center_y_km], dtype=float)
        away = current[:2] - center
        dist = float(np.linalg.norm(away))
        influence = 1.8 * checker.event.radius_km
        if dist < influence and dist > 1e-9:
            combined += (away / dist) * (influence - dist) * 1.2
    if np.linalg.norm(combined) <= 1e-9:
        return sample
    xy = current[:2] + combined / np.linalg.norm(combined) * float(rng.uniform(0.2, 1.2))
    return checker.project_point(float(xy[0]), float(xy[1]), z_hint_m=float(sample[2]))


def run_bi_apf_rrt_star(
    start: np.ndarray,
    goal: np.ndarray,
    checker: FeasibilityChecker,
    rng: np.random.Generator,
    max_iter: int,
    step_km: float,
    goal_bias: float,
    time_limit_ms: float | None = None,
) -> tuple[bool, list[np.ndarray], int, str]:
    start_tree = [start.copy()]
    start_parent = [-1]
    goal_tree = [goal.copy()]
    goal_parent = [-1]
    work_units = 0
    connect_km = step_km * 1.25
    detour_points = [p for path in event_detour_paths(start, goal, checker) for p in path[1:-1]]
    t_start = time.perf_counter()

    direct_path = densify_terrain_following_path([start, goal], checker)
    if checker.path_is_valid(direct_path):
        return True, direct_path, 1, ""
    for seed_path in event_detour_paths(start, goal, checker):
        seeded = densify_terrain_following_path(seed_path, checker)
        if checker.path_is_valid(seeded):
            return True, seeded, len(seeded), ""

    for it in range(max_iter):
        if time_limit_ms is not None and (time.perf_counter() - t_start) * 1000.0 >= float(time_limit_ms):
            return False, [], work_units, "rrt_budget_exhausted"
        grow_start = (it % 2 == 0)
        tree = start_tree if grow_start else goal_tree
        parent = start_parent if grow_start else goal_parent
        other_tree = goal_tree if grow_start else start_tree
        other_parent = goal_parent if grow_start else start_parent
        tree_target = goal if grow_start else start

        nearest_to_goal = tree[nearest_index(tree, tree_target)]
        sample = apf_guided_sample(checker, rng, nearest_to_goal, tree_target, goal_bias, start, goal, detour_points)
        idx_near = nearest_index(tree, sample)
        new_point = steer(checker, tree[idx_near], sample, step_km)
        work_units += 1
        if not local_connection_is_valid(tree[idx_near], new_point, checker):
            continue

        # 轻量 RRT* 重连：在局部邻域中选择路径长度更短且可行的父节点。
        best_parent = idx_near
        best_cost = path_length_km(reconstruct_path(tree, parent, idx_near) + [new_point])
        for cand_idx, cand in enumerate(tree):
            if np.linalg.norm(cand[:2] - new_point[:2]) <= connect_km and local_connection_is_valid(cand, new_point, checker):
                cand_cost = path_length_km(reconstruct_path(tree, parent, cand_idx) + [new_point])
                if cand_cost < best_cost:
                    best_parent = cand_idx
                    best_cost = cand_cost
        tree.append(new_point)
        parent.append(best_parent)
        new_idx = len(tree) - 1

        idx_other = nearest_index(other_tree, new_point)
        if np.linalg.norm(other_tree[idx_other][:2] - new_point[:2]) <= connect_km and local_connection_is_valid(new_point, other_tree[idx_other], checker):
            if grow_start:
                path = reconstruct_path(start_tree, start_parent, new_idx) + list(reversed(reconstruct_path(goal_tree, goal_parent, idx_other)))
            else:
                path = reconstruct_path(start_tree, start_parent, idx_other) + list(reversed(reconstruct_path(goal_tree, goal_parent, new_idx)))
            path = densify_terrain_following_path(path, checker)
            path = shortcut_path(path, checker)
            path = densify_terrain_following_path(path, checker)
            if checker.path_is_valid(path):
                return True, path, work_units, ""

    return False, [], work_units, "rrt_no_connection"


def decode_gwo_individual(individual: np.ndarray, start: np.ndarray, goal: np.ndarray, checker: FeasibilityChecker) -> list[np.ndarray]:
    points = [start.copy()]
    for gene in individual.reshape((-1, 3)):
        x = float(gene[0])
        y = float(gene[1])
        frac = float(gene[2])
        points.append(checker.project_point(x, y, frac=frac))
    points.append(goal.copy())
    return points


def gwo_de_fitness(individual: np.ndarray, start: np.ndarray, goal: np.ndarray, checker: FeasibilityChecker, evaluator: ContinuousPathEvaluator) -> float:
    path = decode_gwo_individual(individual, start, goal, checker)
    base = fast_objective_cost(path, evaluator)
    smooth = 0.0
    for i in range(1, len(path) - 1):
        a = path[i] - path[i - 1]
        b = path[i + 1] - path[i]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na > 1e-9 and nb > 1e-9:
            smooth += max(0.0, 1.0 - float(np.dot(a, b) / (na * nb)))
    return float(base + 120.0 * fast_path_penalty(path, checker) + 0.08 * smooth)


def run_gwo_de(
    start: np.ndarray,
    goal: np.ndarray,
    checker: FeasibilityChecker,
    evaluator: ContinuousPathEvaluator,
    rng: np.random.Generator,
    pop_size: int,
    iterations: int,
    waypoints: int,
) -> tuple[bool, list[np.ndarray], int, str]:
    dim = waypoints * 3
    lo = np.zeros(dim, dtype=float)
    hi = np.zeros(dim, dtype=float)
    for i in range(waypoints):
        lo[3 * i] = 0.0
        hi[3 * i] = checker.x_max_km
        lo[3 * i + 1] = 0.0
        hi[3 * i + 1] = checker.y_max_km
        lo[3 * i + 2] = 0.05
        hi[3 * i + 2] = 0.95

    pop = rng.uniform(lo, hi, size=(pop_size, dim))
    seed_paths: list[list[np.ndarray]] = [[start.copy(), goal.copy()]]
    seed_paths.extend(event_detour_paths(start, goal, checker))
    seeded = 0
    for seed_path in seed_paths[:pop_size]:
        pop[seeded] = path_to_individual(seed_path, checker, waypoints)
        seeded += 1

    for i in range(seeded, pop_size):
        for w in range(waypoints):
            frac = (w + 1) / (waypoints + 1)
            if i < max(4, pop_size // 4):
                xy = start[:2] + frac * (goal[:2] - start[:2])
                xy += rng.normal(0.0, 0.35 + 0.04 * i, size=2)
                pop[i, 3 * w] = float(np.clip(xy[0], 0.0, checker.x_max_km))
                pop[i, 3 * w + 1] = float(np.clip(xy[1], 0.0, checker.y_max_km))
                pop[i, 3 * w + 2] = float(np.clip(0.50 + rng.normal(0.0, 0.12), 0.05, 0.95))

    fitness = np.array([gwo_de_fitness(ind, start, goal, checker, evaluator) for ind in pop], dtype=float)
    evals = int(pop_size)
    for t in range(iterations):
        order = np.argsort(fitness)
        alpha, beta, delta = pop[order[0]].copy(), pop[order[1]].copy(), pop[order[2]].copy()
        a = 2.0 - 2.0 * (t / max(iterations - 1, 1))
        new_pop = pop.copy()
        for i in range(pop_size):
            r = rng.choice(pop_size, size=3, replace=False)
            mutant = pop[r[0]] + 0.65 * (pop[r[1]] - pop[r[2]])
            mutant = np.clip(mutant, lo, hi)
            cross = rng.random(dim) < 0.75
            if not np.any(cross):
                cross[int(rng.integers(0, dim))] = True
            de_trial = np.where(cross, mutant, pop[i])

            gwo_parts = []
            for leader in [alpha, beta, delta]:
                a1 = 2.0 * a * rng.random(dim) - a
                c1 = 2.0 * rng.random(dim)
                d1 = np.abs(c1 * leader - pop[i])
                gwo_parts.append(leader - a1 * d1)
            gwo_trial = np.mean(gwo_parts, axis=0)
            candidate = np.clip(0.55 * gwo_trial + 0.45 * de_trial, lo, hi)
            cand_fit = gwo_de_fitness(candidate, start, goal, checker, evaluator)
            evals += 1
            if cand_fit < fitness[i]:
                new_pop[i] = candidate
                fitness[i] = cand_fit
        pop = new_pop

    best = pop[int(np.argmin(fitness))]
    path = densify_terrain_following_path(decode_gwo_individual(best, start, goal, checker), checker)
    path = shortcut_path(path, checker)
    path = densify_terrain_following_path(path, checker)
    if checker.path_is_valid(path):
        return True, path, evals, ""
    return False, path, evals, "gwo_de_infeasible"


def load_scene_context(workdir: Path, scene: str) -> tuple[Path, dict[str, object], dict[str, np.ndarray | dict[str, object]]]:
    data_root = workdir / "intermediate_artifacts" / "data" / scene
    cfg_path = workdir / "scenarios" / f"{scene}.json"
    cfg = bm.load_scenario_config(str(cfg_path), workdir)
    meta_path = data_root / "Z_crop_meta.json"
    resolution_m = 12.5
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        resolution_m = float(meta.get("resolution_m", resolution_m))
    bm.RESOLUTION = resolution_m
    z_grid = np.asarray(np.load(data_root / "Z_crop.npy"), dtype=float)
    floor = np.asarray(np.load(data_root / "floor.npy"), dtype=float)
    ceiling = np.asarray(np.load(data_root / "ceiling.npy"), dtype=float)
    graph_nodes = np.asarray(np.load(data_root / "graph_nodes.npy"), dtype=float)
    graph_edges = np.asarray(np.load(data_root / "graph_edges.npy"))
    risk_fields = bm.load_risk_fields(data_root, z_grid.shape, cfg)
    return data_root, cfg, {
        "z_grid": z_grid,
        "floor": floor,
        "ceiling": ceiling,
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "risk_fields": risk_fields,
        "resolution_m": resolution_m,
    }


def trial_seed(base_seed: int, scene: str, trial: int, method: str) -> int:
    scene_offset = {"huashan": 1000, "huangshan": 2000, "emeishan": 3000}.get(scene, 0)
    method_offset = 17 if method == "BI_APF_RRT_STAR" else 31
    return int(base_seed + scene_offset + 97 * int(trial) + method_offset)


def run_scene(args: argparse.Namespace, workdir: Path, scene: str) -> list[dict[str, object]]:
    _data_root, _cfg, ctx = load_scene_context(workdir, scene)
    z_grid = ctx["z_grid"]
    floor = ctx["floor"]
    ceiling = ctx["ceiling"]
    graph_nodes = ctx["graph_nodes"]
    graph_edges = ctx["graph_edges"]
    risk_fields = ctx["risk_fields"]
    resolution_m = float(ctx["resolution_m"])
    assert isinstance(z_grid, np.ndarray)
    assert isinstance(floor, np.ndarray)
    assert isinstance(ceiling, np.ndarray)
    assert isinstance(graph_nodes, np.ndarray)
    assert isinstance(graph_edges, np.ndarray)
    assert isinstance(risk_fields, dict)
    layered_graph = bm.build_weighted_graph("external_seed_layered", graph_nodes, graph_edges, z_grid, risk_fields=risk_fields)

    trial_path = workdir / "final_results" / scene / "E1_E2_single_final" / "benchmark_trials.csv"
    mp_rows = [r for r in read_csv(trial_path) if r.get("baseline") == "B4_Proposed_LPA_Layered"]
    mp_rows.sort(key=lambda r: int(float(r.get("trial", "0") or 0)))
    if args.trials > 0:
        mp_rows = mp_rows[: args.trials]

    rows: list[dict[str, object]] = []
    for source_row in mp_rows:
        trial = int(float(source_row.get("trial", "0") or 0))
        event = AreaEventSpec(
            event_type=source_row.get("event_type", "no_fly") or "no_fly",
            center_x_km=to_float(source_row.get("event_center_x_km"), 0.0),
            center_y_km=to_float(source_row.get("event_center_y_km"), 0.0),
            radius_km=to_float(source_row.get("event_radius_km"), 0.8),
            severity=to_float(source_row.get("event_severity"), 1.0),
        )
        checker = FeasibilityChecker(
            z_grid,
            floor,
            ceiling,
            event=event,
            max_climb_angle_deg=args.max_climb_angle_deg,
            samples_per_segment=args.segment_samples,
            resolution_m=resolution_m,
            segment_tolerance=args.segment_tolerance,
        )
        evaluator = ContinuousPathEvaluator(
            checker,
            risk_fields,
            cost_maxima=(layered_graph.t_max, layered_graph.e_max, layered_graph.r_max),
        )
        start_idx = int(float(source_row.get("start_node", "0") or 0))
        goal_idx = int(float(source_row.get("goal_node", "0") or 0))
        start = checker.project_point(
            float(graph_nodes[start_idx, 0]),
            float(graph_nodes[start_idx, 1]),
            z_hint_m=float(graph_nodes[start_idx, 2]),
        )
        goal = checker.project_point(
            float(graph_nodes[goal_idx, 0]),
            float(graph_nodes[goal_idx, 1]),
            z_hint_m=float(graph_nodes[goal_idx, 2]),
        )
        for method in METHODS:
            rng = np.random.default_rng(trial_seed(args.seed, scene, trial, method))
            t0 = time.perf_counter()
            if method == "BI_APF_RRT_STAR":
                ok, path, work_units, failure = run_bi_apf_rrt_star(
                    start,
                    goal,
                    checker,
                    rng,
                    max_iter=args.rrt_iterations,
                    step_km=args.rrt_step_km,
                    goal_bias=args.rrt_goal_bias,
                    time_limit_ms=args.rrt_time_limit_ms,
                )
                path_origin = "continuous_bi_apf_rrt_star" if ok else ""
            else:
                ok, path, work_units, failure = run_gwo_de(
                    start,
                    goal,
                    checker,
                    evaluator,
                    rng,
                    pop_size=args.gwo_population,
                    iterations=args.gwo_iterations,
                    waypoints=args.gwo_waypoints,
                )
                path_origin = "continuous_gwo_de" if ok else ""
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            metrics = evaluator.evaluate(path) if ok else ContinuousPathEvaluator.failed_metrics()
            row: dict[str, object] = {
                "scene": scene,
                "trial": trial,
                "baseline": method,
                "method": METHOD_NAMES[method],
                "task_id": source_row.get("task_id", ""),
                "task_depot": source_row.get("task_depot", ""),
                "task_target": source_row.get("task_target", ""),
                "start_node": start_idx,
                "goal_node": goal_idx,
                "event_type": event.event_type,
                "event_center_x_km": event.center_x_km,
                "event_center_y_km": event.center_y_km,
                "event_radius_km": event.radius_km,
                "event_severity": event.severity,
                "success": bool(ok),
                "failure_reason": "" if ok else failure,
                "path_origin": path_origin,
                "replan_ms": elapsed_ms if ok else float("nan"),
                "work_units": int(work_units),
                "path_cost": metrics["path_cost"],
                "path_energy_kj": metrics["path_energy_kj"],
                "path_len_km": metrics["path_len_km"],
                "min_clearance_m": metrics["min_clearance_m"],
                "risk_exposure_integral": metrics["risk_exposure_integral"],
                "comm_coverage_ratio": metrics["comm_coverage_ratio"],
                "max_comm_loss_time_s": metrics["max_comm_loss_time_s"],
                "max_comm_loss_length_km": metrics["max_comm_loss_length_km"],
                "path_waypoints": len(path) if ok else 0,
            }
            rows.append(row)
            if args.progress:
                status = "成功" if ok else f"失败:{failure}"
                print(f"[{scene}] trial={trial} {METHOD_NAMES[method]} {status} time={elapsed_ms:.1f}ms work={work_units}")
    return rows


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["scene"]), str(row["baseline"])), []).append(row)
    metrics = [
        ("replan_ms", "mean_replan_ms", "std_replan_ms"),
        ("work_units", "mean_work_units", "std_work_units"),
        ("path_cost", "mean_path_cost", "std_path_cost"),
        ("path_len_km", "mean_path_len_km", "std_path_len_km"),
        ("risk_exposure_integral", "mean_risk_exposure_integral", "std_risk_exposure_integral"),
        ("comm_coverage_ratio", "mean_comm_coverage_ratio", "std_comm_coverage_ratio"),
    ]
    for (scene, baseline), subset in sorted(grouped.items()):
        ok = [r for r in subset if bool(r.get("success"))]
        failed = [r for r in subset if not bool(r.get("success"))]
        row: dict[str, object] = {
            "scene": scene,
            "baseline": baseline,
            "method": METHOD_NAMES.get(baseline, baseline),
            "n_trials": len(subset),
            "n_success": len(ok),
            "n_failed": len(failed),
            "success_rate": len(ok) / max(len(subset), 1),
            "failure_reason_top": "",
        }
        if failed:
            reasons: dict[str, int] = {}
            for item in failed:
                reason = str(item.get("failure_reason", "unknown"))
                reasons[reason] = reasons.get(reason, 0) + 1
            top = sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            row["failure_reason_top"] = f"{top[0]}:{top[1]}"
        for key, mean_key, std_key in metrics:
            values = np.asarray([to_float(r.get(key)) for r in ok], dtype=float)
            values = values[np.isfinite(values)]
            row[mean_key] = float(np.mean(values)) if values.size else float("nan")
            row[std_key] = float(np.std(values, ddof=1)) if values.size > 1 else (0.0 if values.size == 1 else float("nan"))
            row[f"ci95_{key}"] = ci95(values)
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 BI-APF-RRT* 与 GWO-DE 外部代表性基线。")
    parser.add_argument("--workdir", type=str, default=".", help="项目根目录。")
    parser.add_argument("--out-dir", type=str, default="final_results/external_baselines", help="输出目录。")
    parser.add_argument("--scenes", type=str, default=",".join(SCENES), help="逗号分隔场景列表。")
    parser.add_argument("--trials", type=int, default=30, help="每个场景使用的 MP 配对 trial 数。")
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--segment-samples", type=int, default=16)
    parser.add_argument("--segment-tolerance", type=float, default=0.10, help="线段检查的数值容差，用于吸收 DEM 栅格插值边界误差。")
    parser.add_argument(
        "--max-climb-angle-deg",
        type=float,
        default=89.0,
        help="连续线段爬升角审计阈值。默认 89° 用于对齐现有主实验图边口径；若所有主方法按 30° 重跑，可改为 30。",
    )
    parser.add_argument("--rrt-iterations", type=int, default=250)
    parser.add_argument("--rrt-time-limit-ms", type=float, default=3000.0, help="BI-APF-RRT* 单个 trial 的搜索时间预算。")
    parser.add_argument("--rrt-step-km", type=float, default=0.35)
    parser.add_argument("--rrt-goal-bias", type=float, default=0.25)
    parser.add_argument("--gwo-population", type=int, default=8)
    parser.add_argument("--gwo-iterations", type=int, default=6)
    parser.add_argument("--gwo-waypoints", type=int, default=5)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    out_dir = (workdir / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    selected_scenes = [s.strip().lower() for s in args.scenes.split(",") if s.strip()]

    all_rows: list[dict[str, object]] = []
    for scene in selected_scenes:
        print(f"[开始] 外部基线场景: {scene}")
        all_rows.extend(run_scene(args, workdir, scene))

    trial_fields = [
        "scene",
        "trial",
        "baseline",
        "method",
        "task_id",
        "task_depot",
        "task_target",
        "start_node",
        "goal_node",
        "event_type",
        "event_center_x_km",
        "event_center_y_km",
        "event_radius_km",
        "event_severity",
        "success",
        "failure_reason",
        "path_origin",
        "replan_ms",
        "work_units",
        "path_cost",
        "path_energy_kj",
        "path_len_km",
        "min_clearance_m",
        "risk_exposure_integral",
        "comm_coverage_ratio",
        "max_comm_loss_time_s",
        "max_comm_loss_length_km",
        "path_waypoints",
    ]
    summary_fields = [
        "scene",
        "baseline",
        "method",
        "n_trials",
        "n_success",
        "n_failed",
        "success_rate",
        "failure_reason_top",
        "mean_replan_ms",
        "std_replan_ms",
        "ci95_replan_ms",
        "mean_work_units",
        "std_work_units",
        "ci95_work_units",
        "mean_path_cost",
        "std_path_cost",
        "ci95_path_cost",
        "mean_path_len_km",
        "std_path_len_km",
        "ci95_path_len_km",
        "mean_risk_exposure_integral",
        "std_risk_exposure_integral",
        "ci95_risk_exposure_integral",
        "mean_comm_coverage_ratio",
        "std_comm_coverage_ratio",
        "ci95_comm_coverage_ratio",
    ]

    write_csv(out_dir / "external_baseline_trials.csv", all_rows, trial_fields)
    write_csv(out_dir / "external_baseline_summary.csv", summarise(all_rows), summary_fields)
    print(f"[完成] 外部基线 trial 输出: {out_dir / 'external_baseline_trials.csv'}")
    print(f"[完成] 外部基线 summary 输出: {out_dir / 'external_baseline_summary.csv'}")


if __name__ == "__main__":
    main()
