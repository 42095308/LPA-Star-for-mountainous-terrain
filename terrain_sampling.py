"""
地形驱动航路节点采样。

骨干层以山脊、宽谷和低风险连通带为主体，规则网格只作为覆盖补充；
支路层围绕终端任务区和终端到主通道的过渡带加密。
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter


def _normalise(a: np.ndarray) -> np.ndarray:
    finite = np.isfinite(a)
    if not np.any(finite):
        return np.zeros_like(a, dtype=float)
    lo = float(np.nanmin(a[finite]))
    hi = float(np.nanmax(a[finite]))
    if hi <= lo + 1e-12:
        return np.zeros_like(a, dtype=float)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def terrain_features(z: np.ndarray, resolution_m: float) -> Dict[str, np.ndarray]:
    zf = np.asarray(z, dtype=float)
    z_smooth = gaussian_filter(zf, sigma=3)
    gy, gx = np.gradient(z_smooth, float(resolution_m), float(resolution_m))
    slope_deg = np.degrees(np.arctan(np.sqrt(gx * gx + gy * gy)))
    local_max = maximum_filter(z_smooth, size=21)
    local_min = minimum_filter(z_smooth, size=21)
    relief = np.maximum(local_max - local_min, 1e-6)
    ridge_score = np.clip((z_smooth - local_min) / relief, 0.0, 1.0)
    valley_score = np.clip((local_max - z_smooth) / relief, 0.0, 1.0)
    open_score = (1.0 - _normalise(slope_deg)) * valley_score
    return {
        "z_smooth": z_smooth,
        "slope_deg": slope_deg,
        "ridge_score": ridge_score,
        "valley_score": valley_score,
        "open_score": open_score,
    }


def _grid_points(rows: int, cols: int, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 2), dtype=int)
    per_axis = max(2, int(np.ceil(np.sqrt(count))))
    rr = np.linspace(0, rows - 1, per_axis, dtype=int)
    cc = np.linspace(0, cols - 1, per_axis, dtype=int)
    r2, c2 = np.meshgrid(rr, cc, indexing="ij")
    pts = np.column_stack([r2.ravel(), c2.ravel()])
    if len(pts) > count:
        idx = np.linspace(0, len(pts) - 1, count, dtype=int)
        pts = pts[idx]
    return pts.astype(int)


def _dedupe(points: Iterable[Tuple[int, int]], rows: int, cols: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    seen = set()
    for r, c in points:
        key = (int(np.clip(r, 0, rows - 1)), int(np.clip(c, 0, cols - 1)))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _select_by_score(
    score: np.ndarray,
    count: int,
    min_spacing_px: int,
    allowed: np.ndarray | None = None,
    exclude: set | None = None,
    role: str = "terrain",
) -> List[Tuple[int, int, str]]:
    if count <= 0:
        return []
    rows, cols = score.shape
    mask = np.isfinite(score)
    if allowed is not None:
        mask &= allowed
    if not np.any(mask):
        return []

    flat = np.argwhere(mask)
    vals = score[mask]
    order = np.argsort(vals)[::-1]
    exclude = exclude or set()
    selected: List[Tuple[int, int, str]] = []
    selected_rc: List[Tuple[int, int]] = []

    spacing2 = float(max(1, min_spacing_px) ** 2)
    for idx in order:
        r, c = int(flat[idx, 0]), int(flat[idx, 1])
        if (r, c) in exclude:
            continue
        ok = True
        for sr, sc in selected_rc:
            if (r - sr) * (r - sr) + (c - sc) * (c - sc) < spacing2:
                ok = False
                break
        if not ok:
            continue
        selected.append((r, c, role))
        selected_rc.append((r, c))
        if len(selected) >= count:
            break

    if len(selected) < count:
        for idx in order:
            r, c = int(flat[idx, 0]), int(flat[idx, 1])
            if (r, c) in exclude or any((r == sr and c == sc) for sr, sc in selected_rc):
                continue
            selected.append((r, c, role))
            selected_rc.append((r, c))
            if len(selected) >= count:
                break
    return selected


def _terminal_distance_maps(rows: int, cols: int, terminals: Sequence[Tuple[int, int]], resolution_m: float) -> Tuple[np.ndarray, np.ndarray]:
    rr, cc = np.indices((rows, cols))
    if not terminals:
        inf = np.full((rows, cols), np.inf, dtype=float)
        return inf, inf

    min_terminal = np.full((rows, cols), np.inf, dtype=float)
    for tr, tc in terminals:
        d = np.sqrt((rr - tr) ** 2 + (cc - tc) ** 2) * resolution_m / 1000.0
        min_terminal = np.minimum(min_terminal, d)

    center_r = float(np.mean([p[0] for p in terminals]))
    center_c = float(np.mean([p[1] for p in terminals]))
    min_line = np.full((rows, cols), np.inf, dtype=float)
    for tr, tc in terminals:
        ar = center_r - tr
        ac = center_c - tc
        denom = ar * ar + ac * ac
        if denom < 1e-9:
            continue
        t = ((rr - tr) * ar + (cc - tc) * ac) / denom
        t = np.clip(t, 0.0, 1.0)
        pr = tr + t * ar
        pc = tc + t * ac
        d = np.sqrt((rr - pr) ** 2 + (cc - pc) ** 2) * resolution_m / 1000.0
        min_line = np.minimum(min_line, d)
    return min_terminal, min_line


def build_terrain_samples(
    z: np.ndarray,
    risk_human: np.ndarray | None,
    terminal_rcs: Sequence[Tuple[int, int]],
    params: Dict[str, Any],
    resolution_m: float | None = None,
    layer_allowed: np.ndarray | None = None,
) -> Tuple[np.ndarray, List[str], np.ndarray, List[str], Dict[str, Any]]:
    if resolution_m is None:
        raise ValueError("build_terrain_samples 需要显式传入 resolution_m。")
    rows, cols = z.shape
    features = terrain_features(z, resolution_m)
    risk = np.zeros((rows, cols), dtype=float)
    if risk_human is not None and risk_human.shape == (rows, cols):
        risk = np.clip(np.asarray(risk_human, dtype=float), 0.0, 1.0)

    branch_budget = int(params.get("branch_node_budget", 3364))
    backbone_budget = int(params.get("backbone_node_budget", 3364))
    terrain_ratio = float(params.get("terrain_ratio", 0.70))
    grid_ratio = float(params.get("supplement_grid_ratio", 0.30))
    min_spacing_px = max(1, int(round(float(params.get("min_spacing_m", 130.0)) / resolution_m)))
    term_radius = float(params.get("branch_terminal_radius_km", 1.4))
    corridor_width = float(params.get("branch_corridor_width_km", 0.55))
    low_risk_threshold = float(params.get("low_risk_threshold", 0.30))

    branch_allowed = layer_allowed[1].astype(bool) if layer_allowed is not None else np.ones((rows, cols), dtype=bool)
    backbone_allowed = layer_allowed[2].astype(bool) if layer_allowed is not None else np.ones((rows, cols), dtype=bool)

    low_risk_score = 1.0 - risk
    backbone_score = (
        0.35 * features["ridge_score"]
        + 0.25 * features["open_score"]
        + 0.25 * low_risk_score
        + 0.15 * features["valley_score"]
    )
    backbone_score = np.where(risk <= max(low_risk_threshold, 0.01), backbone_score, backbone_score * 0.45)

    min_terminal, min_line = _terminal_distance_maps(rows, cols, terminal_rcs, resolution_m)
    terminal_score = np.exp(-np.square(min_terminal / max(term_radius, 1e-6)))
    corridor_score = np.exp(-np.square(min_line / max(corridor_width, 1e-6)))
    branch_score = (
        0.40 * terminal_score
        + 0.30 * corridor_score
        + 0.15 * low_risk_score
        + 0.15 * (1.0 - _normalise(features["slope_deg"]))
    )

    backbone_terrain_n = int(round(backbone_budget * terrain_ratio))
    backbone_grid_n = max(0, backbone_budget - backbone_terrain_n)
    branch_terrain_n = int(round(branch_budget * terrain_ratio))
    branch_grid_n = max(0, branch_budget - branch_terrain_n)

    backbone_selected = _select_by_score(
        backbone_score,
        backbone_terrain_n,
        min_spacing_px,
        allowed=backbone_allowed,
        role="terrain_backbone",
    )
    backbone_used = {(r, c) for r, c, _ in backbone_selected}
    for r, c in _grid_points(rows, cols, backbone_grid_n):
        if len(backbone_selected) >= backbone_budget:
            break
        if not backbone_allowed[int(r), int(c)]:
            continue
        if (int(r), int(c)) in backbone_used:
            continue
        backbone_selected.append((int(r), int(c), "supplement_grid"))
        backbone_used.add((int(r), int(c)))
    if len(backbone_selected) < backbone_budget:
        extra = _select_by_score(
            backbone_score,
            backbone_budget - len(backbone_selected),
            max(1, min_spacing_px // 2),
            allowed=backbone_allowed,
            exclude=backbone_used,
            role="terrain_backbone_fill",
        )
        backbone_selected.extend(extra)

    branch_selected = _select_by_score(
        branch_score,
        branch_terrain_n,
        min_spacing_px,
        allowed=branch_allowed,
        role="task_corridor_branch",
    )
    branch_used = {(r, c) for r, c, _ in branch_selected}
    for r, c in _grid_points(rows, cols, branch_grid_n):
        if len(branch_selected) >= branch_budget:
            break
        if not branch_allowed[int(r), int(c)]:
            continue
        if (int(r), int(c)) in branch_used:
            continue
        branch_selected.append((int(r), int(c), "supplement_grid"))
        branch_used.add((int(r), int(c)))
    if len(branch_selected) < branch_budget:
        extra = _select_by_score(
            branch_score,
            branch_budget - len(branch_selected),
            max(1, min_spacing_px // 2),
            allowed=branch_allowed,
            exclude=branch_used,
            role="task_corridor_branch_fill",
        )
        branch_selected.extend(extra)

    branch_pts = np.asarray([[r, c] for r, c, _ in branch_selected[:branch_budget]], dtype=int)
    branch_roles = [role for _, _, role in branch_selected[:branch_budget]]
    backbone_pts = np.asarray([[r, c] for r, c, _ in backbone_selected[:backbone_budget]], dtype=int)
    backbone_roles = [role for _, _, role in backbone_selected[:backbone_budget]]

    meta = {
        "branch_budget": branch_budget,
        "backbone_budget": backbone_budget,
        "terrain_ratio_target": terrain_ratio,
        "supplement_grid_ratio_target": grid_ratio,
        "branch_role_counts": dict(Counter(branch_roles)),
        "backbone_role_counts": dict(Counter(backbone_roles)),
        "actual_branch_grid_ratio": float(branch_roles.count("supplement_grid") / max(len(branch_roles), 1)),
        "actual_backbone_grid_ratio": float(backbone_roles.count("supplement_grid") / max(len(backbone_roles), 1)),
        "min_spacing_px": int(min_spacing_px),
    }
    return branch_pts, branch_roles, backbone_pts, backbone_roles, meta
