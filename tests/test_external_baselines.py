from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_external_baselines import AreaEventSpec, FeasibilityChecker


def make_checker(event: AreaEventSpec | None = None) -> FeasibilityChecker:
    z_grid = np.zeros((10, 10), dtype=float)
    floor = np.full((10, 10), 30.0, dtype=float)
    ceiling = np.full((10, 10), 120.0, dtype=float)
    return FeasibilityChecker(
        z_grid,
        floor,
        ceiling,
        event=event,
        min_clearance_m=30.0,
        max_climb_angle_deg=30.0,
        samples_per_segment=5,
    )


def test_rejects_point_below_floor() -> None:
    checker = make_checker()
    assert not checker.point_is_valid([0.02, 0.02, 20.0])


def test_rejects_point_above_ceiling() -> None:
    checker = make_checker()
    assert not checker.point_is_valid([0.02, 0.02, 130.0])


def test_rejects_segment_exceeding_climb_angle() -> None:
    checker = make_checker()
    assert not checker.segment_is_valid([0.02, 0.02, 35.0], [0.04, 0.02, 95.0])


def test_rejects_segment_slightly_over_climb_angle() -> None:
    checker = make_checker()
    assert not checker.segment_is_valid([0.02, 0.02, 40.0], [0.10, 0.02, 90.0])


def test_rejects_no_fly_intersection() -> None:
    event = AreaEventSpec(
        event_type="no_fly",
        center_x_km=0.05,
        center_y_km=0.02,
        radius_km=0.015,
        severity=1.0,
    )
    checker = make_checker(event=event)
    assert not checker.segment_is_valid([0.02, 0.02, 60.0], [0.08, 0.02, 60.0])
