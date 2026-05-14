"""Deceleration-rate-required-to-avoid-collision utilities."""
from __future__ import annotations

import numpy as np

from scope.geometry.boxes import relative_closing_speed, swept_min_distance


def drac_from_gap_and_closing(gap_m: float, closing_speed_mps: float, eps: float = 1e-3) -> float:
    if closing_speed_mps <= 0 or not np.isfinite(gap_m) or gap_m <= eps:
        return 0.0 if gap_m > eps else float("inf")
    return float(closing_speed_mps**2 / (2.0 * max(gap_m, eps)))


def trajectory_drac(ego: np.ndarray, agent: np.ndarray) -> float:
    steps = min(len(ego), len(agent))
    best = 0.0
    for t in range(steps):
        if ego[t, 9] < 0.5 or agent[t, 9] < 0.5:
            continue
        gap, _ = swept_min_distance(ego[t : t + 1], agent[t : t + 1])
        closing = relative_closing_speed(ego[t], agent[t])
        val = drac_from_gap_and_closing(gap, closing)
        if np.isfinite(val):
            best = max(best, val)
        else:
            return float("inf")
    return float(best)


def max_deceleration(traj: np.ndarray, dt: float) -> float:
    if len(traj) < 2:
        return 0.0
    speed = np.asarray(traj[:, 5], dtype=float)
    acc = np.diff(speed) / dt
    return float(max(0.0, -np.min(acc))) if acc.size else 0.0


def max_jerk(traj: np.ndarray, dt: float) -> float:
    if len(traj) < 3:
        return 0.0
    speed = np.asarray(traj[:, 5], dtype=float)
    acc = np.diff(speed) / dt
    jerk = np.diff(acc) / dt
    return float(np.max(np.abs(jerk))) if jerk.size else 0.0
