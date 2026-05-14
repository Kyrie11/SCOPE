"""Conflict-zone and route-arrival utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from shapely.geometry import LineString, Point
except Exception:  # pragma: no cover
    LineString = None
    Point = None


@dataclass(frozen=True)
class ConflictZone:
    exists: bool
    point: np.ndarray | None
    ego_progress_m: float | None
    agent_progress_m: float | None
    overlap_area_m2: float = 0.0


def polyline_lengths(polyline: np.ndarray) -> np.ndarray:
    pts = np.asarray(polyline, dtype=float)
    if len(pts) == 0:
        return np.zeros(0)
    seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=-1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def nearest_progress(polyline: np.ndarray, point: np.ndarray) -> float:
    pts = np.asarray(polyline, dtype=float)[:, :2]
    if len(pts) == 0:
        return 0.0
    cum = polyline_lengths(pts)
    best_d = float("inf")
    best_s = 0.0
    p = np.asarray(point, dtype=float)[:2]
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        ab = b - a
        denom = float(ab @ ab)
        t = 0.0 if denom < 1e-9 else float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
        proj = a + t * ab
        d = float(np.linalg.norm(p - proj))
        if d < best_d:
            best_d = d
            best_s = float(cum[i] + t * np.linalg.norm(ab))
    return best_s


def route_tube_overlap(route_a: np.ndarray, route_b: np.ndarray, lateral_buffer_m: float = 1.0) -> tuple[float, np.ndarray | None]:
    a = np.asarray(route_a, dtype=float)
    b = np.asarray(route_b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0, None
    if LineString is not None:
        pa = LineString(a[:, :2]).buffer(lateral_buffer_m, cap_style=2, join_style=2)
        pb = LineString(b[:, :2]).buffer(lateral_buffer_m, cap_style=2, join_style=2)
        inter = pa.intersection(pb)
        if inter.is_empty:
            return 0.0, None
        centroid = np.array([inter.centroid.x, inter.centroid.y], dtype=np.float32)
        return float(inter.area), centroid
    # fallback: use closest route samples as a small overlap proxy
    d = np.linalg.norm(a[:, None, :2] - b[None, :, :2], axis=-1)
    idx = np.unravel_index(np.argmin(d), d.shape)
    if d[idx] <= 2.0 * lateral_buffer_m:
        return float((2.0 * lateral_buffer_m - d[idx]) * lateral_buffer_m), (a[idx[0], :2] + b[idx[1], :2]) * 0.5
    return 0.0, None


def find_conflict_zone(ego_route: np.ndarray, agent_route: np.ndarray, lateral_buffer_m: float = 1.0) -> ConflictZone:
    area, point = route_tube_overlap(ego_route, agent_route, lateral_buffer_m)
    if point is None:
        return ConflictZone(False, None, None, None, 0.0)
    return ConflictZone(True, point, nearest_progress(ego_route, point), nearest_progress(agent_route, point), area)


def constant_speed_arrival_time(route: np.ndarray, root_state: np.ndarray, conflict_progress_m: float | None) -> float | None:
    if conflict_progress_m is None:
        return None
    current_progress = nearest_progress(route, root_state[:2])
    dist = conflict_progress_m - current_progress
    if dist < -1.0:
        return None
    speed = max(float(root_state[5]), 0.1)
    return max(0.0, dist / speed)


def delta_tta(ego_route: np.ndarray, agent_route: np.ndarray, ego_root: np.ndarray, agent_root: np.ndarray) -> float | None:
    cz = find_conflict_zone(ego_route, agent_route)
    if not cz.exists:
        return None
    te = constant_speed_arrival_time(ego_route, ego_root, cz.ego_progress_m)
    ta = constant_speed_arrival_time(agent_route, agent_root, cz.agent_progress_m)
    if te is None or ta is None:
        return None
    return float(te - ta)


def lateral_overlap_at_closest(ego: np.ndarray, agent: np.ndarray) -> float:
    steps = min(len(ego), len(agent))
    if steps == 0:
        return 0.0
    best = 0.0
    for t in range(steps):
        rel = agent[t, :2] - ego[t, :2]
        normal = np.array([-math.sin(float(ego[t, 6])), math.cos(float(ego[t, 6]))])
        lat = abs(float(rel @ normal))
        half_sum = 0.5 * (float(ego[t, 8]) + float(agent[t, 8]))
        best = max(best, max(0.0, half_sum - lat))
    return float(best)
