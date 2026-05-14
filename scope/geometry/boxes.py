"""Oriented-box geometry for collision, distance, and time-to-collision."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from shapely.geometry import Polygon
except Exception:  # pragma: no cover
    Polygon = None


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    yaw: float
    length: float
    width: float

    @classmethod
    def from_state(cls, state: np.ndarray) -> "Box":
        return cls(float(state[0]), float(state[1]), float(state[6]), float(state[7]), float(state[8]))


def box_corners(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
    half_l = length * 0.5
    half_w = width * 0.5
    local = np.array([[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]], dtype=np.float64)
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])


def corners_from_box(box: Box) -> np.ndarray:
    return box_corners(box.x, box.y, box.yaw, box.length, box.width)


def polygon_from_box(box: Box):
    if Polygon is None:
        return None
    return Polygon(corners_from_box(box))


def _axes(corners: np.ndarray) -> list[np.ndarray]:
    edges = np.roll(corners, -1, axis=0) - corners
    axes = []
    for e in edges[:2]:
        n = np.array([-e[1], e[0]], dtype=float)
        norm = np.linalg.norm(n)
        if norm > 1e-9:
            axes.append(n / norm)
    return axes


def _project(corners: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    vals = corners @ axis
    return float(vals.min()), float(vals.max())


def boxes_intersect(a: Box, b: Box, eps: float = 1e-9) -> bool:
    ca = corners_from_box(a)
    cb = corners_from_box(b)
    for axis in _axes(ca) + _axes(cb):
        amin, amax = _project(ca, axis)
        bmin, bmax = _project(cb, axis)
        if amax < bmin - eps or bmax < amin - eps:
            return False
    return True


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _segments_distance(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> float:
    return min(
        _point_segment_distance(a0, b0, b1),
        _point_segment_distance(a1, b0, b1),
        _point_segment_distance(b0, a0, a1),
        _point_segment_distance(b1, a0, a1),
    )


def oriented_box_distance(a: Box, b: Box) -> float:
    if boxes_intersect(a, b):
        return 0.0
    if Polygon is not None:
        pa = polygon_from_box(a)
        pb = polygon_from_box(b)
        return float(pa.distance(pb))
    ca = corners_from_box(a)
    cb = corners_from_box(b)
    d = float("inf")
    for i in range(4):
        for j in range(4):
            d = min(d, _segments_distance(ca[i], ca[(i + 1) % 4], cb[j], cb[(j + 1) % 4]))
    return float(d)


def swept_min_distance(ego: np.ndarray, agent: np.ndarray) -> tuple[float, int]:
    steps = min(len(ego), len(agent))
    best = float("inf")
    best_t = 0
    for t in range(steps):
        if ego[t, 9] < 0.5 or agent[t, 9] < 0.5:
            continue
        d = oriented_box_distance(Box.from_state(ego[t]), Box.from_state(agent[t]))
        if d < best:
            best = d
            best_t = t
    return float(best), best_t


def collision_any(ego: np.ndarray, agent: np.ndarray) -> bool:
    steps = min(len(ego), len(agent))
    for t in range(steps):
        if ego[t, 9] >= 0.5 and agent[t, 9] >= 0.5 and boxes_intersect(Box.from_state(ego[t]), Box.from_state(agent[t])):
            return True
    return False


def min_ttc_discrete(ego: np.ndarray, agent: np.ndarray, dt: float, horizon_s: float = 8.0, substeps: int = 5) -> float | None:
    """Approximate TTC using OBB distance plus constant-velocity projection.

    The routine avoids a costly full nested simulation. For every synchronized
    future step it computes the oriented-box clearance and relative closing
    speed, projects to the first possible contact time, and verifies projected
    OBB separation around that time. This keeps TTC branch labels grounded in
    box geometry rather than center-distance division while remaining practical
    for candidate matrices.
    """
    steps = min(len(ego), len(agent))
    best: float | None = None
    for t in range(steps):
        if ego[t, 9] < 0.5 or agent[t, 9] < 0.5:
            continue
        dist = oriented_box_distance(Box.from_state(ego[t]), Box.from_state(agent[t]))
        if dist <= 0.0:
            return 0.0
        closing = relative_closing_speed(ego[t], agent[t])
        if closing <= 1e-6:
            continue
        tau = dist / closing
        if tau < 0.0 or tau > horizon_s:
            continue
        # Verify around the projected contact time; if exact overlap is missed
        # due to heading simplification, retain the projected TTC when the
        # projected clearance is under one meter.
        verified = False
        min_proj_dist = float("inf")
        for probe in (max(0.0, tau - dt), tau, min(horizon_s, tau + dt)):
            ego_state = ego[t].copy()
            agent_state = agent[t].copy()
            ego_state[:2] = ego[t, :2] + ego[t, 3:5] * probe
            agent_state[:2] = agent[t, :2] + agent[t, 3:5] * probe
            d_probe = oriented_box_distance(Box.from_state(ego_state), Box.from_state(agent_state))
            min_proj_dist = min(min_proj_dist, d_probe)
            if d_probe <= 0.0:
                verified = True
                break
        if verified or min_proj_dist < 1.0:
            best = tau if best is None else min(best, tau)
    return best

def relative_closing_speed(ego_state: np.ndarray, agent_state: np.ndarray) -> float:
    rel = agent_state[:2] - ego_state[:2]
    norm = float(np.linalg.norm(rel))
    if norm < 1e-6:
        return 0.0
    rel_v = agent_state[3:5] - ego_state[3:5]
    return max(0.0, -float(rel @ rel_v) / norm)


def batch_oriented_distance(ego: np.ndarray, agents: Iterable[np.ndarray]) -> np.ndarray:
    return np.asarray([swept_min_distance(ego, a)[0] for a in agents], dtype=np.float32)
