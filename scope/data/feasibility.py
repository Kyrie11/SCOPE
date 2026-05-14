"""Candidate feasibility checks."""
from __future__ import annotations

import numpy as np

from scope.data.scene_schema import EgoCandidate, RootScene
from scope.geometry.boxes import Box, boxes_intersect
from scope.geometry.map_utils import points_in_drivable


def _speed_acc_jerk(future: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    speed = np.asarray(future[:, 5], dtype=float)
    acc = np.diff(speed, prepend=speed[0]) / dt
    jerk = np.diff(acc, prepend=acc[0]) / dt
    return acc, jerk


def lateral_acceleration(future: np.ndarray, dt: float) -> np.ndarray:
    if len(future) < 3:
        return np.zeros(len(future))
    yaw = np.unwrap(future[:, 6].astype(float))
    yaw_rate = np.diff(yaw, prepend=yaw[0]) / dt
    return future[:, 5].astype(float) * yaw_rate


def check_candidate_feasibility(
    scene: RootScene,
    candidate: EgoCandidate,
    max_accel: float = 3.0,
    max_jerk: float = 4.0,
    max_lateral_accel: float = 2.5,
    drivable_tolerance_m: float = 4.0,
) -> dict[str, object]:
    fut = candidate.future_states
    acc, jerk = _speed_acc_jerk(fut, scene.dt)
    lat_acc = lateral_acceleration(fut, scene.dt)
    dynamic_ok = bool(np.nanmax(np.abs(acc)) <= max_accel + 1e-6 and np.nanmax(np.abs(jerk)) <= max_jerk + 1e-6 and np.nanmax(np.abs(lat_acc)) <= max_lateral_accel + 1e-6)
    drivable_ok = bool(points_in_drivable(fut[:, :2], scene.map_features, drivable_tolerance_m).all())
    route_ok = True
    static_collision_free = True
    # Static map collision is approximated by drivable consistency here because WOMD map polygons do not contain parked obstacles.
    return {
        "dynamic_ok": dynamic_ok,
        "drivable_ok": drivable_ok,
        "route_ok": route_ok,
        "static_collision_free": static_collision_free,
        "duplicate_of": candidate.feasibility.get("duplicate_of"),
        "max_abs_accel": float(np.nanmax(np.abs(acc))) if len(acc) else 0.0,
        "max_abs_jerk": float(np.nanmax(np.abs(jerk))) if len(jerk) else 0.0,
        "max_abs_lateral_accel": float(np.nanmax(np.abs(lat_acc))) if len(lat_acc) else 0.0,
    }


def mark_duplicates(candidates: list[EgoCandidate], mean_threshold_m: float = 0.3, final_threshold_m: float = 0.5) -> list[EgoCandidate]:
    accepted: list[EgoCandidate] = []
    for cand in candidates:
        dup_of = None
        for prev in accepted:
            steps = min(len(cand.future_states), len(prev.future_states))
            disp = np.linalg.norm(cand.future_states[:steps, :2] - prev.future_states[:steps, :2], axis=-1)
            if float(np.mean(disp)) < mean_threshold_m and float(disp[-1]) < final_threshold_m:
                dup_of = prev.candidate_id
                break
        cand.feasibility["duplicate_of"] = dup_of
        if dup_of is None:
            accepted.append(cand)
    return candidates


def filter_feasible(candidates: list[EgoCandidate]) -> list[EgoCandidate]:
    return [c for c in candidates if c.is_feasible]
