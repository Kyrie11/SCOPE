"""Ego intervention candidate generation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from scope.data.feasibility import check_candidate_feasibility, mark_duplicates
from scope.data.scene_schema import EgoCandidate, RootScene
from scope.geometry.map_utils import infer_route_from_logged_future


@dataclass
class CandidateConfig:
    candidate_count: int = 32
    timing_shifts_s: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
    speed_multipliers: tuple[float, ...] = (0.75, 0.9, 1.0, 1.1)
    lateral_durations_s: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0)
    lateral_midpoint_shifts_s: tuple[float, ...] = (-0.5, 0.0, 0.5)
    planner_proposals: int = 8
    max_repair_candidates: int = 8


class CandidateGenerator:
    def __init__(self, cfg: CandidateConfig | dict | None = None):
        if cfg is None:
            cfg = CandidateConfig()
        self.cfg = CandidateConfig(**cfg) if isinstance(cfg, dict) else cfg

    def generate(self, scene: RootScene) -> list[EgoCandidate]:
        base = scene.ego_future_logged().copy()
        base[:, 9] = 1.0
        cands: list[EgoCandidate] = []
        cands.append(EgoCandidate("logged", "logged", base, task_cost=0.0))
        for dt_shift in self.cfg.timing_shifts_s:
            cands.append(self._timing_candidate(scene, base, dt_shift))
        for mult in self.cfg.speed_multipliers:
            cands.append(self._speed_candidate(scene, base, mult))
        for dur in self.cfg.lateral_durations_s:
            for mid in self.cfg.lateral_midpoint_shifts_s:
                cands.append(self._lateral_candidate(scene, base, dur, mid))
        for offset in (-6.0, 0.0, 6.0):
            cands.append(self._gap_candidate(scene, base, offset))
        for idx in range(self.cfg.planner_proposals):
            cands.append(self._planner_candidate(scene, base, idx))
        self._apply_feasibility(scene, cands)
        mark_duplicates(cands)
        return self._stratified_keep(cands, self.cfg.candidate_count)

    def _apply_feasibility(self, scene: RootScene, cands: list[EgoCandidate]) -> None:
        for cand in cands:
            cand.feasibility.update(check_candidate_feasibility(scene, cand))

    def _timing_candidate(self, scene: RootScene, base: np.ndarray, shift_s: float) -> EgoCandidate:
        shifted = time_shift_trajectory(base, scene.dt, shift_s)
        return EgoCandidate(f"timing_{shift_s:+.1f}", "timing", shifted, {"delta_t": shift_s}, task_cost=abs(shift_s) * 0.1)

    def _speed_candidate(self, scene: RootScene, base: np.ndarray, multiplier: float) -> EgoCandidate:
        fut = speed_scale_trajectory(base, scene.dt, multiplier)
        return EgoCandidate(f"speed_{multiplier:.2f}", "speed", fut, {"speed_multiplier": multiplier}, task_cost=abs(multiplier - 1.0))

    def _gap_candidate(self, scene: RootScene, base: np.ndarray, longitudinal_offset_m: float) -> EgoCandidate:
        fut = base.copy()
        heading = fut[:, 6]
        tangent = np.stack([np.cos(heading), np.sin(heading)], axis=-1)
        ramp = np.linspace(0.0, 1.0, len(fut))[:, None]
        fut[:, :2] += ramp * longitudinal_offset_m * tangent
        recompute_kinematics(fut, scene.dt)
        name = "current" if abs(longitudinal_offset_m) < 1e-6 else ("later" if longitudinal_offset_m < 0 else "earlier")
        return EgoCandidate(f"gap_{name}_{longitudinal_offset_m:+.1f}", "gap", fut, {"target_gap_id": name, "longitudinal_buffer_m": float(max(0.0, longitudinal_offset_m))}, task_cost=abs(longitudinal_offset_m) * 0.02)

    def _lateral_candidate(self, scene: RootScene, base: np.ndarray, duration_s: float, midpoint_shift_s: float) -> EgoCandidate:
        fut = base.copy()
        steps = len(fut)
        times = np.arange(steps) * scene.dt
        center = max(0.0, 0.5 * duration_s + midpoint_shift_s)
        x = np.clip((times - (center - duration_s / 2.0)) / max(duration_s, 1e-3), 0.0, 1.0)
        smooth = x * x * (3.0 - 2.0 * x)
        # Route-consistent lateral commitment: change lateral offset by at most half a lane and blend back to logged end.
        normal = np.stack([-np.sin(fut[:, 6]), np.cos(fut[:, 6])], axis=-1)
        amp = 0.5 * np.sign(midpoint_shift_s if midpoint_shift_s != 0 else 1.0)
        fut[:, :2] += (amp * smooth * (1.0 - smooth))[:, None] * normal
        recompute_kinematics(fut, scene.dt)
        return EgoCandidate(
            f"lat_T{duration_s:.1f}_mid{midpoint_shift_s:+.1f}",
            "lateral",
            fut,
            {"lane_entry_duration_s": duration_s, "lateral_midpoint_shift_s": midpoint_shift_s},
            task_cost=0.05 * duration_s + abs(midpoint_shift_s) * 0.1,
        )

    def _planner_candidate(self, scene: RootScene, base: np.ndarray, idx: int) -> EgoCandidate:
        fut = base.copy()
        route = scene.route_context.primary_route
        factor = (idx - (self.cfg.planner_proposals - 1) / 2.0) / max(1, self.cfg.planner_proposals - 1)
        mult = 1.0 + 0.15 * factor
        fut = speed_scale_trajectory(fut, scene.dt, mult)
        if route is not None and len(route) >= 2:
            # Keep route-consistent by only nudging along heading.
            fut[:, :2] += (factor * 1.0 * np.linspace(0, 1, len(fut)))[:, None] * np.stack([np.cos(fut[:, 6]), np.sin(fut[:, 6])], axis=-1)
        recompute_kinematics(fut, scene.dt)
        return EgoCandidate(f"planner_{idx:02d}", "planner", fut, {"speed_multiplier": mult}, task_cost=abs(factor) * 0.2)

    def _stratified_keep(self, candidates: list[EgoCandidate], k: int) -> list[EgoCandidate]:
        feasible = [c for c in candidates if c.is_feasible]
        if len(feasible) <= k:
            return feasible
        priority_families = ["logged", "gap", "timing", "speed", "lateral", "planner"]
        kept: list[EgoCandidate] = []
        for fam in priority_families:
            fam_cands = [c for c in feasible if c.family == fam]
            if fam_cands:
                kept.append(sorted(fam_cands, key=lambda c: c.task_cost)[0])
        remaining = [c for c in feasible if c not in kept]
        remaining.sort(key=lambda c: (c.task_cost, c.candidate_id))
        for cand in remaining:
            if len(kept) >= k:
                break
            kept.append(cand)
        kept.sort(key=lambda c: c.candidate_id != "logged")
        return kept


def recompute_kinematics(future: np.ndarray, dt: float) -> None:
    if len(future) < 2:
        return
    vel = np.gradient(future[:, :2], dt, axis=0)
    future[:, 3:5] = vel
    future[:, 5] = np.linalg.norm(vel, axis=-1)
    yaw = np.arctan2(vel[:, 1], vel[:, 0])
    stationary = future[:, 5] < 0.1
    if stationary.any():
        yaw[stationary] = future[stationary, 6]
    future[:, 6] = yaw
    future[:, 9] = 1.0


def time_shift_trajectory(base: np.ndarray, dt: float, shift_s: float) -> np.ndarray:
    steps = len(base)
    old_t = np.arange(steps) * dt
    new_t = np.clip(old_t - shift_s, 0.0, old_t[-1])
    out = base.copy()
    for d in range(min(7, base.shape[1])):
        out[:, d] = np.interp(new_t, old_t, base[:, d])
    out[:, 7:10] = base[:, 7:10]
    recompute_kinematics(out, dt)
    return smooth_acceleration(out, dt)


def speed_scale_trajectory(base: np.ndarray, dt: float, multiplier: float, max_accel: float = 3.0) -> np.ndarray:
    out = base.copy()
    speed = np.clip(base[:, 5] * multiplier, 0.0, None)
    for t in range(1, len(speed)):
        dv = np.clip(speed[t] - speed[t - 1], -max_accel * dt, max_accel * dt)
        speed[t] = speed[t - 1] + dv
    heading = base[:, 6]
    out[0, :2] = base[0, :2]
    for t in range(1, len(out)):
        direction = np.array([math.cos(heading[t - 1]), math.sin(heading[t - 1])])
        out[t, :2] = out[t - 1, :2] + direction * speed[t - 1] * dt
    out[:, 3] = speed * np.cos(heading)
    out[:, 4] = speed * np.sin(heading)
    out[:, 5] = speed
    out[:, 6] = heading
    out[:, 9] = 1.0
    return smooth_acceleration(out, dt)


def smooth_acceleration(future: np.ndarray, dt: float, max_accel: float = 3.0, max_jerk: float = 4.0) -> np.ndarray:
    out = future.copy()
    speed = out[:, 5].copy()
    for _ in range(2):
        for t in range(1, len(speed)):
            speed[t] = speed[t - 1] + np.clip(speed[t] - speed[t - 1], -max_accel * dt, max_accel * dt)
        acc = np.diff(speed, prepend=speed[0]) / dt
        for t in range(1, len(acc)):
            allowed = acc[t - 1] + np.clip(acc[t] - acc[t - 1], -max_jerk * dt, max_jerk * dt)
            speed[t] = speed[t - 1] + allowed * dt
    heading = out[:, 6]
    out[:, 3] = speed * np.cos(heading)
    out[:, 4] = speed * np.sin(heading)
    out[:, 5] = speed
    return out


def repair_candidates_from_violation(scene: RootScene, candidate: EgoCandidate, violations: dict[str, float], max_count: int = 8) -> list[EgoCandidate]:
    repairs: list[EgoCandidate] = []
    base = candidate.future_states
    if violations.get("R", 0.0) > 0:
        repairs.append(EgoCandidate(candidate.candidate_id + "_repair_delay", "repair", time_shift_trajectory(base, scene.dt, 0.5), {**candidate.control_edits, "delta_t": candidate.control_edits.get("delta_t", 0.0) + 0.5}))
        repairs.append(EgoCandidate(candidate.candidate_id + "_repair_slow", "repair", speed_scale_trajectory(base, scene.dt, 0.9), {**candidate.control_edits, "speed_multiplier": 0.9 * candidate.control_edits.get("speed_multiplier", 1.0)}))
    if violations.get("D", 0.0) > 0:
        repairs.append(EgoCandidate(candidate.candidate_id + "_repair_later_gap", "repair", time_shift_trajectory(base, scene.dt, 0.75), {**candidate.control_edits, "target_gap_id": "later"}))
    if violations.get("P_hp", 0.0) > 0:
        fut = base.copy()
        heading = fut[:, 6]
        fut[:, :2] -= np.linspace(0, 5, len(fut))[:, None] * np.stack([np.cos(heading), np.sin(heading)], axis=-1)
        recompute_kinematics(fut, scene.dt)
        repairs.append(EgoCandidate(candidate.candidate_id + "_repair_buffer", "repair", fut, {**candidate.control_edits, "longitudinal_buffer_m": candidate.control_edits.get("longitudinal_buffer_m", 0.0) + 5.0, "time_headway_s": candidate.control_edits.get("time_headway_s", 1.5) + 0.5}))
    for cand in repairs[:max_count]:
        cand.feasibility.update(check_candidate_feasibility(scene, cand))
    return [c for c in repairs[:max_count] if c.is_feasible]
