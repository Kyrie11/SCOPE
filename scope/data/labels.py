"""Operational labels: branch, burden, safety, FD diagnostics, and boundary labels."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scope.data.scene_schema import BRANCH_TO_INDEX, EgoCandidate, LabelMask, ResponseLabel, RootScene, RolloutResult
from scope.geometry.boxes import collision_any, min_ttc_discrete, swept_min_distance
from scope.geometry.conflict_zone import delta_tta, lateral_overlap_at_closest
from scope.geometry.drac import max_deceleration, max_jerk, trajectory_drac
from scope.geometry.map_utils import infer_route_from_logged_future, target_gap_margin


@dataclass
class AnchorStats:
    delta_tta: float = 0.0
    d_min: float = 50.0
    ttc_min: float | None = None
    drac: float = 0.0
    gap_margin: float = 50.0
    lateral_overlap: float = 0.0
    eta_commitment: float = 0.0

    def vector(self) -> np.ndarray:
        return np.array([
            self.delta_tta,
            self.d_min,
            10.0 if self.ttc_min is None else self.ttc_min,
            self.drac,
            self.gap_margin,
            self.lateral_overlap,
            self.eta_commitment,
        ], dtype=np.float32)


def compute_physical_anchors(scene: RootScene, candidate: EgoCandidate, agent_index: int, agent_future: np.ndarray) -> AnchorStats:
    ego_route = scene.route_context.primary_route
    if ego_route is None or len(ego_route) < 2:
        ego_route = infer_route_from_logged_future(candidate.future_states)
    agent_route = infer_route_from_logged_future(agent_future)
    ego_root = scene.root_state(scene.ego_track_index)
    agent_root = scene.root_state(agent_index)
    dtta = delta_tta(ego_route, agent_route, ego_root, agent_root) if len(agent_route) >= 2 and len(ego_route) >= 2 else None
    d_min, _ = swept_min_distance(candidate.future_states, agent_future)
    ttc = min_ttc_discrete(candidate.future_states, agent_future, scene.dt, scene.future_horizon_s)
    drac = trajectory_drac(candidate.future_states, agent_future)
    gap, _headway = target_gap_margin(candidate.future_states, agent_future, ego_route)
    lat = lateral_overlap_at_closest(candidate.future_states, agent_future)
    eta = lateral_commitment(candidate.future_states)
    return AnchorStats(0.0 if dtta is None else float(dtta), float(d_min), ttc, float(drac), float(gap), float(lat), float(eta))


def lateral_commitment(future: np.ndarray) -> float:
    if len(future) < 2:
        return 0.0
    heading = future[:, 6]
    normal = np.stack([-np.sin(heading), np.cos(heading)], axis=-1)
    disp = future[:, :2] - future[0, :2]
    lat = np.abs(np.sum(disp * normal, axis=-1))
    return float(np.clip(np.max(lat) / 3.5, 0.0, 1.0))


def safety_events(scene: RootScene, candidate: EgoCandidate, agent_future: np.ndarray, diagnostics: dict[str, Any] | None = None) -> dict[str, bool]:
    d_min, _ = swept_min_distance(candidate.future_states, agent_future)
    ttc = min_ttc_discrete(candidate.future_states, agent_future, scene.dt, scene.future_horizon_s)
    b_max = max_deceleration(agent_future, scene.dt)
    ego_route = scene.route_context.primary_route
    gap, headway = target_gap_margin(candidate.future_states, agent_future, ego_route)
    events = {
        "collision": bool(collision_any(candidate.future_states, agent_future)),
        "near_collision": bool(d_min < 1.0 or (ttc is not None and ttc < 1.5)),
        "induced_hard_brake": bool(b_max >= 3.5),
        "unsafe_gap": bool(headway < 1.0 or gap < 5.0),
    }
    if diagnostics is not None:
        diagnostics.update({"d_min": float(d_min), "ttc_min": ttc, "b_max": float(b_max), "gap_margin_m": float(gap), "headway_s": float(headway)})
    return events


def label_branch(scene: RootScene, candidate: EgoCandidate, agent_index: int, agent_future: np.ndarray, nominal_future: np.ndarray | None = None) -> tuple[int, dict[str, Any]]:
    ego_future = candidate.future_states
    ego_route = scene.route_context.primary_route
    if ego_route is None or len(ego_route) < 2:
        ego_route = infer_route_from_logged_future(ego_future)
    agent_route = infer_route_from_logged_future(agent_future)
    anchors = compute_physical_anchors(scene, candidate, agent_index, agent_future)
    diagnostics: dict[str, Any] = {
        "delta_tta": None if anchors.delta_tta == 0.0 else anchors.delta_tta,
        "d_min": anchors.d_min,
        "ttc_min": anchors.ttc_min,
        "drac": anchors.drac,
        "lateral_overlap": anchors.lateral_overlap,
        "commitment": anchors.eta_commitment,
        "branch_evidence": {},
    }
    speed_change = 0.0
    final_long_dev = 0.0
    if nominal_future is not None and len(nominal_future) == len(agent_future):
        speed_change = float(np.max(np.abs(agent_future[:, 5] - nominal_future[:, 5])))
        final_long_dev = float(np.linalg.norm(agent_future[-1, :2] - nominal_future[-1, :2]))
    route_low = anchors.lateral_overlap <= 0.05 and (anchors.delta_tta == 0.0 or abs(anchors.delta_tta) > 5.0)
    unaffected = (route_low or anchors.d_min > 30.0) and anchors.d_min > 10.0 and (anchors.ttc_min is None or anchors.ttc_min > 5.0) and speed_change < 1.0 and final_long_dev < 2.0
    diagnostics["branch_evidence"]["unaffected"] = bool(unaffected)
    if unaffected:
        return BRANCH_TO_INDEX["unaffected"], diagnostics
    gap, headway = target_gap_margin(ego_future, agent_future, ego_route)
    b_max = max_deceleration(agent_future, scene.dt)
    dtta = anchors.delta_tta
    maintain = dtta < -0.5 and gap < 8.0 and not behind_ego_after_entry(ego_future, agent_future, scene.dt, duration_s=2.0)
    cede = (gap >= 3.0 or headway >= 0.5) and (dtta >= -0.25 or b_max >= 1.5)
    follow = dtta > 0.0 and behind_ego_after_entry(ego_future, agent_future, scene.dt, duration_s=2.0) and headway > 1.0 and b_max < 3.5
    diagnostics["branch_evidence"].update({"maintain": bool(maintain), "cede": bool(cede), "follow": bool(follow), "gap": float(gap), "headway": float(headway), "b_max": float(b_max)})
    truth_count = int(maintain) + int(cede) + int(follow)
    if truth_count > 1:
        return BRANCH_TO_INDEX["ambiguous"], diagnostics
    if maintain:
        return BRANCH_TO_INDEX["maintain"], diagnostics
    if cede:
        return BRANCH_TO_INDEX["cede"], diagnostics
    if follow:
        return BRANCH_TO_INDEX["follow"], diagnostics
    return BRANCH_TO_INDEX["ambiguous"], diagnostics


def behind_ego_after_entry(ego_future: np.ndarray, agent_future: np.ndarray, dt: float, duration_s: float = 2.0) -> bool:
    steps = min(len(ego_future), len(agent_future), max(1, int(round(duration_s / dt))))
    heading = ego_future[:steps, 6]
    tangent = np.stack([np.cos(heading), np.sin(heading)], axis=-1)
    rel = agent_future[:steps, :2] - ego_future[:steps, :2]
    lon = np.sum(rel * tangent, axis=-1)
    return bool(np.mean(lon < 0.0) > 0.75)


def label_burden(scene: RootScene, candidate: EgoCandidate, agent_future: np.ndarray, branch: int, diagnostics: dict[str, Any] | None = None) -> int:
    ttc = min_ttc_discrete(candidate.future_states, agent_future, scene.dt, scene.future_horizon_s)
    b_max = max_deceleration(agent_future, scene.dt)
    j_max = max_jerk(agent_future, scene.dt)
    b_req = trajectory_drac(candidate.future_states, agent_future)
    gap, headway = target_gap_margin(candidate.future_states, agent_future, None)
    progress_loss = float(max(0.0, agent_future[0, 5] * scene.future_horizon_s - np.linalg.norm(agent_future[-1, :2] - agent_future[0, :2])))
    unsafe_gap = headway < 1.0 or gap < 5.0
    collision = collision_any(candidate.future_states, agent_future)
    if diagnostics is not None:
        diagnostics.update({"b_req": float(b_req), "b_max": float(b_max), "j_max": float(j_max), "gap_margin_m": float(gap), "headway_s": float(headway), "progress_loss": progress_loss})
    if collision or unsafe_gap or b_max >= 3.5 or b_req >= 4.0 or (ttc is not None and ttc < 2.0):
        return 3
    if b_max >= 2.0 or b_req >= 2.0 or (ttc is not None and 2.0 <= ttc <= 3.0) or progress_loss > 8.0:
        return 2
    if b_max >= 1.0 or b_req >= 1.0 or (ttc is not None and ttc <= 5.0) or progress_loss > 3.0:
        return 1
    return 0


def make_response_label(scene: RootScene, candidate: EgoCandidate, agent_index: int, agent_future: np.ndarray, nominal_future: np.ndarray | None = None) -> ResponseLabel:
    branch, diagnostics = label_branch(scene, candidate, agent_index, agent_future, nominal_future)
    burden = label_burden(scene, candidate, agent_future, branch, diagnostics)
    safety = safety_events(scene, candidate, agent_future, diagnostics)
    return ResponseLabel(branch, burden, agent_future, safety, diagnostics)


def diagnostic_forced_dependence(label: ResponseLabel, nonceding_risk_or_unsafe: bool, rho0: int = 2) -> tuple[bool | None, bool]:
    if label.branch != BRANCH_TO_INDEX["cede"]:
        return False, True
    if label.burden < rho0:
        return False, True
    if label.safety.get("collision", False):
        return False, True
    return (True if nonceding_risk_or_unsafe else None), bool(nonceding_risk_or_unsafe)


def boundary_label(label_a: ResponseLabel, label_b: ResponseLabel, rho0: int = 2) -> bool:
    hp_a = label_a.branch == BRANCH_TO_INDEX["cede"] and label_a.burden >= rho0
    hp_b = label_b.branch == BRANCH_TO_INDEX["cede"] and label_b.burden >= rho0
    events = ["collision", "near_collision"]
    return bool(
        label_a.branch != label_b.branch
        or hp_a != hp_b
        or any(label_a.safety.get(e, False) != label_b.safety.get(e, False) for e in events)
        or label_a.diagnostics.get("fd_diagnostic") != label_b.diagnostics.get("fd_diagnostic")
    )


def labels_from_rollout(scene: RootScene, candidate: EgoCandidate, rollout: RolloutResult, policy_variant_id: str, nominal_futures: dict[int, np.ndarray] | None = None) -> tuple[dict[tuple[str, int, str], ResponseLabel], dict[tuple[str, int, str], LabelMask]]:
    labels: dict[tuple[str, int, str], ResponseLabel] = {}
    masks: dict[tuple[str, int, str], LabelMask] = {}
    for agent_id in scene.relevant_agent_indices:
        key = (candidate.candidate_id, agent_id, policy_variant_id)
        if not rollout.valid or agent_id not in rollout.agent_futures:
            masks[key] = LabelMask(False, False, False, False, False, False, False)
            continue
        nominal = nominal_futures.get(agent_id) if nominal_futures else None
        label = make_response_label(scene, candidate, agent_id, rollout.agent_futures[agent_id], nominal)
        labels[key] = label
        masks[key] = LabelMask(True, True, True, True, True, False, False)
    return labels, masks
