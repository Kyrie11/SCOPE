"""Relevant-agent selection for same-root intervention groups."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scope.data.scene_schema import RootScene
from scope.geometry.conflict_zone import delta_tta, route_tube_overlap
from scope.geometry.map_utils import infer_route_from_logged_future


@dataclass
class RelevantAgentScore:
    agent_index: int
    score: float
    components: dict[str, float]


def _agent_route(scene: RootScene, idx: int) -> np.ndarray:
    fut = scene.agent_future_logged(idx)
    if len(fut) >= 2:
        return infer_route_from_logged_future(fut)
    return scene.tracks.states[idx, max(0, scene.current_time_index - 5) : scene.current_time_index + 1, :2]


def score_agent(scene: RootScene, idx: int, weights: dict[str, float] | None = None) -> RelevantAgentScore:
    weights = weights or {
        "overlap": 2.0,
        "dist": 1.5,
        "tta": 2.0,
        "gap": 1.0,
        "cross": 1.0,
        "rear": 1.0,
        "oi": 1.0,
    }
    root_i = scene.root_state(idx)
    root_e = scene.root_state(scene.ego_track_index)
    ego_route = scene.route_context.primary_route
    if ego_route is None:
        ego_route = infer_route_from_logged_future(scene.ego_future_logged())
    agent_route = _agent_route(scene, idx)
    overlap_area, _ = route_tube_overlap(ego_route, agent_route, 1.0) if len(ego_route) >= 2 and len(agent_route) >= 2 else (0.0, None)
    dist = float(np.linalg.norm(root_i[:2] - root_e[:2]))
    tta = delta_tta(ego_route, agent_route, root_e, root_i) if len(ego_route) >= 2 and len(agent_route) >= 2 else None
    heading = np.array([np.cos(root_e[6]), np.sin(root_e[6])])
    rel = root_i[:2] - root_e[:2]
    longitudinal = float(rel @ heading)
    lateral = abs(float(rel @ np.array([-heading[1], heading[0]])))
    rear = 1.0 if longitudinal < 0 and (abs(longitudinal) < 25.0 or abs(longitudinal) / max(root_i[5], 0.1) < 2.0) else 0.0
    gap = 1.0 if lateral < 5.0 and abs(longitudinal) < 35.0 else 0.0
    cross = 1.0 if overlap_area > 0.5 else 0.0
    oi = 1.0 if idx in scene.metadata.get("objects_of_interest", []) or idx in scene.metadata.get("tracks_to_predict", []) else 0.0
    components = {
        "overlap": min(overlap_area / 20.0, 1.0),
        "dist": float(np.exp(-dist / 30.0)),
        "tta": float(np.exp(-abs(tta) / 3.0)) if tta is not None else 0.0,
        "gap": gap,
        "cross": cross,
        "rear": rear,
        "oi": oi,
    }
    score = sum(weights[k] * components[k] for k in components)
    return RelevantAgentScore(idx, float(score), components)


def select_relevant_agents(scene: RootScene, top_n: int = 12, agent_types: list[str] | None = None, weights: dict[str, float] | None = None) -> list[RelevantAgentScore]:
    agent_types = agent_types or ["vehicle", "pedestrian", "cyclist", "TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST"]
    scores: list[RelevantAgentScore] = []
    for idx in range(scene.tracks.states.shape[0]):
        if idx == scene.ego_track_index:
            continue
        if scene.tracks.object_types[idx] not in agent_types:
            continue
        if not bool(scene.tracks.valid[idx, scene.current_time_index]):
            continue
        scores.append(score_agent(scene, idx, weights))
    scores.sort(key=lambda s: s.score, reverse=True)
    return scores[:top_n]
