"""Root-state mining for interactive same-root groups."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scope.data.relevant_agents import select_relevant_agents
from scope.data.scene_schema import RootScene
from scope.geometry.conflict_zone import delta_tta, route_tube_overlap
from scope.geometry.map_utils import infer_route_from_logged_future


@dataclass
class MiningResult:
    keep: bool
    tags: list[str]
    debug: dict[str, float]


def mine_interaction_conditions(scene: RootScene) -> MiningResult:
    ego_route = scene.route_context.primary_route
    if ego_route is None or len(ego_route) < 2:
        ego_route = infer_route_from_logged_future(scene.ego_future_logged())
    ego_root = scene.root_state(scene.ego_track_index)
    tags: set[str] = set()
    debug: dict[str, float] = {}
    for idx in range(scene.tracks.states.shape[0]):
        if idx == scene.ego_track_index or not scene.tracks.valid[idx, scene.current_time_index]:
            continue
        root_i = scene.root_state(idx)
        agent_route = infer_route_from_logged_future(scene.agent_future_logged(idx))
        dist = float(np.linalg.norm(root_i[:2] - ego_root[:2]))
        if dist <= 80.0:
            tags.add("lane_or_target_gap_interaction")
        area, _ = route_tube_overlap(ego_route, agent_route, 1.0) if len(agent_route) >= 2 and len(ego_route) >= 2 else (0.0, None)
        if area > 0.1:
            tags.add("conflict_zone_overlap")
            debug["max_overlap_area"] = max(debug.get("max_overlap_area", 0.0), area)
        dtta = delta_tta(ego_route, agent_route, ego_root, root_i) if len(agent_route) >= 2 and len(ego_route) >= 2 else None
        if dtta is not None:
            debug["min_abs_delta_tta"] = min(debug.get("min_abs_delta_tta", float("inf")), abs(dtta))
            if abs(dtta) <= 3.0:
                tags.add("close_arrival")
        heading = np.array([np.cos(ego_root[6]), np.sin(ego_root[6])])
        rel = root_i[:2] - ego_root[:2]
        longitudinal = float(rel @ heading)
        lateral = abs(float(rel @ np.array([-heading[1], heading[0]])))
        headway = abs(longitudinal) / max(float(root_i[5]), 0.1)
        if lateral < 5.0 and (abs(longitudinal) < 25.0 or headway < 2.5):
            tags.add("dense_gap")
        if longitudinal < 0 and (abs(longitudinal) < 20.0 or headway < 2.0):
            tags.add("close_following")
    if any("unprotected" in x or "ambiguous" in x for x in scene.scenario_tags):
        tags.add("ambiguous_priority")
    keep = bool(tags)
    return MiningResult(keep, sorted(tags), debug)


def mine_root_scenes(scenes: list[RootScene], top_n_agents: int = 12, min_candidates_per_scene: int = 1) -> list[RootScene]:
    kept: list[RootScene] = []
    for scene in scenes:
        result = mine_interaction_conditions(scene)
        if not result.keep:
            continue
        rel = select_relevant_agents(scene, top_n_agents)
        if not rel:
            continue
        scene.scenario_tags = sorted(set(scene.scenario_tags) | set(result.tags))
        scene.relevant_agent_indices = [s.agent_index for s in rel]
        scene.metadata["root_mining_debug"] = result.debug
        scene.metadata["relevant_agent_scores"] = [s.__dict__ for s in rel]
        kept.append(scene)
    return kept
