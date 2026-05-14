"""Feasible-neighbor graph N_eta built from interpretable candidate edits."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from scope.data.scene_schema import EgoCandidate, NeighborEdge


EDIT_KEYS = [
    "delta_t",
    "speed_multiplier",
    "target_gap_id",
    "longitudinal_buffer_m",
    "time_headway_s",
    "lane_entry_duration_s",
    "lateral_midpoint_shift_s",
]


def edit_difference(a: EgoCandidate, b: EgoCandidate) -> tuple[str | None, float]:
    diffs: list[tuple[str, float]] = []
    ea, eb = a.control_edits, b.control_edits
    for key in EDIT_KEYS:
        va, vb = ea.get(key), eb.get(key)
        if va == vb:
            continue
        if key == "target_gap_id":
            d = 1.0
        else:
            try:
                d = abs(float(va) - float(vb))
            except (TypeError, ValueError):
                d = 1.0
        diffs.append((key, d))
    if len(diffs) != 1:
        return None, float("inf")
    key, raw = diffs[0]
    scale = {
        "delta_t": 0.5,
        "speed_multiplier": 0.15,
        "target_gap_id": 1.0,
        "longitudinal_buffer_m": 5.0,
        "time_headway_s": 0.5,
        "lane_entry_duration_s": 1.0,
        "lateral_midpoint_shift_s": 0.5,
    }[key]
    return key, float(raw / max(scale, 1e-6))


def is_small_edit(edit_type: str | None, normalized_distance: float) -> bool:
    if edit_type is None:
        return False
    return normalized_distance <= 1.000001


def anchor_distance(a: EgoCandidate, b: EgoCandidate) -> float:
    steps = min(len(a.future_states), len(b.future_states))
    disp = np.linalg.norm(a.future_states[:steps, :2] - b.future_states[:steps, :2], axis=-1)
    return float(np.mean(disp) / 10.0 + disp[-1] / 20.0)


def build_neighbor_graph(candidates: list[EgoCandidate], max_neighbors: int = 8) -> list[NeighborEdge]:
    feasible = [c for c in candidates if c.is_feasible]
    per_node: dict[str, list[NeighborEdge]] = defaultdict(list)
    for i, a in enumerate(feasible):
        for b in feasible[i + 1 :]:
            edit_type, edit_dist = edit_difference(a, b)
            if not is_small_edit(edit_type, edit_dist):
                continue
            dist = float(edit_dist + anchor_distance(a, b))
            per_node[a.candidate_id].append(NeighborEdge(a.candidate_id, b.candidate_id, edit_type or "unknown", dist))
            per_node[b.candidate_id].append(NeighborEdge(b.candidate_id, a.candidate_id, edit_type or "unknown", dist))
    edges: list[NeighborEdge] = []
    seen: set[tuple[str, str]] = set()
    for node, node_edges in per_node.items():
        node_edges.sort(key=lambda e: e.normalized_distance)
        for edge in node_edges[:max_neighbors]:
            key = tuple(sorted([edge.source, edge.target]))
            if key in seen:
                continue
            seen.add(key)
            edges.append(edge)
    return edges


def edge_lookup(edges: list[NeighborEdge]) -> dict[str, list[NeighborEdge]]:
    out: dict[str, list[NeighborEdge]] = defaultdict(list)
    for e in edges:
        out[e.source].append(e)
        out[e.target].append(NeighborEdge(e.target, e.source, e.edit_type, e.normalized_distance))
    return dict(out)
