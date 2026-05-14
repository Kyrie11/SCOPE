"""PyTorch dataloader preserving same-root group structure."""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from scope.data.labels import compute_physical_anchors
from scope.data.scene_schema import BRANCH_TO_INDEX, SAFETY_EVENTS, SameRootGroup, read_groups_jsonl


class SameRootDataset(Dataset):
    def __init__(self, dataset_dir: str | Path, split: str = "train", max_groups: int | None = None):
        paths = sorted(glob.glob(str(Path(dataset_dir) / f"{split}_groups.jsonl*")))
        if not paths:
            raise FileNotFoundError(f"No {split}_groups.jsonl* files under {dataset_dir}")
        groups: list[SameRootGroup] = []
        for path in paths:
            groups.extend(read_groups_jsonl(path))
        self.groups = groups[:max_groups] if max_groups else groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> SameRootGroup:
        return self.groups[idx]


def group_to_training_examples(group: SameRootGroup) -> dict[str, Any]:
    scene = group.root_scene
    examples = []
    for cand in group.candidate_set:
        for agent_idx in scene.relevant_agent_indices:
            for policy_id in {k[1] for k in group.rollout_matrix.keys()}:
                key = (cand.candidate_id, agent_idx, policy_id)
                label = group.labels.get(key)
                mask = group.masks.get(key)
                if label is None or mask is None or not mask.valid_rollout:
                    continue
                anchors = compute_physical_anchors(scene, cand, agent_idx, label.agent_future).vector()
                ctrl = control_vector(cand.control_edits)
                safety = np.array([float(label.safety[e]) for e in SAFETY_EVENTS], dtype=np.float32)
                examples.append({
                    "candidate_id": cand.candidate_id,
                    "agent_index": agent_idx,
                    "policy_id": policy_id,
                    "ctrl": ctrl,
                    "anchors": anchors,
                    "ego_future": cand.future_states[:, :7].astype(np.float32),
                    "agent_future": label.agent_future[:, :7].astype(np.float32),
                    "branch": label.branch,
                    "burden": label.burden,
                    "safety": safety,
                    "neighbor_ids": [e.target for e in group.neighbor_edges if e.source == cand.candidate_id] + [e.source for e in group.neighbor_edges if e.target == cand.candidate_id],
                })
    return {"group": group, "examples": examples}


def control_vector(edits: dict[str, Any]) -> np.ndarray:
    gap = edits.get("target_gap_id")
    gap_code = {None: 0.0, "current": 0.0, "earlier": -1.0, "later": 1.0}.get(gap, 0.0)
    return np.array([
        float(edits.get("delta_t", 0.0)),
        float(edits.get("speed_multiplier", 1.0)),
        gap_code,
        0.0 if edits.get("target_lane_id") is None else float(hash(edits.get("target_lane_id")) % 997) / 997.0,
        float(edits.get("longitudinal_buffer_m", 0.0)),
        float(edits.get("time_headway_s", 1.5)),
        float(edits.get("lane_entry_duration_s", 3.0)),
        float(edits.get("lateral_midpoint_shift_s", 0.0)),
    ], dtype=np.float32)


def collate_same_root_groups(groups: list[SameRootGroup]) -> dict[str, Any]:
    converted = [group_to_training_examples(g) for g in groups]
    batch_examples = [ex for item in converted for ex in item["examples"]]
    if not batch_examples:
        raise ValueError("Batch contains no valid examples")
    max_q = max(len(item["examples"]) for item in converted)
    b = len(converted)
    t = batch_examples[0]["ego_future"].shape[0]
    ctrl = np.zeros((b, max_q, 8), dtype=np.float32)
    anchors = np.zeros((b, max_q, 7), dtype=np.float32)
    ego_future = np.zeros((b, max_q, t, 7), dtype=np.float32)
    branch = np.zeros((b, max_q), dtype=np.int64)
    burden = np.zeros((b, max_q), dtype=np.int64)
    safety = np.zeros((b, max_q, len(SAFETY_EVENTS)), dtype=np.float32)
    traj = np.zeros((b, max_q, t, 7), dtype=np.float32)
    mask = np.zeros((b, max_q), dtype=np.float32)
    scene_features = np.zeros((b, 16), dtype=np.float32)
    for bi, item in enumerate(converted):
        g = item["group"]
        scene_features[bi] = scene_summary_features(g)
        for qi, ex in enumerate(item["examples"]):
            ctrl[bi, qi] = ex["ctrl"]
            anchors[bi, qi] = ex["anchors"]
            ego_future[bi, qi] = ex["ego_future"]
            branch[bi, qi] = ex["branch"]
            burden[bi, qi] = ex["burden"]
            safety[bi, qi] = ex["safety"]
            traj[bi, qi] = ex["agent_future"]
            mask[bi, qi] = 1.0
    return {
        "groups": groups,
        "scene_features": torch.from_numpy(scene_features),
        "query_ctrl": torch.from_numpy(ctrl),
        "query_anchors": torch.from_numpy(anchors),
        "query_future_tokens": torch.from_numpy(ego_future),
        "branch": torch.from_numpy(branch),
        "burden": torch.from_numpy(burden),
        "safety": torch.from_numpy(safety),
        "trajectory": torch.from_numpy(traj),
        "mask": torch.from_numpy(mask),
    }


def scene_summary_features(group: SameRootGroup) -> np.ndarray:
    scene = group.root_scene
    ego = scene.root_state(scene.ego_track_index)
    rel_states = [scene.root_state(i) for i in scene.relevant_agent_indices]
    feats = np.zeros(16, dtype=np.float32)
    feats[0:4] = [ego[3], ego[4], ego[5], ego[6]]
    if rel_states:
        rel = np.stack([s[:2] - ego[:2] for s in rel_states])
        dist = np.linalg.norm(rel, axis=-1)
        feats[4:8] = [float(np.min(dist)), float(np.mean(dist)), float(len(rel_states)), float(len(group.candidate_set))]
    feats[8] = float("dense_gap" in scene.scenario_tags)
    feats[9] = float("conflict_zone_overlap" in scene.scenario_tags)
    feats[10] = float("close_following" in scene.scenario_tags)
    feats[11] = scene.history_horizon_s
    feats[12] = scene.future_horizon_s
    feats[13] = scene.dt
    feats[14] = len(scene.map_features.features)
    feats[15] = len(group.neighbor_edges)
    return feats
