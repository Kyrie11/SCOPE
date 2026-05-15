"""Dataset summary diagnostics for same-root SCOPE groups."""
from __future__ import annotations

import argparse
import glob
from collections import Counter
from pathlib import Path

import pandas as pd

from scope.data.scene_schema import BRANCHES, read_groups_jsonl
from scope.utils.logging import write_json


def summarize_dataset(dataset_dir: str | Path) -> dict:
    paths = sorted(glob.glob(str(Path(dataset_dir) / "*_groups.jsonl*")))
    rows = []
    branch = Counter()
    burden = Counter()
    tags = Counter()
    valid_rollouts = 0
    total_rollouts = 0
    fd_pos = 0
    fd_valid = 0
    boundary_pos = 0
    boundary_valid = 0
    duplicate = 0
    candidates = 0
    route_inferred = 0
    route_total = 0
    sdc_paths_count = []
    empty_relevant = 0
    empty_neighbor_edges = 0
    empty_labels = 0
    candidate_shape_bad = 0
    agent_future_shape_bad = 0
    groups_with_fd_valid = 0
    groups_with_boundary_valid = 0
    for path in paths:
        groups = read_groups_jsonl(path)
        split = Path(path).name.split("_")[0]
        for g in groups:
            route_total += 1
            route_inferred += int(g.root_scene.route_context.inferred)
            sdc_paths_count.append(len(g.root_scene.route_context.sdc_paths))

            empty_relevant += int(len(g.root_scene.relevant_agent_indices) == 0)
            empty_neighbor_edges += int(len(g.neighbor_edges) == 0)
            empty_labels += int(len(g.labels) == 0)

            expected_steps = int(round(g.root_scene.future_horizon_s / g.root_scene.dt))

            for c in g.candidate_set:
                if c.future_states.shape[0] != expected_steps:
                    candidate_shape_bad += 1

            for label in g.labels.values():
                if label.agent_future.shape[0] != expected_steps:
                    agent_future_shape_bad += 1

            groups_with_fd_valid += int(any(m.fd_diag_valid for m in g.masks.values()))
            groups_with_boundary_valid += int(any(m.boundary_valid for m in g.masks.values()))
            rows.append({"split": split, "scene_id": g.scene_id, "candidates": len(g.candidate_set), "valid_rollouts": valid_rollouts, "labels": len(g.labels)})
    groups_n = len(rows)
    summary.update({
        "route_inferred_rate": route_inferred / max(route_total, 1),
        "sdc_paths_per_group_mean": float(np.mean(sdc_paths_count)) if sdc_paths_count else 0.0,
        "empty_relevant_agent_rate": empty_relevant / max(groups_n, 1),
        "empty_neighbor_edge_rate": empty_neighbor_edges / max(groups_n, 1),
        "empty_label_group_rate": empty_labels / max(groups_n, 1),
        "candidate_shape_bad": candidate_shape_bad,
        "agent_future_shape_bad": agent_future_shape_bad,
        "groups_with_fd_valid_rate": groups_with_fd_valid / max(groups_n, 1),
        "groups_with_boundary_valid_rate": groups_with_boundary_valid / max(groups_n, 1),
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    args = parser.parse_args()
    summary = summarize_dataset(args.dataset_dir)
    print(pd.Series(summary).to_string())
    write_json(Path(args.dataset_dir) / "diagnostics.json", summary)


if __name__ == "__main__":
    main()
