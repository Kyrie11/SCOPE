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
    for path in paths:
        groups = read_groups_jsonl(path)
        split = Path(path).name.split("_")[0]
        for g in groups:
            candidates += len(g.candidate_set)
            duplicate += sum(1 for c in g.candidate_set if c.feasibility.get("duplicate_of") is not None)
            tags.update(g.root_scene.scenario_tags)
            total_rollouts += len(g.masks)
            for mask in g.masks.values():
                valid_rollouts += int(mask.valid_rollout)
                fd_valid += int(mask.fd_diag_valid)
                boundary_valid += int(mask.boundary_valid)
            for label in g.labels.values():
                branch[BRANCHES[label.branch]] += 1
                burden[label.burden] += 1
                fd = label.diagnostics.get("fd_diagnostic")
                if fd is True:
                    fd_pos += 1
                for item in label.diagnostics.get("boundary_positive_edges", []):
                    boundary_pos += int(bool(item))
            rows.append({"split": split, "scene_id": g.scene_id, "candidates": len(g.candidate_set), "valid_rollouts": valid_rollouts, "labels": len(g.labels)})
    scenes = len({r["scene_id"] for r in rows})
    groups_n = len(rows)
    summary = {
        "scenes": scenes,
        "root_groups": groups_n,
        "candidates_per_group": candidates / max(groups_n, 1),
        "valid_rollouts": valid_rollouts,
        "simulator_failure_rate": 1.0 - valid_rollouts / max(total_rollouts, 1),
        "branch_distribution": dict(branch),
        "burden_distribution": {str(k): v for k, v in burden.items()},
        "scenario_type_counts": dict(tags),
        "fd_positive_rate": fd_pos / max(fd_valid, 1),
        "boundary_pair_positive_rate": boundary_pos / max(boundary_valid, 1),
        "duplicate_candidate_rate": duplicate / max(candidates, 1),
        "calibration_set_coverage": sum(1 for r in rows if r["split"] in {"val", "validation"}) / max(groups_n, 1),
    }
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
