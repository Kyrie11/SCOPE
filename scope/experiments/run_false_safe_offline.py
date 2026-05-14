"""Experiment 3: coercive false-safe offline selection."""
from __future__ import annotations

import argparse

from scope.data.scene_schema import read_groups_jsonl
from scope.utils.config import add_config_args, load_config_from_args
from scope.utils.logging import write_json


def run(config: dict, dataset_file: str) -> dict:
    groups = read_groups_jsonl(dataset_file)
    triples = 0
    false_safe = 0
    fd_rate = 0
    induced_brake = 0
    for g in groups:
        labels = list(g.labels.values())
        coercive = [l for l in labels if l.branch == 0 and l.burden >= config.get("rho0", 2) and not l.safety.get("collision", False)]
        comfortable = [l for l in labels if l.burden <= 1 and not l.safety.get("collision", False)]
        unsafe = [l for l in labels if l.safety.get("collision", False) or l.safety.get("near_collision", False)]
        if coercive and comfortable and unsafe:
            triples += 1
            false_safe += int(len(coercive) > 0)
            fd_rate += int(any(l.diagnostics.get("fd_diagnostic") is True for l in coercive))
            induced_brake += int(any(l.safety.get("induced_hard_brake", False) for l in coercive))
    return {
        "candidate_triples": triples,
        "false_safe_selection_rate_proxy": false_safe / max(triples, 1),
        "FD_rate_proxy": fd_rate / max(triples, 1),
        "induced_hard_braking_proxy": induced_brake / max(triples, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--dataset_file")
    p.add_argument("--output", default="outputs/experiments/false_safe_offline.json")
    args = p.parse_args()
    cfg = load_config_from_args(args)
    metrics = run(cfg, args.dataset_file or cfg.get("dataset_file"))
    write_json(args.output, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
