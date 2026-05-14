"""Closed-loop SCOPE planning loop skeleton with scene-only operator queries."""
from __future__ import annotations

import argparse
from pathlib import Path

from scope.data.candidates import CandidateGenerator
from scope.data.rollout_waymax import DEFAULT_POLICY_FAMILY, make_rollout_backend
from scope.data.womd_loader import WOMDWaymaxLoader
from scope.planning.selector import PlanningThresholds, lagrangian_fallback, select_mechanism_feasible
from scope.utils.config import add_config_args, load_config_from_args
from scope.utils.logging import write_json


def run_closed_loop(config: dict) -> dict:
    loader = WOMDWaymaxLoader(config.get("data", {}))
    backend = make_rollout_backend(config.get("rollout", {}).get("backend", "reactive"), True)
    generator = CandidateGenerator(config.get("candidates", {}))
    thresholds = PlanningThresholds(**{k: v for k, v in config.get("planning", {}).items() if k in PlanningThresholds.__dataclass_fields__})
    metrics = {"scenes": 0, "route_success": 0, "collisions": 0, "near_collisions": 0, "progress": 0.0, "feasible_selection_rate": 0.0}
    # This loop performs the data-side closed-loop mechanics. Model scoring is
    # delegated to experiment scripts when a checkpoint is available.
    for scene in loader.iter_scenes():
        cands = generator.generate(scene)
        if not cands:
            continue
        policy = DEFAULT_POLICY_FAMILY[0]
        rollout = backend.rollout(scene, cands[0], policy)
        metrics["scenes"] += 1
        metrics["collisions"] += int(any(trace.get("collision", False) for trace in rollout.traces.values()))
        metrics["progress"] += float(cands[0].future_states[-1, 0] - cands[0].future_states[0, 0])
    n = max(metrics["scenes"], 1)
    metrics["progress"] /= n
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--output", default="outputs/experiments/closed_loop.json")
    args = p.parse_args()
    cfg = load_config_from_args(args)
    metrics = run_closed_loop(cfg)
    write_json(args.output, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
