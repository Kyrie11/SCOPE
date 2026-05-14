"""Experiment 1: same-root held-out response prediction."""
from __future__ import annotations

import argparse

from scope.training.evaluate_response import evaluate
from scope.utils.config import add_config_args, load_config_from_args
from scope.utils.logging import write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--checkpoint", required=False)
    p.add_argument("--data", required=False)
    p.add_argument("--output", default="outputs/experiments/heldout_response.json")
    args = p.parse_args()
    cfg = load_config_from_args(args)
    checkpoint = args.checkpoint or cfg.get("checkpoint")
    data = args.data or cfg.get("data_dir")
    if not checkpoint or not data:
        raise ValueError("checkpoint and data_dir/--data are required")
    metrics = evaluate(checkpoint, data, cfg.get("split", "val"))
    write_json(args.output, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
