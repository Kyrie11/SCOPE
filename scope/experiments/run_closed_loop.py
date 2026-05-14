"""Experiment 4: closed-loop non-coercive planning and robustness."""
from __future__ import annotations

import argparse

from scope.planning.closed_loop import run_closed_loop
from scope.utils.config import add_config_args, load_config_from_args
from scope.utils.logging import write_json


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
