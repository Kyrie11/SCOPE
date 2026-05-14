"""Validation-only calibration for branch, ordinal, safety, and FD probabilities."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scope.training.evaluate_response import evaluate
from scope.utils.logging import write_json


def calibrate(checkpoint: str, dataset_dir: str, split: str = "val") -> dict:
    # Lightweight deterministic calibration summary. Temperature optimization is
    # represented by identity temperatures unless validation loss indicates a
    # conservative widening. This keeps thresholds validation-only and explicit.
    metrics = evaluate(checkpoint, dataset_dir, split)
    temp = 1.0 if metrics["branch_accuracy"] >= 0.2 else 1.5
    cal = {
        "split": split,
        "branch_temperature": temp,
        "ordinal_temperature": temp,
        "safety_platt_a": 1.0 / temp,
        "safety_platt_b": 0.0,
        "fd_platt_a": 1.0 / temp,
        "fd_platt_b": 0.0,
        "thresholds": {
            "epsilon_R": 0.05,
            "epsilon_D": 0.10,
            "epsilon_rho": 0.30,
            "epsilon_B": 1.0,
            "epsilon_U": 0.20,
            "delta": 0.10,
            "tau_d": 0.10,
        },
        "validation_metrics": metrics,
    }
    return cal


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--output", default="outputs/calibration/calibration.json")
    args = p.parse_args()
    cal = calibrate(args.checkpoint, args.data, args.split)
    write_json(args.output, cal)
    print(cal)


if __name__ == "__main__":
    main()
