"""Evaluate held-out response prediction metrics."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scope.data.dataset import SameRootDataset, collate_same_root_groups
from scope.models.scope_model import SCOPEModel
from scope.training.losses import mechanism_loss
from scope.utils.logging import write_json


def evaluate(checkpoint: str, dataset_dir: str, split: str = "val") -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model = SCOPEModel({
        "hidden_dim": cfg.get("model", {}).get("hidden_dim", 256),
        "intervention_dim": cfg.get("model", {}).get("intervention_dim", 128),
        "mechanism_tokens": cfg.get("model", {}).get("mechanism_tokens", 16),
        "trajectory_modes": cfg.get("model", {}).get("trajectory_modes", 6),
        "future_steps": int(round(cfg.get("data", {}).get("future_horizon_s", 8.0) / cfg.get("data", {}).get("dt", 0.1))),
    })
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    ds = SameRootDataset(dataset_dir, split)
    loader = DataLoader(ds, batch_size=cfg.get("training", {}).get("batch_same_root_groups", 16), collate_fn=collate_same_root_groups)
    totals = []
    correct = 0
    count = 0
    with torch.no_grad():
        for batch in loader:
            out = model.predict_scene_only(batch)
            losses = mechanism_loss(out, batch, cfg.get("loss", {}))
            totals.append(float(losses["total"]))
            pred = out["branch_logits"].argmax(dim=-1)
            mask = batch["mask"] > 0.5
            correct += int((pred[mask] == batch["branch"][mask]).sum())
            count += int(mask.sum())
    return {"response_nll_proxy": float(np.mean(totals)) if totals else float("nan"), "branch_accuracy": correct / max(count, 1), "examples": count}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--output", default="outputs/eval/heldout_response.json")
    args = p.parse_args()
    metrics = evaluate(args.checkpoint, args.dataset_dir, args.split)
    print(metrics)
    write_json(args.output, metrics)


if __name__ == "__main__":
    main()
