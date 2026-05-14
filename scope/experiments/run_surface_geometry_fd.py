"""Experiment 2: surface geometry, boundary, and forced-dependence diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scope.data.dataset import SameRootDataset, collate_same_root_groups
from scope.models.scope_model import SCOPEModel
from scope.planning.estimators import forced_dependence, marginal_collision_risk
from scope.utils.config import add_config_args, load_config_from_args
from scope.utils.logging import write_json
from scope.utils.metrics import auroc


def run(config: dict, checkpoint: str, data_dir: str) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = SCOPEModel({
        "hidden_dim": ckpt["config"].get("model", {}).get("hidden_dim", 256),
        "intervention_dim": ckpt["config"].get("model", {}).get("intervention_dim", 128),
        "mechanism_tokens": ckpt["config"].get("model", {}).get("mechanism_tokens", 16),
        "future_steps": int(round(ckpt["config"].get("data", {}).get("future_horizon_s", 8.0) / ckpt["config"].get("data", {}).get("dt", 0.1))),
    })
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    ds = SameRootDataset(data_dir, config.get("split", "val"))
    loader = DataLoader(ds, batch_size=config.get("batch_size", 8), collate_fn=collate_same_root_groups)
    fd_scores, fd_labels, risk_scores, collision_labels = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model.predict_scene_only(batch)
            fd = forced_dependence(out, config.get("rho0", 2), config.get("delta", 0.1), config.get("tau_d", 0.1))
            risk = marginal_collision_risk(out)
            mask = batch["mask"] > 0.5
            fd_scores.extend(fd[mask].cpu().numpy().tolist())
            risk_scores.extend(risk[mask].cpu().numpy().tolist())
            labels = ((batch["branch"] == 0) & (batch["burden"] >= config.get("rho0", 2)) & (batch["safety"][..., 0] < 0.5)).float()
            fd_labels.extend(labels[mask].cpu().numpy().astype(int).tolist())
            collision_labels.extend(batch["safety"][..., 0][mask].cpu().numpy().astype(int).tolist())
    return {"FD_AUROC": auroc(fd_scores, fd_labels), "risk_AUROC": auroc(risk_scores, collision_labels), "examples": len(fd_scores)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--checkpoint")
    p.add_argument("--data")
    p.add_argument("--output", default="outputs/experiments/surface_geometry_fd.json")
    args = p.parse_args()
    cfg = load_config_from_args(args)
    metrics = run(cfg, args.checkpoint or cfg.get("checkpoint"), args.data or cfg.get("data_dir"))
    write_json(args.output, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
