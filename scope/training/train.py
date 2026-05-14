"""Train SCOPE or internal baselines."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scope.data.dataset import SameRootDataset, collate_same_root_groups
from scope.models.baselines import EgoCondResponse, EgoCondTraj, switches_from_config
from scope.models.scope_model import SCOPEModel
from scope.training.losses import mechanism_loss, total_scope_loss
from scope.training.support_query import SupportQuerySampler
from scope.utils.config import add_config_args, ensure_dir, load_config_from_args
from scope.utils.random import seed_all


def build_model(cfg: dict):
    name = cfg.get("model", {}).get("name", "scope")
    model_cfg = cfg.get("model", {})
    future_steps = int(round(cfg.get("data", {}).get("future_horizon_s", 8.0) / cfg.get("data", {}).get("dt", 0.1)))
    if name == "egocond_traj":
        return EgoCondTraj(model_cfg.get("hidden_dim", 256), future_steps, model_cfg.get("trajectory_modes", 6))
    if name == "egocond_response":
        return EgoCondResponse(model_cfg.get("hidden_dim", 256), future_steps, model_cfg.get("trajectory_modes", 6))
    mcfg = {
        "hidden_dim": model_cfg.get("hidden_dim", 256),
        "intervention_dim": model_cfg.get("intervention_dim", 128),
        "mechanism_tokens": model_cfg.get("mechanism_tokens", 16),
        "support_updater_blocks": model_cfg.get("support_updater_blocks", 3),
        "trajectory_modes": model_cfg.get("trajectory_modes", 6),
        "future_steps": future_steps,
        "dropout": model_cfg.get("dropout", 0.1),
        "use_support_query": model_cfg.get("use_support_query", True),
        "use_operator_tokens": model_cfg.get("use_operator_tokens", True),
    }
    return SCOPEModel(mcfg)


def train(cfg: dict, data_dir: str) -> Path:
    seed_all(cfg.get("seed", 17))
    torch.set_num_threads(int(cfg.get("training", {}).get("torch_num_threads", min(8, os.cpu_count() or 1))))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("training", {}).get("cuda", True) else "cpu")
    dataset = SameRootDataset(data_dir, cfg.get("data", {}).get("split", "train"), cfg.get("training", {}).get("max_groups"))
    loader = DataLoader(dataset, batch_size=cfg.get("training", {}).get("batch_same_root_groups", 32), shuffle=True, collate_fn=collate_same_root_groups)
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("training", {}).get("lr", 2e-4), weight_decay=cfg.get("training", {}).get("weight_decay", 1e-4))
    sampler = SupportQuerySampler(cfg.get("training", {}))
    switches = switches_from_config(cfg)
    max_epochs = int(cfg.get("training", {}).get("max_epochs", 30))
    max_steps = int(cfg.get("training", {}).get("max_steps", 200000))
    step = 0
    model.train()
    for _epoch in range(max_epochs):
        for batch in tqdm(loader, desc="train"):
            batch = _to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            if isinstance(model, SCOPEModel):
                support, query_batch = sampler.split_batch(batch)
                scene_out = model(query_batch, None, "scene_only")
                support_out = None
                if switches.use_support_query and support is not None and switches.use_support_adapted_loss:
                    support = _to_device(support, device)
                    support_out = model(query_batch, support, "support_adapted")
                losses = total_scope_loss(scene_out, query_batch, support_out, scene_out if switches.use_distillation else None, cfg.get("loss", {}))
                loss = losses["total"]
            else:
                out = model(batch)
                if "branch_logits" in out:
                    loss = mechanism_loss(out, batch, cfg.get("loss", {}))["total"]
                else:
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["safety_logits"], batch["safety"].float()).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("training", {}).get("grad_clip_norm", 5.0))
            opt.step()
            step += 1
            if step >= max_steps:
                break
        if step >= max_steps:
            break
    out_dir = ensure_dir(cfg.get("output", {}).get("checkpoint_dir", "outputs/checkpoints"))
    path = out_dir / f"{cfg.get('model', {}).get('name', 'scope')}.pt"
    torch.save({"model": model.state_dict(), "config": cfg}, path)
    return path


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return obj
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    cfg = load_config_from_args(args)
    ckpt = train(cfg, args.data)
    print(ckpt)


if __name__ == "__main__":
    main()
