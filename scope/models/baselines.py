"""Internal baselines and ablation model adapters."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from scope.models.heads import ResponseHeads
from scope.models.scene_encoder import MLP, SceneEncoder


class EgoCondTraj(nn.Module):
    """Candidate-wise trajectory and unconditional risk predictor without branch/burden/operator."""

    def __init__(self, hidden_dim: int = 256, future_steps: int = 80, modes: int = 6):
        super().__init__()
        self.scene = SceneEncoder(hidden_dim=hidden_dim)
        self.u = MLP(8 + 7, hidden_dim, hidden_dim, 3, 0.1)
        self.traj = MLP(hidden_dim, hidden_dim, modes * future_steps * 5, 2, 0.1)
        self.risk = MLP(hidden_dim, hidden_dim, 4, 2, 0.1)
        self.modes = modes
        self.future_steps = future_steps

    def forward(self, batch):
        ctx = self.scene(batch).mean(dim=1).unsqueeze(1)
        x = torch.cat([batch["query_ctrl"], batch["query_anchors"]], dim=-1)
        h = self.u(x) + ctx
        b, q, _ = h.shape
        return {"trajectory_loc": self.traj(h).reshape(b, q, self.modes, self.future_steps, 5), "safety_logits": self.risk(h)}


class EgoCondResponse(nn.Module):
    """Candidate-wise branch/burden/traj/safety predictor with no shared same-root operator."""

    def __init__(self, hidden_dim: int = 256, future_steps: int = 80, modes: int = 6):
        super().__init__()
        self.scene = SceneEncoder(hidden_dim=hidden_dim)
        self.u = MLP(8 + 7, hidden_dim, hidden_dim, 3, 0.1)
        self.heads = ResponseHeads(hidden_dim, 5, modes, future_steps, 5, 0.1)

    def forward(self, batch):
        ctx = self.scene(batch).mean(dim=1).unsqueeze(1)
        x = torch.cat([batch["query_ctrl"], batch["query_anchors"]], dim=-1)
        return self.heads(self.u(x) + ctx)


@dataclass
class AblationSwitches:
    use_support_query: bool = True
    use_scene_only_loss: bool = True
    use_support_adapted_loss: bool = True
    use_distillation: bool = True
    use_manifold_loss: bool = True
    use_forced_dependence: bool = True
    use_boundary_constraint: bool = True
    risk_pressure_penalty_only: bool = False


def switches_from_config(cfg: dict) -> AblationSwitches:
    model = cfg.get("model", {})
    planning = cfg.get("planning", {})
    baseline = cfg.get("baseline", {})
    return AblationSwitches(
        use_support_query=model.get("use_support_query", True),
        use_scene_only_loss=model.get("use_scene_only_loss", True),
        use_support_adapted_loss=model.get("use_support_adapted_loss", True),
        use_distillation=model.get("use_distillation", True),
        use_manifold_loss=model.get("use_manifold_loss", True),
        use_forced_dependence=planning.get("use_forced_dependence", True),
        use_boundary_constraint=planning.get("use_boundary_constraint", True),
        risk_pressure_penalty_only=baseline.get("risk_pressure_penalty_only", False),
    )
