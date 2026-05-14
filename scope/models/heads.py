"""Response heads: branch, ordinal burden, branch-conditioned trajectories, safety."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from scope.models.scene_encoder import MLP


class BranchHead(nn.Module):
    def __init__(self, hidden_dim: int = 256, branches: int = 5, dropout: float = 0.1):
        super().__init__()
        self.net = MLP(hidden_dim, hidden_dim, branches, 2, dropout)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r)


class OrdinalBurdenHead(nn.Module):
    def __init__(self, hidden_dim: int = 256, branches: int = 5, dropout: float = 0.1):
        super().__init__()
        self.branches = branches
        self.score = MLP(hidden_dim, hidden_dim, branches * 3, 2, dropout)
        self.raw_thresholds = nn.Parameter(torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32))

    def thresholds(self) -> torch.Tensor:
        diffs = F.softplus(self.raw_thresholds)
        return torch.cumsum(diffs, dim=0) - diffs.mean()

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        b, q, _ = r.shape
        g = self.score(r).reshape(b, q, self.branches, 3)
        return g - self.thresholds().view(1, 1, 1, 3)

    @staticmethod
    def ge_logits_to_pmf(ge_logits: torch.Tensor) -> torch.Tensor:
        p_ge = torch.sigmoid(ge_logits)
        p0 = 1.0 - p_ge[..., 0]
        p1 = p_ge[..., 0] - p_ge[..., 1]
        p2 = p_ge[..., 1] - p_ge[..., 2]
        p3 = p_ge[..., 2]
        pmf = torch.stack([p0, p1, p2, p3], dim=-1)
        return torch.clamp(pmf, min=1e-6) / torch.clamp(pmf.sum(dim=-1, keepdim=True), min=1e-6)


class TrajectoryMixtureHead(nn.Module):
    def __init__(self, hidden_dim: int = 256, branches: int = 5, modes: int = 6, future_steps: int = 80, state_dim: int = 5, dropout: float = 0.1):
        super().__init__()
        self.branches = branches
        self.modes = modes
        self.future_steps = future_steps
        self.state_dim = state_dim
        self.branch_emb = nn.Embedding(branches, hidden_dim)
        self.logits = MLP(hidden_dim, hidden_dim, branches * modes, 2, dropout)
        self.loc = MLP(hidden_dim, hidden_dim, branches * modes * future_steps * state_dim, 2, dropout)
        self.log_scale = MLP(hidden_dim, hidden_dim, branches * modes * future_steps * state_dim, 2, dropout)

    def forward(self, r: torch.Tensor) -> dict[str, torch.Tensor]:
        b, q, _ = r.shape
        logits = self.logits(r).reshape(b, q, self.branches, self.modes)
        loc = self.loc(r).reshape(b, q, self.branches, self.modes, self.future_steps, self.state_dim)
        log_scale = torch.clamp(self.log_scale(r).reshape_as(loc), -5.0, 3.0)
        return {"mode_logits": logits, "loc": loc, "log_scale": log_scale}


class SafetyHead(nn.Module):
    def __init__(self, hidden_dim: int = 256, branches: int = 5, events: int = 4, dropout: float = 0.1):
        super().__init__()
        self.branches = branches
        self.events = events
        self.net = MLP(hidden_dim, hidden_dim, branches * events, 2, dropout)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        b, q, _ = r.shape
        return self.net(r).reshape(b, q, self.branches, self.events)


class ResponseHeads(nn.Module):
    def __init__(self, hidden_dim: int = 256, branches: int = 5, modes: int = 6, future_steps: int = 80, traj_state_dim: int = 5, dropout: float = 0.1):
        super().__init__()
        self.branch = BranchHead(hidden_dim, branches, dropout)
        self.burden = OrdinalBurdenHead(hidden_dim, branches, dropout)
        self.trajectory = TrajectoryMixtureHead(hidden_dim, branches, modes, future_steps, traj_state_dim, dropout)
        self.safety = SafetyHead(hidden_dim, branches, 4, dropout)

    def forward(self, r: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        return {
            "branch_logits": self.branch(r),
            "burden_ge_logits": self.burden(r),
            "burden_pmf": OrdinalBurdenHead.ge_logits_to_pmf(self.burden(r)),
            "trajectory": self.trajectory(r),
            "safety_logits": self.safety(r),
        }


def branch_conditioned_collision_risk(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.sigmoid(outputs["safety_logits"])[..., 0]
