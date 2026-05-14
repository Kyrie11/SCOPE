"""Structured intervention encoder z_ctrl, z_rel, z_ctx."""
from __future__ import annotations

import torch
from torch import nn

from scope.models.scene_encoder import MLP


class InterventionEncoder(nn.Module):
    def __init__(self, ctrl_dim: int = 8, anchor_dim: int = 7, future_dim: int = 7, hidden_dim: int = 256, intervention_dim: int = 128, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.ctrl_mlp = MLP(ctrl_dim, hidden_dim, hidden_dim, 2, dropout)
        self.anchor_mlp = MLP(anchor_dim, hidden_dim, hidden_dim, 2, dropout)
        self.future_mlp = MLP(future_dim, hidden_dim, hidden_dim, 2, dropout)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, intervention_dim), nn.LayerNorm(intervention_dim))
        self.anchor_mean = nn.Parameter(torch.zeros(anchor_dim), requires_grad=False)
        self.anchor_scale = nn.Parameter(torch.ones(anchor_dim), requires_grad=False)

    def set_anchor_stats(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        self.anchor_mean.data.copy_(mean.to(self.anchor_mean))
        self.anchor_scale.data.copy_(torch.clamp(scale.to(self.anchor_scale), min=1e-3))

    def forward(self, ctrl: torch.Tensor, anchors: torch.Tensor, future_tokens: torch.Tensor, scene_tokens: torch.Tensor) -> torch.Tensor:
        b, q, _ = ctrl.shape
        z_ctrl = self.ctrl_mlp(ctrl.float())
        norm_anchors = (anchors.float() - self.anchor_mean.view(1, 1, -1)) / self.anchor_scale.view(1, 1, -1)
        z_rel = self.anchor_mlp(torch.clamp(norm_anchors, -10.0, 10.0))
        fut = self.future_mlp(future_tokens.float()).mean(dim=2)
        query = fut.reshape(b * q, 1, -1)
        context = scene_tokens[:, None, :, :].expand(b, q, scene_tokens.shape[1], scene_tokens.shape[2]).reshape(b * q, scene_tokens.shape[1], scene_tokens.shape[2])
        z_ctx, _ = self.cross_attn(query, context, context, need_weights=False)
        z_ctx = z_ctx.reshape(b, q, -1)
        return self.out(torch.cat([z_ctrl, z_rel, z_ctx], dim=-1))
