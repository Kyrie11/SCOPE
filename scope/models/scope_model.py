"""Full SCOPE model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from scope.models.heads import ResponseHeads
from scope.models.intervention_encoder import InterventionEncoder
from scope.models.operator import OperatorQuery, ResponseMechanismOperator, SupportEncoder
from scope.models.scene_encoder import SceneEncoder


@dataclass
class SCOPEModelConfig:
    scene_feature_dim: int = 16
    hidden_dim: int = 256
    intervention_dim: int = 128
    mechanism_tokens: int = 16
    support_updater_blocks: int = 3
    num_attention_heads: int = 8
    dropout: float = 0.1
    trajectory_modes: int = 6
    future_steps: int = 80
    traj_state_dim: int = 5
    use_support_query: bool = True
    use_operator_tokens: bool = True
    use_structured_intervention: bool = True
    use_contextual_residual: bool = True


class SCOPEModel(nn.Module):
    def __init__(self, cfg: SCOPEModelConfig | dict[str, Any] | None = None):
        super().__init__()
        if cfg is None:
            cfg = SCOPEModelConfig()
        self.cfg = SCOPEModelConfig(**cfg) if isinstance(cfg, dict) else cfg
        self.scene_encoder = SceneEncoder(self.cfg.scene_feature_dim, self.cfg.hidden_dim, 4, self.cfg.num_attention_heads, self.cfg.dropout)
        self.intervention_encoder = InterventionEncoder(8, 7, 7, self.cfg.hidden_dim, self.cfg.intervention_dim, self.cfg.num_attention_heads, self.cfg.dropout)
        self.operator = ResponseMechanismOperator(self.cfg.hidden_dim, self.cfg.mechanism_tokens, self.cfg.support_updater_blocks, self.cfg.num_attention_heads, self.cfg.dropout)
        self.support_encoder = SupportEncoder(self.cfg.intervention_dim, self.cfg.hidden_dim, dropout=self.cfg.dropout)
        self.query = OperatorQuery(self.cfg.intervention_dim, self.cfg.hidden_dim, self.cfg.num_attention_heads, self.cfg.dropout)
        self.heads = ResponseHeads(self.cfg.hidden_dim, 5, self.cfg.trajectory_modes, self.cfg.future_steps, self.cfg.traj_state_dim, self.cfg.dropout)

    def encode_scene(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.scene_encoder(batch)

    def encode_interventions(self, batch: dict[str, torch.Tensor], scene_tokens: torch.Tensor, prefix: str = "query") -> torch.Tensor:
        return self.intervention_encoder(batch[f"{prefix}_ctrl"], batch[f"{prefix}_anchors"], batch[f"{prefix}_future_tokens"], scene_tokens)

    def forward(self, batch: dict[str, torch.Tensor], support: dict[str, torch.Tensor] | None = None, mode: str = "scene_only") -> dict[str, Any]:
        scene_tokens = self.encode_scene(batch)
        query_u = self.encode_interventions(batch, scene_tokens, "query")
        omega_empty = self.operator.scene_only(scene_tokens)
        omega = omega_empty
        support_tokens = None
        if mode == "support_adapted" and support is not None:
            support_u = self.intervention_encoder(support["ctrl"], support["anchors"], support["future_tokens"], scene_tokens)
            support_tokens = self.support_encoder(support_u, support["labels"], support["masks"])
            omega = self.operator.adapt(scene_tokens, support_tokens)
        r = self.query(query_u, scene_tokens, omega)
        outputs = self.heads(r)
        outputs.update({"scene_tokens": scene_tokens, "query_u": query_u, "omega": omega, "omega_empty": omega_empty, "support_tokens": support_tokens})
        return outputs

    @torch.no_grad()
    def predict_scene_only(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        self.eval()
        return self.forward(batch, None, mode="scene_only")
