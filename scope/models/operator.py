"""Response-mechanism operator with scene-only and support-adapted modes."""
from __future__ import annotations

import torch
from torch import nn

from scope.models.scene_encoder import MLP


class SupportEncoder(nn.Module):
    def __init__(self, intervention_dim: int, hidden_dim: int = 256, branches: int = 5, safety_events: int = 4, dropout: float = 0.1):
        super().__init__()
        self.branch_emb = nn.Embedding(branches + 1, hidden_dim)
        self.burden_emb = nn.Embedding(5, hidden_dim)
        self.type_emb = nn.Embedding(8, hidden_dim)
        self.mask_mlp = MLP(8, hidden_dim, hidden_dim, 2, dropout)
        self.u_mlp = MLP(intervention_dim, hidden_dim, hidden_dim, 2, dropout)
        self.safety_mlp = MLP(safety_events, hidden_dim, hidden_dim, 2, dropout)
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim * 4), nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

    def forward(self, support_u: torch.Tensor, support_labels: dict[str, torch.Tensor], support_masks: dict[str, torch.Tensor]) -> torch.Tensor:
        branch = support_labels.get("branch").long().clamp_min(0)
        burden = support_labels.get("burden").long().clamp(0, 3)
        safety = support_labels.get("safety").float()
        valid = support_masks.get("valid", torch.ones_like(branch, dtype=torch.float32)).float()
        branch = torch.where(valid > 0.5, branch, torch.full_like(branch, 5))
        burden = torch.where(valid > 0.5, burden, torch.full_like(burden, 4))
        mask_vec = torch.stack([
            valid,
            support_masks.get("branch", valid).float(),
            support_masks.get("burden", valid).float(),
            support_masks.get("trajectory", valid).float(),
            support_masks.get("safety", valid).float(),
            support_masks.get("fd_diag", torch.zeros_like(valid)).float(),
            support_masks.get("boundary", torch.zeros_like(valid)).float(),
            torch.ones_like(valid),
        ], dim=-1)
        return self.out(torch.cat([self.u_mlp(support_u), self.branch_emb(branch), self.burden_emb(burden) + self.safety_mlp(safety), self.mask_mlp(mask_vec)], dim=-1))


class SupportUpdaterBlock(nn.Module):
    def __init__(self, hidden_dim: int = 256, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.scene_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.support_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.mlp = MLP(hidden_dim, hidden_dim * 4, hidden_dim, 2, dropout)
        self.n1 = nn.LayerNorm(hidden_dim)
        self.n2 = nn.LayerNorm(hidden_dim)
        self.n3 = nn.LayerNorm(hidden_dim)
        self.n4 = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor, scene_tokens: torch.Tensor, support_tokens: torch.Tensor | None = None) -> torch.Tensor:
        x = tokens + self.self_attn(self.n1(tokens), self.n1(tokens), self.n1(tokens), need_weights=False)[0]
        x = x + self.scene_attn(self.n2(x), scene_tokens, scene_tokens, need_weights=False)[0]
        if support_tokens is not None and support_tokens.shape[1] > 0:
            x = x + self.support_attn(self.n3(x), support_tokens, support_tokens, need_weights=False)[0]
        x = x + self.mlp(self.n4(x))
        return x


class ResponseMechanismOperator(nn.Module):
    def __init__(self, hidden_dim: int = 256, mechanism_tokens: int = 16, support_blocks: int = 3, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mechanism_tokens = mechanism_tokens
        self.base_tokens = nn.Parameter(torch.randn(1, mechanism_tokens, hidden_dim) * 0.02)
        self.init_mlp = MLP(hidden_dim, hidden_dim, hidden_dim, 2, dropout)
        self.blocks = nn.ModuleList([SupportUpdaterBlock(hidden_dim, heads, dropout) for _ in range(support_blocks)])

    def scene_only(self, scene_tokens: torch.Tensor) -> torch.Tensor:
        pooled = scene_tokens.mean(dim=1)
        shift = self.init_mlp(pooled).unsqueeze(1)
        tokens = self.base_tokens.expand(scene_tokens.shape[0], -1, -1) + shift
        for block in self.blocks:
            tokens = block(tokens, scene_tokens, None)
        return tokens

    def adapt(self, scene_tokens: torch.Tensor, support_tokens: torch.Tensor | None) -> torch.Tensor:
        tokens = self.scene_only(scene_tokens)
        if support_tokens is None or support_tokens.shape[1] == 0:
            return tokens
        for block in self.blocks:
            tokens = block(tokens, scene_tokens, support_tokens)
        return tokens


class OperatorQuery(nn.Module):
    def __init__(self, intervention_dim: int = 128, hidden_dim: int = 256, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.u_mlp = MLP(intervention_dim, hidden_dim, hidden_dim, 2, dropout)
        self.context_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.operator_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

    def forward(self, u: torch.Tensor, scene_tokens: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        b, q, _ = u.shape
        query = self.u_mlp(u)
        c, _ = self.context_attn(query, scene_tokens, scene_tokens, need_weights=False)
        o, _ = self.operator_attn(query + c, omega, omega, need_weights=False)
        return self.out(query + c + o)
