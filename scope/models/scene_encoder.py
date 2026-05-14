"""Scene/context encoder for SCOPE."""
from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        mods = []
        dim = in_dim
        for _ in range(max(0, layers - 1)):
            mods.extend([nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            dim = hidden_dim
        mods.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SceneEncoder(nn.Module):
    """Encode root-scene tokens into agent-centric context tokens.

    The production dataloader can provide full agent histories and map polylines;
    the compact batch path used by smoke tests provides scene_summary_features.
    Both paths end in a Transformer token set C_i.
    """

    def __init__(self, scene_feature_dim: int = 16, hidden_dim: int = 256, layers: int = 4, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.summary_mlp = MLP(scene_feature_dim, hidden_dim, hidden_dim, 3, dropout)
        self.history_gru = nn.GRU(input_size=10, hidden_size=hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.map_point_mlp = MLP(4, hidden_dim, hidden_dim, 2, dropout)
        enc_layer = nn.TransformerEncoderLayer(hidden_dim, heads, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.type_embedding = nn.Embedding(8, hidden_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if "agent_histories" in batch:
            return self._forward_full(batch)
        summary = batch["scene_features"].float()
        b = summary.shape[0]
        token = self.summary_mlp(summary).unsqueeze(1)
        cls = self.cls.expand(b, -1, -1)
        tokens = torch.cat([cls, token + self.type_embedding.weight[0].view(1, 1, -1)], dim=1)
        return self.transformer(tokens)

    def _forward_full(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        histories = batch["agent_histories"].float()  # [B,A,T,F]
        b, a, t, f = histories.shape
        hist_flat = histories.reshape(b * a, t, f)
        _, h = self.history_gru(hist_flat)
        hist_tok = torch.cat([h[0], h[1]], dim=-1).reshape(b, a, self.hidden_dim)
        tokens = [self.cls.expand(b, -1, -1), hist_tok + self.type_embedding.weight[1].view(1, 1, -1)]
        if "map_polylines" in batch:
            mp = batch["map_polylines"].float()  # [B,P,L,4]
            map_tok = self.map_point_mlp(mp).mean(dim=2)
            tokens.append(map_tok + self.type_embedding.weight[2].view(1, 1, -1))
        if "traffic_tokens" in batch:
            tr = batch["traffic_tokens"].float()
            if tr.shape[-1] != self.hidden_dim:
                pad = torch.zeros(*tr.shape[:-1], self.hidden_dim - tr.shape[-1], device=tr.device, dtype=tr.dtype)
                tr = torch.cat([tr, pad], dim=-1)
            tokens.append(tr + self.type_embedding.weight[3].view(1, 1, -1))
        return self.transformer(torch.cat(tokens, dim=1))
