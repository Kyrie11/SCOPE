"""Same-root support/query sampling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SupportQueryConfig:
    support_sizes: tuple[int, ...] = (0, 1, 2, 4, 8)
    prob_empty_support: float = 0.4
    no_support_query_overlap: bool = True
    seed: int = 19


class SupportQuerySampler:
    def __init__(self, cfg: SupportQueryConfig | dict | None = None):
        if cfg is None:
            cfg = SupportQueryConfig()
        
        if isinstance(cfg, dict):
            allowed = set(SupportQueryConfig.__dataclass_fields__.keys())
            self.cfg = SupportQueryConfig(**{k: v for k, v in cfg.items() if k in allowed})
        else:
            self.cfg = cfg
        self.rng = np.random.default_rng(self.cfg.seed)

    def sample_indices(self, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.flatnonzero(valid_mask.astype(bool))
        if idx.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        if self.rng.random() < self.cfg.prob_empty_support:
            size = 0
        else:
            size = int(self.rng.choice([s for s in self.cfg.support_sizes if s > 0]))
        size = min(size, max(0, idx.size - 1)) if self.cfg.no_support_query_overlap else min(size, idx.size)
        support = self.rng.choice(idx, size=size, replace=False) if size > 0 else np.array([], dtype=int)
        query = np.setdiff1d(idx, support, assume_unique=False) if self.cfg.no_support_query_overlap else idx
        return support.astype(int), query.astype(int)

    def split_batch(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor] | None, dict[str, torch.Tensor]]:
        mask = batch["mask"].detach().cpu().numpy()
        b, q = mask.shape
        support_lists = []
        query_mask = torch.zeros_like(batch["mask"])
        max_s = 0
        for bi in range(b):
            s_idx, q_idx = self.sample_indices(mask[bi])
            support_lists.append(s_idx)
            max_s = max(max_s, len(s_idx))
            if len(q_idx):
                query_mask[bi, torch.as_tensor(q_idx, dtype=torch.long, device=query_mask.device)] = 1.0
        query = dict(batch)
        query["mask"] = batch["mask"] * query_mask
        if max_s == 0:
            return None, query
        def gather(name: str, fill: float = 0.0):
            src = batch[name]
            out_shape = (b, max_s) + tuple(src.shape[2:])
            out = torch.full(out_shape, fill, dtype=src.dtype, device=src.device)
            for bi, idx in enumerate(support_lists):
                if len(idx):
                    out[bi, : len(idx)] = src[bi, torch.as_tensor(idx, dtype=torch.long, device=src.device)]
            return out
        valid = torch.zeros((b, max_s), dtype=batch["mask"].dtype, device=batch["mask"].device)
        for bi, idx in enumerate(support_lists):
            if len(idx):
                valid[bi, : len(idx)] = 1.0
        support = {
            "ctrl": gather("query_ctrl"),
            "anchors": gather("query_anchors"),
            "future_tokens": gather("query_future_tokens"),
            "labels": {"branch": gather("branch"), "burden": gather("burden"), "safety": gather("safety")},
            "masks": {"valid": valid, "branch": valid, "burden": valid, "trajectory": valid, "safety": valid, "fd_diag": torch.zeros_like(valid), "boundary": torch.zeros_like(valid)},
        }
        return support, query
