"""Random seeding helpers."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class SeedBundle:
    data: int = 13
    model: int = 17
    support: int = 19
    rollout: int = 23


def seed_all(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)
