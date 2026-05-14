"""Prediction and planning metrics used by SCOPE experiments."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def accuracy(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    pred_label = np.asarray(pred).argmax(axis=-1) if pred.ndim > target.ndim else np.asarray(pred)
    ok = pred_label == target
    if mask is not None:
        ok = ok[np.asarray(mask).astype(bool)]
    return float(ok.mean()) if ok.size else float("nan")


def nll_from_probs(probs: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-8) -> float:
    p = np.asarray(probs)
    idx = np.asarray(target).astype(int)
    picked = np.take_along_axis(p, np.expand_dims(idx, -1), axis=-1).squeeze(-1)
    loss = -np.log(np.clip(picked, eps, 1.0))
    if mask is not None:
        loss = loss[np.asarray(mask).astype(bool)]
    return float(loss.mean()) if loss.size else float("nan")


def brier_score(prob: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    err = (np.asarray(prob) - np.asarray(target)) ** 2
    if mask is not None:
        err = err[np.asarray(mask).astype(bool)]
    return float(err.mean()) if err.size else float("nan")


def expected_calibration_error(prob: np.ndarray, target: np.ndarray, bins: int = 15) -> float:
    p = np.asarray(prob).reshape(-1)
    y = np.asarray(target).reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(p), 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if m.any():
            ece += m.sum() / total * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(ece)


def min_ade_fde(pred_modes: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    pm = np.asarray(pred_modes)[..., :2]
    tgt = np.asarray(target)[..., :2]
    dist = np.linalg.norm(pm - tgt[:, None, :, :], axis=-1)
    if mask is not None:
        tm = np.asarray(mask).astype(bool)
        dist = np.where(tm[:, None, :], dist, np.nan)
    ade_modes = np.nanmean(dist, axis=-1)
    fde_modes = dist[..., -1]
    best = np.nanargmin(ade_modes, axis=1)
    ade = ade_modes[np.arange(len(best)), best]
    fde = fde_modes[np.arange(len(best)), best]
    return float(np.nanmean(ade)), float(np.nanmean(fde))


def auroc(scores: Iterable[float], labels: Iterable[int]) -> float:
    s = np.asarray(list(scores), dtype=float)
    y = np.asarray(list(labels), dtype=int)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for val in pos:
        wins += float(np.sum(val > neg)) + 0.5 * float(np.sum(val == neg))
    return wins / (len(pos) * len(neg))


def cvar(samples: np.ndarray, alpha: float = 0.1) -> float:
    arr = np.sort(np.asarray(samples, dtype=float).reshape(-1))
    if arr.size == 0:
        return float("nan")
    tail = max(1, int(math.ceil(alpha * arr.size)))
    return float(arr[-tail:].mean())
