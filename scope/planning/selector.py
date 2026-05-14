"""Mechanism-feasible candidate selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from scope.data.scene_schema import EgoCandidate


@dataclass
class PlanningThresholds:
    epsilon_R: float = 0.05
    epsilon_D: float = 0.10
    epsilon_rho: float = 0.30
    epsilon_B: float = 1.0
    epsilon_U: float = 0.20
    beta: float = 1.0
    cvar_alpha: float = 0.1


@dataclass
class CandidateEstimate:
    candidate_id: str
    R: float
    D: float
    P_hp: float
    B: float
    U: float
    task_cost: float
    metadata: dict | None = None


def is_mechanism_feasible(est: CandidateEstimate, th: PlanningThresholds) -> bool:
    return bool(est.R <= th.epsilon_R and est.D <= th.epsilon_D and est.P_hp <= th.epsilon_rho and est.B <= th.epsilon_B and est.U <= th.epsilon_U)


def select_mechanism_feasible(candidates: Sequence[EgoCandidate], estimates: Sequence[CandidateEstimate], thresholds: PlanningThresholds) -> tuple[EgoCandidate | None, list[CandidateEstimate]]:
    by_id = {c.candidate_id: c for c in candidates}
    feasible = [e for e in estimates if e.candidate_id in by_id and is_mechanism_feasible(e, thresholds)]
    if not feasible:
        return None, []
    best = min(feasible, key=lambda e: (e.task_cost, e.R, e.D, e.B))
    return by_id[best.candidate_id], feasible


def lagrangian_fallback(candidates: Sequence[EgoCandidate], estimates: Sequence[CandidateEstimate], thresholds: PlanningThresholds, lambdas: dict[str, float] | None = None) -> EgoCandidate:
    lambdas = lambdas or {"R": 10.0, "D": 10.0, "P_hp": 2.0, "B": 1.0, "U": 1.0}
    by_id = {c.candidate_id: c for c in candidates}
    best_id = None
    best_val = float("inf")
    for e in estimates:
        if e.candidate_id not in by_id:
            continue
        val = e.task_cost
        val += lambdas.get("R", 0.0) * max(0.0, e.R - thresholds.epsilon_R)
        val += lambdas.get("D", 0.0) * max(0.0, e.D - thresholds.epsilon_D)
        val += lambdas.get("P_hp", 0.0) * max(0.0, e.P_hp - thresholds.epsilon_rho)
        val += lambdas.get("B", 0.0) * max(0.0, e.B - thresholds.epsilon_B)
        val += lambdas.get("U", 0.0) * max(0.0, e.U - thresholds.epsilon_U)
        if val < best_val:
            best_val = val
            best_id = e.candidate_id
    if best_id is None:
        raise ValueError("No candidate estimates match candidates")
    return by_id[best_id]


def active_violations(est: CandidateEstimate, thresholds: PlanningThresholds) -> dict[str, float]:
    return {
        "R": max(0.0, est.R - thresholds.epsilon_R),
        "D": max(0.0, est.D - thresholds.epsilon_D),
        "P_hp": max(0.0, est.P_hp - thresholds.epsilon_rho),
        "B": max(0.0, est.B - thresholds.epsilon_B),
        "U": max(0.0, est.U - thresholds.epsilon_U),
    }
