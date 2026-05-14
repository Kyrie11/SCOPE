"""Surface-guided candidate repair."""
from __future__ import annotations

from scope.data.candidates import repair_candidates_from_violation
from scope.data.scene_schema import EgoCandidate, RootScene
from scope.planning.selector import CandidateEstimate, PlanningThresholds, active_violations


def repair_candidate(scene: RootScene, candidate: EgoCandidate, estimate: CandidateEstimate, thresholds: PlanningThresholds, max_count: int = 8) -> list[EgoCandidate]:
    violations = active_violations(estimate, thresholds)
    return repair_candidates_from_violation(scene, candidate, violations, max_count=max_count)


def repair_pool(scene: RootScene, candidates: list[EgoCandidate], estimates: list[CandidateEstimate], thresholds: PlanningThresholds, max_total: int = 8) -> list[EgoCandidate]:
    by_id = {c.candidate_id: c for c in candidates}
    ordered = sorted(estimates, key=lambda e: sum(active_violations(e, thresholds).values()), reverse=True)
    repaired: list[EgoCandidate] = []
    for est in ordered:
        cand = by_id.get(est.candidate_id)
        if cand is None:
            continue
        repaired.extend(repair_candidate(scene, cand, est, thresholds, max_count=max_total - len(repaired)))
        if len(repaired) >= max_total:
            break
    return repaired[:max_total]
