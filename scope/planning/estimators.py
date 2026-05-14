"""Planning estimators: risk, forced-dependence, boundary, CVaR, uncertainty."""
from __future__ import annotations

import torch
import torch.nn.functional as F


BRANCH_CEDE = 0


def burden_ge_probability(burden_ge_logits: torch.Tensor, rho0: int = 2) -> torch.Tensor:
    """Return P(rho >= rho0) from cumulative logits.

    burden_ge_logits shape: [..., branches, 3] for thresholds rho>=1,2,3.
    """
    if rho0 <= 0:
        return torch.ones_like(burden_ge_logits[..., 0])
    if rho0 > 3:
        return torch.zeros_like(burden_ge_logits[..., 0])
    return torch.sigmoid(burden_ge_logits[..., rho0 - 1])


def branch_conditioned_risk(outputs: dict, collision_event_index: int = 0) -> torch.Tensor:
    return torch.sigmoid(outputs["safety_logits"])[..., collision_event_index]


def marginal_collision_risk(outputs: dict) -> torch.Tensor:
    p_branch = torch.softmax(outputs["branch_logits"], dim=-1)
    risk_m = branch_conditioned_risk(outputs)
    return (p_branch * risk_m).sum(dim=-1)


def nonceding_risk(outputs: dict, eps: float = 1e-6) -> torch.Tensor:
    p_branch = torch.softmax(outputs["branch_logits"], dim=-1)
    risk_m = branch_conditioned_risk(outputs)
    mask = torch.ones_like(p_branch)
    mask[..., BRANCH_CEDE] = 0.0
    num = (p_branch * mask * risk_m).sum(dim=-1)
    den = (p_branch * mask).sum(dim=-1)
    return num / torch.clamp(den, min=eps)


def forced_dependence(
    outputs: dict,
    rho0: int = 2,
    delta: float = 0.1,
    tau_d: float = 0.1,
) -> torch.Tensor:
    """Branch-conditioned forced-dependence estimator from the paper.

    D_i(k) = P(cede) * P(rho>=rho0 | cede) * (1 - R_cede)
             * sigmoid((R_not_cede - R_cede - delta) / tau_d)
    """
    p_branch = torch.softmax(outputs["branch_logits"], dim=-1)
    p_cede = p_branch[..., BRANCH_CEDE]
    p_hp_cede = burden_ge_probability(outputs["burden_ge_logits"], rho0)[..., BRANCH_CEDE]
    risk_m = branch_conditioned_risk(outputs)
    r_cede = risk_m[..., BRANCH_CEDE]
    r_not = nonceding_risk(outputs)
    return p_cede * p_hp_cede * (1.0 - r_cede) * torch.sigmoid((r_not - r_cede - delta) / max(tau_d, 1e-6))


def aggregate_noisy_or(agent_values: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.prod(1.0 - torch.clamp(agent_values, 0.0, 1.0), dim=-1)


def aggregate_candidate_quantities(outputs_by_agent: list[dict], rho0: int = 2, delta: float = 0.1, tau_d: float = 0.1) -> dict[str, torch.Tensor]:
    risks = torch.stack([marginal_collision_risk(o) for o in outputs_by_agent], dim=-1)
    fd = torch.stack([forced_dependence(o, rho0, delta, tau_d) for o in outputs_by_agent], dim=-1)
    p_branch = [torch.softmax(o["branch_logits"], dim=-1) for o in outputs_by_agent]
    p_hp = torch.stack([p[..., BRANCH_CEDE] * burden_ge_probability(o["burden_ge_logits"], rho0)[..., BRANCH_CEDE] for p, o in zip(p_branch, outputs_by_agent)], dim=-1)
    return {"R": aggregate_noisy_or(risks), "D": aggregate_noisy_or(fd), "P_hp": aggregate_noisy_or(p_hp)}


def cvar_upper_tail(samples: torch.Tensor, alpha: float = 0.1, dim: int = -1) -> torch.Tensor:
    sorted_vals, _ = torch.sort(samples, dim=dim)
    n = sorted_vals.shape[dim]
    tail = max(1, int(torch.ceil(torch.tensor(alpha * n)).item()))
    return sorted_vals.narrow(dim, n - tail, tail).mean(dim=dim)


def ordinal_w1(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.cumsum(p, dim=-1) - torch.cumsum(q, dim=-1)).sum(dim=-1)


def js_distance(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    return 0.5 * (p * (p.log() - m.log())).sum(dim=-1) + 0.5 * (q * (q.log() - m.log())).sum(dim=-1)


def branch_marginal_burden(outputs: dict) -> torch.Tensor:
    p_branch = torch.softmax(outputs["branch_logits"], dim=-1)
    burden_pmf = outputs["burden_pmf"]
    return (p_branch[..., None] * burden_pmf).sum(dim=-2)


def expected_risk(outputs: dict) -> torch.Tensor:
    return marginal_collision_risk(outputs)


def surface_distance(outputs_a: dict, outputs_b: dict, lambda_rho_b: float = 0.5, lambda_R_b: float = 1.0) -> torch.Tensor:
    p_a = torch.softmax(outputs_a["branch_logits"], dim=-1)
    p_b = torch.softmax(outputs_b["branch_logits"], dim=-1)
    return js_distance(p_a, p_b) + lambda_rho_b * ordinal_w1(branch_marginal_burden(outputs_a), branch_marginal_burden(outputs_b)) + lambda_R_b * torch.abs(expected_risk(outputs_a) - expected_risk(outputs_b))


def boundary_sensitivity(outputs: dict, neighbor_outputs: list[dict], neighbor_distances: list[float], eps: float = 1e-6) -> torch.Tensor:
    if not neighbor_outputs:
        return torch.zeros_like(outputs["branch_logits"][..., 0])
    vals = []
    for out_n, d in zip(neighbor_outputs, neighbor_distances):
        vals.append(surface_distance(outputs, out_n) / (float(d) + eps))
    return torch.stack(vals, dim=-1).max(dim=-1).values


def ensemble_uncertainty(output_list: list[dict]) -> torch.Tensor:
    if len(output_list) <= 1:
        return torch.zeros_like(output_list[0]["branch_logits"][..., 0])
    probs = torch.stack([torch.softmax(o["branch_logits"], dim=-1) for o in output_list], dim=0)
    mean = probs.mean(dim=0)
    ent_mean = -(mean * torch.clamp(mean, min=1e-6).log()).sum(dim=-1)
    ent_members = -(probs * torch.clamp(probs, min=1e-6).log()).sum(dim=-1).mean(dim=0)
    return ent_mean - ent_members


def upper_confidence_bound(mean: torch.Tensor, uncertainty: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    return mean + beta * uncertainty
