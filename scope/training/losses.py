"""Training losses for SCOPE."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from scope.models.heads import OrdinalBurdenHead, branch_conditioned_collision_risk


def masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    while mask.dim() < value.dim():
        mask = mask.unsqueeze(-1)
    return (value * mask).sum() / torch.clamp(mask.sum(), min=eps)


def branch_ce_loss(outputs: dict, target: torch.Tensor, mask: torch.Tensor, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    logits = outputs["branch_logits"]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none", weight=class_weights).reshape_as(target).float()
    return masked_mean(loss, mask)


def ordinal_burden_loss(outputs: dict, target: torch.Tensor, branch_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = outputs["burden_ge_logits"]
    gather_idx = branch_target[..., None, None].expand(*branch_target.shape, 1, 3)
    selected = torch.gather(logits, 2, gather_idx).squeeze(2)
    ge_target = torch.stack([(target >= r).float() for r in (1, 2, 3)], dim=-1)
    loss = F.binary_cross_entropy_with_logits(selected, ge_target, reduction="none").sum(dim=-1)
    return masked_mean(loss, mask)


def safety_bce_loss(outputs: dict, target: torch.Tensor, branch_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = outputs["safety_logits"]
    gather_idx = branch_target[..., None, None].expand(*branch_target.shape, 1, logits.shape[-1])
    selected = torch.gather(logits, 2, gather_idx).squeeze(2)
    loss = F.binary_cross_entropy_with_logits(selected, target.float(), reduction="none").sum(dim=-1)
    return masked_mean(loss, mask)


def trajectory_mixture_nll(outputs: dict, target: torch.Tensor, branch_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    traj = outputs["trajectory"]
    loc = traj["loc"]
    log_scale = traj["log_scale"]
    mode_logits = traj["mode_logits"]
    state_dim = loc.shape[-1]
    target = target[..., :state_dim]
    branch_idx = branch_target[..., None, None, None, None].expand(*branch_target.shape, 1, loc.shape[3], loc.shape[4], loc.shape[5])
    loc_sel = torch.gather(loc, 2, branch_idx).squeeze(2)
    scale_sel = torch.exp(torch.gather(log_scale, 2, branch_idx).squeeze(2))
    mode_logits_sel = torch.gather(mode_logits, 2, branch_target[..., None, None].expand(*branch_target.shape, 1, mode_logits.shape[-1])).squeeze(2)
    err = torch.abs(target[:, :, None, :, :] - loc_sel) / torch.clamp(scale_sel, min=1e-4) + torch.log(torch.clamp(scale_sel, min=1e-4))
    nll_mode = err.sum(dim=(-1, -2)) - F.log_softmax(mode_logits_sel, dim=-1)
    nll = -torch.logsumexp(-nll_mode, dim=-1)
    return masked_mean(nll, mask)


def mechanism_loss(outputs: dict, batch: dict, weights: dict | None = None) -> dict[str, torch.Tensor]:
    weights = weights or {}
    mask = batch["mask"].float()
    branch = batch["branch"].long()
    burden = batch["burden"].long()
    losses = {
        "branch": branch_ce_loss(outputs, branch, mask),
        "burden": ordinal_burden_loss(outputs, burden, branch, mask),
        "trajectory": trajectory_mixture_nll(outputs, batch["trajectory"].float(), branch, mask),
        "safety": safety_bce_loss(outputs, batch["safety"].float(), branch, mask),
    }
    total = losses["branch"] + weights.get("lambda_rho", 1.0) * losses["burden"] + weights.get("lambda_tau", 1.0) * losses["trajectory"] + weights.get("lambda_c", 1.0) * losses["safety"]
    losses["total"] = total
    return losses


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    return 0.5 * (p * (p.log() - m.log())).sum(dim=-1) + 0.5 * (q * (q.log() - m.log())).sum(dim=-1)


def ordinal_w1(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.cumsum(p, dim=-1) - torch.cumsum(q, dim=-1)).sum(dim=-1)


def expected_risk(outputs: dict) -> torch.Tensor:
    branch_p = torch.softmax(outputs["branch_logits"], dim=-1)
    risk_m = branch_conditioned_collision_risk(outputs)
    return (branch_p * risk_m).sum(dim=-1)


def distillation_loss(outputs_s: dict, outputs_empty: dict, mask: torch.Tensor, lambda_rho_d: float = 1.0, lambda_R_d: float = 1.0) -> torch.Tensor:
    p_s = torch.softmax(outputs_s["branch_logits"], dim=-1)
    p_e = torch.softmax(outputs_empty["branch_logits"], dim=-1)
    branch_js = js_divergence(p_s, p_e)
    # Use branch-marginal burden distribution for W1.
    b_s = outputs_s["burden_pmf"]
    b_e = outputs_empty["burden_pmf"]
    marg_s = (p_s[..., None] * b_s).sum(dim=-2)
    marg_e = (p_e[..., None] * b_e).sum(dim=-2)
    w1 = ordinal_w1(marg_s, marg_e)
    risk = torch.abs(expected_risk(outputs_s) - expected_risk(outputs_empty))
    return masked_mean(branch_js + lambda_rho_d * w1 + lambda_R_d * risk, mask.float())


def manifold_regularizer(query_u: torch.Tensor, edit_distance: torch.Tensor, pair_index: torch.Tensor) -> torch.Tensor:
    if pair_index.numel() == 0:
        return query_u.sum() * 0.0
    b_idx, a_idx, c_idx = pair_index[:, 0], pair_index[:, 1], pair_index[:, 2]
    du = torch.linalg.norm(query_u[b_idx, a_idx] - query_u[b_idx, c_idx], dim=-1)
    return torch.mean(torch.abs(du - edit_distance.to(query_u.device)))


def total_scope_loss(scene_outputs: dict, batch: dict, support_outputs: dict | None = None, empty_outputs_for_distill: dict | None = None, cfg: dict | None = None) -> dict[str, torch.Tensor]:
    cfg = cfg or {}
    mech = mechanism_loss(scene_outputs, batch, cfg)
    total = mech["total"]
    losses = {f"scene_{k}": v for k, v in mech.items()}
    if support_outputs is not None:
        sup = mechanism_loss(support_outputs, batch, cfg)
        total = total + sup["total"]
        losses.update({f"support_{k}": v for k, v in sup.items()})
    if support_outputs is not None and empty_outputs_for_distill is not None and cfg.get("lambda_dist", 0.1) > 0:
        dist = distillation_loss(support_outputs, empty_outputs_for_distill, batch["mask"], cfg.get("lambda_rho_d", 1.0), cfg.get("lambda_R_d", 1.0))
        total = total + cfg.get("lambda_dist", 0.1) * dist
        losses["distill"] = dist
    losses["total"] = total
    return losses
