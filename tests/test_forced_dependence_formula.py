import torch

from scope.planning.estimators import forced_dependence, nonceding_risk


def test_forced_dependence_uses_branch_conditioned_nonceding_risk():
    out = {
        "branch_logits": torch.tensor([[[4.0, 0.0, 0.0, -1.0, -2.0]]]),
        "burden_ge_logits": torch.zeros(1, 1, 5, 3),
        "safety_logits": torch.full((1, 1, 5, 4), -5.0),
    }
    out["burden_ge_logits"][..., 0, 1] = 5.0  # P(rho>=2 | cede)
    out["safety_logits"][..., 0, 0] = -5.0  # cede safe
    out["safety_logits"][..., 1:, 0] = 5.0  # non-ceding risky
    d = forced_dependence(out, rho0=2, delta=0.1, tau_d=0.1)
    assert d.item() > 0.5
    out["safety_logits"][..., 1:, 0] = -5.0
    d2 = forced_dependence(out, rho0=2, delta=0.1, tau_d=0.1)
    assert d2.item() < d.item() * 0.4
