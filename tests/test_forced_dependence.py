from scope.planning.forced_dependence import forced_dependence_score
import torch

def test_fd_increases():
    surf={'outcome_prob':torch.tensor([[.9,.05,.02,.02,.01]]),'pressure_exceedance_prob':torch.tensor([[[1.,.9,.1]]]),'collision_risk_by_outcome':torch.tensor([[.0,.9,.9,.9,.9]])}
    assert forced_dependence_score(surf).item()>.5
