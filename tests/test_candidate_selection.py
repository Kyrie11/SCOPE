from scope.planning.selector import select_candidate
import torch

def test_selector_rejects_risk():
    surf={'collision_prob':torch.tensor([.9,.01]),'outcome_prob':torch.ones(2,5)/5,'pressure_exceedance_prob':torch.zeros(2,3),'collision_risk_by_outcome':torch.zeros(2,5),'surface_sensitivity':torch.zeros(2),'uncertainty':torch.zeros(2)}
    ego=torch.zeros(2,5,11); idx,_=select_candidate(surf,ego,{'eps_collision':.05,'eps_fd':.1,'eps_surface':.5,'eps_uncertainty':.5}); assert idx==1
