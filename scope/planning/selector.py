import torch
from scope.planning.forced_dependence import forced_dependence_score
from scope.planning.task_cost import task_cost

def select_candidate(surface, ego_candidates, config):
    p=config.get('planning',config)
    collision=surface['collision_prob']; fd=forced_dependence_score(surface, p.get('forced_dependence_delta',.1), p.get('forced_dependence_temperature',.05)); sens=surface.get('surface_sensitivity', torch.zeros_like(collision)); unc=surface.get('operator_uncertainty', surface.get('uncertainty', torch.zeros_like(collision)))
    # supports [K] or [K,N]; aggregate agents by max if needed
    for name in ['collision','fd','sens','unc']:
        v=locals()[name]
        if v.ndim>1: locals()[name]=v.max(-1).values
    cost=task_cost(ego_candidates)
    feasible=(collision<=p.get('eps_collision',.05))&(fd<=p.get('eps_fd',.1))&(sens<=p.get('eps_surface',.5))&(unc<=p.get('eps_uncertainty',.5))
    if feasible.any():
        idx=torch.where(feasible, cost, torch.full_like(cost, float('inf'))).argmin()
    else:
        lex=collision*1e6+fd*1e4+sens*1e2+unc*10+cost
        idx=lex.argmin()
    return int(idx), {'collision':collision,'forced_dependence':fd,'surface_sensitivity':sens,'uncertainty':unc,'task_cost':cost,'feasible':feasible}
