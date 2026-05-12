import torch
from scope.data.schema import OUTCOME

def estimate_conditional_high_pressure(surface, k=None, i=None):
    # use P(pressure>=2) as operational high-pressure probability
    return surface['pressure_exceedance_prob'][...,1]

def aggregate_noncede_risk(risk_by_outcome):
    idx=[v for name,v in OUTCOME.items() if name!='cede']
    return risk_by_outcome[...,idx].max(-1).values

def forced_dependence_score(surface, delta=0.10, tau=0.05):
    p_cede=surface['outcome_prob'][...,OUTCOME['cede']]
    p_hp=estimate_conditional_high_pressure(surface)
    risk_cede=surface['collision_risk_by_outcome'][...,OUTCOME['cede']]
    risk_non=aggregate_noncede_risk(surface['collision_risk_by_outcome'])
    dep=torch.sigmoid((risk_non-risk_cede-delta)/tau)
    return p_cede*p_hp*dep
