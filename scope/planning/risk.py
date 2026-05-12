import torch

def cvar(values, alpha=.9, dim=-1):
    q=torch.quantile(values, alpha, dim=dim, keepdim=True)
    tail=values.masked_fill(values<q, float('nan'))
    return torch.nanmean(tail, dim=dim)

def collision_cvar_from_prob(p): return p
