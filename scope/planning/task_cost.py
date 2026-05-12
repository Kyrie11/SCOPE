import torch

def task_cost(ego_candidates):
    final=ego_candidates[:,-1,:2] if ego_candidates.ndim==3 else ego_candidates[..., -1, :2]
    progress=-final[...,0]; comfort=torch.linalg.norm(ego_candidates[...,4:6],dim=-1).mean(-1)
    return progress + 0.1*comfort
