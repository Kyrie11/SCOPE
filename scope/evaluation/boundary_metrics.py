import torch
from scope.training.losses import js_divergence

def boundary_score(outputs,batch,lambda_pressure=.25,lambda_risk=.25):
    pairs=batch['neighbor_pairs']; B,E,_=pairs.shape; idxa=pairs[...,0]; idxb=pairs[...,1]
    def gather(x):
        shape=x.shape[2:]; return torch.gather(x,1,idxa.reshape(B,E,*([1]*len(shape))).expand(B,E,*shape)), torch.gather(x,1,idxb.reshape(B,E,*([1]*len(shape))).expand(B,E,*shape))
    pa,pb=gather(outputs['outcome_prob']); pra,prb=gather(outputs['pressure_exceedance_prob']); ca,cb=gather(outputs['collision_prob'].unsqueeze(-1))
    return js_divergence(pa,pb)+lambda_pressure*(pra-prb).abs().mean(-1)+lambda_risk*(ca.squeeze(-1)-cb.squeeze(-1)).abs()
