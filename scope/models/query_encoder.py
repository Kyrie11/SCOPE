import torch
from torch import nn
from scope.models.encoders import MLP
from scope.data.state import STATE_DIM

class InterventionQueryEncoder(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.1):
        super().__init__(); self.traj=MLP(STATE_DIM+4, hidden_dim, hidden_dim, dropout=dropout); self.cross=nn.MultiheadAttention(hidden_dim,4,batch_first=True); self.norm=nn.LayerNorm(hidden_dim)
    def forward(self, ego_candidates, ego_candidate_valid, context_tokens, context_mask):
        B,K,T,D=ego_candidates.shape
        xy=ego_candidates[...,:2]; delta=xy[:,:, -1]-xy[:,:,0]; dist=torch.linalg.norm(delta,dim=-1,keepdim=True); mean_speed=torch.linalg.norm(ego_candidates[...,2:4],dim=-1).masked_fill(~ego_candidate_valid.bool(),0).sum(-1,keepdim=True)/ego_candidate_valid.float().sum(-1,keepdim=True).clamp_min(1)
        feats=torch.cat([ego_candidates, delta[:,:,None,:].expand(B,K,T,2), dist[:,:,None,:].expand(B,K,T,1), mean_speed[:,:,None,:].expand(B,K,T,1)],-1)
        z=self.traj(feats).masked_fill(~ego_candidate_valid[...,None].bool(),0).sum(2)/(ego_candidate_valid.float().sum(2).clamp_min(1)[...,None])
        attn,_=self.cross(z,context_tokens,context_tokens,key_padding_mask=~context_mask.bool())
        return self.norm(z+attn)
