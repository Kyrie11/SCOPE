import torch
from torch import nn
from scope.data.state import STATE_DIM

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, layers=2, dropout=0.0):
        super().__init__(); mods=[]; d=in_dim
        for _ in range(max(1,layers-1)):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]; d=hidden
        mods.append(nn.Linear(d,out_dim)); self.net=nn.Sequential(*mods)
    def forward(self,x): return self.net(x)

class ContextEncoder(nn.Module):
    def __init__(self, hidden_dim=256, map_dim=4, dropout=0.1):
        super().__init__(); self.agent=MLP(STATE_DIM, hidden_dim, hidden_dim, dropout=dropout); self.map=MLP(map_dim, hidden_dim, hidden_dim, dropout=dropout)
        enc=nn.TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True)
        self.fuse=nn.TransformerEncoder(enc, num_layers=2)
    def forward(self, history, history_valid, map_polylines, map_valid):
        B,H,A,D=history.shape
        agent_tokens=self.agent(history).masked_fill(~history_valid[...,None].bool(),0).sum(1)/(history_valid.float().sum(1).clamp_min(1)[...,None])
        B,P,L,M=map_polylines.shape
        map_tokens=self.map(map_polylines).masked_fill(~map_valid[...,None].bool(),0).sum(2)/(map_valid.float().sum(2).clamp_min(1)[...,None])
        tokens=torch.cat([agent_tokens,map_tokens],1); mask=torch.cat([history_valid.any(1), map_valid.any(2)],1)
        tokens=self.fuse(tokens, src_key_padding_mask=~mask.bool())
        return tokens, mask
