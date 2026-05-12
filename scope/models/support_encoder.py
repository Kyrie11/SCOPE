import torch
from torch import nn

class SupportEncoder(nn.Module):
    def __init__(self, hidden_dim=256, num_outcomes=5):
        super().__init__(); self.out=nn.Embedding(num_outcomes+1,hidden_dim); self.pres=nn.Embedding(5,hidden_dim); self.proj=nn.Sequential(nn.Linear(hidden_dim*3,hidden_dim), nn.GELU(), nn.Linear(hidden_dim,hidden_dim)); self.empty=nn.Parameter(torch.zeros(hidden_dim))
    def forward(self, support_candidate_embeddings, support_labels, support_masks):
        # support_candidate_embeddings [B,S,D]
        if support_candidate_embeddings.numel()==0 or support_candidate_embeddings.shape[1]==0:
            return self.empty[None,:].expand(support_candidate_embeddings.shape[0],-1)
        out_y=support_labels.get('outcome'); pr_y=support_labels.get('pressure_ordinal')
        out_m=support_masks.get('outcome').bool(); pr_m=support_masks.get('pressure').bool()
        out_emb=self.out(torch.where(out_m,out_y,torch.full_like(out_y,5)))
        pr_emb=self.pres(torch.where(pr_m,pr_y+1,torch.zeros_like(pr_y)))
        z=self.proj(torch.cat([support_candidate_embeddings,out_emb,pr_emb],-1))
        m=(out_m|pr_m).float()
        if m.sum() == 0: return self.empty[None,:].expand(z.shape[0],-1)
        return (z*m[...,None]).sum(1)/(m.sum(1,keepdim=True).clamp_min(1))
