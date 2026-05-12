import torch
from torch import nn

class ResponseOperator(nn.Module):
    def __init__(self, hidden_dim=256, operator_slots=8, dropout=0.1):
        super().__init__(); self.operator_slots=operator_slots; self.base=nn.Parameter(torch.randn(operator_slots,hidden_dim)*0.02); self.ctx=nn.Linear(hidden_dim,hidden_dim); self.sup=nn.Linear(hidden_dim,hidden_dim); self.read=nn.MultiheadAttention(hidden_dim,4,batch_first=True); self.norm=nn.LayerNorm(hidden_dim)
    def generate_slots(self, context_tokens, context_mask=None, support_embedding=None):
        ctx=context_tokens.mean(1); slots=self.base[None].expand(context_tokens.shape[0],-1,-1)+self.ctx(ctx)[:,None,:]
        if support_embedding is not None: slots=slots+self.sup(support_embedding)[:,None,:]
        return slots
    def readout(self, query_embeddings, operator_slots):
        out,attn=self.read(query_embeddings, operator_slots, operator_slots, need_weights=True)
        return self.norm(query_embeddings+out), attn
