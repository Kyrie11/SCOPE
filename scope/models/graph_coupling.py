from torch import nn
class ConflictGraphCoupling(nn.Module):
    def __init__(self, hidden_dim=256, enabled=True): super().__init__(); self.enabled=enabled; self.ff=nn.Sequential(nn.Linear(hidden_dim,hidden_dim), nn.GELU(), nn.Linear(hidden_dim,hidden_dim))
    def forward(self, response_embeddings, batch=None): return response_embeddings + self.ff(response_embeddings) if self.enabled else response_embeddings
