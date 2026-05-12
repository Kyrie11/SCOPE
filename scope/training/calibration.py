import torch
class TemperatureScaler(torch.nn.Module):
    def __init__(self): super().__init__(); self.log_t=torch.nn.Parameter(torch.zeros(()))
    def forward(self, logits): return logits / self.log_t.exp().clamp_min(1e-3)

def fit_temperature(logits, labels, mask, steps=100):
    scaler=TemperatureScaler(); opt=torch.optim.LBFGS(scaler.parameters(), lr=.1, max_iter=steps)
    def closure():
        opt.zero_grad(); loss=torch.nn.functional.cross_entropy(scaler(logits)[mask.bool()], labels[mask.bool()]) if mask.any() else scaler.log_t*0; loss.backward(); return loss
    opt.step(closure); return {'temperature': float(scaler.log_t.exp().detach())}
