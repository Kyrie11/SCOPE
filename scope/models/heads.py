import torch
from torch import nn
from scope.data.state import STATE_DIM

class ResponseHeads(nn.Module):
    def __init__(self, hidden_dim=256, num_outcomes=5, num_traj_modes=1, future_steps=80):
        super().__init__(); self.num_outcomes=num_outcomes; self.future_steps=future_steps
        self.outcome=nn.Linear(hidden_dim,num_outcomes); self.pressure=nn.Linear(hidden_dim,3); self.traj=nn.Linear(hidden_dim,future_steps*STATE_DIM); self.collision=nn.Linear(hidden_dim,1); self.near=nn.Linear(hidden_dim,1); self.hard=nn.Linear(hidden_dim,1); self.high=nn.Linear(hidden_dim,1); self.fd=nn.Linear(hidden_dim,1); self.risk_by_out=nn.Linear(hidden_dim,num_outcomes); self.unc=nn.Linear(hidden_dim,1)
    def forward(self,z):
        B,K,D=z.shape; out_logits=self.outcome(z); p_logits=self.pressure(z)
        p_ge=torch.sigmoid(p_logits); pressure_mean=(p_ge.sum(-1)).clamp(0,3)
        traj_mean=self.traj(z).view(B,K,self.future_steps,STATE_DIM); traj_samples=traj_mean[:,:,None]
        return {'outcome_logits':out_logits,'outcome_prob':torch.softmax(out_logits,-1),'pressure_exceedance_logits':p_logits,'pressure_exceedance_prob':p_ge,'pressure_mean':pressure_mean,'traj_params':{'mean':traj_mean},'traj_samples':traj_samples,'collision_prob':torch.sigmoid(self.collision(z)).squeeze(-1),'near_collision_prob':torch.sigmoid(self.near(z)).squeeze(-1),'hard_brake_prob':torch.sigmoid(self.hard(z)).squeeze(-1),'high_pressure_prob':torch.sigmoid(self.high(z)).squeeze(-1),'forced_dependence_prob':torch.sigmoid(self.fd(z)).squeeze(-1),'collision_risk_by_outcome':torch.sigmoid(self.risk_by_out(z)),'uncertainty':torch.sigmoid(self.unc(z)).squeeze(-1)}
