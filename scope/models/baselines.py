from typing import Any, Dict
from torch import nn
from scope.models.scope_surface import SCOPEResponseSurface

class EgoCondResponse(SCOPEResponseSurface):
    def forward(self,batch,support_mode='scene_only'):
        return super().forward(batch,'scene_only')
class EgoCondTraj(EgoCondResponse): pass
class SharedCtxSlots(EgoCondResponse): pass
class SCOPE_NoSupport(SCOPEResponseSurface):
    def forward(self,batch,support_mode='scene_only'): return super().forward(batch,'scene_only')
class SCOPE_NoBoundary(SCOPEResponseSurface): pass
class SCOPE_NoFD(SCOPEResponseSurface): pass
class SCOPE_Full(SCOPEResponseSurface): pass
class PhysicalProbe(nn.Module):
    def __init__(self, config=None): super().__init__(); self.config=config or {}
    def forward(self,batch,support_mode='scene_only'):
        import torch
        B,K=batch['ego_candidates'].shape[:2]; M=5; dev=batch['ego_candidates'].device
        return {'outcome_logits':torch.zeros(B,K,M,device=dev),'outcome_prob':torch.ones(B,K,M,device=dev)/M,'pressure_exceedance_logits':torch.zeros(B,K,3,device=dev),'pressure_exceedance_prob':torch.zeros(B,K,3,device=dev),'pressure_mean':torch.zeros(B,K,device=dev),'traj_params':{},'traj_samples':batch['ego_candidates'][:,:,None],'collision_prob':torch.zeros(B,K,device=dev),'near_collision_prob':torch.zeros(B,K,device=dev),'hard_brake_prob':torch.zeros(B,K,device=dev),'high_pressure_prob':torch.zeros(B,K,device=dev),'forced_dependence_prob':torch.zeros(B,K,device=dev),'collision_risk_by_outcome':torch.zeros(B,K,M,device=dev),'operator_slots':torch.zeros(B,1,1,device=dev),'operator_attention':torch.zeros(B,K,1,device=dev),'query_embeddings':torch.zeros(B,K,1,device=dev),'response_embeddings':torch.zeros(B,K,1,device=dev),'surface_sensitivity':torch.zeros(B,K,device=dev),'uncertainty':torch.zeros(B,K,device=dev)}

def build_model(config: Dict[str,Any]):
    name=config.get('model_name','scope_full')
    cls={'ego_cond_traj':EgoCondTraj,'ego_cond_response':EgoCondResponse,'shared_ctx_slots':SharedCtxSlots,'scope_no_support':SCOPE_NoSupport,'scope_no_boundary':SCOPE_NoBoundary,'scope_no_fd':SCOPE_NoFD,'scope_full':SCOPE_Full,'physical_probe':PhysicalProbe}.get(name, SCOPE_Full)
    return cls(config)
