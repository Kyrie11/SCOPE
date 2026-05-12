from typing import Any, Dict
import torch
from torch import nn
from scope.models.encoders import ContextEncoder
from scope.models.query_encoder import InterventionQueryEncoder
from scope.models.support_encoder import SupportEncoder
from scope.models.response_operator import ResponseOperator
from scope.models.graph_coupling import ConflictGraphCoupling
from scope.models.heads import ResponseHeads

class SCOPEResponseSurface(nn.Module):
    def __init__(self, config: Dict[str,Any]):
        super().__init__(); h=int(config.get('hidden_dim',256)); self.config=config
        self.context=ContextEncoder(h, dropout=config.get('dropout',.1)); self.query=InterventionQueryEncoder(h, dropout=config.get('dropout',.1)); self.support=SupportEncoder(h, config.get('num_outcomes',5)); self.operator=ResponseOperator(h, config.get('operator_slots',8), config.get('dropout',.1)); self.graph=ConflictGraphCoupling(h, config.get('use_graph_coupling',True)); self.heads=ResponseHeads(h, config.get('num_outcomes',5), config.get('num_traj_modes',1), config.get('future_steps',80))
    def _gather(self, x, ids, mask):
        B,S=ids.shape; D=x.shape[-1]; idx=ids.clamp(0,x.shape[1]-1)[...,None].expand(B,S,D); return torch.gather(x,1,idx)*mask[...,None].float()
    def forward(self, batch: Dict[str,Any], support_mode: str='scene_only') -> Dict[str,Any]:
        ctx,ctx_mask=self.context(batch['history'],batch['history_valid'],batch['map_polylines'],batch['map_valid'])
        q=self.query(batch['ego_candidates'], batch['ego_candidate_valid'], ctx, ctx_mask)
        sup_emb=None
        if support_mode in ('support_adapted','factual_adapted') and batch.get('support_ids') is not None:
            se=self._gather(q,batch['support_ids'],batch['support_mask']); ids=batch['support_ids'].clamp(0,q.shape[1]-1); labels={}; masks={}
            for key,val in batch['labels'].items():
                if val.ndim>=2 and val.shape[1]==q.shape[1] and key in ('outcome','pressure_ordinal'):
                    labels[key]=torch.gather(val,1,ids)
            for key,val in batch['masks'].items():
                if val.ndim>=2 and val.shape[1]==q.shape[1] and key in ('outcome','pressure'):
                    masks[key]=torch.gather(val,1,ids)
            sup_emb=self.support(se,labels,masks)
        slots=self.operator.generate_slots(ctx,ctx_mask,sup_emb); z,attn=self.operator.readout(q,slots); z=self.graph(z,batch); outputs=self.heads(z)
        outputs.update({'operator_slots':slots,'operator_attention':attn,'query_embeddings':q,'response_embeddings':z,'surface_sensitivity':outputs['uncertainty']})
        return outputs
