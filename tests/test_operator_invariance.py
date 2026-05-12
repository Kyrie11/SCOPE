import torch
from scope.data.collate import collate_scope
from scope.models.scope_surface import SCOPEResponseSurface

def test_scene_only_permutation(group):
    b=collate_scope([group]); m=SCOPEResponseSurface({'hidden_dim':32,'operator_slots':2,'future_steps':b['ego_candidates'].shape[2]}); out1=m(b,'scene_only')
    perm=torch.arange(b['ego_candidates'].shape[1]-1,-1,-1); b2=dict(b); b2['ego_candidates']=b['ego_candidates'][:,perm]; b2['ego_candidate_valid']=b['ego_candidate_valid'][:,perm]
    out2=m(b2,'scene_only'); assert out1['operator_slots'].shape==out2['operator_slots'].shape
