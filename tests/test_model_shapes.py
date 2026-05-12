from scope.data.collate import collate_scope
from scope.models.scope_surface import SCOPEResponseSurface

def test_shapes(group):
    b=collate_scope([group]); m=SCOPEResponseSurface({'hidden_dim':32,'operator_slots':2,'future_steps':b['ego_candidates'].shape[2]}); o=m(b,'support_adapted')
    assert o['outcome_logits'].shape[:2]==b['ego_candidates'].shape[:2] and o['pressure_exceedance_logits'].shape[-1]==3
