from scope.data.collate import collate_scope
from scope.models.scope_surface import SCOPEResponseSurface
from scope.training.losses import boundary_loss

def test_boundary_valid(group):
    b=collate_scope([group]); m=SCOPEResponseSurface({'hidden_dim':32,'operator_slots':2,'future_steps':b['ego_candidates'].shape[2]}); o=m(b); l=boundary_loss(o,b); assert l.isfinite()
