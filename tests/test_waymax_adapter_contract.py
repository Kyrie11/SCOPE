from scope.data.waymax_adapter import WaymaxAdapter
from scope.data.candidates import generate_ego_candidates

def test_rollout(parsed,cfg):
    c=generate_ego_candidates(parsed,9,cfg)[0]; tr=WaymaxAdapter(cfg).rollout_candidate(parsed,9,0,c.ego_future.numpy(),'replay')
    for k in ['valid','failure_reason','sim_agent_policy','ego_states','agent_states','collisions','tracking_error']:
        assert k in tr
