from scope.data.candidates import generate_ego_candidates
from scope.data.rollouts import run_rollouts_for_candidates
from scope.data.labels import fill_candidate_labels, derive_pressure

def test_invalid_no_labels(parsed,cfg):
    c=generate_ego_candidates(parsed,9,cfg)[0]; tr={'valid':False}; fill_candidate_labels(c,tr,1,cfg); assert c.labels.outcome is None and not c.masks.outcome

def test_near_collision_pressure(cfg):
    tr={'valid':True,'max_decel':{1:0},'min_ttc':{1:.5},'min_distances':{1:1.0},'agent_delay':{1:0},'priority_consistency':{1:1},'post_encroachment_gap':{1:10}}
    p,ind=derive_pressure(tr,1,cfg); assert p==3
