import pytest, torch
from scope.data.womd_adapter import WOMDAdapter
from scope.data.candidates import generate_ego_candidates
from scope.data.neighbors import build_feasible_neighbor_graph
from scope.data.rollouts import run_rollouts_for_candidates
from scope.data.labels import fill_candidate_labels
from scope.data.dataset import make_group
from scope.data.collate import collate_scope
from scope.utils.io import load_yaml

@pytest.fixture
def cfg(): return load_yaml('configs/data/womd_waymax_debug.yaml')
@pytest.fixture
def parsed(cfg): return WOMDAdapter('/tmp/no_such','validation',max_scenarios=1).parse_scenario(next(WOMDAdapter('/tmp/no_such','validation',max_scenarios=1).iter_scenarios()))
@pytest.fixture
def group(parsed,cfg):
    root={'scenario_id':parsed['scenario_id'],'root_time_index':9,'ego_id':0,'relevant_agent_ids':[1],'interaction_tags_by_agent':{1:['car_following_conflict']}}
    cs=generate_ego_candidates(parsed,9,cfg); build_feasible_neighbor_graph(cs,cfg); traces=run_rollouts_for_candidates(parsed,9,cs,cfg,'waymax_reactive')
    cs=[fill_candidate_labels(c,traces[c.candidate_id],1,cfg,'waymax_reactive') for c in cs]
    return make_group(parsed,root,1,cs,cfg,'waymax_reactive')
