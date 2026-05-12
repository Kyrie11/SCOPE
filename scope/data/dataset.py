from pathlib import Path
from typing import Any, Dict, List, Optional
import random, torch
from torch.utils.data import Dataset
from scope.utils.io import list_shards, load_torch_shard, save_torch_shard
from scope.data.schema import SameRootGroup, CandidateItem, SCHEMA_VERSION
from scope.data.labels import estimate_nonceding_risk, derive_boundary_pairs

class SCOPEDataset(Dataset):
    def __init__(self, root: str, mode: str='train_surface', max_groups: Optional[int]=None):
        self.root=Path(root); self.mode=mode; self.groups=[]
        for p in list_shards(root):
            payload=load_torch_shard(p); self.groups.extend(payload.get('groups', []))
            if max_groups and len(self.groups)>=max_groups: break
        if max_groups: self.groups=self.groups[:max_groups]
    def __len__(self): return len(self.groups)
    def __getitem__(self, idx):
        g=self.groups[idx]
        if g.support_query_splits:
            g.metadata=dict(g.metadata); g.metadata['_active_split']=random.choice(g.support_query_splits)
        return g

def build_support_query_splits(candidates: List[CandidateItem], seed:int=0):
    valid=[c.candidate_id for c in candidates if c.masks.outcome or c.masks.trajectory]
    logged=[c.candidate_id for c in candidates if c.candidate_type=='logged']
    rng=random.Random(seed); splits=[]
    splits.append({'support_ids':[], 'query_ids':valid[:]})
    if logged: splits.append({'support_ids':logged[:1], 'query_ids':[i for i in valid if i not in logged[:1]]})
    for _ in range(2):
        ids=valid[:]; rng.shuffle(ids); cut=max(0,min(len(ids)-1, len(ids)//3)); splits.append({'support_ids':ids[:cut], 'query_ids':ids[cut:]})
    return splits

def make_group(parsed, root, agent_id, candidates, config, simulator_policy):
    H=config.get('horizon',{}).get('history_steps',10); T=config.get('horizon',{}).get('future_steps',80); r=root['root_time_index']; ego=root['ego_id']
    hist=torch.as_tensor(parsed['tracks'][:, r-H+1:r+1], dtype=torch.float32).permute(1,0,2)
    hv=torch.as_tensor(parsed['track_valid'][:, r-H+1:r+1], dtype=torch.bool).permute(1,0)
    fut=torch.as_tensor(parsed['tracks'][:, r+1:r+1+T], dtype=torch.float32).permute(1,0,2)
    fv=torch.as_tensor(parsed['track_valid'][:, r+1:r+1+T], dtype=torch.bool).permute(1,0)
    g=SameRootGroup(SCHEMA_VERSION, parsed['scenario_id'], int(r), int(ego), int(agent_id), root.get('interaction_tags_by_agent',{}).get(str(agent_id), root.get('interaction_tags_by_agent',{}).get(agent_id, [])), simulator_policy, hist, hv, fut, fv, torch.as_tensor(parsed['map_polylines'], dtype=torch.float32), torch.as_tensor(parsed['map_valid'], dtype=torch.bool), None if parsed.get('traffic_lights') is None else torch.as_tensor(parsed['traffic_lights']), None if parsed.get('route') is None else torch.as_tensor(parsed['route'], dtype=torch.float32), candidates, None, {'source':'build_intervention_groups'})
    for c in g.candidates:
        risk=estimate_nonceding_risk(g,c.candidate_id,agent_id,config)
        if risk is not None and c.labels.high_pressure_ceding is not None:
            c.labels.forced_dependence=bool(c.labels.high_pressure_ceding and risk>=config.get('labels',{}).get('noncede_risk_threshold',.5) and not (c.labels.collision or c.labels.near_collision))
            c.masks.forced_dependence=True
    boundary=derive_boundary_pairs(g.candidates, config)
    for c in g.candidates: c.labels.boundary_pairs=boundary; c.masks.boundary=bool(boundary)
    g.support_query_splits=build_support_query_splits(g.candidates, seed=int(r)+int(agent_id))
    return g
