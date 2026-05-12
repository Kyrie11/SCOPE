import math
from typing import Any, Dict, List
import numpy as np
from scope.data.state import StateIndex

def _dist(a,b): return float(np.linalg.norm(np.asarray(a[:2])-np.asarray(b[:2])))

def mine_relevant_agents(parsed_scenario: Dict[str, Any], root_time: int, ego_id: int, max_agents: int=16) -> List[int]:
    tracks=parsed_scenario['tracks']; valid=parsed_scenario['track_valid']; A=tracks.shape[0]
    if not valid[ego_id, root_time]: return []
    ego=tracks[ego_id, root_time]
    scored=[]
    for a in range(A):
        if a==ego_id or not valid[a, root_time]: continue
        d=_dist(ego, tracks[a,root_time]);
        if d < 80.0:
            rel_speed=np.linalg.norm(tracks[a,root_time,2:4]-ego[2:4])
            score=d - 0.5*rel_speed
            scored.append((score,a))
    return [a for _,a in sorted(scored)[:max_agents]]

def assign_interaction_tags(parsed_scenario, root_time:int, ego_id:int, agent_id:int):
    e=parsed_scenario['tracks'][ego_id,root_time]; a=parsed_scenario['tracks'][agent_id,root_time]
    dx=a[0]-e[0]; dy=abs(a[1]-e[1]); tags=[]
    if dy < 2.5 and abs(dx) < 30: tags.append('car_following_conflict')
    if 2.5 <= dy < 6.0 and abs(dx) < 35: tags.append('gap_insertion')
    if abs(dx) < 20 and dy > 2.0: tags.append('ambiguous_priority')
    if not tags: tags.append('intersection_yield' if np.linalg.norm(a[:2]-e[:2]) < 25 else 'dense_lane_change')
    return tags

def mine_roots(parsed_scenario, config):
    H=config.get('horizon',{}).get('history_steps',10); T=config.get('horizon',{}).get('future_steps',80)
    max_roots=config.get('mining',{}).get('max_roots_per_scenario',4); ego=int(parsed_scenario['ego_id'])
    roots=[]; total=parsed_scenario['tracks'].shape[1]
    candidates=list(range(H-1, max(H, total-T-1), max(1,(total-H-T)//max_roots or 1)))[:max_roots]
    for r in candidates:
        agents=mine_relevant_agents(parsed_scenario,r,ego,config.get('data',{}).get('max_relevant_agents',8))
        roots.append({'scenario_id':parsed_scenario['scenario_id'],'root_time_index':int(r),'ego_id':ego,'relevant_agent_ids':agents,'interaction_tags_by_agent':{int(a):assign_interaction_tags(parsed_scenario,r,ego,a) for a in agents},'quality_flags':{}})
    return roots
