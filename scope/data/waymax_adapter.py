from typing import Any, Dict
import numpy as np
from scope.data.state import STATE_DIM, StateIndex
from scope.utils.collision import min_distance_traj, collision_any
from scope.utils.ttc import min_ttc_pair

class WaymaxAdapter:
    def __init__(self, config: Dict[str, Any]): self.config=config or {}
    def make_env(self, parsed_scenario: Dict[str, Any]) -> Any: return {'parsed_scenario': parsed_scenario}
    def rollout_candidate(self, parsed_scenario, root_time:int, ego_id:int, ego_future, sim_agent_policy:str='waymax_reactive') -> Dict[str, Any]:
        # Isolated fallback simulator. If real Waymax is installed, replace only this method.
        T=int(self.config.get('waymax',{}).get('rollout_steps', len(ego_future)))
        tracks=parsed_scenario['tracks']; valid=parsed_scenario['track_valid']; A=tracks.shape[0]
        ego_states=np.asarray(ego_future[:T], dtype=np.float32).copy(); ego_valid=ego_states[:,StateIndex.VALID]>.5
        agent_states=np.zeros((A,T,STATE_DIM),np.float32); agent_valid=np.zeros((A,T),bool)
        for a in range(A):
            f=tracks[a, root_time+1:root_time+1+T].copy(); v=valid[a, root_time+1:root_time+1+T].copy()
            if len(f)<T:
                pad=np.repeat(f[-1:], T-len(f), axis=0) if len(f) else np.zeros((T,STATE_DIM),np.float32); f=np.concatenate([f,pad]); v=np.concatenate([v,np.zeros(T-len(v),bool)])
            if sim_agent_policy in ('waymax_reactive','idm_reactive') and a != ego_id:
                # crude reactive ceding: slow down if close in front/lateral conflict
                d=np.linalg.norm(f[:,:2]-ego_states[:len(f),:2],axis=-1); close=d<8.0
                if close.any():
                    first=int(np.argmax(close)); f[first:,0]-=np.linspace(0,3.0,T-first)
                    f[first:,2]*=.75; f[first:,4]-=1.0
            agent_states[a]=f; agent_valid[a]=v
        collisions={}; near={}; mind={}; ttc={}; max_decel={}; max_jerk={}; delay={}; pri={}; peg={}; cad={}
        for a in range(A):
            if a==ego_id: continue
            md=min_distance_traj(ego_states, agent_states[a], ego_valid, agent_valid[a]); mind[a]=md; near[a]=md<2.0; collisions[a]=collision_any(ego_states, agent_states[a]); ttc[a]=min_ttc_pair(ego_states, agent_states[a])
            ax=agent_states[a,:,4]; max_decel[a]=float(max(0.0, -np.min(ax))) if len(ax) else 0.0; max_jerk[a]=float(np.max(np.abs(np.diff(ax)/.1))) if len(ax)>1 else 0.0
            delay[a]=float(max(0.0, parsed_scenario['tracks'][a,min(root_time+T,tracks.shape[1]-1),0]-agent_states[a,-1,0]))
            pri[a]=float(1.0 if cad.get(a,0)>=0 else 0.0); peg[a]=float(md); cad[a]=float(np.argmin(np.linalg.norm(agent_states[a,:,:2]-ego_states[:,:2],axis=-1))*0.1)
        tracking=float(np.max(np.linalg.norm(ego_states[:,:2]-np.asarray(ego_future[:T])[:,:2],axis=-1)))
        valid_rollout=tracking <= self.config.get('waymax',{}).get('tracking_error_threshold_m',1.0)
        if sim_agent_policy == 'replay': valid_rollout=True
        return {'valid':bool(valid_rollout),'failure_reason':None if valid_rollout else 'tracking_error','sim_agent_policy':sim_agent_policy,'ego_states':ego_states,'ego_valid':ego_valid,'agent_states':agent_states,'agent_valid':agent_valid,'collisions':collisions,'near_collisions':near,'min_distances':mind,'min_ttc':ttc,'max_decel':max_decel,'max_jerk':max_jerk,'agent_delay':delay,'priority_consistency':pri,'post_encroachment_gap':peg,'conflict_arrival_delta':cad,'tracking_error':tracking,'tracking_yaw_error':0.0,'metadata':{'fallback_simulator':True}}
