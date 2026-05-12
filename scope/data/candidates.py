from typing import Any, Dict, List
import numpy as np, torch
from scope.data.schema import CandidateItem
from scope.data.state import StateIndex
from scope.data.feasibility import mark_feasibility

def _recompute_kinematics(traj, dt=0.1):
    x=traj.copy(); xy=x[:,:2]
    if len(x) > 1:
        v=np.gradient(xy, dt, axis=0); a=np.gradient(v, dt, axis=0)
        x[:,2:4]=v; x[:,4:6]=a
        yaw=np.arctan2(v[:,1], v[:,0]); x[:,6]=yaw; x[:,7]=np.gradient(yaw, dt)
    x[:,10]=1.0
    return x

def _time_shift(future, steps):
    out=future.copy(); T=len(out)
    if steps>0:
        out[steps:]=future[:-steps]; out[:steps]=future[0]
    elif steps<0:
        s=-steps; out[:-s]=future[s:]; out[-s:]=future[-1]
    return _recompute_kinematics(out)

def _speed_scale(future, scale, dt=0.1):
    base=future.copy(); p0=base[0,:2].copy(); deltas=np.diff(base[:,:2], axis=0, prepend=base[0:1,:2]) * scale
    xy=p0 + np.cumsum(deltas, axis=0); base[:,:2]=xy
    return _recompute_kinematics(base, dt)

def _lateral_scale(future, scale):
    base=future.copy(); y0=base[0,1]; base[:,1]=y0 + (base[:,1]-y0)*scale
    return _recompute_kinematics(base)

def generate_ego_candidates(parsed_scenario: Dict[str,Any], root_time:int, config:Dict[str,Any]) -> List[CandidateItem]:
    T=config.get('horizon',{}).get('future_steps',80); dt=config.get('horizon',{}).get('dt',0.1); ego=parsed_scenario['ego_id']
    future=parsed_scenario['tracks'][ego, root_time+1:root_time+1+T].copy()
    valid=parsed_scenario['track_valid'][ego, root_time+1:root_time+1+T].copy()
    if len(future)<T:
        pad=np.repeat(future[-1:], T-len(future), axis=0) if len(future) else np.zeros((T,11),np.float32)
        future=np.concatenate([future,pad]); valid=np.concatenate([valid, np.zeros(T-len(valid),bool)])
    specs=[('logged', future), ('delay_0p5', _time_shift(future, int(round(.5/dt)))), ('delay_1p0', _time_shift(future, int(round(1.0/dt)))), ('advance_0p5', _time_shift(future, -int(round(.5/dt)))), ('slowdown_mild', _speed_scale(future,.9,dt)), ('slowdown_strong', _speed_scale(future,.8,dt)), ('speedup_mild', _speed_scale(future,1.1,dt)), ('lateral_soften', _lateral_scale(future,.8)), ('lateral_sharpen', _lateral_scale(future,1.15)), ('gap_shift_early', _time_shift(_speed_scale(future,1.05,dt), -3)), ('gap_shift_late', _time_shift(_speed_scale(future,.9,dt), 3)), ('hold_lane', _lateral_scale(_speed_scale(future,.75,dt),0.0)), ('repair_delay', _time_shift(future,5)), ('repair_slowdown', _speed_scale(future,.7,dt)), ('repair_late_gap', _time_shift(_speed_scale(future,.75,dt),6)), ('repair_lateral_soften', _lateral_scale(_speed_scale(future,.85,dt),.6))]
    out=[]; seen=[]
    for i,(typ,tr) in enumerate(specs[:config.get('data',{}).get('candidates_per_scene',32)]):
        cand=CandidateItem(i, typ, torch.tensor(tr,dtype=torch.float32), torch.tensor(valid,dtype=torch.bool))
        mark_feasibility(cand, config)
        key=np.round(tr[:,:2],2).tobytes(); cand.diagnostics['duplicate']=key in seen; seen.append(key)
        out.append(cand)
    return out
