from typing import Dict, List, Tuple
import numpy as np
from scope.data.state import StateIndex
CandidateFailureReason = str

def check_candidate_feasible(ego_future, valid, config) -> Tuple[bool, List[str]]:
    c=config.get('candidate',{}) if config else {}; reasons=[]
    if valid is None: valid=ego_future[:,StateIndex.VALID] > .5
    pts=np.asarray(ego_future); mask=np.asarray(valid).astype(bool)
    if mask.sum() < c.get('min_valid_horizon', 10): reasons.append('insufficient_validity')
    if mask.any():
        sp=np.linalg.norm(pts[mask][:,2:4], axis=-1)
        if sp.max(initial=0) > c.get('max_speed_mps',35.0): reasons.append('max_speed')
        acc=np.linalg.norm(pts[mask][:,4:6], axis=-1)
        if acc.max(initial=0) > c.get('max_accel_mps2',4.0) or pts[mask][:,4].min(initial=0) < c.get('min_accel_mps2',-6.0): reasons.append('max_accel')
        if len(pts[mask]) > 2:
            jerk=np.linalg.norm(np.diff(pts[mask][:,4:6], axis=0)/0.1, axis=-1)
            if jerk.max(initial=0) > c.get('max_jerk_mps3',8.0)*3: reasons.append('max_jerk')
        if np.abs(pts[mask][:,StateIndex.YAW_RATE]).max(initial=0) > c.get('max_yaw_rate_radps',0.8)*2: reasons.append('max_yaw_rate')
        jumps=np.linalg.norm(np.diff(pts[mask][:,:2],axis=0),axis=-1)
        if jumps.max(initial=0) > 8.0: reasons.append('discontinuity')
    return len(reasons)==0, reasons

def mark_feasibility(candidate, config):
    ok, reasons=check_candidate_feasible(candidate.ego_future.detach().cpu().numpy(), candidate.ego_future_valid.detach().cpu().numpy(), config)
    candidate.diagnostics['feasible']=ok; candidate.diagnostics['failure_reasons']=reasons
    return ok
