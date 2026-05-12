from typing import Any, Dict, Optional
import numpy as np, torch
from scope.data.schema import OUTCOME, ResponseLabels, SupervisionMasks

def compute_conflict_features(trace, ego_candidate, agent_id, config):
    md=trace['min_distances'].get(agent_id, float('inf')); ttc=trace['min_ttc'].get(agent_id, float('inf'))
    return {'valid': np.isfinite(md) and md < config.get('mining',{}).get('conflict_distance_m',15.0), 'min_distance':md, 'min_ttc':ttc, 'delay': trace['agent_delay'].get(agent_id,0.0), 'arrival_delta': trace['conflict_arrival_delta'].get(agent_id,0.0)}

def derive_outcome(trace, ego_candidate, agent_id, config) -> Optional[int]:
    if not trace.get('valid', False): return None
    cf=compute_conflict_features(trace, ego_candidate, agent_id, config)
    if not cf['valid']: return OUTCOME['unaffected']
    if cf['delay'] > config.get('labels',{}).get('delay_threshold_s',1.0) and not trace['near_collisions'].get(agent_id,False): return OUTCOME['cede']
    if cf['arrival_delta'] < 0.5 and cf['min_distance'] < 6.0: return OUTCOME['maintain']
    if cf['min_distance'] < 8.0 and cf['delay'] > .2: return OUTCOME['follow']
    if cf['min_distance'] > 12.0: return OUTCOME['unaffected']
    return OUTCOME['ambiguous']

def derive_pressure(trace, agent_id, config):
    if not trace.get('valid', False): return None, {}
    l=config.get('labels',{})
    max_decel=trace['max_decel'].get(agent_id,0.0); min_ttc=trace['min_ttc'].get(agent_id,float('inf')); md=trace['min_distances'].get(agent_id,float('inf'))
    delay=trace['agent_delay'].get(agent_id,0.0); pc=trace['priority_consistency'].get(agent_id,1.0); gap=trace['post_encroachment_gap'].get(agent_id,float('inf'))
    ind={
        'hard_brake': max_decel > l.get('hard_brake_threshold_mps2',3.5),
        'late_brake': min_ttc < l.get('ttc_late_threshold_s',1.5) and max_decel > l.get('mild_brake_threshold_mps2',1.5),
        'large_delay': delay > l.get('delay_threshold_s',1.0),
        'near_collision': md < l.get('near_collision_distance_m',2.0) or min_ttc < l.get('near_collision_ttc_s',1.0),
        'priority_inconsistent': pc < l.get('priority_threshold',.5),
        'small_post_encroachment_gap': gap < l.get('post_encroachment_gap_threshold_m',3.0),
        'mild_adaptation': delay > .2 or max_decel > .7,
    }
    p=0
    if ind['mild_adaptation']: p=max(p,1)
    if ind['hard_brake'] or ind['late_brake'] or ind['large_delay']: p=max(p,2)
    if ind['near_collision'] or max_decel > l.get('extreme_brake_threshold_mps2',5.0) or (ind['priority_inconsistent'] and ind['late_brake']) or ind['small_post_encroachment_gap']: p=3
    return p, ind

def estimate_nonceding_risk(group, candidate_id, agent_id, config):
    risks=[]
    for c in group.candidates:
        if c.candidate_id == candidate_id: continue
        if c.masks.outcome and c.labels.outcome != OUTCOME['cede'] and c.masks.safety:
            risks.append(float(bool(c.labels.collision or c.labels.near_collision)))
    return max(risks) if risks else None

def fill_candidate_labels(candidate, trace, agent_id, config, simulator_policy='waymax_reactive'):
    labels=ResponseLabels(); masks=SupervisionMasks()
    candidate.simulator_trace=trace
    reactive=simulator_policy != 'replay'
    if not trace.get('valid',False) or not reactive:
        if simulator_policy == 'replay': masks.factual_trajectory=True
        candidate.labels=labels; candidate.masks=masks; return candidate
    outcome=derive_outcome(trace, candidate.ego_future, agent_id, config); pressure, ind=derive_pressure(trace, agent_id, config)
    if outcome is not None: labels.outcome=int(outcome); masks.outcome=True
    if pressure is not None: labels.pressure_ordinal=int(pressure); labels.pressure_normalized=float(pressure)/3.0; masks.pressure=True
    labels.trajectory=torch.as_tensor(trace['agent_states'][agent_id], dtype=torch.float32); masks.trajectory=True
    labels.collision=bool(trace['collisions'].get(agent_id,False)); labels.near_collision=bool(trace['near_collisions'].get(agent_id,False)); labels.hard_brake=bool(ind.get('hard_brake',False))
    labels.high_pressure=bool(pressure is not None and pressure >= config.get('labels',{}).get('high_pressure_threshold',2)); labels.high_pressure_ceding=bool(labels.outcome==OUTCOME['cede'] and labels.high_pressure)
    masks.safety=masks.collision=masks.hard_brake=masks.high_pressure=masks.sim_response=masks.calibration=True
    candidate.labels=labels; candidate.masks=masks; candidate.diagnostics.update(ind); return candidate

def derive_boundary_pairs(candidates, config):
    pairs=[]; pd=config.get('labels',{}).get('pressure_boundary_delta',1)
    for c in candidates:
        for nb in c.feasible_neighbor_ids:
            if c.candidate_id < nb:
                d=next(x for x in candidates if x.candidate_id==nb)
                valid=c.masks.outcome and d.masks.outcome and c.masks.pressure and d.masks.pressure and c.masks.safety and d.masks.safety
                if not valid: continue
                boundary=(c.labels.outcome!=d.labels.outcome or abs(c.labels.pressure_ordinal-d.labels.pressure_ordinal)>=pd or c.labels.collision!=d.labels.collision or c.labels.near_collision!=d.labels.near_collision)
                pairs.append((c.candidate_id, d.candidate_id, bool(boundary)))
    return pairs
