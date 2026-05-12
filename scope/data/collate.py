from typing import Any, Dict, List
import torch
from scope.data.state import STATE_DIM

def _pad_tensor(xs, shape, dtype=torch.float32):
    out=torch.zeros(shape, dtype=dtype)
    return out

def collate_scope(batch: List[Any]) -> Dict[str, Any]:
    B=len(batch); K=max(len(g.candidates) for g in batch); H=max(g.history.shape[0] for g in batch); A=max(g.history.shape[1] for g in batch); T=max(c.ego_future.shape[0] for g in batch for c in g.candidates); P=max(g.map_polylines.shape[0] for g in batch); Lp=max(g.map_polylines.shape[1] for g in batch); md=max(g.map_polylines.shape[2] for g in batch)
    hist=torch.zeros(B,H,A,STATE_DIM); hv=torch.zeros(B,H,A,dtype=torch.bool); maps=torch.zeros(B,P,Lp,md); mv=torch.zeros(B,P,Lp,dtype=torch.bool)
    ego=torch.zeros(B,K,T,STATE_DIM); ev=torch.zeros(B,K,T,dtype=torch.bool); cm=torch.zeros(B,K,dtype=torch.bool)
    labels={k:torch.zeros(B,K,dtype=torch.float32) for k in ['pressure_normalized','collision','near_collision','hard_brake','high_pressure','high_pressure_ceding','forced_dependence']}
    labels['outcome']=torch.zeros(B,K,dtype=torch.long); labels['pressure_ordinal']=torch.zeros(B,K,dtype=torch.long); labels['trajectory']=torch.zeros(B,K,T,STATE_DIM)
    masks={k:torch.zeros(B,K,dtype=torch.bool) for k in ['outcome','pressure','trajectory','safety','forced_dependence','factual_trajectory','query','support']}
    neighbor_lists=[]; boundary_labels=[]
    support_ids=[]; query_ids=[]; candidate_types=[]
    for b,g in enumerate(batch):
        h,a=g.history.shape[:2]; hist[b,:h,:a]=g.history; hv[b,:h,:a]=g.history_valid.bool(); p,l,dm=g.map_polylines.shape; maps[b,:p,:l,:dm]=g.map_polylines; mv[b,:p,:l]=g.map_valid.bool()
        active=g.metadata.get('_active_split') if hasattr(g,'metadata') else None
        if active is None and g.support_query_splits: active=g.support_query_splits[0]
        supp=set(active.get('support_ids',[]) if active else []); quer=set(active.get('query_ids',[]) if active else [c.candidate_id for c in g.candidates])
        id_to_idx={c.candidate_id:i for i,c in enumerate(g.candidates)}; candidate_types.append([c.candidate_type for c in g.candidates])
        for k,c in enumerate(g.candidates):
            tt=c.ego_future.shape[0]; ego[b,k,:tt]=c.ego_future; ev[b,k,:tt]=c.ego_future_valid.bool(); cm[b,k]=True
            if c.candidate_id in supp: masks['support'][b,k]=True
            if c.candidate_id in quer: masks['query'][b,k]=True
            lab=c.labels; m=c.masks
            if lab.outcome is not None: labels['outcome'][b,k]=int(lab.outcome)
            if lab.pressure_ordinal is not None: labels['pressure_ordinal'][b,k]=int(lab.pressure_ordinal); labels['pressure_normalized'][b,k]=float(lab.pressure_normalized)
            if lab.trajectory is not None: labels['trajectory'][b,k,:lab.trajectory.shape[0]]=lab.trajectory
            for name in ['collision','near_collision','hard_brake','high_pressure','high_pressure_ceding','forced_dependence']:
                v=getattr(lab,name); labels[name][b,k]=0.0 if v is None else float(v)
            masks['outcome'][b,k]=m.outcome; masks['pressure'][b,k]=m.pressure; masks['trajectory'][b,k]=m.trajectory; masks['safety'][b,k]=m.safety; masks['forced_dependence'][b,k]=m.forced_dependence; masks['factual_trajectory'][b,k]=m.factual_trajectory
        pairs=[]; bl=[]
        boundary = g.candidates[0].labels.boundary_pairs if g.candidates and g.candidates[0].labels.boundary_pairs else []
        for a_id,b_id,val in boundary:
            if a_id in id_to_idx and b_id in id_to_idx: pairs.append([id_to_idx[a_id], id_to_idx[b_id]]); bl.append(float(val))
        neighbor_lists.append(pairs); boundary_labels.append(bl); support_ids.append([id_to_idx[i] for i in supp if i in id_to_idx]); query_ids.append([id_to_idx[i] for i in quer if i in id_to_idx])
    E=max([len(x) for x in neighbor_lists]+[1]); neighbor_pairs=torch.zeros(B,E,2,dtype=torch.long); npm=torch.zeros(B,E,dtype=torch.bool); labels['boundary']=torch.zeros(B,E); masks['boundary']=torch.zeros(B,E,dtype=torch.bool)
    for b,pairs in enumerate(neighbor_lists):
        for e,pair in enumerate(pairs): neighbor_pairs[b,e]=torch.tensor(pair); npm[b,e]=True; labels['boundary'][b,e]=boundary_labels[b][e]; masks['boundary'][b,e]=True
    S=max([len(x) for x in support_ids]+[1]); Q=max([len(x) for x in query_ids]+[1]); sid=torch.zeros(B,S,dtype=torch.long); sm=torch.zeros(B,S,dtype=torch.bool); qid=torch.zeros(B,Q,dtype=torch.long); qm=torch.zeros(B,Q,dtype=torch.bool)
    for b,x in enumerate(support_ids):
        if x: sid[b,:len(x)]=torch.tensor(x); sm[b,:len(x)]=True
    for b,x in enumerate(query_ids):
        if x: qid[b,:len(x)]=torch.tensor(x); qm[b,:len(x)]=True
    return {'scenario_id':[g.scenario_id for g in batch], 'root_time_index':torch.tensor([g.root_time_index for g in batch]), 'ego_id':torch.tensor([g.ego_id for g in batch]), 'agent_id':torch.tensor([g.agent_id for g in batch]), 'interaction_tags':[g.interaction_tags for g in batch], 'history':hist, 'history_valid':hv, 'map_polylines':maps, 'map_valid':mv, 'traffic_lights':None, 'route':None, 'ego_candidates':ego, 'ego_candidate_valid':ev, 'candidate_mask':cm, 'candidate_types':candidate_types, 'neighbor_pairs':neighbor_pairs, 'neighbor_pair_mask':npm, 'support_ids':sid, 'support_mask':sm, 'query_ids':qid, 'query_mask':qm, 'labels':labels, 'masks':masks, 'diagnostics':{}}
