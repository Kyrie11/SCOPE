from typing import Dict, List
import numpy as np

def build_feasible_neighbor_graph(candidates, config) -> Dict[int, List[int]]:
    feasible=[c for c in candidates if c.diagnostics.get('feasible', True) and not c.diagnostics.get('duplicate', False)]
    graph={int(c.candidate_id):[] for c in candidates}
    positions={c.candidate_id:c.ego_future.detach().cpu().numpy()[:,:2] for c in feasible}
    ids=list(positions)
    for i,a in enumerate(ids):
        for b in ids[i+1:]:
            d=float(np.mean(np.linalg.norm(positions[a]-positions[b], axis=-1)))
            if d < config.get('candidate',{}).get('neighbor_distance_m', 8.0):
                graph[a].append(b); graph[b].append(a)
    for c in candidates: c.feasible_neighbor_ids=graph.get(c.candidate_id, [])
    return graph
