from scope.data.candidates import generate_ego_candidates
from scope.data.neighbors import build_feasible_neighbor_graph

def test_graph(parsed,cfg):
    cs=generate_ego_candidates(parsed,9,cfg); g=build_feasible_neighbor_graph(cs,cfg)
    for a,bs in g.items():
        for b in bs: assert a in g[b]
