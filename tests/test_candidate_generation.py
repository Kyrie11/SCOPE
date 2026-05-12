from scope.data.candidates import generate_ego_candidates

def test_candidates(parsed,cfg):
    cs=generate_ego_candidates(parsed,9,cfg); types={c.candidate_type for c in cs}
    assert {'logged','delay_0p5','slowdown_mild','repair_delay'} <= types
    assert all('failure_reasons' in c.diagnostics for c in cs)
