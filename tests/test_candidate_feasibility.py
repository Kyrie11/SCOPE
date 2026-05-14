from scope.data.candidates import CandidateGenerator
from scope.data.womd_loader import make_synthetic_scene


def test_candidate_feasibility_grid_contains_families():
    scene = make_synthetic_scene(1)
    cands = CandidateGenerator({"candidate_count": 32, "planner_proposals": 3}).generate(scene)
    families = {c.family for c in cands}
    assert "logged" in families
    assert "timing" in families or "speed" in families
    assert all(c.feasibility["dynamic_ok"] for c in cands)
    assert len(cands) <= 32
