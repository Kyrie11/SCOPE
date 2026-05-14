from scope.data.candidates import CandidateGenerator
from scope.data.labels import label_burden
from scope.data.womd_loader import make_synthetic_scene


def test_burden_hard_brake_label():
    scene = make_synthetic_scene(3)
    cand = CandidateGenerator({"candidate_count": 4, "planner_proposals": 1}).generate(scene)[0]
    agent = scene.agent_future_logged(1).copy()
    agent[:, 5] = list(reversed([max(0, 8 - t * 0.6) for t in range(len(agent))]))
    # Put the agent close to ego to force unsafe/high burden conditions.
    agent[:, :2] = cand.future_states[:, :2]
    burden = label_burden(scene, cand, agent, 0, {})
    assert burden == 3
