import numpy as np

from scope.data.candidates import CandidateGenerator
from scope.data.labels import label_branch
from scope.data.scene_schema import BRANCH_TO_INDEX
from scope.data.womd_loader import make_synthetic_scene


def test_unaffected_branch_far_agent():
    scene = make_synthetic_scene(2)
    cand = CandidateGenerator({"candidate_count": 4, "planner_proposals": 1}).generate(scene)[0]
    far = scene.agent_future_logged(1).copy()
    far[:, 0] += 100.0
    far[:, 1] += 100.0
    branch, diag = label_branch(scene, cand, 1, far, far)
    assert branch == BRANCH_TO_INDEX["unaffected"]
    assert diag["branch_evidence"]["unaffected"] is True
