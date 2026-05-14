import numpy as np

from scope.data.candidates import CandidateGenerator
from scope.data.neighbor_graph import build_neighbor_graph
from scope.data.rollout_waymax import DEFAULT_POLICY_FAMILY, ReactiveRolloutBackend
from scope.data.labels import labels_from_rollout
from scope.data.scene_schema import SameRootGroup
from scope.data.womd_loader import make_synthetic_scene


def test_same_root_group_integrity():
    scene = make_synthetic_scene(0)
    cands = CandidateGenerator({"candidate_count": 8, "planner_proposals": 2}).generate(scene)
    edges = build_neighbor_graph(cands, 4)
    backend = ReactiveRolloutBackend()
    group = SameRootGroup("g", scene.scene_id, "idm", scene, cands, edges)
    nominal = {idx: scene.agent_future_logged(idx) for idx in scene.relevant_agent_indices}
    for cand in cands[:3]:
        rollout = backend.rollout(scene, cand, DEFAULT_POLICY_FAMILY[0])
        group.rollout_matrix[(cand.candidate_id, DEFAULT_POLICY_FAMILY[0].policy_variant_id)] = rollout
        labels, masks = labels_from_rollout(scene, cand, rollout, DEFAULT_POLICY_FAMILY[0].policy_variant_id, nominal)
        group.labels.update(labels)
        group.masks.update(masks)
    group.validate_same_root_integrity()
    assert group.root_scene.scene_id == group.scene_id
    assert len({tuple(c.future_states[0, :2]) for c in group.candidate_set}) >= 1
