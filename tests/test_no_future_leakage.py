import torch

from scope.models.scope_model import SCOPEModel


def test_scene_only_forward_does_not_require_labels():
    model = SCOPEModel({"hidden_dim": 64, "intervention_dim": 32, "mechanism_tokens": 4, "support_updater_blocks": 1, "num_attention_heads": 4, "future_steps": 10, "trajectory_modes": 2})
    batch = {
        "scene_features": torch.randn(2, 16),
        "query_ctrl": torch.randn(2, 3, 8),
        "query_anchors": torch.randn(2, 3, 7),
        "query_future_tokens": torch.randn(2, 3, 10, 7),
    }
    out = model.predict_scene_only(batch)
    assert out["branch_logits"].shape == (2, 3, 5)
