import torch

from scope.training.support_query import SupportQuerySampler


def test_support_query_no_overlap():
    batch = {
        "mask": torch.ones(2, 10),
        "query_ctrl": torch.randn(2, 10, 8),
        "query_anchors": torch.randn(2, 10, 7),
        "query_future_tokens": torch.randn(2, 10, 80, 7),
        "branch": torch.zeros(2, 10, dtype=torch.long),
        "burden": torch.zeros(2, 10, dtype=torch.long),
        "safety": torch.zeros(2, 10, 4),
    }
    sampler = SupportQuerySampler({"support_sizes": (2,), "prob_empty_support": 0.0, "seed": 5})
    support, query = sampler.split_batch(batch)
    assert support is not None
    assert support["masks"]["valid"].sum().item() == 4
    assert query["mask"].sum().item() == 16
