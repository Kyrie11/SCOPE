from scope.models.baselines import switches_from_config


def test_ablation_switches():
    cfg = {"model": {"use_support_query": False, "use_distillation": False}, "planning": {"use_forced_dependence": False}, "baseline": {"risk_pressure_penalty_only": True}}
    s = switches_from_config(cfg)
    assert not s.use_support_query
    assert not s.use_distillation
    assert not s.use_forced_dependence
    assert s.risk_pressure_penalty_only
