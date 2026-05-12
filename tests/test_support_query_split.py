from scope.training.support_query import assert_disjoint

def test_disjoint(group):
    for s in group.support_query_splits: assert assert_disjoint(s)
