def assert_disjoint(split):
    return set(split.get('support_ids',[])).isdisjoint(set(split.get('query_ids',[])))
