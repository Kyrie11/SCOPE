from scope.data.collate import collate_scope

def test_collate(group):
    b=collate_scope([group]); assert b['ego_candidates'].ndim==4 and b['labels']['outcome'].shape[:2]==b['candidate_mask'].shape
    assert (b['masks']['support'] & b['masks']['query']).sum()==0
