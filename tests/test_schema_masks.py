from scope.data.schema import CandidateItem, ResponseLabels, SupervisionMasks, validate_masks
import torch, pytest

def test_missing_label_false_mask():
    c=CandidateItem(0,'x',torch.zeros(2,11),torch.ones(2,dtype=torch.bool)); assert validate_masks(c)
    c.masks.outcome=True
    with pytest.raises(ValueError): validate_masks(c)
