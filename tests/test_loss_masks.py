import torch
from scope.training.losses import masked_mean, ordinal_targets

def test_masked_mean():
    assert masked_mean(torch.tensor([1.,100.]), torch.tensor([1,0])).item() < 2

def test_ordinal():
    t=ordinal_targets(torch.tensor([0,2,3])); assert t.tolist()==[[0,0,0],[1,1,0],[1,1,1]]
