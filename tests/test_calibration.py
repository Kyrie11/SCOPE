from scope.training.calibration import TemperatureScaler
import torch

def test_temperature():
    s=TemperatureScaler(); x=torch.ones(2,3); assert s(x).shape==x.shape
