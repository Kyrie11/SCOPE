from scope.data.feasibility import check_candidate_feasible
import numpy as np

def test_speed_limit(cfg):
    x=np.zeros((10,11),dtype='float32'); x[:,2]=100; x[:,10]=1
    ok,reasons=check_candidate_feasible(x,x[:,10]>0,cfg); assert not ok and 'max_speed' in reasons
