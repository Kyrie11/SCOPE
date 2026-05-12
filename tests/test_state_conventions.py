import numpy as np
from scope.data.state import StateIndex, STATE_DIM, make_state, valid_mask
from scope.utils.geometry import transform_to_local, transform_to_global, wrap_angle

def test_state_indexes(): assert int(StateIndex.VALID)==10 and STATE_DIM==11

def test_transforms_roundtrip():
    p=np.array([[1.,2.],[3.,4.]]); q=transform_to_local(p,[1,1],.3); r=transform_to_global(q,[1,1],.3); assert np.allclose(p,r)

def test_yaw_wrap(): assert abs(float(wrap_angle(3*np.pi))+np.pi) < 1e-6

def test_invalid_mask():
    s=make_state(valid=0)[None]; assert not valid_mask(s)[0]
