from enum import IntEnum
import math
import numpy as np
import torch

class StateIndex(IntEnum):
    X = 0; Y = 1; VX = 2; VY = 3; AX = 4; AY = 5; YAW = 6; YAW_RATE = 7; LENGTH = 8; WIDTH = 9; VALID = 10
STATE_DIM = 11

def validate_state_array(x):
    if x.shape[-1] != STATE_DIM:
        raise ValueError(f"Expected last dim {STATE_DIM}, got {x.shape[-1]}")
    return x

def valid_mask(states):
    validate_state_array(states)
    return states[..., StateIndex.VALID] > 0.5

def wrap_angle(a):
    if isinstance(a, torch.Tensor):
        
        out = torch.atan2(torch.sin(a), torch.cos(a))
        return torch.where(torch.isclose(out, torch.tensor(torch.pi, device=out.device, dtype=out.dtype)), -torch.pi * torch.ones_like(out), out)
    
    out = np.arctan2(np.sin(a), np.cos(a))
    return np.where(np.isclose(out, np.pi), -np.pi, out)

def speed(states):
    return torch.linalg.norm(states[..., [StateIndex.VX, StateIndex.VY]], dim=-1) if isinstance(states, torch.Tensor) else np.linalg.norm(states[..., [StateIndex.VX, StateIndex.VY]], axis=-1)

def make_state(x=0.0,y=0.0,vx=0.0,vy=0.0,ax=0.0,ay=0.0,yaw=0.0,yaw_rate=0.0,length=4.5,width=1.9,valid=1.0, dtype=np.float32):
    return np.array([x,y,vx,vy,ax,ay,yaw,yaw_rate,length,width,valid], dtype=dtype)
