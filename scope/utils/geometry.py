import math
import numpy as np
import torch

def wrap_angle(a):
    if isinstance(a, torch.Tensor):
        
        out = torch.atan2(torch.sin(a), torch.cos(a))
        return torch.where(torch.isclose(out, torch.tensor(torch.pi, device=out.device, dtype=out.dtype)), -torch.pi * torch.ones_like(out), out)
    
    out = np.arctan2(np.sin(a), np.cos(a))
    return np.where(np.isclose(out, np.pi), -np.pi, out)

def pairwise_dist_xy(a, b):
    return np.linalg.norm(np.asarray(a)[..., :2] - np.asarray(b)[..., :2], axis=-1)

def transform_to_local(points, origin_xy, yaw):
    pts = np.asarray(points)
    c, s = math.cos(-yaw), math.sin(-yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    out = pts.copy()
    out[..., :2] = (pts[..., :2] - np.asarray(origin_xy)) @ R.T
    return out

def transform_to_global(points, origin_xy, yaw):
    pts = np.asarray(points)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    out = pts.copy()
    out[..., :2] = pts[..., :2] @ R.T + np.asarray(origin_xy)
    return out

def boxes_overlap_approx(state_a, state_b, margin=0.0):
    dx = float(state_a[0] - state_b[0]); dy = float(state_a[1] - state_b[1])
    dist = math.hypot(dx, dy)
    ra = 0.5 * math.hypot(float(state_a[8]), float(state_a[9]))
    rb = 0.5 * math.hypot(float(state_b[8]), float(state_b[9]))
    return dist <= (ra + rb + margin)
