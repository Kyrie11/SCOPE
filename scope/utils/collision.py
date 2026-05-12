import numpy as np
from scope.data.state import StateIndex
from scope.utils.geometry import boxes_overlap_approx

def min_distance_traj(a, b, valid_a=None, valid_b=None):
    a=np.asarray(a); b=np.asarray(b); T=min(len(a), len(b)); vals=[]
    for t in range(T):
        if valid_a is not None and not valid_a[t]: continue
        if valid_b is not None and not valid_b[t]: continue
        vals.append(np.linalg.norm(a[t,:2]-b[t,:2]))
    return float(min(vals)) if vals else float('inf')

def collision_any(a,b,margin=0.0):
    a=np.asarray(a); b=np.asarray(b); T=min(len(a),len(b))
    for t in range(T):
        if a[t,StateIndex.VALID] > .5 and b[t,StateIndex.VALID] > .5 and boxes_overlap_approx(a[t], b[t], margin): return True
    return False
