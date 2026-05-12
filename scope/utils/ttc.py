import numpy as np

def min_ttc_pair(ego, agent, eps=1e-3):
    vals=[]
    T=min(len(ego), len(agent))
    for t in range(T):
        p=agent[t,:2]-ego[t,:2]; v=agent[t,2:4]-ego[t,2:4]
        closing=-float(np.dot(p, v))/(np.linalg.norm(p)+eps)
        if closing > eps:
            vals.append(np.linalg.norm(p)/closing)
    return float(min(vals)) if vals else float('inf')
