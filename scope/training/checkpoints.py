import torch, os
from pathlib import Path

def save_checkpoint(path, model, optimizer=None, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True); torch.save({'model':model.state_dict(),'optimizer':None if optimizer is None else optimizer.state_dict(),'extra':extra or {}}, path)

def load_checkpoint(path, model, map_location='cpu'):
    ck=torch.load(path,map_location=map_location,weights_only=False); model.load_state_dict(ck['model'], strict=False); return ck
