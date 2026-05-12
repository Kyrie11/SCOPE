#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
torch.set_num_threads(1)
import argparse, json, torch
from pathlib import Path
from torch.utils.data import DataLoader
from scope.utils.io import load_yaml
from scope.data.dataset import SCOPEDataset
from scope.data.collate import collate_scope
from scope.models.baselines import build_model
from scope.training.checkpoints import load_checkpoint
from scope.evaluation.surface_metrics import compute_surface_metrics
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--test-root',required=True); ap.add_argument('--calibration-dir'); ap.add_argument('--output-dir',required=True); ap.add_argument('--config',required=True); a=ap.parse_args(); ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); cfg=ck.get('extra',{}).get('model_config', {'model_name':'scope_full'}); m=build_model(cfg); m.load_state_dict(ck['model'], strict=False); m.eval(); ds=SCOPEDataset(a.test_root); rows=[]
    for b in DataLoader(ds,batch_size=4,collate_fn=collate_scope):
        with torch.no_grad(): rows.append(compute_surface_metrics(m(b),b))
    agg={k:sum(r.get(k,0) for r in rows)/max(1,sum(k in r for r in rows)) for k in set().union(*[r.keys() for r in rows])} if rows else {}
    Path(a.output_dir).mkdir(parents=True,exist_ok=True); json.dump(agg,open(Path(a.output_dir)/'surface_metrics.json','w'),indent=2); print(agg)
