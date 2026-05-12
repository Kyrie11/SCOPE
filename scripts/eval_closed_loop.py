#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
torch.set_num_threads(1)
import argparse, json
from pathlib import Path
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--scenario-root',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--config',required=True); ap.add_argument('--num-scenarios',type=int,default=1000); a=ap.parse_args(); Path(a.output_dir).mkdir(parents=True,exist_ok=True); json.dump({'route_success':None,'collision_rate':None,'num_scenarios':a.num_scenarios},open(Path(a.output_dir)/'closed_loop_metrics.json','w'),indent=2); print('closed-loop placeholder metrics written')
