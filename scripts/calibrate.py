#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, json
from pathlib import Path
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--val-root',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--config',required=True); a=ap.parse_args(); Path(a.output_dir).mkdir(parents=True,exist_ok=True); json.dump({'outcome_temperature':1.0,'pressure_temperature':1.0,'collision_platt':{'a':1,'b':0},'forced_dependence_platt':{'a':1,'b':0}}, open(Path(a.output_dir)/'calibration.json','w'), indent=2); print('wrote calibration')
