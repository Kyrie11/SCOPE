#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
torch.set_num_threads(1)
import argparse, json
from pathlib import Path
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--test-root',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--config',required=True); a=ap.parse_args(); Path(a.output_dir).mkdir(parents=True,exist_ok=True); json.dump({'boundary_auroc':None,'same_regime_smoothness':None},open(Path(a.output_dir)/'boundary_metrics.json','w'),indent=2); print('boundary metrics written')
