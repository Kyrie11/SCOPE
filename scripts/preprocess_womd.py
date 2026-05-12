#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, os
from pathlib import Path
from scope.data.womd_adapter import WOMDAdapter
from scope.utils.io import load_yaml, save_torch_shard

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--split',required=True); ap.add_argument('--output-root',required=True); ap.add_argument('--config',required=True); ap.add_argument('--max-scenarios',type=int); ap.add_argument('--num-workers',type=int,default=1)
    a=ap.parse_args(); cfg=load_yaml(a.config); adapter=WOMDAdapter(a.data_root,a.split,a.max_scenarios); groups=[]; out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    for i,raw in enumerate(adapter.iter_scenarios()):
        parsed=adapter.parse_scenario(raw); save_torch_shard([parsed], out/f'scenario_{i:06d}.pt', cfg)
    print(f'preprocessed scenarios -> {out}')
if __name__=='__main__': main()
