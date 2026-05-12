#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, glob
from scope.utils.io import load_yaml, load_torch_shard, write_jsonl
from scope.data.scenario_mining import mine_roots
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--preprocessed-root',required=True); ap.add_argument('--output',required=True); ap.add_argument('--config',required=True); ap.add_argument('--num-workers',type=int,default=1); a=ap.parse_args(); cfg=load_yaml(a.config); roots=[]
    for p in sorted(glob.glob(a.preprocessed_root+'/*.pt')):
        for parsed in load_torch_shard(p)['groups']: roots+=mine_roots(parsed,cfg)
    write_jsonl(roots,a.output); print(f'wrote {len(roots)} roots to {a.output}')
