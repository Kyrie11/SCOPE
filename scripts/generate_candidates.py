#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, glob
from pathlib import Path
from scope.utils.io import load_yaml, load_torch_shard, read_jsonl, save_torch_shard
from scope.data.candidates import generate_ego_candidates
from scope.data.neighbors import build_feasible_neighbor_graph
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--preprocessed-root',required=True); ap.add_argument('--root-index',required=True); ap.add_argument('--output-root',required=True); ap.add_argument('--config',required=True); ap.add_argument('--num-workers',type=int,default=1); a=ap.parse_args(); cfg=load_yaml(a.config); parsed={}
    for p in glob.glob(a.preprocessed_root+'/*.pt'):
        for s in load_torch_shard(p)['groups']: parsed[s['scenario_id']]=s
    out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True); n=0
    for root in read_jsonl(a.root_index):
        s=parsed[root['scenario_id']]; c=generate_ego_candidates(s,root['root_time_index'],cfg); build_feasible_neighbor_graph(c,cfg); save_torch_shard([{'root':root,'candidates':c}], out/f'candidates_{n:06d}.pt', cfg); n+=1
    print(f'wrote {n} candidate groups')
