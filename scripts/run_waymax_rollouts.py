#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, glob
from pathlib import Path
from scope.utils.io import load_yaml, load_torch_shard, save_torch_shard
from scope.data.rollouts import run_rollouts_for_candidates
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--preprocessed-root',required=True); ap.add_argument('--candidate-root',required=True); ap.add_argument('--output-root',required=True); ap.add_argument('--config',required=True); ap.add_argument('--sim-agent-policy',default='waymax_reactive'); ap.add_argument('--max-groups',type=int); ap.add_argument('--num-workers',type=int,default=1); a=ap.parse_args(); cfg=load_yaml(a.config); parsed={}
    for p in glob.glob(a.preprocessed_root+'/*.pt'):
        for s in load_torch_shard(p)['groups']: parsed[s['scenario_id']]=s
    out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True); n=0
    for cp in sorted(glob.glob(a.candidate_root+'/*.pt')):
        if a.max_groups and n>=a.max_groups: break
        item=load_torch_shard(cp)['groups'][0]; root=item['root']; traces=run_rollouts_for_candidates(parsed[root['scenario_id']],root['root_time_index'],item['candidates'],cfg,a.sim_agent_policy); save_torch_shard([{'root':root,'traces':traces}], out/f'rollouts_{n:06d}.pt', cfg); n+=1
    print(f'wrote {n} rollout groups')
