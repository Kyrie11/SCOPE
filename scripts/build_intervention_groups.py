#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, glob
from pathlib import Path
from scope.utils.io import load_yaml, load_torch_shard, save_torch_shard
from scope.data.labels import fill_candidate_labels
from scope.data.dataset import make_group
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--preprocessed-root',required=True); ap.add_argument('--candidate-root',required=True); ap.add_argument('--rollout-root',required=True); ap.add_argument('--output-root',required=True); ap.add_argument('--config',required=True); ap.add_argument('--num-workers',type=int,default=1); a=ap.parse_args(); cfg=load_yaml(a.config); parsed={}
    for p in glob.glob(a.preprocessed_root+'/*.pt'):
        for s in load_torch_shard(p)['groups']: parsed[s['scenario_id']]=s
    out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True); n=0
    cps=sorted(glob.glob(a.candidate_root+'/*.pt')); rps=sorted(glob.glob(a.rollout_root+'/*.pt'))
    for cp,rp in zip(cps,rps):
        cand=load_torch_shard(cp)['groups'][0]; roll=load_torch_shard(rp)['groups'][0]; root=cand['root']; s=parsed[root['scenario_id']]
        for agent_id in root['relevant_agent_ids']:
            cs=[]
            for c in cand['candidates']:
                trace=roll['traces'][c.candidate_id]; cs.append(fill_candidate_labels(c,trace,agent_id,cfg,trace.get('sim_agent_policy','waymax_reactive')))
            g=make_group(s,root,agent_id,cs,cfg,cs[0].simulator_trace.get('sim_agent_policy','waymax_reactive') if cs and cs[0].simulator_trace else 'unknown')
            save_torch_shard([g], out/f'group_{n:06d}.pt', cfg); n+=1
    print(f'wrote {n} same-root groups')
