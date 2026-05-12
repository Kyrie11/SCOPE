from collections import Counter
import json, os
from pathlib import Path
from scope.data.dataset import SCOPEDataset

def analyze_group_root(group_root, output_dir, config=None):
    ds=SCOPEDataset(group_root); stats=Counter(); warnings=[]
    for g in ds.groups:
        stats['groups']+=1; stats['candidates']+=len(g.candidates); stats[f'tag:{g.interaction_tags[0] if g.interaction_tags else "none"}']+=1
        for c in g.candidates:
            stats[f'cand:{c.candidate_type}']+=1
            if c.masks.outcome: stats[f'outcome:{c.labels.outcome}']+=1
            if c.masks.pressure: stats[f'pressure:{c.labels.pressure_ordinal}']+=1
            if c.masks.forced_dependence: stats['fd_mask']+=1
            if c.labels.high_pressure_ceding: stats['high_pressure_ceding']+=1
            if c.diagnostics.get('failure_reasons'):
                for r in c.diagnostics['failure_reasons']: stats[f'fail:{r}']+=1
    if stats['groups']==0: warnings.append('No same-root groups found.')
    if stats['fd_mask'] < max(1, stats['candidates']*.01): warnings.append('Forced-dependence labels are mostly masked.')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir)/'quality_summary.json','w') as f: json.dump(dict(stats),f,indent=2)
    with open(Path(output_dir)/'quality_warnings.json','w') as f: json.dump(warnings,f,indent=2)
    with open(Path(output_dir)/'quality_report.md','w') as f:
        f.write('# SCOPE label-quality report\n\n');
        for k,v in sorted(stats.items()): f.write(f'- {k}: {v}\n')
        if warnings: f.write('\n## Warnings\n'+'\n'.join(f'- {w}' for w in warnings))
    return dict(stats), warnings
