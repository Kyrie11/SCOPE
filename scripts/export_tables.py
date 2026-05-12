#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from scope.evaluation.table_export import export_table
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--eval-root',required=True); ap.add_argument('--output-root',required=True); ap.add_argument('--format',default='all'); a=ap.parse_args();
    export_table([{'model':'SCOPE_Full','outcome_nll':None,'pressure_mae':None,'risk_brier':None}], f'{a.output_root}/heldout_intervention_prediction')
    export_table([{'model':'SCOPE_Full','support_js':None,'boundary_auroc':None}], f'{a.output_root}/operator_boundary')
    export_table([{'selector':'scope_full','collision':None,'forced_dependence_rate':None,'progress':None}], f'{a.output_root}/offline_candidate_selection')
    export_table([{'planner':'scope_full','success':None,'collision':None,'cvar':None}], f'{a.output_root}/closed_loop_waymax')
    print('tables exported')
