#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
from scope.evaluation.plotting import save_placeholder_plot
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--group-root',required=True); ap.add_argument('--scenario-id',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--config',required=True); a=ap.parse_args(); Path(a.output_dir).mkdir(parents=True,exist_ok=True); save_placeholder_plot(Path(a.output_dir)/f'{a.scenario_id}_surface.png','SCOPE surface'); print('figure written')
