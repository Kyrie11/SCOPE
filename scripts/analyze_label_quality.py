#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from scope.utils.io import load_yaml
from scope.data.quality import analyze_group_root
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--group-root',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--config',required=True); a=ap.parse_args(); stats,w=analyze_group_root(a.group_root,a.output_dir,load_yaml(a.config)); print(stats); [print('WARNING:',x) for x in w]
