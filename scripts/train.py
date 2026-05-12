#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
torch.set_num_threads(1)
import argparse
from scope.utils.io import load_yaml
from scope.utils.seed import seed_all
from scope.training.trainer import Trainer
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--model-config',required=True); ap.add_argument('--train-root',required=True); ap.add_argument('--val-root',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--max-train-groups',type=int); ap.add_argument('--max-val-groups',type=int); ap.add_argument('--overfit-debug',action='store_true'); a=ap.parse_args(); seed_all(a.seed); Trainer(load_yaml(a.config),load_yaml(a.model_config),a.train_root,a.val_root,a.output_dir,max_train_groups=a.max_train_groups,max_val_groups=a.max_val_groups).train(a.overfit_debug)
