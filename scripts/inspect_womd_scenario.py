#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from scope.utils.io import load_torch_shard
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('shard'); a=ap.parse_args(); p=load_torch_shard(a.shard)['groups'][0]; print({k:(v.shape if hasattr(v,'shape') else type(v).__name__) for k,v in p.items()})
