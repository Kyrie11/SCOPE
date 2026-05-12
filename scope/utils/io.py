import json, os, glob, hashlib, time
from pathlib import Path
import torch, yaml

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f: return yaml.safe_load(f) or {}

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f: json.dump(obj, f, indent=2, ensure_ascii=False)

def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def write_jsonl(items, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in items: f.write(json.dumps(item, ensure_ascii=False) + '\n')

def list_shards(root, suffix='.pt'):
    return sorted(glob.glob(os.path.join(str(root), f'*{suffix}')))

def config_hash(config):
    return hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]

def save_torch_shard(groups, path, config=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {'schema_version':'scope.v1','creation_time':time.time(),'config_hash':config_hash(config or {}),'num_groups':len(groups),'groups':groups}
    torch.save(payload, path)

def load_torch_shard(path):
    return torch.load(path, map_location='cpu', weights_only=False)
