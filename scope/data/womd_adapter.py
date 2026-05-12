from pathlib import Path
from typing import Any, Dict, Iterator, Optional
import glob, json
import numpy as np
import torch
from scope.data.state import STATE_DIM, StateIndex, make_state

class WOMDAdapter:
    def __init__(self, data_root: str, split: str, max_scenarios: Optional[int]=None, source_format: str='auto'):
        self.data_root=Path(data_root); self.split=split; self.max_scenarios=max_scenarios; self.source_format=source_format

    def iter_scenarios(self) -> Iterator[Any]:
        candidates=[]
        for pat in ['*.npz','*.pt','*.json','*.jsonl']:
            candidates += glob.glob(str(self.data_root / self.split / pat)) + glob.glob(str(self.data_root / pat))
        if not candidates:
            for i in range(self.max_scenarios or 1): yield self._mock_raw(i)
            return
        n=0
        for p in sorted(set(candidates)):
            if self.max_scenarios is not None and n>=self.max_scenarios: break
            if p.endswith('.npz'): yield dict(np.load(p, allow_pickle=True))
            elif p.endswith('.pt'): yield torch.load(p, map_location='cpu', weights_only=False)
            elif p.endswith('.jsonl'):
                with open(p) as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line); n+=1
                            if self.max_scenarios is not None and n>=self.max_scenarios: return
                continue
            else:
                with open(p) as f: yield json.load(f)
            n+=1

    def _mock_raw(self, idx:int) -> Dict[str, Any]:
        T,A=100,5; tracks=np.zeros((A,T,STATE_DIM),np.float32); valid=np.ones((A,T),bool)
        for a in range(A):
            x0=a*8.0; y0=(a%2)*3.5
            for t in range(T):
                tracks[a,t]=make_state(x0+0.8*t, y0, vx=8.0, vy=0.0, yaw=0.0, valid=1.0)
        tracks[1,:,1]=3.5; tracks[1,:,0]=50-0.35*np.arange(T)
        return {'scenario_id':f'mock_{idx}','timestamps':np.arange(T)*0.1,'tracks':tracks,'track_valid':valid,'object_types':np.ones(A,dtype=np.int64),'ego_id':0,'sdc_track_index':0,'map_polylines':np.zeros((8,20,4),np.float32),'map_valid':np.ones((8,20),bool),'traffic_lights':None,'route':np.zeros((20,2),np.float32),'metadata':{'mock':True}}

    def parse_scenario(self, raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict) and all(k in raw for k in ['scenario_id','tracks','track_valid']):
            d=dict(raw)
            d['tracks']=np.asarray(d['tracks'], dtype=np.float32)
            d['track_valid']=np.asarray(d['track_valid'], dtype=bool)
            d['timestamps']=np.asarray(d.get('timestamps', np.arange(d['tracks'].shape[1])*.1), dtype=np.float32)
            d['object_types']=np.asarray(d.get('object_types', np.ones(d['tracks'].shape[0])), dtype=np.int64)
            d['ego_id']=int(d.get('ego_id', d.get('sdc_track_index',0))); d['sdc_track_index']=int(d.get('sdc_track_index', d['ego_id']))
            d['map_polylines']=np.asarray(d.get('map_polylines', np.zeros((1,1,4))), dtype=np.float32)
            d['map_valid']=np.asarray(d.get('map_valid', np.ones(d['map_polylines'].shape[:2])), dtype=bool)
            d.setdefault('traffic_lights', None); d.setdefault('route', None); d.setdefault('metadata', {})
            self._validate_contract(d); return d
        raise ValueError('Unsupported raw scenario. Convert Waymax/WOMD proto externally or provide parsed npz/pt/json.')

    def _validate_contract(self, d):
        required=['scenario_id','timestamps','tracks','track_valid','object_types','ego_id','sdc_track_index','map_polylines','map_valid','traffic_lights','route','metadata']
        for k in required:
            if k not in d: raise KeyError(k)
        if d['tracks'].shape[-1] != STATE_DIM: raise ValueError('tracks last dim must equal STATE_DIM')
