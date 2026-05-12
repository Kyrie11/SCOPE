from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import torch

SCHEMA_VERSION = 'scope.v1'
OUTCOME = {'cede':0,'maintain':1,'follow':2,'unaffected':3,'ambiguous':4}
OUTCOME_NAME = {v:k for k,v in OUTCOME.items()}
PRESSURE_CLASS = {'none':0,'mild':1,'high':2,'critical':3}

@dataclass
class ResponseLabels:
    outcome: Optional[int] = None
    pressure_ordinal: Optional[int] = None
    pressure_normalized: Optional[float] = None
    trajectory: Optional[torch.Tensor] = None
    collision: Optional[bool] = None
    near_collision: Optional[bool] = None
    hard_brake: Optional[bool] = None
    high_pressure: Optional[bool] = None
    high_pressure_ceding: Optional[bool] = None
    forced_dependence: Optional[bool] = None
    branch_collision_risk: Optional[torch.Tensor] = None
    boundary_pairs: Optional[List[Tuple[int, int, bool]]] = None

@dataclass
class SupervisionMasks:
    factual_trajectory: bool = False
    sim_response: bool = False
    outcome: bool = False
    pressure: bool = False
    trajectory: bool = False
    safety: bool = False
    collision: bool = False
    hard_brake: bool = False
    high_pressure: bool = False
    forced_dependence: bool = False
    boundary: bool = False
    calibration: bool = False

@dataclass
class CandidateItem:
    candidate_id: int
    candidate_type: str
    ego_future: torch.Tensor
    ego_future_valid: torch.Tensor
    agent_future: Optional[torch.Tensor] = None
    agent_future_valid: Optional[torch.Tensor] = None
    simulator_trace: Optional[Dict[str, Any]] = None
    feasible_neighbor_ids: List[int] = field(default_factory=list)
    labels: ResponseLabels = field(default_factory=ResponseLabels)
    masks: SupervisionMasks = field(default_factory=SupervisionMasks)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SameRootGroup:
    schema_version: str
    scenario_id: str
    root_time_index: int
    ego_id: int
    agent_id: int
    interaction_tags: List[str]
    simulator_policy: str
    history: torch.Tensor
    history_valid: torch.Tensor
    future: Optional[torch.Tensor]
    future_valid: Optional[torch.Tensor]
    map_polylines: torch.Tensor
    map_valid: torch.Tensor
    traffic_lights: Optional[torch.Tensor]
    route: Optional[torch.Tensor]
    candidates: List[CandidateItem]
    support_query_splits: Optional[List[Dict[str, List[int]]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

def tensorize(x, dtype=torch.float32):
    if x is None or isinstance(x, torch.Tensor): return x
    return torch.as_tensor(x, dtype=dtype)

def candidate_valid_for_sim(c: CandidateItem) -> bool:
    return bool(c.masks.sim_response and c.masks.outcome and c.masks.pressure)

def validate_masks(candidate: CandidateItem):
    if candidate.labels.outcome is None and candidate.masks.outcome:
        raise ValueError('outcome mask true but label missing')
    if candidate.labels.pressure_ordinal is None and candidate.masks.pressure:
        raise ValueError('pressure mask true but label missing')
    return True
