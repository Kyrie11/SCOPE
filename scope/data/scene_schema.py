"""Dataclasses and serialization for same-root SCOPE datasets."""
from __future__ import annotations

import dataclasses
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

BRANCHES = ["cede", "maintain", "follow", "unaffected", "ambiguous"]
BRANCH_TO_INDEX = {name: idx for idx, name in enumerate(BRANCHES)}
INDEX_TO_BRANCH = {idx: name for name, idx in BRANCH_TO_INDEX.items()}
SAFETY_EVENTS = ["collision", "near_collision", "induced_hard_brake", "unsafe_gap"]
STATE_DIM = 10  # x, y, z, vx, vy, speed, yaw, length, width, valid


def as_array(value: Any, dtype: Any = np.float32) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def array_to_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if dataclasses.is_dataclass(value):
        return dataclass_to_dict(value)
    if isinstance(value, dict):
        return {k: array_to_list(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [array_to_list(v) for v in value]
    return value


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    raw = dataclasses.asdict(obj)
    return array_to_list(raw)


@dataclass
class AgentTrackTensor:
    states: np.ndarray  # [N, T, STATE_DIM]
    object_ids: list[int] = field(default_factory=list)
    object_types: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.states = as_array(self.states)
        if self.states.ndim != 3:
            raise ValueError(f"AgentTrackTensor.states must be [N,T,D], got {self.states.shape}")
        n = self.states.shape[0]
        if not self.object_ids:
            self.object_ids = list(range(n))
        if not self.object_types:
            self.object_types = ["vehicle"] * n
        if len(self.object_ids) != n or len(self.object_types) != n:
            raise ValueError("object_ids and object_types must match number of tracks")

    @property
    def valid(self) -> np.ndarray:
        return self.states[..., 9] > 0.5

    @property
    def xy(self) -> np.ndarray:
        return self.states[..., :2]

    @property
    def yaw(self) -> np.ndarray:
        return self.states[..., 6]

    @property
    def length_width(self) -> np.ndarray:
        return self.states[..., 7:9]

    def slice_time(self, start: int, end: int) -> "AgentTrackTensor":
        return AgentTrackTensor(self.states[:, start:end], list(self.object_ids), list(self.object_types))


@dataclass
class MapFeature:
    feature_id: int | str
    feature_type: str
    polyline: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.polyline = as_array(self.polyline)
        if self.polyline.ndim != 2 or self.polyline.shape[1] < 2:
            raise ValueError("MapFeature.polyline must be [P,2+] coordinates")


@dataclass
class MapFeatureSet:
    features: list[MapFeature] = field(default_factory=list)

    def by_type(self, feature_type: str) -> list[MapFeature]:
        return [f for f in self.features if f.feature_type == feature_type]

    def all_polylines(self) -> list[np.ndarray]:
        return [f.polyline for f in self.features]


@dataclass
class LaneStateTensor:
    lane_ids: list[int] = field(default_factory=list)
    states: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int64))

    def __post_init__(self) -> None:
        self.states = np.asarray(self.states, dtype=np.int64)


@dataclass
class RouteContext:
    route_polylines: list[np.ndarray] = field(default_factory=list)
    sdc_paths: list[np.ndarray] = field(default_factory=list)
    inferred: bool = True

    def __post_init__(self) -> None:
        self.route_polylines = [as_array(p) for p in self.route_polylines]
        self.sdc_paths = [as_array(p) for p in self.sdc_paths]

    @property
    def primary_route(self) -> np.ndarray | None:
        if self.route_polylines:
            return self.route_polylines[0]
        if self.sdc_paths:
            return self.sdc_paths[0]
        return None


@dataclass
class RootScene:
    scene_id: str
    split: Literal["train", "val", "test", "stress"]
    source: str
    womd_version: str
    current_time_index: int
    dt: float
    history_horizon_s: float
    future_horizon_s: float
    ego_track_index: int
    tracks: AgentTrackTensor
    map_features: MapFeatureSet = field(default_factory=MapFeatureSet)
    traffic_lights: LaneStateTensor = field(default_factory=LaneStateTensor)
    route_context: RouteContext = field(default_factory=RouteContext)
    scenario_tags: list[str] = field(default_factory=list)
    relevant_agent_indices: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.current_time_index < 0:
            raise ValueError("current_time_index must be non-negative")
        if not (0 <= self.ego_track_index < self.tracks.states.shape[0]):
            raise ValueError("ego_track_index is out of bounds")
        if self.dt <= 0:
            raise ValueError("dt must be positive")

    @property
    def future_steps(self) -> int:
        return int(round(self.future_horizon_s / self.dt))

    @property
    def history_steps(self) -> int:
        return int(round(self.history_horizon_s / self.dt))

    def root_state(self, agent_index: int) -> np.ndarray:
        return self.tracks.states[agent_index, self.current_time_index]

    def ego_future_logged(self) -> np.ndarray:
        start = min(self.current_time_index + 1, self.tracks.states.shape[1])
        end = min(start + self.future_steps, self.tracks.states.shape[1])
        fut = self.tracks.states[self.ego_track_index, start:end]
        if fut.shape[0] < self.future_steps:
            fut = pad_future_from_last(fut, self.future_steps, self.dt)
        return fut

    def agent_future_logged(self, agent_index: int) -> np.ndarray:
        start = min(self.current_time_index + 1, self.tracks.states.shape[1])
        end = min(start + self.future_steps, self.tracks.states.shape[1])
        fut = self.tracks.states[agent_index, start:end]
        if fut.shape[0] < self.future_steps:
            fut = pad_future_from_last(fut, self.future_steps, self.dt)
        return fut


def pad_future_from_last(future: np.ndarray, steps: int, dt: float) -> np.ndarray:
    future = as_array(future)
    if future.shape[0] == steps:
        return future
    if future.shape[0] == 0:
        raise ValueError("Cannot pad an empty future without a root state")
    out = np.zeros((steps, future.shape[1]), dtype=np.float32)
    out[: future.shape[0]] = future
    last = future[-1].copy()
    for t in range(future.shape[0], steps):
        last = last.copy()
        last[0] += last[3] * dt
        last[1] += last[4] * dt
        out[t] = last
    return out


@dataclass
class EgoCandidate:
    candidate_id: str
    family: str
    future_states: np.ndarray
    control_edits: dict[str, Any] = field(default_factory=dict)
    feasibility: dict[str, Any] = field(default_factory=dict)
    task_cost: float = 0.0

    def __post_init__(self) -> None:
        self.future_states = as_array(self.future_states)
        if self.future_states.ndim != 2 or self.future_states.shape[1] < 7:
            raise ValueError("EgoCandidate.future_states must be [T, state_dim>=7]")
        defaults = {
            "delta_t": 0.0,
            "speed_multiplier": 1.0,
            "target_gap_id": None,
            "target_lane_id": None,
            "longitudinal_buffer_m": 0.0,
            "time_headway_s": 1.5,
            "lane_entry_duration_s": 3.0,
            "lateral_midpoint_shift_s": 0.0,
        }
        defaults.update(self.control_edits)
        self.control_edits = defaults
        fdefs = {
            "dynamic_ok": True,
            "drivable_ok": True,
            "route_ok": True,
            "static_collision_free": True,
            "duplicate_of": None,
        }
        fdefs.update(self.feasibility)
        self.feasibility = fdefs

    @property
    def is_feasible(self) -> bool:
        return bool(
            self.feasibility.get("dynamic_ok", False)
            and self.feasibility.get("drivable_ok", False)
            and self.feasibility.get("route_ok", False)
            and self.feasibility.get("static_collision_free", False)
            and self.feasibility.get("duplicate_of") is None
        )


@dataclass
class ResponseLabel:
    branch: int
    burden: int
    agent_future: np.ndarray
    safety: dict[str, bool]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.agent_future = as_array(self.agent_future)
        if self.branch not in INDEX_TO_BRANCH:
            raise ValueError(f"Invalid branch index {self.branch}")
        if self.burden not in (0, 1, 2, 3):
            raise ValueError("burden must be 0,1,2,3")
        for event in SAFETY_EVENTS:
            self.safety.setdefault(event, False)


@dataclass
class LabelMask:
    valid_rollout: bool = True
    branch_valid: bool = True
    burden_valid: bool = True
    trajectory_valid: bool = True
    safety_valid: bool = True
    fd_diag_valid: bool = False
    boundary_valid: bool = False

    def all_response_false(self) -> "LabelMask":
        return LabelMask(False, False, False, False, False, False, False)


@dataclass
class NeighborEdge:
    source: str
    target: str
    edit_type: str
    normalized_distance: float


@dataclass
class RolloutResult:
    candidate_id: str
    policy_variant_id: str
    ego_future: np.ndarray
    agent_futures: dict[int, np.ndarray]
    traces: dict[int, dict[str, Any]] = field(default_factory=dict)
    valid: bool = True
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        self.ego_future = as_array(self.ego_future)
        self.agent_futures = {int(k): as_array(v) for k, v in self.agent_futures.items()}


@dataclass
class SameRootGroup:
    group_id: str
    scene_id: str
    policy_family_id: str
    root_scene: RootScene
    candidate_set: list[EgoCandidate]
    neighbor_edges: list[NeighborEdge]
    rollout_matrix: dict[tuple[str, str], RolloutResult] = field(default_factory=dict)
    labels: dict[tuple[str, int, str], ResponseLabel] = field(default_factory=dict)
    masks: dict[tuple[str, int, str], LabelMask] = field(default_factory=dict)

    def validate_same_root_integrity(self) -> None:
        if not self.candidate_set:
            raise ValueError("SameRootGroup must contain at least one candidate")
        scene_ids = {self.scene_id, self.root_scene.scene_id}
        if len(scene_ids) != 1:
            raise ValueError("group scene_id and root_scene.scene_id differ")
        agent_set = set(self.root_scene.relevant_agent_indices)
        for (_candidate, agent_id, _policy) in self.labels:
            if agent_id not in agent_set:
                raise ValueError(f"label agent {agent_id} not in relevant_agent_indices")
        ids = [c.candidate_id for c in self.candidate_set]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate_ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "scene_id": self.scene_id,
            "policy_family_id": self.policy_family_id,
            "root_scene": root_scene_to_dict(self.root_scene),
            "candidate_set": [dataclass_to_dict(c) for c in self.candidate_set],
            "neighbor_edges": [dataclass_to_dict(e) for e in self.neighbor_edges],
            "rollout_matrix": [
                {"candidate_id": k[0], "policy_variant_id": k[1], "result": dataclass_to_dict(v)}
                for k, v in self.rollout_matrix.items()
            ],
            "labels": [
                {"candidate_id": k[0], "agent_id": k[1], "policy_variant_id": k[2], "label": dataclass_to_dict(v)}
                for k, v in self.labels.items()
            ],
            "masks": [
                {"candidate_id": k[0], "agent_id": k[1], "policy_variant_id": k[2], "mask": dataclass_to_dict(v)}
                for k, v in self.masks.items()
            ],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "SameRootGroup":
        root = root_scene_from_dict(data["root_scene"])
        candidates = [EgoCandidate(**c) for c in data["candidate_set"]]
        edges = [NeighborEdge(**e) for e in data.get("neighbor_edges", [])]
        rollouts = {}
        for item in data.get("rollout_matrix", []):
            result = item["result"]
            rollouts[(item["candidate_id"], item["policy_variant_id"])] = RolloutResult(
                candidate_id=result["candidate_id"],
                policy_variant_id=result["policy_variant_id"],
                ego_future=result["ego_future"],
                agent_futures={int(k): v for k, v in result["agent_futures"].items()},
                traces={int(k): v for k, v in result.get("traces", {}).items()},
                valid=result.get("valid", True),
                invalid_reason=result.get("invalid_reason"),
            )
        labels = {}
        for item in data.get("labels", []):
            key = (item["candidate_id"], int(item["agent_id"]), item["policy_variant_id"])
            labels[key] = ResponseLabel(**item["label"])
        masks = {}
        for item in data.get("masks", []):
            key = (item["candidate_id"], int(item["agent_id"]), item["policy_variant_id"])
            masks[key] = LabelMask(**item["mask"])
        group = SameRootGroup(
            group_id=data["group_id"],
            scene_id=data["scene_id"],
            policy_family_id=data["policy_family_id"],
            root_scene=root,
            candidate_set=candidates,
            neighbor_edges=edges,
            rollout_matrix=rollouts,
            labels=labels,
            masks=masks,
        )
        group.validate_same_root_integrity()
        return group


def root_scene_to_dict(scene: RootScene) -> dict[str, Any]:
    return {
        "scene_id": scene.scene_id,
        "split": scene.split,
        "source": scene.source,
        "womd_version": scene.womd_version,
        "current_time_index": scene.current_time_index,
        "dt": scene.dt,
        "history_horizon_s": scene.history_horizon_s,
        "future_horizon_s": scene.future_horizon_s,
        "ego_track_index": scene.ego_track_index,
        "tracks": dataclass_to_dict(scene.tracks),
        "map_features": {"features": [dataclass_to_dict(f) for f in scene.map_features.features]},
        "traffic_lights": dataclass_to_dict(scene.traffic_lights),
        "route_context": dataclass_to_dict(scene.route_context),
        "scenario_tags": list(scene.scenario_tags),
        "relevant_agent_indices": list(scene.relevant_agent_indices),
        "metadata": scene.metadata,
    }


def root_scene_from_dict(data: dict[str, Any]) -> RootScene:
    tracks = AgentTrackTensor(**data["tracks"])
    features = [MapFeature(**f) for f in data.get("map_features", {}).get("features", [])]
    return RootScene(
        scene_id=data["scene_id"],
        split=data["split"],
        source=data["source"],
        womd_version=data.get("womd_version", "unknown"),
        current_time_index=int(data["current_time_index"]),
        dt=float(data["dt"]),
        history_horizon_s=float(data["history_horizon_s"]),
        future_horizon_s=float(data["future_horizon_s"]),
        ego_track_index=int(data["ego_track_index"]),
        tracks=tracks,
        map_features=MapFeatureSet(features),
        traffic_lights=LaneStateTensor(**data.get("traffic_lights", {})),
        route_context=RouteContext(**data.get("route_context", {})),
        scenario_tags=list(data.get("scenario_tags", [])),
        relevant_agent_indices=[int(x) for x in data.get("relevant_agent_indices", [])],
        metadata=data.get("metadata", {}),
    )


def write_groups_jsonl(path: str | Path, groups: list[SameRootGroup], compress: bool | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    use_gzip = p.suffix == ".gz" if compress is None else compress
    opener = gzip.open if use_gzip else open
    mode = "wt" if use_gzip else "w"
    with opener(p, mode, encoding="utf-8") as f:
        for group in groups:
            group.validate_same_root_integrity()
            f.write(json.dumps(group.to_dict(), ensure_ascii=False) + "\n")


def read_groups_jsonl(path: str | Path) -> list[SameRootGroup]:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    groups = []
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                groups.append(SameRootGroup.from_dict(json.loads(line)))
    return groups


def make_state(x: float, y: float, speed: float, yaw: float, length: float = 4.8, width: float = 2.0, valid: bool = True) -> np.ndarray:
    return np.array([x, y, 0.0, speed * np.cos(yaw), speed * np.sin(yaw), speed, yaw, length, width, float(valid)], dtype=np.float32)
