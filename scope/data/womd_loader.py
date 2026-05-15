"""WOMD and Waymax loading utilities.

The loader keeps WOMD global coordinates and converts Waymax SimulatorState or
Waymo Scenario protos into RootScene objects. Waymax is imported lazily so the
rest of the package, tests, and synthetic smoke runs work without a local WOMD
credentialed setup.
"""
from __future__ import annotations

import glob
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from scope.data.scene_schema import (
    AgentTrackTensor,
    LaneStateTensor,
    MapFeature,
    MapFeatureSet,
    RootScene,
    RouteContext,
    make_state,
)
from scope.geometry.map_utils import infer_route_from_logged_future


@dataclass
class LoaderConfig:
    split: str = "train"
    womd_version: str = "1.3.1"
    data_path: str | None = None
    max_num_objects: int = 128
    max_num_rg_points: int = 30000
    history_horizon_s: float = 2.0
    future_horizon_s: float = 8.0
    dt: float = 0.1
    include_sdc_paths: bool = True
    num_paths: int | None = 45
    num_points_per_path: int | None = 800
    allow_synthetic: bool = False
    synthetic_count: int = 4


class WOMDWaymaxLoader:
    """Iterate RootScene objects from Waymax configs, serialized JSON, or synthetic scenes."""

    def __init__(self, cfg: LoaderConfig | dict[str, Any]):
        
        if isinstance(cfg, dict):
            allowed = set(LoaderConfig.__dataclass_fields__.keys())
            self.cfg = LoaderConfig(**{k: v for k, v in cfg.items() if k in allowed})
        else:
            self.cfg = cfg

    def __iter__(self) -> Iterator[RootScene]:
        yield from self.iter_scenes()

    def iter_scenes(self) -> Iterator[RootScene]:
        if self.cfg.data_path:
            path = Path(self.cfg.data_path)
            if path.exists() and path.is_file() and path.suffix in {".json", ".jsonl"}:
                yield from self._iter_json(path)
                return
            if path.exists() and path.is_dir() and list(path.glob("*.json*")):
                for item in sorted(path.glob("*.json*")):
                    yield from self._iter_json(item)
                return
        try:
            yield from self._iter_waymax()
            return
        except Exception as exc:
            if not self.cfg.allow_synthetic:
                raise RuntimeError(
                    "Unable to load WOMD/Waymax scenes. Install Waymax and configure WOMD access, "
                    "or set allow_synthetic=true for smoke tests. Original error: " + str(exc)
                ) from exc
        for idx in range(self.cfg.synthetic_count):
            yield make_synthetic_scene(idx, split=self.cfg.split, dt=self.cfg.dt, future_horizon_s=self.cfg.future_horizon_s)

    def _iter_json(self, path: Path) -> Iterator[RootScene]:
        from scope.data.scene_schema import root_scene_from_dict

        opener = open
        if path.suffix == ".gz":
            import gzip

            opener = gzip.open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if "root_scene" in data:
                        data = data["root_scene"]
                    yield root_scene_from_dict(data)

    def _iter_waymax(self) -> Iterator[RootScene]:
        waymax_config = importlib.import_module("waymax.config")
        dataloader = importlib.import_module("waymax.dataloader")
        import dataclasses

        split_key = {
            "train": "TRAINING",
            "training": "TRAINING",
            "val": "VALIDATION",
            "validation": "VALIDATION",
            "test": "TESTING",
            "testing": "TESTING",
        }.get(self.cfg.split, "TRAINING")

        version_key = self.cfg.womd_version.replace(".", "_")
        cfg_name = f"WOD_{version_key}_{split_key}"

        if hasattr(waymax_config, cfg_name):
            dataset_cfg = getattr(waymax_config, cfg_name)
        else:
            # 兼容旧 Waymax：没有 WOD_1_3_1_* 时手动构造 DatasetConfig
            dataset_cfg = waymax_config.DatasetConfig(
                path=self.cfg.data_path,
                max_num_objects=self.cfg.max_num_objects,
                max_num_rg_points=self.cfg.max_num_rg_points,
                include_sdc_paths=self.cfg.include_sdc_paths,
                num_paths=self.cfg.num_paths,
                num_points_per_path=self.cfg.num_points_per_path,
            )

        kwargs = {
            "max_num_objects": self.cfg.max_num_objects,
            "max_num_rg_points": self.cfg.max_num_rg_points,
            "include_sdc_paths": self.cfg.include_sdc_paths,
        }

        if self.cfg.data_path:
            kwargs["path"] = self.cfg.data_path

        if self.cfg.include_sdc_paths:
            if self.cfg.num_paths is None or self.cfg.num_points_per_path is None:
                raise ValueError(
                    "include_sdc_paths=true requires num_paths and num_points_per_path. "
                    "For WOMD v1.3.1 use num_paths=45 and num_points_per_path=800."
                )
            kwargs["num_paths"] = self.cfg.num_paths
            kwargs["num_points_per_path"] = self.cfg.num_points_per_path

        dataset_cfg = dataclasses.replace(
            dataset_cfg,
            **{k: v for k, v in kwargs.items() if hasattr(dataset_cfg, k)}
        )

        gen = dataloader.simulator_state_generator(config=dataset_cfg)
        for state in gen:
            yield self.simulator_state_to_root_scene(state)

    def simulator_state_to_root_scene(self, state: Any) -> RootScene:
        sim_traj = getattr(state, "sim_trajectory", None) or getattr(state, "log_trajectory", None)
        if sim_traj is None:
            raise ValueError("Waymax SimulatorState has neither sim_trajectory nor log_trajectory")
        arrays = _extract_trajectory_arrays(sim_traj)
        states = _stack_state_arrays(arrays)
        n, t, _ = states.shape
        object_ids, object_types = _extract_metadata(getattr(state, "object_metadata", None), n)
        current_time_index = int(getattr(state, "timestep", getattr(state, "current_time_index", 10)))
        if current_time_index <= 0:
            current_time_index = min(10, t - 1)
        ego_idx = int(_safe_scalar(getattr(state, "sdc_track_index", 0), 0))
        if ego_idx < 0 or ego_idx >= n:
            ego_idx = 0
        timestamps = getattr(state, "timestamps_seconds", None)
        dt = self.cfg.dt
        if timestamps is not None:
            ts = np.asarray(timestamps)
            if ts.size >= 2:
                dt = float(np.median(np.diff(ts)))
        map_features = _extract_map_features(state)
        traffic = _extract_traffic_lights(state)
        route_context = _extract_route_context(state, states[ego_idx], current_time_index, self.cfg.future_horizon_s, dt)
        available_hist = current_time_index * dt
        hist = min(self.cfg.history_horizon_s, available_hist)
        return RootScene(
            scene_id=str(getattr(state, "scenario_id", getattr(state, "scenario_id_str", "waymax_scene"))),
            split=self.cfg.split if self.cfg.split in {"train", "val", "test", "stress"} else "train",
            source="womd",
            womd_version=self.cfg.womd_version,
            current_time_index=current_time_index,
            dt=dt,
            history_horizon_s=hist,
            future_horizon_s=self.cfg.future_horizon_s,
            ego_track_index=ego_idx,
            tracks=AgentTrackTensor(states, object_ids, object_types),
            map_features=map_features,
            traffic_lights=traffic,
            route_context=route_context,
            scenario_tags=[],
            relevant_agent_indices=[],
            metadata={
                "loader": "waymax",
                "history_fallback_applied": hist < self.cfg.history_horizon_s,
                "waymax_state": state,
            },
        )


def _safe_scalar(value: Any, default: Any) -> Any:
    try:
        arr = np.asarray(value)
        if arr.size == 1:
            return arr.reshape(()).item()
    except Exception:
        pass
    return default


def _extract_trajectory_arrays(traj: Any) -> dict[str, np.ndarray]:
    names = ["x", "y", "z", "vel_x", "vel_y", "speed", "yaw", "length", "width", "valid"]
    alt = {"vel_x": ["vx", "velocity_x"], "vel_y": ["vy", "velocity_y"], "yaw": ["heading"], "valid": ["valid"]}
    out: dict[str, np.ndarray] = {}
    for name in names:
        candidates = [name] + alt.get(name, [])
        val = None
        for c in candidates:
            if hasattr(traj, c):
                val = getattr(traj, c)
                break
        if val is not None:
            out[name] = np.asarray(val, dtype=np.float32)
    if "speed" not in out and "vel_x" in out and "vel_y" in out:
        out["speed"] = np.sqrt(out["vel_x"] ** 2 + out["vel_y"] ** 2)
    if "z" not in out and "x" in out:
        out["z"] = np.zeros_like(out["x"])
    if "valid" not in out and "x" in out:
        out["valid"] = np.isfinite(out["x"]).astype(np.float32)
    return out


def _normalize_nt(arr: np.ndarray, n: int | None = None, t: int | None = None) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        if n is not None and a.shape[0] == n:
            a = np.repeat(a[:, None], t or 1, axis=1)
        elif t is not None and a.shape[0] == t:
            a = np.repeat(a[None, :], n or 1, axis=0)
    if a.ndim != 2:
        raise ValueError(f"Expected trajectory array [N,T], got {a.shape}")
    return a


def _stack_state_arrays(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if "x" not in arrays or "y" not in arrays:
        raise ValueError("trajectory arrays require x and y")
    x = np.asarray(arrays["x"], dtype=np.float32)
    y = np.asarray(arrays["y"], dtype=np.float32)
    if x.ndim != 2 and x.ndim >= 3:
        x = np.squeeze(x)
        y = np.squeeze(y)
    n, t = x.shape
    def arr(name: str, fill: float = 0.0) -> np.ndarray:
        if name not in arrays:
            return np.full((n, t), fill, dtype=np.float32)
        return _normalize_nt(arrays[name], n, t)

    z = arr("z")
    vx = arr("vel_x")
    vy = arr("vel_y")
    speed = arr("speed")
    yaw = arr("yaw")
    length = arr("length", 4.8)
    width = arr("width", 2.0)
    valid = arr("valid")
    return np.stack([x, y, z, vx, vy, speed, yaw, length, width, valid], axis=-1).astype(np.float32)


def _extract_metadata(metadata: Any, n: int) -> tuple[list[int], list[str]]:
    ids = list(range(n))
    types = ["vehicle"] * n
    if metadata is None:
        return ids, types
    for attr, target in [("ids", ids), ("object_ids", ids), ("object_types", types), ("types", types)]:
        if hasattr(metadata, attr):
            vals = list(np.asarray(getattr(metadata, attr)).reshape(-1))[:n]
            if "type" in attr:
                types = [str(v) for v in vals] + types[len(vals) :]
            else:
                ids = [int(v) for v in vals] + ids[len(vals) :]
    return ids, types


def _extract_map_features(state: Any) -> MapFeatureSet:
    rg = getattr(state, "roadgraph_points", None)
    features: list[MapFeature] = []
    if rg is not None:
        x = getattr(rg, "x", None)
        y = getattr(rg, "y", None)
        ids = getattr(rg, "ids", None)
        types = getattr(rg, "types", None)
        if x is not None and y is not None:
            pts = np.stack([np.asarray(x), np.asarray(y)], axis=-1).reshape(-1, 2)
            id_arr = np.asarray(ids).reshape(-1) if ids is not None else np.arange(len(pts))
            type_arr = np.asarray(types).reshape(-1) if types is not None else np.zeros(len(pts), dtype=int)
            for fid in np.unique(id_arr):
                mask = id_arr == fid
                if mask.sum() >= 2:
                    ftype = "lane" if int(type_arr[mask][0]) in {1, 2, 3} else "roadgraph"
                    features.append(MapFeature(int(fid), ftype, pts[mask]))
    return MapFeatureSet(features)


def _extract_traffic_lights(state: Any) -> LaneStateTensor:
    tl = getattr(state, "log_traffic_light", getattr(state, "traffic_lights", None))
    if tl is None:
        return LaneStateTensor([], np.zeros((0, 0), dtype=np.int64))
    lane_ids = []
    states = None
    for lane_attr in ["lane_ids", "lane_id", "ids"]:
        if hasattr(tl, lane_attr):
            lane_ids = [int(x) for x in np.asarray(getattr(tl, lane_attr)).reshape(-1)]
            break
    for state_attr in ["state", "states", "lane_states"]:
        if hasattr(tl, state_attr):
            states = np.asarray(getattr(tl, state_attr), dtype=np.int64)
            break
    return LaneStateTensor(lane_ids, states if states is not None else np.zeros((0, 0), dtype=np.int64))


def _extract_route_context(state: Any, ego_track: np.ndarray, current_idx: int, horizon: float, dt: float) -> RouteContext:
    paths = []
    sdc_paths = getattr(state, "sdc_paths", None)
    if sdc_paths is not None:
        for attr in ["xy", "positions", "paths", "x"]:
            if hasattr(sdc_paths, attr):
                raw = np.asarray(getattr(sdc_paths, attr))
                if raw.ndim >= 3:
                    for p in raw.reshape((-1, raw.shape[-2], raw.shape[-1])):
                        if p.shape[-1] >= 2:
                            paths.append(p[:, :2].astype(np.float32))
                break
    start = min(current_idx + 1, len(ego_track))
    steps = int(round(horizon / dt))
    logged = ego_track[start : start + steps]
    if len(logged) >= 2:
        route = infer_route_from_logged_future(logged)
    else:
        route = ego_track[max(0, current_idx - 5) : current_idx + 1, :2]
    return RouteContext(route_polylines=[route], sdc_paths=paths, inferred=not bool(paths))


def make_synthetic_scene(idx: int = 0, split: str = "train", dt: float = 0.1, future_horizon_s: float = 8.0) -> RootScene:
    total = int(round((1.0 + future_horizon_s) / dt)) + 1
    current = 10
    n = 4
    states = np.zeros((n, total, 10), dtype=np.float32)
    for t in range(total):
        tau = (t - current) * dt
        states[0, t] = make_state(tau * 8.0, 0.0, 8.0, 0.0, valid=True)
        states[1, t] = make_state(26.0 - tau * 6.0, -1.5, 6.0, np.pi, valid=True)
        states[2, t] = make_state(tau * 7.0 + 8.0, 3.5, 7.0, 0.0, valid=True)
        states[3, t] = make_state(5.0, -18.0 + tau * 5.0, 5.0, np.pi / 2, valid=True)
    lane0 = np.stack([np.linspace(-20, 80, 30), np.zeros(30)], axis=-1)
    lane1 = np.stack([np.linspace(-20, 80, 30), np.full(30, 3.5)], axis=-1)
    cross = np.stack([np.full(30, 5.0), np.linspace(-30, 40, 30)], axis=-1)
    scene = RootScene(
        scene_id=f"synthetic_{idx:04d}",
        split=split if split in {"train", "val", "test", "stress"} else "train",
        source="synthetic_womd_like",
        womd_version="synthetic",
        current_time_index=current,
        dt=dt,
        history_horizon_s=min(1.0, current * dt),
        future_horizon_s=future_horizon_s,
        ego_track_index=0,
        tracks=AgentTrackTensor(states, [100, 101, 102, 103], ["vehicle"] * n),
        map_features=MapFeatureSet([
            MapFeature("lane0", "lane", lane0),
            MapFeature("lane1", "lane", lane1),
            MapFeature("cross", "lane", cross),
        ]),
        traffic_lights=LaneStateTensor([], np.zeros((0, total), dtype=np.int64)),
        route_context=RouteContext([lane0], inferred=True),
        scenario_tags=["conflict_zone", "close_following"],
        relevant_agent_indices=[1, 2, 3],
        metadata={"synthetic": True},
    )
    return scene
