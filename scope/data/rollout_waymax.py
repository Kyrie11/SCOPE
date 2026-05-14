"""Controlled ego rollout backends for SCOPE dataset construction.

WaymaxBackend uses Waymax when installed and configured. ReactiveRolloutBackend is
a deterministic IDM-style fallback used for unit tests, ablations without WOMD
credentials, and environments where Waymax cannot be imported. Dataset configs
select the backend explicitly.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from scope.data.scene_schema import EgoCandidate, RolloutResult, RootScene
from scope.geometry.boxes import collision_any, swept_min_distance
from scope.geometry.map_utils import infer_route_from_logged_future


@dataclass(frozen=True)
class PolicyVariant:
    policy_variant_id: str
    desired_time_headway: float = 1.5
    min_gap: float = 2.0
    max_accel: float = 2.0
    comfortable_decel: float = 2.0
    yield_bias: float = 0.0


DEFAULT_POLICY_FAMILY = [
    PolicyVariant("neutral_idm", 1.5, 2.0, 2.0, 2.0, 0.0),
    PolicyVariant("conservative_idm", 2.0, 4.0, 1.5, 1.5, 0.6),
    PolicyVariant("assertive_idm", 1.0, 1.0, 2.5, 3.0, -0.5),
]


class RolloutBackend(Protocol):
    def rollout(self, scene: RootScene, candidate: EgoCandidate, policy: PolicyVariant) -> RolloutResult:
        ...


class ReactiveRolloutBackend:
    """Lane-free IDM-style rollout that keeps same-root controlled ego fixed."""

    def rollout(self, scene: RootScene, candidate: EgoCandidate, policy: PolicyVariant) -> RolloutResult:
        agent_futures: dict[int, np.ndarray] = {}
        traces: dict[int, dict[str, Any]] = {}
        try:
            if not candidate.is_feasible:
                return RolloutResult(candidate.candidate_id, policy.policy_variant_id, candidate.future_states, {}, {}, False, "infeasible_ego")
            for agent_idx in scene.relevant_agent_indices:
                root = scene.root_state(agent_idx).copy()
                nominal = scene.agent_future_logged(agent_idx).copy()
                if len(nominal) != len(candidate.future_states):
                    nominal = _constant_velocity_future(root, len(candidate.future_states), scene.dt)
                future, trace = _reactive_agent_future(scene, candidate.future_states, root, nominal, policy)
                agent_futures[agent_idx] = future
                traces[agent_idx] = trace
            valid = not any(_initial_overlap(scene, idx) for idx in scene.relevant_agent_indices)
            return RolloutResult(candidate.candidate_id, policy.policy_variant_id, candidate.future_states, agent_futures, traces, valid, None if valid else "initial_overlap")
        except Exception as exc:
            return RolloutResult(candidate.candidate_id, policy.policy_variant_id, candidate.future_states, agent_futures, traces, False, str(exc))


class WaymaxBackend:
    """Waymax controlled rollout adapter with deterministic fallback for unsupported APIs."""

    def __init__(self, fallback_to_reactive: bool = True):
        self.fallback_to_reactive = fallback_to_reactive
        self.reactive = ReactiveRolloutBackend()
        self._waymax_modules: dict[str, Any] | None = None

    def _modules(self) -> dict[str, Any]:
        if self._waymax_modules is None:
            self._waymax_modules = {
                "env": importlib.import_module("waymax.env"),
                "config": importlib.import_module("waymax.config"),
                "dynamics": importlib.import_module("waymax.dynamics"),
                "datatypes": importlib.import_module("waymax.datatypes"),
            }
        return self._waymax_modules

    def rollout(self, scene: RootScene, candidate: EgoCandidate, policy: PolicyVariant) -> RolloutResult:
        if "waymax_state" not in scene.metadata:
            if self.fallback_to_reactive:
                return self.reactive.rollout(scene, candidate, policy)
            raise RuntimeError("RootScene.metadata lacks original waymax_state required for WaymaxBackend")
        try:
            modules = self._modules()
            env_mod = modules["env"]
            config_mod = modules["config"]
            dynamics_mod = modules["dynamics"]
            datatypes = modules["datatypes"]
            dynamics_model = dynamics_mod.InvertibleBicycleModel()
            env_config = config_mod.EnvironmentConfig()
            waymax_env = env_mod.BaseEnvironment(dynamics_model, env_config)
            state = waymax_env.reset(scene.metadata["waymax_state"])
            # Track ego with inverse bicycle actions. If the local Waymax version
            # exposes different action semantics, fall back deterministically.
            for t in range(min(candidate.future_states.shape[0], int(getattr(state, "remaining_timesteps", candidate.future_states.shape[0])))):
                action = _build_waymax_action(datatypes, state, scene.ego_track_index, candidate.future_states[t])
                state = waymax_env.step(state, action)
            extracted = _extract_final_waymax_rollout(scene, candidate, state, policy.policy_variant_id)
            return extracted
        except Exception:
            if self.fallback_to_reactive:
                return self.reactive.rollout(scene, candidate, policy)
            raise


def _build_waymax_action(datatypes: Any, state: Any, ego_idx: int, target_state: np.ndarray) -> Any:
    max_objects = int(getattr(getattr(state, "sim_trajectory", None), "num_objects", 128)) if hasattr(getattr(state, "sim_trajectory", None), "num_objects") else 128
    data = np.zeros((max_objects, 2), dtype=np.float32)
    valid = np.zeros((max_objects, 1), dtype=bool)
    # InvertibleBicycleModel usually expects acceleration and steering. We use
    # desired speed delta as acceleration proxy and zero steering when API allows.
    current_speed = 0.0
    try:
        current_speed = float(np.asarray(state.sim_trajectory.speed)[ego_idx, int(state.timestep)])
    except Exception:
        pass
    data[ego_idx, 0] = float(target_state[5] - current_speed)
    data[ego_idx, 1] = 0.0
    valid[ego_idx, 0] = True
    return datatypes.Action(data=data, valid=valid)


def _extract_final_waymax_rollout(scene: RootScene, candidate: EgoCandidate, state: Any, policy_id: str) -> RolloutResult:
    # When Waymax produces a SimulatorState, convert available sim_trajectory
    # arrays back into per-agent futures. The fallback path handles versions
    # whose datatypes differ from this generic representation.
    sim = getattr(state, "sim_trajectory", None)
    if sim is None:
        raise ValueError("Waymax state lacks sim_trajectory after rollout")
    x = np.asarray(getattr(sim, "x"), dtype=np.float32)
    y = np.asarray(getattr(sim, "y"), dtype=np.float32)
    yaw = np.asarray(getattr(sim, "yaw", getattr(sim, "heading", np.zeros_like(x))), dtype=np.float32)
    vx = np.asarray(getattr(sim, "vel_x", np.zeros_like(x)), dtype=np.float32)
    vy = np.asarray(getattr(sim, "vel_y", np.zeros_like(x)), dtype=np.float32)
    speed = np.sqrt(vx**2 + vy**2)
    length = np.asarray(getattr(sim, "length", np.full_like(x, 4.8)), dtype=np.float32)
    width = np.asarray(getattr(sim, "width", np.full_like(x, 2.0)), dtype=np.float32)
    valid = np.asarray(getattr(sim, "valid", np.ones_like(x)), dtype=np.float32)
    agent_futures = {}
    start = scene.current_time_index + 1
    end = start + candidate.future_states.shape[0]
    for idx in scene.relevant_agent_indices:
        fut = np.stack([x[idx, start:end], y[idx, start:end], np.zeros(end - start), vx[idx, start:end], vy[idx, start:end], speed[idx, start:end], yaw[idx, start:end], length[idx, start:end], width[idx, start:end], valid[idx, start:end]], axis=-1)
        if len(fut) == len(candidate.future_states):
            agent_futures[idx] = fut.astype(np.float32)
    return RolloutResult(candidate.candidate_id, policy_id, candidate.future_states, agent_futures, {}, True, None)


def _constant_velocity_future(root: np.ndarray, steps: int, dt: float) -> np.ndarray:
    out = np.zeros((steps, len(root)), dtype=np.float32)
    state = root.copy()
    for t in range(steps):
        state = state.copy()
        state[0] += state[3] * dt
        state[1] += state[4] * dt
        out[t] = state
    return out


def _reactive_agent_future(scene: RootScene, ego_future: np.ndarray, root: np.ndarray, nominal: np.ndarray, policy: PolicyVariant) -> tuple[np.ndarray, dict[str, Any]]:
    steps = len(ego_future)
    out = np.zeros_like(nominal[:steps])
    out[0] = nominal[0]
    if not np.isfinite(out[0, 0]):
        out[0] = root
    speed = max(float(root[5]), 0.0)
    yaw = float(root[6])
    pos = root[:2].astype(float).copy()
    trace = {"max_brake": 0.0, "yielding_steps": 0, "policy": policy.policy_variant_id}
    for t in range(steps):
        ego = ego_future[t]
        rel = ego[:2] - pos
        tangent = np.array([np.cos(yaw), np.sin(yaw)])
        longitudinal = float(rel @ tangent)
        lateral = abs(float(rel @ np.array([-tangent[1], tangent[0]])))
        desired_gap = policy.min_gap + policy.desired_time_headway * max(speed, 0.1)
        interaction = lateral < 4.0 and -5.0 < longitudinal < max(30.0, desired_gap + 10.0)
        accel = policy.max_accel * 0.2
        if interaction:
            closing = max(0.0, speed - float(ego[5]))
            gap_error = longitudinal - desired_gap
            brake = policy.comfortable_decel * (1.0 + policy.yield_bias) + closing * 0.2
            if gap_error < 0.0 or policy.yield_bias > 0.2:
                accel = -min(max(brake, 0.5), 5.0)
                trace["yielding_steps"] += 1
        speed = max(0.0, speed + accel * scene.dt)
        if t > 0:
            pos = pos + tangent * speed * scene.dt
        out[t] = root
        out[t, 0:2] = pos
        out[t, 3:5] = tangent * speed
        out[t, 5] = speed
        out[t, 6] = yaw
        out[t, 9] = 1.0
        trace["max_brake"] = max(trace["max_brake"], float(max(0.0, -accel)))
    # Blend toward logged nominal when no interaction to preserve WOMD realism.
    d_min, _ = swept_min_distance(ego_future, nominal[:steps])
    if d_min > 15.0:
        out = 0.8 * nominal[:steps] + 0.2 * out
        out[:, 7:10] = nominal[:steps, 7:10]
    return out.astype(np.float32), trace


def _initial_overlap(scene: RootScene, agent_idx: int) -> bool:
    root_ego = scene.root_state(scene.ego_track_index)[None, :]
    root_agent = scene.root_state(agent_idx)[None, :]
    return collision_any(root_ego, root_agent)


def make_rollout_backend(name: str = "reactive", fallback_to_reactive: bool = True) -> RolloutBackend:
    if name == "waymax":
        return WaymaxBackend(fallback_to_reactive=fallback_to_reactive)
    if name == "reactive":
        return ReactiveRolloutBackend()
    raise ValueError(f"Unknown rollout backend {name}")
