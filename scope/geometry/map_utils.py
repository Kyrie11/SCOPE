"""Map and lane helper functions for vectorized WOMD/Waymax scenes."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

try:
    from shapely.geometry import Point, Polygon, LineString
except Exception:  # pragma: no cover
    Point = None
    Polygon = None
    LineString = None

from scope.data.scene_schema import MapFeature, MapFeatureSet
from scope.geometry.conflict_zone import nearest_progress


def infer_route_from_logged_future(future: np.ndarray) -> np.ndarray:
    pts = np.asarray(future, dtype=float)[:, :2]
    if len(pts) < 2:
        return pts
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 0.5:
            keep.append(i)
    if keep[-1] != len(pts) - 1:
        keep.append(len(pts) - 1)
    return pts[keep].astype(np.float32)


def drivable_polygons(map_features: MapFeatureSet) -> list:
    polygons = []
    if Polygon is None:
        return polygons
    for feat in map_features.features:
        if feat.feature_type in {"road_edge", "road_boundary", "driveway", "crosswalk", "drivable_area"} and len(feat.polyline) >= 3:
            try:
                poly = Polygon(feat.polyline[:, :2])
                if poly.is_valid and poly.area > 1e-3:
                    polygons.append(poly)
            except Exception:
                continue
    return polygons


def points_in_drivable(points: np.ndarray, map_features: MapFeatureSet, tolerance_m: float = 3.0) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    lane_lines = [f.polyline[:, :2] for f in map_features.features if f.feature_type in {"lane", "road_line", "lane_center"}]
    polys = drivable_polygons(map_features)
    mask = np.ones(len(pts), dtype=bool) if not lane_lines and not polys else np.zeros(len(pts), dtype=bool)
    if Point is not None:
        for i, p in enumerate(pts[:, :2]):
            point = Point(float(p[0]), float(p[1]))
            if any(poly.buffer(tolerance_m).contains(point) for poly in polys):
                mask[i] = True
                continue
            if LineString is not None and any(LineString(line).distance(point) <= tolerance_m for line in lane_lines if len(line) >= 2):
                mask[i] = True
    else:
        for i, p in enumerate(pts[:, :2]):
            if any(np.min(np.linalg.norm(line - p[:2], axis=-1)) <= tolerance_m for line in lane_lines):
                mask[i] = True
    return mask


def lane_relative_features(route: np.ndarray, future: np.ndarray) -> np.ndarray:
    """Return [lateral_offset, route_progress] per future state."""
    route = np.asarray(route, dtype=float)
    fut = np.asarray(future, dtype=float)
    out = np.zeros((len(fut), 2), dtype=np.float32)
    if len(route) < 2:
        return out
    for t, state in enumerate(fut):
        p = state[:2]
        best_d = float("inf")
        best_lat = 0.0
        best_s = 0.0
        cum = 0.0
        for i in range(len(route) - 1):
            a, b = route[i, :2], route[i + 1, :2]
            ab = b - a
            seg_len = float(np.linalg.norm(ab))
            if seg_len < 1e-9:
                continue
            tangent = ab / seg_len
            normal = np.array([-tangent[1], tangent[0]])
            tau = float(np.clip(((p - a) @ ab) / (seg_len**2), 0.0, 1.0))
            proj = a + tau * ab
            d = float(np.linalg.norm(p - proj))
            if d < best_d:
                best_d = d
                best_lat = float((p - proj) @ normal)
                best_s = cum + tau * seg_len
            cum += seg_len
        out[t] = [best_lat, best_s]
    return out


def target_gap_margin(ego_future: np.ndarray, agent_future: np.ndarray, route: np.ndarray | None = None) -> tuple[float, float]:
    """Approximate target-lane clearance as minimum longitudinal gap and headway."""
    steps = min(len(ego_future), len(agent_future))
    if steps == 0:
        return float("inf"), float("inf")
    if route is not None and len(route) >= 2:
        ego_s = lane_relative_features(route, ego_future[:steps])[:, 1]
        agent_s = lane_relative_features(route, agent_future[:steps])[:, 1]
        gap = np.abs(agent_s - ego_s) - 0.5 * (ego_future[:steps, 7] + agent_future[:steps, 7])
    else:
        rel = agent_future[:steps, :2] - ego_future[:steps, :2]
        heading = ego_future[:steps, 6]
        tangent = np.stack([np.cos(heading), np.sin(heading)], axis=-1)
        gap = np.abs(np.sum(rel * tangent, axis=-1)) - 0.5 * (ego_future[:steps, 7] + agent_future[:steps, 7])
    min_gap = float(np.nanmin(gap))
    rel_speed = np.maximum(ego_future[:steps, 5], 0.1)
    headway = float(np.nanmin(np.maximum(gap, 0.0) / rel_speed))
    return min_gap, headway


def traffic_priority_uncertain(lane_state: int | None) -> bool:
    if lane_state is None:
        return True
    # WOMD traffic signal enum: unknown/stop/caution variants are not decisive go-priority.
    return int(lane_state) in {0, 2, 4, 5, 7, 8}
