"""Configuration loading and recursive overrides."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "defaults" in data:
        merged: dict[str, Any] = {}
        for item in data.pop("defaults") or []:
            sub = load_yaml(Path(path).parent / item)
            merged = deep_update(merged, sub)
        data = deep_update(merged, data)
    return data


def parse_key_value(value: str) -> tuple[list[str], Any]:
    if "=" not in value:
        raise ValueError(f"Override must be key=value, got {value!r}")
    key, raw = value.split("=", 1)
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        parsed = raw
    return key.split("."), parsed


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    out = copy.deepcopy(config)
    for item in overrides or []:
        path, value = parse_key_value(item)
        cursor = out
        for key in path[:-1]:
            cursor = cursor.setdefault(key, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Override path {'.'.join(path)} crosses non-dict key {key}")
        cursor[path[-1]] = value
    return out


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--override", action="append", default=[], help="key=value recursive override")
    return parser


def load_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return apply_overrides(load_yaml(args.config), getattr(args, "override", []))


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
