"""Build same-root intervention-response datasets from WOMD/Waymax roots."""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from scope.data.candidates import CandidateGenerator
from scope.data.labels import labels_from_rollout
from scope.data.neighbor_graph import build_neighbor_graph
from scope.data.rollout_waymax import DEFAULT_POLICY_FAMILY, make_rollout_backend
from scope.data.root_mining import mine_root_scenes
from scope.data.scene_schema import SameRootGroup, write_groups_jsonl
from scope.data.womd_loader import WOMDWaymaxLoader
from scope.utils.config import add_config_args, ensure_dir, load_config_from_args
from scope.utils.logging import get_logger, write_json

LOGGER = get_logger(__name__)


def build_dataset(config: dict, split: str | None = None) -> list[SameRootGroup]:
    data_cfg = dict(config.get("data", {}))
    if split is not None:
        data_cfg["split"] = split
    loader = WOMDWaymaxLoader(data_cfg)
    raw_scenes = list(tqdm(loader.iter_scenes(), desc="load root scenes", total=data_cfg.get("limit_scenes")))
    if data_cfg.get("limit_scenes"):
        raw_scenes = raw_scenes[: int(data_cfg["limit_scenes"])]
    scenes = mine_root_scenes(raw_scenes, top_n_agents=int(data_cfg.get("relevant_agents", 12)))
    generator = CandidateGenerator(config.get("candidates", {}))
    backend_cfg = config.get("rollout", {})
    backend = make_rollout_backend(backend_cfg.get("backend", "reactive"), backend_cfg.get("fallback_to_reactive", True))
    policies = _policies_from_config(backend_cfg.get("policy_family"))
    groups: list[SameRootGroup] = []
    for scene in tqdm(scenes, desc="same-root groups"):
        candidates = generator.generate(scene)
        if not candidates:
            continue
        edges = build_neighbor_graph(candidates, int(data_cfg.get("neighbor_count", 8)))
        group = SameRootGroup(
            group_id=f"{scene.scene_id}:{backend_cfg.get('policy_family_id','default')}:{scene.current_time_index}",
            scene_id=scene.scene_id,
            policy_family_id=backend_cfg.get("policy_family_id", "idm_default"),
            root_scene=scene,
            candidate_set=candidates,
            neighbor_edges=edges,
        )
        for cand_idx, cand in enumerate(candidates):
            if backend_cfg.get("multi_policy", False):
                policy_list = policies
            else:
                policy_list = [policies[cand_idx % len(policies)]]
            nominal_futures = {idx: scene.agent_future_logged(idx) for idx in scene.relevant_agent_indices}
            for policy in policy_list:
                rollout = backend.rollout(scene, cand, policy)
                key = (cand.candidate_id, policy.policy_variant_id)
                group.rollout_matrix[key] = rollout
                labels, masks = labels_from_rollout(scene, cand, rollout, policy.policy_variant_id, nominal_futures)
                group.labels.update(labels)
                group.masks.update(masks)
        group.validate_same_root_integrity()
        groups.append(group)
    return groups


def _policies_from_config(policy_cfg):
    if not policy_cfg:
        return DEFAULT_POLICY_FAMILY
    from scope.data.rollout_waymax import PolicyVariant

    policies = []
    for name, params in policy_cfg.items():
        policies.append(PolicyVariant(policy_variant_id=name, **params))
    return policies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--split", default=None, help="train/val/test override")
    parser.add_argument("--output", default=None, help="Output dataset directory")
    args = parser.parse_args()
    cfg = load_config_from_args(args)
    groups = build_dataset(cfg, args.split)
    out_dir = ensure_dir(args.output or cfg.get("output", {}).get("dataset_dir", "outputs/datasets/scope_womd_waymax"))
    split = args.split or cfg.get("data", {}).get("split", "train")
    out_path = out_dir / f"{split}_groups.jsonl.gz"
    write_groups_jsonl(out_path, groups, compress=True)
    write_json(out_dir / f"{split}_metadata.json", {"groups": len(groups), "split": split})
    LOGGER.info("wrote %d groups to %s", len(groups), out_path)


if __name__ == "__main__":
    main()
