from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scope.data.scene_schema import BRANCHES, read_groups_jsonl


def plot_group(g, out_path: Path, max_candidates: int = 12):
    scene = g.root_scene
    fig, ax = plt.subplots(figsize=(9, 9))

    # map
    for mf in scene.map_features.features[:300]:
        p = np.asarray(mf.polyline)
        if p.ndim == 2 and p.shape[0] >= 2:
            ax.plot(p[:, 0], p[:, 1], linewidth=0.4, alpha=0.35)

    # sdc paths / route
    for p in scene.route_context.sdc_paths[:8]:
        p = np.asarray(p)
        if p.ndim == 2 and p.shape[0] >= 2:
            ax.plot(p[:, 0], p[:, 1], linestyle="--", linewidth=1.0, alpha=0.5)

    if scene.route_context.primary_route is not None:
        p = np.asarray(scene.route_context.primary_route)
        ax.plot(p[:, 0], p[:, 1], linewidth=2.0, label="primary route")

    # all agents current positions
    cur = scene.current_time_index
    xy = scene.tracks.xy[:, cur]
    valid = scene.tracks.valid[:, cur]
    ax.scatter(xy[valid, 0], xy[valid, 1], s=8, alpha=0.4, label="agents")

    # relevant agents
    for idx in scene.relevant_agent_indices:
        fut = scene.agent_future_logged(idx)
        ax.plot(fut[:, 0], fut[:, 1], linewidth=1.5)
        ax.text(fut[0, 0], fut[0, 1], f"a{idx}", fontsize=8)

    # ego logged
    ego = scene.ego_future_logged()
    ax.plot(ego[:, 0], ego[:, 1], linewidth=2.5, label="ego logged")

    # candidates
    for c in g.candidate_set[:max_candidates]:
        fs = np.asarray(c.future_states)
        ax.plot(fs[:, 0], fs[:, 1], linewidth=1.0, alpha=0.7)

    ax.set_title(
        f"{g.group_id}\n"
        f"cands={len(g.candidate_set)}, labels={len(g.labels)}, "
        f"route_inferred={scene.route_context.inferred}, "
        f"sdc_paths={len(scene.route_context.sdc_paths)}"
    )
    ax.axis("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    paths = sorted(Path(args.dataset_dir).glob(f"{args.split}_groups.jsonl*"))
    groups = []
    for p in paths:
        groups.extend(read_groups_jsonl(p))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, g in enumerate(random.sample(groups, min(args.num_samples, len(groups)))):
        plot_group(g, out_dir / f"sample_{i:03d}.png")


if __name__ == "__main__":
    main()