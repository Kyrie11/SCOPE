# SCOPE: Surface-Constrained Operator Planning

This repository implements **SCOPE: Surface-Constrained Operator Planning for Non-Coercive Interactive Driving** with PyTorch, WOMD-style parsed scenarios, and an isolated Waymax adapter. The code keeps the response operator shared across same-root interventions, uses mask-aware losses, stores auditable simulator diagnostics, and includes runnable mock/fallback paths for tests and debug reproduction.

## 1. Environment setup

```bash
conda env create -f environment.yml
conda activate scope
pip install -e .
```

Or:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Optional real simulator integration:

```bash
pip install waymax waymo-open-dataset-tf-2-12-0
```

The fallback simulator in `scope/data/waymax_adapter.py` is intentionally isolated. Replace only `WaymaxAdapter.rollout_candidate` when binding to a local Waymax version.

## 2. WOMD and Waymax data assumptions

`WOMDAdapter` consumes already parsed `.npz`, `.pt`, `.json`, or `.jsonl` records with this contract:

```text
scenario_id, timestamps, tracks[A,T,11], track_valid[A,T], object_types[A], ego_id,
sdc_track_index, map_polylines[P,Lp,map_dim], map_valid[P,Lp], traffic_lights, route, metadata
```

If no files are found, the adapter produces deterministic mock scenarios so the debug pipeline and tests can run before raw WOMD is mounted.

## 3. End-to-end quick debug run

```bash
# 1. Preprocess a tiny validation subset
python scripts/preprocess_womd.py \
  --data-root $WOMD_ROOT \
  --split validation \
  --output-root data/debug/womd_preprocessed/validation \
  --config configs/data/womd_waymax_debug.yaml \
  --max-scenarios 32 \
  --num-workers 1

# 2. Mine root scenes
python scripts/mine_root_scenes.py \
  --preprocessed-root data/debug/womd_preprocessed/validation \
  --output data/debug/root_index/validation_roots.jsonl \
  --config configs/data/womd_waymax_debug.yaml \
  --num-workers 1

# 3. Generate candidates
python scripts/generate_candidates.py \
  --preprocessed-root data/debug/womd_preprocessed/validation \
  --root-index data/debug/root_index/validation_roots.jsonl \
  --output-root data/debug/candidates/validation \
  --config configs/data/womd_waymax_debug.yaml \
  --num-workers 1

# 4. Run Waymax rollouts
python scripts/run_waymax_rollouts.py \
  --preprocessed-root data/debug/womd_preprocessed/validation \
  --candidate-root data/debug/candidates/validation \
  --output-root data/debug/rollouts/validation \
  --config configs/data/womd_waymax_debug.yaml \
  --sim-agent-policy waymax_reactive \
  --num-workers 1

# 5. Build same-root groups
python scripts/build_intervention_groups.py \
  --preprocessed-root data/debug/womd_preprocessed/validation \
  --candidate-root data/debug/candidates/validation \
  --rollout-root data/debug/rollouts/validation \
  --output-root data/debug/groups/validation \
  --config configs/data/womd_waymax_debug.yaml \
  --num-workers 1

# 6. Analyze label quality
python scripts/analyze_label_quality.py \
  --group-root data/debug/groups/validation \
  --output-dir outputs/debug/quality_validation \
  --config configs/data/womd_waymax_debug.yaml

# 7. Overfit SCOPE on the tiny subset
python scripts/train.py \
  --config configs/train/train_surface.yaml \
  --model-config configs/model/scope_full.yaml \
  --train-root data/debug/groups/validation \
  --val-root data/debug/groups/validation \
  --output-dir outputs/debug/scope_overfit \
  --max-train-groups 64 \
  --max-val-groups 64 \
  --overfit-debug \
  --seed 42

# 8. Evaluate surface metrics
python scripts/eval_surface.py \
  --checkpoint outputs/debug/scope_overfit/best.ckpt \
  --test-root data/debug/groups/validation \
  --output-dir outputs/debug/eval_surface \
  --config configs/eval/eval_surface.yaml

# 9. Run unit tests
pytest tests -q
```

## 4. Full data preprocessing

```bash
python scripts/preprocess_womd.py \
  --data-root $WOMD_ROOT \
  --split training \
  --output-root data/cache/womd_preprocessed/training \
  --config configs/data/womd_waymax.yaml \
  --num-workers 8
```

Debug command:

```bash
python scripts/preprocess_womd.py \
  --data-root $WOMD_ROOT \
  --split validation \
  --output-root data/debug/womd_preprocessed/validation \
  --config configs/data/womd_waymax_debug.yaml \
  --max-scenarios 32 \
  --num-workers 1
```

## 5. Root-scene mining

```bash
python scripts/mine_root_scenes.py \
  --preprocessed-root data/cache/womd_preprocessed/training \
  --output data/cache/root_index/training_roots.jsonl \
  --config configs/data/womd_waymax.yaml \
  --num-workers 8
```

## 6. Candidate generation

```bash
python scripts/generate_candidates.py \
  --preprocessed-root data/cache/womd_preprocessed/training \
  --root-index data/cache/root_index/training_roots.jsonl \
  --output-root data/cache/candidates/training \
  --config configs/data/womd_waymax.yaml \
  --num-workers 8
```

## 7. Waymax rollout generation

```bash
python scripts/run_waymax_rollouts.py \
  --preprocessed-root data/cache/womd_preprocessed/training \
  --candidate-root data/cache/candidates/training \
  --output-root data/cache/rollouts/training \
  --config configs/data/womd_waymax.yaml \
  --sim-agent-policy waymax_reactive \
  --num-workers 4
```

Replay debug command:

```bash
python scripts/run_waymax_rollouts.py \
  --preprocessed-root data/debug/womd_preprocessed/validation \
  --candidate-root data/debug/candidates/validation \
  --output-root data/debug/rollouts_replay/validation \
  --config configs/data/womd_waymax_debug.yaml \
  --sim-agent-policy replay \
  --max-groups 32 \
  --num-workers 1
```

## 8. Intervention-group construction

```bash
python scripts/build_intervention_groups.py \
  --preprocessed-root data/cache/womd_preprocessed/training \
  --candidate-root data/cache/candidates/training \
  --rollout-root data/cache/rollouts/training \
  --output-root data/cache/groups/training \
  --config configs/data/womd_waymax.yaml \
  --num-workers 8
```

## 9. Label-quality analysis

```bash
python scripts/analyze_label_quality.py \
  --group-root data/cache/groups/training \
  --output-dir outputs/quality/training \
  --config configs/data/womd_waymax.yaml
```

Review `quality_warnings.json` before training. Warnings are produced for sparse forced-dependence labels, empty group roots, low label coverage, and related failure modes.

## 10. Training baselines

```bash
python scripts/train.py \
  --config configs/train/train_baseline.yaml \
  --model-config configs/model/ego_cond_response.yaml \
  --train-root data/cache/groups/training \
  --val-root data/cache/groups/validation \
  --output-dir outputs/ego_cond_response \
  --seed 42
```

Other baseline configs:

```bash
python scripts/train.py --config configs/train/train_baseline.yaml --model-config configs/model/ego_cond_traj.yaml --train-root data/cache/groups/training --val-root data/cache/groups/validation --output-dir outputs/ego_cond_traj --seed 42
python scripts/train.py --config configs/train/train_baseline.yaml --model-config configs/model/shared_ctx_slots.yaml --train-root data/cache/groups/training --val-root data/cache/groups/validation --output-dir outputs/shared_ctx_slots --seed 42
```

## 11. Training SCOPE

```bash
python scripts/train.py \
  --config configs/train/train_surface.yaml \
  --model-config configs/model/scope_full.yaml \
  --train-root data/cache/groups/training \
  --val-root data/cache/groups/validation \
  --output-dir outputs/scope_full \
  --seed 42
```

Debug overfit:

```bash
python scripts/train.py \
  --config configs/train/train_surface.yaml \
  --model-config configs/model/scope_full.yaml \
  --train-root data/debug/groups/training \
  --val-root data/debug/groups/validation \
  --output-dir outputs/debug_scope_overfit \
  --max-train-groups 64 \
  --max-val-groups 64 \
  --overfit-debug \
  --seed 42
```

## 12. Calibration

```bash
python scripts/calibrate.py \
  --checkpoint outputs/scope_full/best.ckpt \
  --val-root data/cache/groups/validation \
  --output-dir outputs/scope_full/calibration \
  --config configs/train/train_calibration.yaml
```

## 13. Surface evaluation

```bash
python scripts/eval_surface.py \
  --checkpoint outputs/scope_full/best.ckpt \
  --test-root data/cache/groups/test \
  --calibration-dir outputs/scope_full/calibration \
  --output-dir outputs/eval/scope_full_surface \
  --config configs/eval/eval_surface.yaml
```

## 14. Operator-invariance evaluation

```bash
python scripts/eval_operator_invariance.py \
  --checkpoint outputs/scope_full/best.ckpt \
  --test-root data/cache/groups/test \
  --output-dir outputs/eval/scope_full_operator_invariance \
  --config configs/eval/eval_operator_invariance.yaml
```

## 15. Boundary evaluation

```bash
python scripts/eval_boundary.py \
  --checkpoint outputs/scope_full/best.ckpt \
  --test-root data/cache/groups/test \
  --output-dir outputs/eval/scope_full_boundary \
  --config configs/eval/eval_boundary.yaml
```

## 16. Offline candidate selection

```bash
python scripts/eval_candidate_selection.py \
  --checkpoints \
      scope_full=outputs/scope_full/best.ckpt \
      scope_no_fd=outputs/scope_full/best.ckpt \
      ego_cond_response=outputs/ego_cond_response/best.ckpt \
  --test-root data/cache/groups/test \
  --output-dir outputs/eval/candidate_selection \
  --config configs/eval/eval_selection.yaml
```

## 17. Closed-loop evaluation

```bash
python scripts/eval_closed_loop.py \
  --checkpoint outputs/scope_full/best.ckpt \
  --scenario-root data/cache/womd_preprocessed/test \
  --output-dir outputs/eval/scope_full_closed_loop \
  --config configs/eval/eval_closed_loop.yaml \
  --num-scenarios 1000
```

## 18. Visualization

```bash
python scripts/visualize_surface.py \
  --checkpoint outputs/scope_full/best.ckpt \
  --group-root data/cache/groups/validation \
  --scenario-id SCENARIO_ID \
  --output-dir outputs/figures/surface_examples \
  --config configs/eval/eval_surface.yaml
```

## 19. Exporting paper tables

```bash
python scripts/export_tables.py \
  --eval-root outputs/eval \
  --output-root outputs/tables \
  --format all
```

## 20. Full reproduction command block

```bash
for split in training validation test; do
  python scripts/preprocess_womd.py --data-root $WOMD_ROOT --split $split --output-root data/cache/womd_preprocessed/$split --config configs/data/womd_waymax.yaml --num-workers 8
  python scripts/mine_root_scenes.py --preprocessed-root data/cache/womd_preprocessed/$split --output data/cache/root_index/${split}_roots.jsonl --config configs/data/womd_waymax.yaml --num-workers 8
  python scripts/generate_candidates.py --preprocessed-root data/cache/womd_preprocessed/$split --root-index data/cache/root_index/${split}_roots.jsonl --output-root data/cache/candidates/$split --config configs/data/womd_waymax.yaml --num-workers 8
  python scripts/run_waymax_rollouts.py --preprocessed-root data/cache/womd_preprocessed/$split --candidate-root data/cache/candidates/$split --output-root data/cache/rollouts/$split --config configs/data/womd_waymax.yaml --sim-agent-policy waymax_reactive --num-workers 4
  python scripts/build_intervention_groups.py --preprocessed-root data/cache/womd_preprocessed/$split --candidate-root data/cache/candidates/$split --rollout-root data/cache/rollouts/$split --output-root data/cache/groups/$split --config configs/data/womd_waymax.yaml --num-workers 8
  python scripts/analyze_label_quality.py --group-root data/cache/groups/$split --output-dir outputs/quality/$split --config configs/data/womd_waymax.yaml
done

python scripts/train.py --config configs/train/train_baseline.yaml --model-config configs/model/ego_cond_response.yaml --train-root data/cache/groups/training --val-root data/cache/groups/validation --output-dir outputs/ego_cond_response --seed 42
python scripts/train.py --config configs/train/train_surface.yaml --model-config configs/model/scope_full.yaml --train-root data/cache/groups/training --val-root data/cache/groups/validation --output-dir outputs/scope_full --seed 42
python scripts/calibrate.py --checkpoint outputs/scope_full/best.ckpt --val-root data/cache/groups/validation --output-dir outputs/scope_full/calibration --config configs/train/train_calibration.yaml
python scripts/eval_surface.py --checkpoint outputs/scope_full/best.ckpt --test-root data/cache/groups/test --calibration-dir outputs/scope_full/calibration --output-dir outputs/eval/scope_full_surface --config configs/eval/eval_surface.yaml
python scripts/eval_operator_invariance.py --checkpoint outputs/scope_full/best.ckpt --test-root data/cache/groups/test --output-dir outputs/eval/scope_full_operator_invariance --config configs/eval/eval_operator_invariance.yaml
python scripts/eval_boundary.py --checkpoint outputs/scope_full/best.ckpt --test-root data/cache/groups/test --output-dir outputs/eval/scope_full_boundary --config configs/eval/eval_boundary.yaml
python scripts/eval_candidate_selection.py --checkpoints scope_full=outputs/scope_full/best.ckpt scope_no_fd=outputs/scope_full/best.ckpt ego_cond_response=outputs/ego_cond_response/best.ckpt --test-root data/cache/groups/test --output-dir outputs/eval/candidate_selection --config configs/eval/eval_selection.yaml
python scripts/eval_closed_loop.py --checkpoint outputs/scope_full/best.ckpt --scenario-root data/cache/womd_preprocessed/test --output-dir outputs/eval/scope_full_closed_loop --config configs/eval/eval_closed_loop.yaml --num-scenarios 1000
python scripts/export_tables.py --eval-root outputs/eval --output-root outputs/tables --format all
pytest tests -q
```

## 21. Running tests

```bash
pytest tests -q
```

## 22. Expected output directory structure

```text
outputs/
  quality/{training,validation,test}/
  debug/{scope_overfit,eval_surface}/
  scope_full/{best.ckpt,calibration/}
  ego_cond_response/best.ckpt
  eval/{scope_full_surface,scope_full_operator_invariance,scope_full_boundary,candidate_selection,scope_full_closed_loop}/
  figures/surface_examples/
  tables/{heldout_intervention_prediction,operator_boundary,offline_candidate_selection,closed_loop_waymax}.{csv,md,tex}
```

## 23. Troubleshooting Waymax issues

- If reactive behavior is close to replay, inspect `reactive-vs-replay` divergence in quality reports and use `idm_reactive` or a declared learned policy.
- If candidate tracking error exceeds thresholds, labels are masked and failure reason `tracking_error` is stored.
- If boundary-pair rate is low, increase candidate diversity or mine denser interactions before training.
- If forced-dependence labels are sparse, keep `L_forced_dependence` low and rely on calibrated planning functional rather than fabricated labels.
- Replay mode is only for factual alignment/debugging. It sets simulator-response masks false.
