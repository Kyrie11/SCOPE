# SCOPE: Surface-Conditioned Operator for Coercion-aware Planning and Evaluation

This repository implements **Learning Intervention-Response Surfaces for Non-Coercive Autonomous Driving** as a same-root intervention-response surface system. The code is organized around fixed root scenes and candidate-agent response functions rather than independent candidate rows. It includes WOMD/Waymax loading adapters, same-root dataset construction, branch/burden/safety labels, scene-only and support-adapted response-mechanism operators, support-query training, branch-conditioned forced-dependence estimation, mechanism-feasible selection, repair, ablations, experiments, and tests.

## What is implemented

- `scope.data`: WOMD/Waymax root loading, interaction root mining, relevant-agent selection, timing/speed/gap/lateral/planner candidate generation, feasibility filters, interpretable neighbor graph, controlled rollout backends, labels, dataset writer, diagnostics, and group-preserving dataloaders.
- `scope.geometry`: oriented-box collision/distance/TTC, route-tube conflict zones and TTA, DRAC, gap/headway and drivable helpers.
- `scope.models`: PyTorch scene encoder, structured intervention encoder (`z_ctrl`, `z_rel`, `z_ctx`), exchangeable mechanism-token operator, support updater, branch/ordinal-burden/trajectory-mixture/safety heads, and internal baselines.
- `scope.training`: same-root support/query split, masked mechanism losses, distillation, calibration entrypoint, trainer, and response evaluator.
- `scope.planning`: branch-conditioned risk, exact paper forced-dependence formula, conservative aggregation primitives, surface distance/boundary sensitivity, uncertainty, mechanism-feasible selection, Lagrangian fallback, and repairs.
- `scope.experiments`: entrypoints for held-out response prediction, surface geometry + FD diagnostics, false-safe offline benchmark, and closed-loop smoke/robustness loop.
- `tests`: eight unit tests for same-root integrity, candidate feasibility, label rules, FD formula, support/query split, no future leakage, and ablation switches.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# Optional for WOMD + Waymax rollouts:
pip install -e '.[waymax]'
```

Waymax and WOMD require Waymo Open Dataset access and Google Cloud authentication. The `rollout.backend=waymax` path attempts the current Waymax API and falls back to the deterministic reactive backend when `fallback_to_reactive=true`; set `fallback_to_reactive=false` to require native Waymax execution.

## Smoke test without WOMD credentials

```bash
python -m scope.data.dataset_writer \
  --config configs/data/womd_waymax.yaml \
  --override data.allow_synthetic=true \
  --override rollout.backend=reactive \
  --override data.synthetic_count=4 \
  --split train

python -m scope.data.dataset_diagnostics \
  --dataset_dir outputs/datasets/scope_womd_waymax

pytest -q
```

## Reproducibility and leakage rule

Closed-loop/planning code uses only scene-only operators. Support-adapted operators are used by training and support-query evaluation only; future labels, support labels, query labels, diagnostic boundary labels, and diagnostic forced-dependence labels are not inputs to deployment-time planning.

---

# SCOPE 代码落地指导

> 目标：让代码生成模型完整实现论文 **Learning Intervention-Response Surfaces for Non-Coercive Autonomous Driving** 中的 SCOPE（Surface-Conditioned Operator for Coercion-aware Planning and Evaluation），尤其是 WOMD + Waymax 的 same-root intervention dataset 构造、响应机制模型、训练、规划、消融与实验开关。  
> 本文件应原样写入项目 `README.md`，并作为实现验收清单使用。禁止只实现普通 ego-conditioned trajectory predictor、禁止用单一风险分数替代 branch-conditioned forced-dependence、禁止把 support-query 训练省略为普通 supervised learning。

---

## 0. 总体实现原则

1. **核心对象必须是 same-root intervention-response surface**：固定同一个 root scene `s`，只改变 ego candidate/intervention `u`，对每个 relevant surrounding agent `i` 学习一个可查询函数：
   `u -> p(m, rho, tau, c | u, s)`。
2. **部署时只能使用 scene-only operator**：测试/闭环规划时不能读取未来 simulator labels、support labels 或 query labels；support-adapted operator 只用于训练中的函数监督和消融。
3. **每个 candidate-agent pair 必须输出四类内容**：
   - response branch `m in {cede, maintain, follow, unaffected, ambiguous}`；
   - ordinal burden `rho in {0,1,2,3}`；
   - branch-conditioned future trajectory；
   - safety events：collision、near collision、induced hard braking、unsafe gap insertion。
4. **forced-dependence 必须 branch-conditioned**：不能用“high pressure probability”或“collision risk”直接代替。必须计算 ceding branch risk、non-ceding normalized risk，并按论文公式组合。
5. **所有模型、数据构造、实验必须参数化**：支持用 config 开关复现 full model、ablation、baseline、threshold sweep、candidate density sweep、policy shift 和 scenario stress。
6. **README.md 必须包含本指导中所有指令、运行命令、数据格式、配置项、实验矩阵、验收测试**。代码生成模型完成实现后，应把本文件内容整合到 README.md，而不是只保留简短说明。

---

## 1. 外部资料与工程依赖

### 1.1 必须参考的公开资料

实现时需要阅读并引用下列资料，原因如下：

- **Waymo Open Motion Dataset / WOMD**：用于 root scene、tracks、map features、traffic signal states、SDC track、objects of interest、tracks to predict、current_time_index 等字段解析。WOMD 坐标为全局坐标，X East、Y North、Z up，所有单位为米。标准 Motion scenario 包含 `tracks`、`dynamic_map_states/lane_states`、`map_features`、`sdc_track_index`、`objects_of_interest`、`tracks_to_predict`、`current_time_index` 等核心字段。
- **WOMD paper**：数据包含大规模交互场景，适合 merge、unprotected turn 等互动预测任务；论文介绍了 20 秒、10Hz、超过 100k scenes 的交互运动数据。
- **Waymax paper / repo / docs**：Waymax 是基于 JAX 的 WOMD 多智能体模拟器，支持从 WOMD 加载场景、闭环仿真、metrics、log playback 与 IDM agents，适合作为 SCOPE 的 controlled rollout backend。

### 1.2 推荐依赖

- Python >= 3.10
- JAX / jaxlib（按 GPU/CPU 环境安装）
- Waymax：`pip install git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax`
- TensorFlow / TFDS 或 Waymo Open Dataset 支持代码（用于 proto/tfrecord 读取）
- PyTorch 或 JAX/Flax（二选一作为模型训练框架；如果 Waymax rollout 用 JAX，模型也可用 JAX 以减少拷贝）
- numpy、scipy、shapely 或自定义 oriented-box geometry、pandas、pyarrow、hydra/omegaconf、tqdm、wandb/tensorboard

---

## 2. 代码仓库结构要求

必须实现如下结构。命名可略有不同，但功能不可缺失。

```text
scope/
  README.md                         # 必须包含本指导全文或等价完整说明
  configs/
    data/womd_waymax.yaml
    model/scope_full.yaml
    model/egocond_traj.yaml
    model/egocond_response.yaml
    experiment/heldout_response.yaml
    experiment/surface_geometry_fd.yaml
    experiment/false_safe_offline.yaml
    experiment/closed_loop.yaml
    ablation/*.yaml
  scope/
    data/
      womd_loader.py                # WOMD/Waymax scenario loading
      scene_schema.py               # RootScene, AgentState, MapFeature schemas
      root_mining.py                # interaction-condition mining
      relevant_agents.py            # top-N relevant agent selector
      candidates.py                 # candidate grid + repair candidate generation
      feasibility.py                # ego dynamic/map feasibility checks
      neighbor_graph.py             # feasible-neighbor graph N_eta
      rollout_waymax.py             # controlled ego injection + reactive agent rollout
      labels.py                     # branch, burden, safety, FD, boundary labels
      dataset_writer.py             # group-level parquet/zarr/tfrecord writer
      dataset.py                    # training dataloader preserving same-root groups
    geometry/
      boxes.py                      # oriented-box distance/collision/TTC
      conflict_zone.py              # route tube overlap, TTA, conflict zone arrival
      drac.py                       # deceleration rate required to avoid collision
      map_utils.py                  # lanes/routes/gaps/traffic controls
    models/
      scene_encoder.py
      intervention_encoder.py
      operator.py                   # scene-only and support-adapted mechanism tokens
      heads.py                      # branch/burden/traj/safety heads
      scope_model.py
      baselines.py
    training/
      losses.py
      support_query.py
      calibration.py
      train.py
      evaluate_response.py
    planning/
      estimators.py                 # risk, CVaR, FD, uncertainty, boundary score
      selector.py                   # mechanism-feasible set and fallback
      repair.py
      closed_loop.py
    experiments/
      run_heldout_response.py
      run_surface_geometry_fd.py
      run_false_safe_offline.py
      run_closed_loop.py
    utils/
      config.py
      random.py
      logging.py
      metrics.py
  tests/
    test_same_root_group_integrity.py
    test_candidate_feasibility.py
    test_branch_labels.py
    test_burden_labels.py
    test_forced_dependence_formula.py
    test_support_query_split.py
    test_no_future_leakage.py
    test_ablation_switches.py
```

---

## 3. 数据 schema：必须围绕 same-root group 组织

### 3.1 RootScene

每个 root scene `s` 固定：

```python
RootScene = {
  "scene_id": str,
  "split": "train|val|test|stress",
  "source": "womd|carla",
  "womd_version": str,
  "current_time_index": int,
  "dt": float,                    # usually 0.1 s
  "history_horizon_s": float,
  "future_horizon_s": float,
  "ego_track_index": int,          # WOMD sdc_track_index
  "tracks": AgentTrackTensor,      # positions, velocities, yaw, bbox, valid mask
  "map_features": MapFeatureSet,   # lanes, boundaries, crosswalks, stop signs, etc.
  "traffic_lights": LaneStateTensor,
  "route_context": RouteContext,   # from sdc_paths if available, else inferred route candidates
  "scenario_tags": list[str],      # lane_change, merge, unprotected_turn, ambiguous_priority...
  "relevant_agent_indices": list[int],
}
```

### 3.2 SameRootGroup

每个 root scene 可以产生多个 group，但每个 group 必须固定 root context、relevant agent set 和 simulator policy family，只改变 ego candidates。

```python
SameRootGroup = {
  "group_id": str,
  "scene_id": str,
  "policy_family_id": str,
  "root_scene": RootSceneRef,
  "candidate_set": list[EgoCandidate],
  "neighbor_edges": list[tuple[candidate_id, candidate_id, edit_type]],
  "rollout_matrix": dict[(candidate_id, policy_variant_id), RolloutResult],
  "labels": dict[(candidate_id, agent_id, policy_variant_id), ResponseLabel],
  "masks": dict[(candidate_id, agent_id, policy_variant_id), LabelMask],
}
```

### 3.3 EgoCandidate

```python
EgoCandidate = {
  "candidate_id": str,
  "family": "logged|timing|speed|gap|lateral|planner|repair",
  "future_states": Tensor[T_f, ego_state_dim],   # x, y, yaw, vx, vy/speed, ax, jerk if available
  "control_edits": {
    "delta_t": float,
    "speed_multiplier": float,
    "target_gap_id": str | None,
    "target_lane_id": str | None,
    "longitudinal_buffer_m": float,
    "time_headway_s": float,
    "lane_entry_duration_s": float,
    "lateral_midpoint_shift_s": float,
  },
  "feasibility": {
    "dynamic_ok": bool,
    "drivable_ok": bool,
    "route_ok": bool,
    "static_collision_free": bool,
    "duplicate_of": str | None,
  }
}
```

### 3.4 ResponseLabel

```python
ResponseLabel = {
  "branch": int,                  # 0 cede, 1 maintain, 2 follow, 3 unaffected, 4 ambiguous
  "burden": int,                  # 0,1,2,3
  "agent_future": Tensor[T_f, state_dim],
  "safety": {
    "collision": bool,
    "near_collision": bool,
    "induced_hard_brake": bool,
    "unsafe_gap": bool,
  },
  "diagnostics": {
    "d_min": float,
    "ttc_min": float | None,
    "drac": float,
    "b_max": float,
    "b_req": float,
    "j_max": float,
    "gap_margin_m": float,
    "headway_s": float,
    "delta_tta": float | None,
    "lateral_overlap": float,
    "commitment": float,
    "fd_diagnostic": bool | None,
    "boundary_positive_edges": list[str],
  }
}
```

### 3.5 LabelMask

必须显式区分缺失标签和负标签。

```python
LabelMask = {
  "valid_rollout": bool,
  "branch_valid": bool,
  "burden_valid": bool,
  "trajectory_valid": bool,
  "safety_valid": bool,
  "fd_diag_valid": bool,
  "boundary_valid": bool,
}
```

---

## 4. WOMD + Waymax 数据集构造步骤

### 4.1 数据读取

1. 通过 Waymax dataloader 读取 WOMD scenario：
   - 默认使用 Waymax `config.WOD_1_1_0_TRAINING` / validation / test config。
   - 若使用 WOMD >= 1.3.1，应读取 `sdc_paths` 作为 route context；否则根据 SDC logged future、lane graph 和 route consistency 推断 route candidates。
2. 每个 scenario 解析：
   - `tracks`：位置、速度、heading、length、width、height、valid mask、object type。
   - `sdc_track_index`：ego/SDC。
   - `map_features`：lane centers、boundaries、road boundaries、crosswalks、speed bumps、stop signs。
   - `dynamic_map_states/lane_states`：traffic signal state 与 lane id。
   - `current_time_index`：root planning time。
3. 历史和未来 horizon：
   - 论文默认 `T_h=2s, T_f=8s, dt=0.1s`。
   - 标准 WOMD 9 秒片段通常提供约 1s history + 8s future；实现必须自动检测可用历史长度：
     - 如果可用 history >= 2s，使用 2s；
     - 如果标准 WOMD 只提供 1s，设置 `history_horizon_s=1` 并记录在 dataset metadata 中；
     - 若要严格复现论文默认 2s history，需要使用 20s 数据或自行从长序列切窗。
4. 坐标处理：
   - 原始 WOMD 坐标保持 global frame 保存。
   - 模型输入对每个 relevant agent 使用 focal agent frame；ego-centric frame 可作为消融。
   - 保存所有 frame transform，确保 trajectory decoding 可回到 global frame 计算碰撞和 metrics。

### 4.2 Root-state mining：筛选互动 root scenes

从每个 scenario 的 `current_time_index` 或长序列 sliding root 中筛选 root state。满足以下任一条件即保留：

1. **Lane / target-gap interaction**：ego route 在未来 horizon 内与某 agent 当前或目标 lane 有纵向 80m 内交互。
2. **Conflict-zone overlap**：ego route tube 与 agent route tube 在 horizon 内重叠；route tube 使用 1.0m lateral buffer。
3. **Close arrival**：常速估计的 conflict-zone arrival difference `|Delta TTA| <= 3.0s`。
4. **Dense gap**：目标 lane front/rear vehicle 形成 gap < 25m 或 time headway < 2.5s。
5. **Ambiguous priority**：无信号/无保护 crossing，traffic-control priority 不确定或不完全决定让行关系。
6. **Close following**：后车 time headway < 2.0s 或距离 < 20m。

每个保留 root scene 必须记录触发条件、candidate families、relevant agents 和可视化 debug 信息。不要只随机抽样 WOMD；否则 high-pressure ceding 和 FD positive 会极度稀疏。

### 4.3 Relevant agent selection

对每个 root scene，按以下分数排序并保留 top `N_r=12`，支持 sweep `8,12,16`：

```text
score_i =
  w_overlap * route_tube_overlap
+ w_dist    * exp(-distance / 30m)
+ w_tta     * exp(-abs(delta_tta) / 3s)
+ w_gap     * target_gap_membership
+ w_cross   * crossing_conflict
+ w_rear    * rear_following_status
+ w_oi      * WOMD_objects_of_interest_or_tracks_to_predict
```

要求：

- relevant agent 必须有 root history valid state。
- 对未来才出现但 history invalid 的 object，不能作为模型预测目标；可以用于可视化或 background，但不作为 relevant agent label。
- pedestrian/cyclist 可保留，但如果首版只实现 vehicle interactions，必须在 README 中声明并用 config `agent_types=[vehicle]` 控制；不能静默丢弃。

### 4.4 Candidate families：构造 ego intervention candidates

每个 root scene 至少包含 logged continuation 和 route-consistent variants。默认 grid：

1. **Timing**：maneuver start shift `Delta t in {-1.0, -0.5, 0, 0.5, 1.0, 1.5}s`。
2. **Speed**：target speed multiplier `gamma_v in {0.75, 0.9, 1.0, 1.1}`，必须做 acceleration-limited smoothing。
3. **Gap**：lane-change/merge 场景使用 current target gap、one earlier gap、one later gap。
4. **Lateral commitment**：lane-entry duration `T_lat in {2.0, 3.0, 4.0, 5.0}s`，lateral midpoint shift `{-0.5,0,0.5}s`。
5. **Planner proposals**：最多 8 个 route-consistent lattice 或 learned-planner proposals。
6. **Repair**：规划时根据 constraint violation 生成最多 8 个 additional candidates。

每个候选轨迹必须通过 feasibility：

- `|a_ego| <= 3.0 m/s^2`
- `|j_ego| <= 4.0 m/s^3`
- lateral acceleration `< 2.5 m/s^2`
- drivable-area consistency
- route consistency
- no static-map collision
- 去重：mean displacement < 0.3m 且 final displacement < 0.5m 的候选视为重复。

若经过 grid 后候选数超过 `K`，按 scenario tag 分层保留：logged、earlier/later gap、fast/slow、early/late timing、lateral commitment extremes 都要覆盖。默认 `K=32`，支持 `16,32,64`。

### 4.5 Feasible-neighbor graph `N_eta`

用于 boundary sensitivity 和 repair。两个 candidates 若只相差一个小 edit，连接边：

- `|Delta t| <= 0.5s`
- speed multiplier difference `<= 0.15`
- adjacent target gap
- lane-entry duration difference `<= 1.0s`
- lateral midpoint shift `<= 0.5s`

每个 candidate 最多保留 8 个 normalized anchor distance 最近邻。边上保存 edit type 和 normalized edit distance。该 graph 不可由模型 embedding 后验构造；必须由可解释 candidate edits 构造，再用于 manifold loss 和 boundary probing。

### 4.6 Waymax rollout matrix：受控 ego 注入 + reactive policy variants

对每个 valid candidate：

1. 从 root WOMD scenario reset Waymax environment。
2. Ego/SDC 使用 candidate future 作为 controlled trajectory：
   - 每个 step 将 ego action 或 state 更新为 candidate 对应状态；
   - 如果 Waymax dynamics 要求 action，使用 inverse bicycle / tracking controller 生成 action，并校验 tracking error。
3. Relevant agents inside interaction neighborhood 使用 declared reactive policy。
4. Interaction neighborhood 外的 agents 使用 log playback；若 replay 与受控 ego 产生物理冲突或离谱交互，则改为 reactive fallback 或 mask invalid，并记录原因。
5. 默认 policy family：

```yaml
policy_family:
  neutral_idm:
    desired_time_headway: 1.5
    min_gap: 2.0
    max_accel: 2.0
    comfortable_decel: 2.0
    politeness_or_yield_bias: 0.0
  conservative_idm:
    desired_time_headway: 2.0
    min_gap: 4.0
    max_accel: 1.5
    comfortable_decel: 1.5
    yield_bias: high
  assertive_idm:
    desired_time_headway: 1.0
    min_gap: 1.0
    max_accel: 2.5
    comfortable_decel: 3.0
    yield_bias: low
```

6. 默认 training 每个 candidate 采样一个 policy variant；diagnostic/calibration subset 和 test diagnostic 对每个 candidate 跑多 policy variants，形成 rollout matrix。
7. Invalid rollout 条件：
   - injected ego violates feasibility；
   - simulator diverges；
   - actor 因非交互原因离开 drivable region；
   - root initialization oriented-box overlap；
   - ego tracking candidate error 超阈值。
   Invalid rollout 的 branch/burden/safety/diagnostic mask 全部设为 false，不能当负样本。

### 4.7 物理 anchors 与 intervention coordinate

对每个 `(candidate k, relevant agent i)` 计算：

```text
anchors a_i^k = [
  Delta_TTA,          # conflict-zone arrival-time difference
  d_min,              # minimum oriented-box distance
  TTC_min,            # minimum time-to-collision
  DRAC,               # deceleration rate required to avoid collision
  gap_margin,         # target-gap margin
  lateral_overlap,    # ego-agent lateral overlap
  eta_commitment      # lateral commitment / lane-entry progress
]
```

实现要求：

- `Delta_TTA` 基于 ego route tube 与 agent route tube 的 first conflict-zone arrival time。
- `d_min` 使用 oriented bounding boxes，不允许只用 center distance。
- `TTC_min` 需要考虑相对速度和 box overlap projection；不可简单除以 Euclidean distance。
- `DRAC` 必须有数值稳定处理，relative closing speed <= 0 时设为 0 或 undefined mask。
- `gap_margin` 对 lane-change/merge 使用 target-lane front/rear gap；对 crossing 可用 conflict-zone clearance margin。
- 所有 anchors clip 到训练集 1st/99th percentile，并用训练统计量 normalize。

Intervention encoder 输入三部分：

```text
u_i^k = LN(W_u [z_ctrl^k, z_rel,i^k, z_ctx,i^k]) in R^d_u
```

- `z_ctrl`：timing delay、speed profile、target gap、target lane、longitudinal buffer、lateral commitment。
- `z_rel`：anchors 的 MLP embedding。
- `z_ctx`：ego candidate tokens cross-attend 到 agent-centric scene tokens 的 learned contextual residual。

---

## 5. Operational labels：分支、负担、安全、FD、Boundary

### 5.1 Response branches

分支集合固定为：

```python
BRANCHES = ["cede", "maintain", "follow", "unaffected", "ambiguous"]
```

决策树：

1. `unaffected`：route/conflict relevance low，`d_min > 10m`，`TTC_min > 5s` 或 undefined，max speed change < 1.0m/s，final longitudinal deviation < 2.0m。
2. 对剩余 rollout：
   - `maintain`：agent 比 ego 早到 conflict zone 至少 0.5s，关闭或保持 relevant gap，且 ego entry 后 2.0s 内没有稳定跟在 ego 后方。
   - `cede`：agent 在 ego 到达 conflict zone 前将 usable gap 增加至少 3.0m 或 0.5s headway，或 deceleration >= 1.5m/s^2 且为 ego 保持 collision-free gap。
   - `follow`：ego 先进入，agent 在 ego 后保持至少 2.0s，time headway > 1.0s，且没有尝试 pass/close gap。
   - `ambiguous`：多个条件以小 margin 同时成立，或 trace 不足。

实现时必须输出每条 branch 的中间证据字段，方便 debug。

### 5.2 Ordinal response burden

计算：

- `b_req`：preserve safety required deceleration。
- `b_max`：agent 最大制动幅值。
- `j_max`：agent 最大 jerk 幅值。
- `TTC_min`：最小 TTC。
- progress loss / delay / gap margin。

默认标签：

```text
rho=0: unaffected or comfortable, b_max < 1.0, TTC_min > 5s, gap above comfort threshold
rho=1: mild adjustment, 1.0 <= b_max < 2.0 or small delay, TTC_min > 3s
rho=2: clear yielding burden, 2.0 <= b_max < 3.5, TTC_min in [2,3]s, or significant progress loss
rho=3: hard braking / near emergency, b_max >= 3.5, b_req >= 4.0, TTC_min < 2s, unsafe gap, or collision-avoidance fallback
```

默认 high-pressure threshold `rho_0=2`，实验中必须支持 `rho_0=3` sensitivity。

### 5.3 Safety outcomes

必须实现四类 binary safety event：

1. `collision`：oriented-box collision。
2. `near_collision`：`d_min < 1.0m` 或 `TTC_min < 1.5s`。
3. `induced_hard_brake`：agent `b_max >= 3.5m/s^2`。
4. `unsafe_gap`：ego entry 后 final headway < 1.0s 或 gap < 5m。

### 5.4 Diagnostic forced-dependence label

只用于 evaluation，不作为主训练必需标签。positive 条件：

```text
branch == cede
rho >= rho_0
no collision under ceding rollout
matched non-ceding policy variant or non-ceding branch envelope shows high risk / unsafe outcome
```

若没有 matched non-ceding evidence，`fd_diag_valid=false`，不得当 negative。

### 5.5 Diagnostic boundary labels

只用于 evaluation。neighbor pair `(a,b)` positive 若 simulator rollouts 中任一项变化：

- response branch 改变；
- high-pressure indicator 改变；
- collision 或 near-collision 改变；
- diagnostic FD 改变。

---

## 6. 模型实现要求

### 6.1 Scene/context encoder

输入：agent histories、ego history、map polylines、traffic signal tokens、route tokens。

默认设置：

```yaml
hidden_dim: 256
num_transformer_layers: 4
num_attention_heads: 8
dropout: 0.1
pre_layer_norm: true
coordinate_frame: focal_agent
```

要求：

- histories 用 temporal Transformer 或 GRU+Transformer；
- map polylines 用 point-wise MLP + polyline pooling；
- traffic-control 和 route tokens append 到 scene set；
- 输出 agent-centric context `C_i`。

### 6.2 Intervention encoder

输入 ego future tokens：position、velocity、acceleration、yaw、lane-relative offset、route progress、target lane、target gap、maneuver phase。

默认：

```yaml
intervention_dim: 128
anchor_mlp_layers: 2
anchor_clip_percentile: [1, 99]
```

必须实现 `z_ctrl`、`z_rel`、`z_ctx` 三部分，不能只拼接 raw trajectory。

### 6.3 Response-mechanism operator

默认 operator：

```yaml
mechanism_tokens: 16
hidden_dim: 256
support_updater_blocks: 3
```

两种模式：

1. **Scene-only deployment operator**：
   `C_i = E_scene(...)`，`Omega_empty = G_phi(C_i)`。
2. **Support-adapted training operator**：
   `Omega_S = U_phi(Omega_empty, {E_sup(u, y, mask) for support})`。

实现要求：

- mechanism tokens 必须是 exchangeable token set；
- updater 使用 token self-attention + cross-attention to scene/support tokens + MLP；
- support label embedding 必须包含 type embedding 和 mask embedding；
- missing label 不能编码成 valid negative。

### 6.4 Response heads

输出 factorization：

```text
p(y | u, Omega, s)
= p(m | r)
  p(rho | r, m)
  p(tau | r, m, rho)
  p(c | r, m, rho)
```

必须实现：

- branch categorical classifier over 5 branches；
- burden cumulative ordinal logits：`P(rho >= r) = sigmoid(g_rho(r_i,m_i)-b_r)` for r=1,2,3；
- trajectory head：`L_tau=6` Gaussian/Laplace mixture modes，输出 position、heading、velocity；
- safety head：collision、near collision、hard braking、unsafe gap probabilities；
- branch-conditioned risk：在给定 branch 下评估 safety head + differentiable collision checks。

---

## 7. 训练实现

### 7.1 Support-query sampling

每个 mini-batch 必须保留完整 same-root group。对每个 group：

```yaml
support_sizes: [0, 1, 2, 4, 8]
prob_empty_support: 0.4
query_policy: heldout_valid_interventions
no_support_query_overlap: true
```

要求：

- scene-only task 总是运行；
- support-adapted task 在 support size > 0 时运行，support size = 0 时应退化为 scene-only 或显式测试空 support；
- query interventions 不能与 support 重叠；
- support 和 query 必须来自同一个 root group。

### 7.2 Loss

总 loss：

```text
L = L_mech(Omega_empty; Q)
  + L_mech(Omega_S; Q)
  + lambda_dist * L_dist(Omega_S, Omega_empty; A)
  + lambda_mani * L_mani
```

`L_mech`：

```text
CE(branch)
+ lambda_rho * ordinal_burden_loss
+ lambda_tau * trajectory_mixture_NLL
+ lambda_c * safety_BCE
```

所有项必须按 `LabelMask` mask。

默认 weights：

```yaml
lambda_rho: 1.0
lambda_tau: 1.0
lambda_c: 1.0
lambda_dist: 0.1
lambda_mani: 0.02
```

### 7.3 Distillation loss

在 anchor-query set `A` 上：

```text
L_dist = JS(p_S(m|u), p_empty(m|u))
       + lambda_rho_d * W1(p_S(rho|u), p_empty(rho|u))
       + lambda_R_d * |Rbar_S(u) - Rbar_empty(u)|
```

`Rbar(u)=sum_m p(m|u) R(u,m)`。

### 7.4 Manifold metric regularizer

对 same-root candidate pair：

```text
L_mani = sum | d_u(u_a,u_b) - normalized_edit_distance(a,b) |
```

normalized edit distance 来自 timing、speed、gap、buffer、lateral commitment edits。该 loss 只能作为 weak regularizer，不能把模型退化成 hand-crafted TTC/DRAC。

### 7.5 Optional burden ranking

仅当物理证据差异可靠时加入 ranking：

- DRAC difference > 1.0 m/s^2；
- TTC difference > 1.0s；
- gap difference > 3.0m；
- delay difference > 0.5s。

### 7.6 Optimization

默认：

```yaml
optimizer: AdamW
learning_rate: 2.0e-4
weight_decay: 1.0e-4
warmup_steps: 5000
schedule: cosine
grad_clip_norm: 5.0
mixed_precision: true
batch_same_root_groups: 32
max_epochs: 30
max_steps: 200000
early_stopping: validation_response_nll_plus_ece
class_weight_clip: [0.5, 5.0]
```

### 7.7 Calibration

训练后在 validation split 上做：

- branch logits temperature scaling；
- ordinal logits scalar temperature；
- safety probabilities Platt 或 isotonic calibration；
- forced-dependence probabilities calibration。

所有 planning thresholds `epsilon_R, epsilon_D, epsilon_rho, epsilon_B, epsilon_U, delta, tau_d` 必须只在 validation 上选择，test 前固定。

---

## 8. Planning estimators 与机制可行选择

### 8.1 Branch-conditioned risk

对每个 candidate `k`、agent `i`：

```text
R_coll_i^k = sum_m p_i^k(m) R_i(u_i^k, m)
```

non-ceding risk：

```text
R_coll_i^{k, not_cede}
= sum_{m != cede} p(m) R(u,m) / (sum_{m != cede} p(m) + eps)
```

### 8.2 Forced-dependence estimator

必须按公式实现：

```text
D_i(k) = P(m=cede)
       * P(rho >= rho0 | m=cede)
       * (1 - R_coll_i^{k, cede})
       * sigmoid((R_coll_i^{k, not_cede} - R_coll_i^{k, cede} - delta) / tau_d)
```

解释：

- likely cede；
- ceding high burden；
- ceding branch ego safe；
- non-ceding branch substantially riskier。

禁止实现成 `P(cede) * P(high_pressure)` 或 `risk + pressure penalty`。

### 8.3 Conservative aggregation

```text
R^k = 1 - prod_i(1 - R_coll_i^k)
D^k = 1 - prod_i(1 - D_i(k))
P_hp^k = 1 - prod_i(1 - P_i(m=cede) P_i(rho >= rho0 | m=cede))
B^k = max_i B_i(u_i^k)
```

### 8.4 Boundary sensitivity

对 feasible neighbors：

```text
D_surf(u,u') = JS(p(m|u),p(m|u'))
             + lambda_rho_b * W1(p(rho|u),p(rho|u'))
             + lambda_R_b * |Rbar(u)-Rbar(u')|
B_i(u) = max_{u' in N_eta(u)} D_surf(u,u') / (d_u(u,u') + eps)
```

默认 `lambda_rho_b=0.5`，`lambda_R_b=1.0`。Wasserstein over ordinal bins `{0,1,2,3}`。

### 8.5 Epistemic uncertainty

必须支持至少一种：

- ensemble size 默认 5，sweep 3/5；
- MC dropout；
- distributional head。

`U^k` 用于 mechanism-feasible set。

### 8.6 Mechanism-feasible set

候选必须满足：

```text
CVaR_alpha[R^k] <= epsilon_R
UCB_beta[D^k] <= epsilon_D
UCB_beta[P_hp^k] <= epsilon_rho
B^k <= epsilon_B
U^k <= epsilon_U
```

默认 `alpha=0.1`。满足条件的候选中选择 task cost 最小：

```text
argmin J_task(tau_ego^k)
```

### 8.7 Surface-guided repair

若 feasible set 为空，按 active violation 生成 repair candidates：

- `R > epsilon_R`：`+0.5s` delay、`0.9` speed multiplier、one larger target gap。
- `D > epsilon_D`：earlier-yield 或 later-gap variants。
- `P_hp > epsilon_rho`：longitudinal buffer +5m 或 target time headway +0.5s。
- `B > epsilon_B`：移动到 boundary score 最低的 feasible neighbor，并在当前 candidate 与低 boundary neighbor 之间插值一个 candidate。
- `U > epsilon_U`：在 task cost 相近 alternatives 中选择更 conservative candidate。

repair 后必须重新跑 same scene-only operators query 和 feasibility filters。若仍无 feasible candidate，使用 calibrated Lagrangian fallback：

```text
argmin_k J_task(k) + sum_g lambda_g [g(k)-epsilon_g]_+
```

---

## 9. Baselines 与 ablation 开关

### 9.1 内部可实现 baselines：必须支持

1. **EgoCond-Traj**
   - candidate-wise trajectory + unconditional risk prediction；
   - 无 branch、无 burden、无 operator、无 FD。
2. **EgoCond-Response**
   - 输入 `(C_i, u_i^k)`，candidate-wise 预测 branch/burden/traj/safety；
   - 无 shared same-root operator；无 support-query functional supervision。
3. **Surface w/o Support**
   - 有 scene-only operator；
   - 不使用 support-adapted task；
   - `lambda_support=0`。
4. **Surface w/o Distill**
   - 有 support-adapted operator；
   - 无 scene-only distillation；
   - 注意部署仍只能用 scene-only，所以该消融会暴露 train-test operator gap。
5. **Risk + Pressure Penalty**
   - 规划目标使用 risk 和 high-pressure probability additive costs；
   - 不计算 branch-conditioned FD。
6. **SCOPE w/o FD**
   - full surface；planning 去掉 `D <= epsilon_D` 约束。
7. **SCOPE w/o Boundary**
   - full surface；planning 去掉 `B <= epsilon_B` 和 boundary repair。
8. **SCOPE Full**
   - structured intervention、operator、support-query、distill、branch-conditioned risk、FD、boundary、uncertainty、repair 全开。

### 9.2 外部 baselines：可以跳过但要留接口

可以只保留 adapter/config，不强制实现：

- DTPP-style ego-conditioned prediction and cost evaluation；
- GameFormer-style interaction prediction/planning；
- RACP-style risk-aware contingency planning；
- M2I / behavior latent / Categorical Traffic Transformer。

README 中要说明：外部 baseline 需要第三方代码时可跳过，但内部 ablation 必须完整实现。

### 9.3 Config 开关示例

```yaml
model:
  name: scope
  use_structured_intervention: true
  use_contextual_residual: true
  use_operator_tokens: true
  use_support_query: true
  use_scene_only_loss: true
  use_support_adapted_loss: true
  use_distillation: true
  use_manifold_loss: true
  branch_head: true
  burden_head: ordinal
  trajectory_head: mixture
  safety_head: true
planning:
  use_forced_dependence: true
  use_boundary_constraint: true
  use_uncertainty_constraint: true
  use_repair: true
  fallback: lagrangian
baseline:
  candidate_wise_only: false
  risk_pressure_penalty_only: false
```

---

## 10. 实验设计与运行要求

### 10.1 Dataset summary diagnostics

训练前必须生成表格：

- scenes；
- root groups；
- candidates/group；
- policy variants；
- valid rollouts；
- high-pressure cede rate；
- FD positive rate；
- scenario type counts；
- branch distribution；
- burden distribution；
- simulator failure rate；
- boundary-pair positive rate；
- duplicate-candidate rate；
- calibration-set coverage。

### 10.2 Experiment 1：Same-root held-out response prediction

目的：验证 operator 学到 intervention function，而不是 candidate-wise labeler。

Procedure：

1. 对每个 root group 做 support/query split。
2. 模型预测 held-out query responses。
3. 对支持 support 的模型，同时报告 scene-only 和 support-adapted 结果；部署主结果用 scene-only。
4. 比较 EgoCond-Response vs SCOPE Full。

Metrics：

- Branch NLL / accuracy；
- Ordinal burden MAE；
- High-pressure AUROC；
- Trajectory minADE/minFDE；
- Risk Brier；
- ECE。

Models：EgoCond-Traj、EgoCond-Response、Surface w/o Support、Surface w/o Distill、SCOPE Full。

### 10.3 Experiment 2：Surface geometry, boundary, forced-dependence

目的：验证 local surface probes 是否对应真实 response transitions，以及 FD 是否区别 comfortable ceding 和 coercive ceding。

Procedure：

1. 用 `N_eta` 取 feasible-neighbor pairs。
2. 用 diagnostic boundary labels 评估 `B_i(u)`。
3. 用 multi-policy rollout matrix 评估 diagnostic FD。
4. 比较 pressure penalty 与 branch-conditioned FD。

Metrics：

- Boundary AUROC；
- response-transition localization；
- burden Kendall / ranking；
- HP AUROC；
- FD AUROC；
- FD ECE；
- false-safe rate。

Models：Physical TTC/DRAC heuristic、EgoCond-Response、Risk + Pressure Penalty、SCOPE w/o Boundary、SCOPE Full。

### 10.4 Experiment 3：Coercive false-safe benchmark + offline selection

构造 candidate triples：

- A：collision-free only after high-pressure ceding；
- B：collision-free with low-pressure cooperation or self-preserved margin；
- C：unsafe under non-ceding。

Selector 只能看 pre-execution information 和 model predictions；simulator outcomes 只用于 evaluation。

Metrics：

- false-safe selection rate；
- FD rate；
- induced hard braking；
- collision / near collision；
- progress；
- comfort；
- oracle regret。

Selectors：Task cost only、EgoCond risk、Risk + Pressure Penalty、SCOPE w/o FD、SCOPE w/o Boundary、SCOPE Full。

### 10.5 Experiment 4：Closed-loop non-coercive planning and robustness

Procedure：

每个 planning step：

1. 生成 candidates；
2. encode interventions；
3. query scene-only response operators；
4. 计算 risk、FD、boundary、uncertainty；
5. mechanism-feasible selection；
6. 必要时 repair；
7. 执行动作并 step simulator。

Metrics：

- route success；
- collisions；
- off-route events；
- progress；
- comfort；
- induced hard braking；
- unsafe-gap insertion；
- FD rate；
- boundary violation；
- CVaR tail risk。

Robustness settings：

- default policy；
- aggressive agents；
- conservative agents；
- mixed reactive agents；
- low candidate density；
- scenario stress shifts。

Planners：Rule/IDM planner、EgoCond planner、Risk-aware planner、Risk + Pressure Penalty、SCOPE w/o FD、SCOPE Full。

---

## 11. 必须实现的 metrics

### 11.1 Prediction / surface metrics

- branch NLL；
- branch accuracy；
- burden-exceedance AUROC；
- ordinal burden MAE；
- burden ranking accuracy / Kendall；
- boundary AUROC；
- support-consistency JS distance；
- trajectory minADE / minFDE；
- collision-risk Brier score；
- expected calibration error；
- negative log-likelihood。

### 11.2 Planning metrics

- collision rate；
- near-collision rate；
- induced hard-braking rate；
- unsafe-gap rate；
- high-burden ceding rate；
- forced-dependence rate；
- progress；
- comfort；
- feasible-selection rate；
- boundary-violation rate；
- CVaR tail risk；
- oracle regret for offline benchmark。

---

## 12. 默认超参数与 sweep

```yaml
data:
  dt: 0.1
  history_horizon_s: 2.0        # fallback to 1.0 if standard WOMD lacks 2s history
  future_horizon_s: 8.0
  candidate_count: 32
  relevant_agents: 12
  neighbor_count: 8
model:
  intervention_dim: 128
  mechanism_tokens: 16
  hidden_dim: 256
  trajectory_modes: 6
  ensemble_size: 5
labels:
  high_burden_threshold_rho0: 2
training:
  support_sizes: [0,1,2,4,8]
  batch_same_root_groups: 32
  lr: 2.0e-4
  weight_decay: 1.0e-4
  warmup_steps: 5000
  max_epochs: 30
  max_steps: 200000
planning:
  cvar_alpha: 0.1
  epsilon_R: validation_calibrated
  epsilon_D: validation_calibrated
  epsilon_rho: validation_calibrated
  epsilon_B: validation_calibrated
  epsilon_U: validation_calibrated
sweeps:
  history_horizon_s: [1.0, 2.0]
  future_horizon_s: [6.0, 8.0]
  candidate_count: [16, 32, 64]
  relevant_agents: [8, 12, 16]
  intervention_dim: [64, 128]
  mechanism_tokens: [8, 16]
  hidden_dim: [128, 256]
  trajectory_modes: [3, 6]
  neighbor_count: [4, 8, 12]
  ensemble_size: [3, 5]
  high_burden_threshold_rho0: [2, 3]
```

---

## 13. 防偷懒验收清单

实现完成前，必须通过以下检查：

### 13.1 数据完整性

- [ ] Dataset 存储单位是 same-root group，不是独立 candidate rows。
- [ ] 每个 group 中 root scene、map、traffic lights、relevant agents 固定，只有 ego candidate 改变。
- [ ] 每个 candidate-agent pair 有 branch/burden/traj/safety labels 或显式 mask。
- [ ] Invalid rollout 被 mask，不被当 negative。
- [ ] Candidate grid 包含 timing/speed/gap/lateral/planner/logged families。
- [ ] Neighbor graph 来自 small feasible edits。
- [ ] 多 policy rollout matrix 可用于 FD diagnostics。

### 13.2 模型完整性

- [ ] Intervention encoder 包含 `z_ctrl`、`z_rel`、`z_ctx`。
- [ ] Operator 有 scene-only 和 support-adapted 两种模式。
- [ ] Deployment/evaluation 使用 scene-only operator。
- [ ] Response heads 输出 branch、ordinal burden、trajectory mixture、safety。
- [ ] Branch-conditioned risk 可按 branch 查询。
- [ ] Support labels 有 mask embedding，missing 不等于 false。

### 13.3 Loss 与训练

- [ ] support/query split 同 root 且无 overlap。
- [ ] 同时训练 scene-only 与 support-adapted query prediction。
- [ ] distillation 对齐 support-adapted teacher 与 scene-only deployment operator。
- [ ] manifold regularizer 使用 edit distance。
- [ ] calibration 只在 validation 上做。

### 13.4 Planning

- [ ] FD 按完整公式计算。
- [ ] high-pressure ceding 不等于 coercion；只有 forced-dependence 高才被机制约束惩罚。
- [ ] boundary sensitivity 来自 local surface slope，不是单独 classifier。
- [ ] mechanism-feasible set 先过滤再按 task cost 选。
- [ ] repair 后重新 query operator。
- [ ] fallback 是 calibrated Lagrangian，不是直接忽略 constraints。

### 13.5 实验

- [ ] 四个主实验都可通过 config 运行。
- [ ] 所有内部 baselines/ablations 可运行。
- [ ] 外部 baseline 可跳过但 adapter/config 留出接口。
- [ ] 输出 tables 与 paper 中指标对应。
- [ ] 所有 thresholds test 前固定。

---

## 14. 推荐运行命令

```bash
# 1. 构造 same-root intervention dataset
python -m scope.data.dataset_writer \
  --config configs/data/womd_waymax.yaml \
  --split train

python -m scope.data.dataset_writer \
  --config configs/data/womd_waymax.yaml \
  --split val

python -m scope.data.dataset_writer \
  --config configs/data/womd_waymax.yaml \
  --split test

# 2. 输出 dataset diagnostics
python -m scope.data.dataset_diagnostics \
  --dataset_dir outputs/datasets/scope_womd_waymax

# 3. 训练 full SCOPE
python -m scope.training.train \
  --config configs/model/scope_full.yaml \
  --data outputs/datasets/scope_womd_waymax

# 4. calibration
python -m scope.training.calibration \
  --checkpoint outputs/checkpoints/scope_full.pt \
  --split val

# 5. 实验 1
python -m scope.experiments.run_heldout_response \
  --config configs/experiment/heldout_response.yaml

# 6. 实验 2
python -m scope.experiments.run_surface_geometry_fd \
  --config configs/experiment/surface_geometry_fd.yaml

# 7. 实验 3
python -m scope.experiments.run_false_safe_offline \
  --config configs/experiment/false_safe_offline.yaml

# 8. 实验 4
python -m scope.experiments.run_closed_loop \
  --config configs/experiment/closed_loop.yaml
```

---

## 15. README.md 必须包含的内容

代码生成模型最终必须把以下内容写进 `README.md`：

1. SCOPE 的核心思想：same-root intervention-response surface，而不是 candidate-wise risk scorer。
2. WOMD/Waymax 数据获取、安装、认证、版本说明。
3. 数据构造流程：root mining、candidate generation、Waymax rollout matrix、labels、masks、neighbor graph。
4. 数据 schema 和磁盘格式。
5. 模型结构：scene encoder、intervention encoder、operator、heads。
6. Loss、support-query sampling、distillation、manifold regularization、calibration。
7. Planning：risk、FD、boundary、uncertainty、repair、fallback。
8. Baselines 和 ablation 开关。
9. 四个实验的运行命令与指标。
10. 防偷懒验收清单和 unit tests。
11. 已知限制：标准 WOMD history 长度问题、外部 baselines 依赖、CARLA stress optional、reactive policy realism limitation。

---

## 16. 关键实现提醒

- 不要把 logged future 直接当所有 candidate 的周围车 response；必须通过 Waymax rollout 注入 ego intervention 后重新生成 agent response。
- 不要把 `cede` 全部视为坏；只有 high-pressure ceding 且 non-ceding unsafe / substantially riskier 时才是 forced-dependence。
- 不要把 support evidence 用到 test-time planning；test-time planning 只能用 scene-only operator。
- 不要用 center distance 替代 oriented-box collision / distance。
- 不要只报告 trajectory ADE/FDE；必须报告 branch、burden、safety、FD、boundary 和 planning metrics。
- 不要让 candidate generation 只产生 logged-like variants；必须覆盖 timing、speed、gap、lateral commitment 的局部 surface。
- 不要把 diagnostic FD label 作为主训练标签；它主要用于 calibration/evaluation。
- 不要在 test 上调 threshold；所有 threshold validation-calibrated。

---

## 17. 最小可接受实现（MVP）与完整实现边界

### MVP 必须包含

- WOMD/Waymax root loading；
- root mining；
- candidate grid；
- controlled ego rollout；
- branch/burden/safety labels；
- same-root dataset writer；
- scene-only SCOPE operator；
- support-query training；
- FD estimator；
- offline selection experiments 1-3；
- internal ablations。

### 完整实现包含

- multi-policy rollout matrix；
- calibration subset；
- closed-loop Waymax planning；
- surface-guided repair；
- uncertainty ensemble；
- robustness under policy shifts；
- optional CARLA stress scenarios；
- external baseline adapters。

MVP 也不得退化成 candidate-wise predictor；只要没有 same-root group + operator + support-query + branch-conditioned FD，就不算 SCOPE 实现。
