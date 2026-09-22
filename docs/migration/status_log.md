# Migration Status Log

## 2026-09-22: Reject device-resident dependency routes

Latest state:

- Prepare -> accessor -> Batch -> dependency access -> model -> task -> state
  update remains the common path.  The candidate aligned dependency-ID
  placement with an already CUDA-resident feature cache before the shared
  feature launcher, then was removed after the timing gate failed.
- The existing prefetch stream and Torch transfer are reused.  CPU feature
  caches remain unchanged; no cache API, route type, collective, model path,
  DGL graph, or native operator is added.

Verification:

- Focused loader/feature/model checks passed 46 with 3 skips; compile and diff
  checks passed while the candidate was present.
- Four-A40 WIKI/TGN epochs 2--10 measured 0.31957 and 0.32772 s/epoch for the
  candidate versus 0.32248 for the adjacent control.  The pooled candidate
  mean is 0.32365, so moving IDs early is not retained.  Accuracy and DCRNN
  campaigns were skipped after the timing gate failed.  Outputs are
  `/tmp/starrygl_tgn_device_route_{candidate,candidate2,adjacent_control}_e10`.

## 2026-09-22: Reject fused CUDA Adam

Latest state:

- Event/Snapshot and node/edge training share one framework optimizer builder
  and one post-backward optimizer call. The current profile reports about
  50 ms in 30 TGN foreach-Adam steps, but includes asynchronous effects.
- The bounded candidate selected public Torch fused Adam/AdamW only when the
  framework constructs an optimizer for CUDA parameters. CPU and explicitly
  supplied optimizers remain unchanged. No config, model path, loop, cache or
  custom kernel was introduced. It was removed; production and tests remain
  byte-identical to `3cf4d38`.

Verification:

- The focused optimizer-lowering test passed. A five-step CUDA Adam trajectory
  matched the existing optimizer at rtol 2e-6/atol 2e-7 with identical values
  and step counters.
- Four-A40 WIKI/TGN epochs 2--10 averaged 0.33782 s/epoch with fused Adam versus
  0.32967 for the adjacent control, a 2.47% regression. The candidate failed
  the first timing gate, so DCRNN/full-accuracy campaigns were not run. Output:
  `/tmp/starrygl_tgn_fused_adam_e10`.

## 2026-09-22: Reject persistent gradient views

Latest state:

- The canonical Prepare -> accessor -> Batch -> dependencies -> model -> task
  -> backward -> optimizer -> state-update path already meets at
  `sync_gradients`. The earlier dense bucket packed/copied every step; native
  coalescing retained separate buffers. Neither reused a flat gradient buffer
  as the parameter `.grad` storage.
- The candidate bound one public Torch buffer per device/dtype at epoch setup,
  uses parameter-shaped views for backward and one post-backward all-reduce per
  buffer. It introduces no public API, model specialization, DDP wrapper, cache
  manager or custom kernel. Missing gradients become stable zero views, keeping
  collective order identical on empty owners. The implementation was removed;
  production and tests remain byte-identical to `2d1faac`.

Verification:

- Local focused tests passed 70 with 9 skipped. Two-rank Gloo passed 3 tests per
  rank; the split two-rank NCCL empty/optimizer check passed 2 tests per rank.
- WIKI/TGN candidate means for epochs 2--10 were 0.33035 and 0.32078 s/epoch.
  Pooled candidate time was 0.32556 versus 0.32663 across three controls, only
  0.33% faster, and individual pairs moved in opposite directions. The change
  fails the end-to-end retention gate despite reducing collective count.
- No accuracy or DCRNN campaign was run after the TGN timing gate failed.
  Outputs are `/tmp/starrygl_tgn_persistent_grad_views*` and
  `/tmp/starrygl_tgn_2d1faac_adjacent_control2_e10`.

## 2026-09-22: Reject direct native DDP event lowering

Latest state:

- The shared Prepare -> accessor -> Batch -> dependencies -> model -> task ->
  backward -> optimizer -> state-update spine remains. Native MemShare uses
  PyTorch DDP for TGN; StarryGL currently performs 27 post-backward parameter
  reductions per TGN batch.
- The bounded candidate bound native DDP once for distributed Event training
  and kept runtime-owned Snapshot recurrent scans on the existing
  synchronization path.
  It reuses `StarryModel.forward(Batch)` and keeps state commits on the
  underlying model. No public option, model adapter, cache or second loop is
  introduced. The candidate was removed; production remains byte-identical to
  `9438496b`.

Verification:

- Focused model/Event regressions passed: 72 passed, 11 skipped.
- The first four-A40 smoke deadlocked before epoch 1 with DDP on the default
  group. The retry placed DDP on a separate cached all-rank group and also
  deadlocked before epoch 1. Both runs were terminated and left no GPU
  processes. This Event configuration does not use endpoint-embedding autograd
  exchange, so the earlier endpoint-specific attribution is withdrawn; exact
  reducer/rank divergence remains unresolved.
- A future overlap path needs a complete per-rank forward/backward collective
  trace. Another wrapper, cache or process group does not solve the observed
  boundary. No timing or accuracy result is claimed because neither
  distributed candidate completed an epoch.

## 2026-09-22: Reject native gradient coalescing

Latest state:

- The common backward -> `sync_gradients` -> optimizer boundary currently
  performs one all-reduce and division per trainable parameter. The TGN profile
  records 27 float reductions per batch; Event/Snapshot and node/edge paths all
  reuse this function.
- A bounded candidate used PyTorch's distributed coalescing context per
  device/dtype and native foreach division. It added no flat packing,
  configuration, DDP wrapper, cache, model branch or execution path, but was
  removed after validation. Production remains byte-identical to `61192b5`.

Verification:

- Two-rank CPU/Gloo and two-rank NCCL gradient/empty-owner checks each passed 6
  tests per rank. Two adjacent four-A40 WIKI/TGN pairs measured 0.33106 ->
  0.32660 and 0.31918 -> 0.31835 s/epoch; pooled means improved only 0.81%,
  within visible run jitter.
- Candidate test AP/AUC was 0.91145/0.90607 after ten epochs. Although random
  samples are intentionally unpaired, the candidate also changes collective
  floating-point ordering and relies on private `torch.distributed` API. The
  small timing result does not justify those risks. Outputs are
  `/tmp/starrygl_tgn_{61192b5_control,native_coalescing}*`.

## 2026-09-22: Reuse static supervision scheduling for Event

Latest state:

- The common Prepare task slice -> accessor -> Batch -> dependencies -> model ->
  task -> state-update path already caches a globally reduced supervision
  schedule for built-in full-snapshot node tasks. Built-in event-edge tasks have
  the same prepared `task_ptr` invariant; only accessor, negative, model and
  state behavior specialize.
- The runtime condition now also covers exact built-in
  `EdgePredictionTask + event_window + neighbor`. Custom tasks, callbacks and
  all dynamic supervision keep the per-step activity collective. No new cache,
  public option, loader, model branch or communication primitive was added.

Efficiency alternatives considered:

- Reuse one vectorized Torch slice, one existing globally ordered
  `CommScheduler` reduction and the existing runtime cache. A DGL or native
  operator cannot improve this ten-element WIKI control vector; another cache
  interface would duplicate the scheduling fact.

Files modified:

- `src/starrygl/runtime/loop.py`
- `tests/test_empty_supervision_step.py`
- `docs/design/memshare_temporal_target_identity_20260913.md`

Verification:

- Focused runtime/TGN checks: 14 passed. Full suite: 543 passed, 45 skipped and
  the same nine pre-existing stale-default and retired-smoothing failures.
  Four-rank WIKI/TGN training completed through the ordinary Trainer path.
- Two adjacent four-A40 control/candidate pairs over rank-max epochs 2--10 were
  0.33555 -> 0.32255 and 0.32403 -> 0.32735 s/epoch. Their pooled means are
  0.32979 -> 0.32495 (-1.47%); the benefit is modest relative to run jitter.
- Two complete candidate train -> validation -> test runs expose the expected
  unpaired native-sampler variation. The stronger repeat ended at test AP/AUC
  0.91328/0.90875, within 0.558/0.652 percentage points of native MemShare and
  0.258/0.189 points of the unchanged StarryGL reference. Outputs are
  `/tmp/starrygl_tgn_{8e0df7f_control,static_event_activity}*`.

Unresolved risks:

- This removes repeated control communication but does not close TGN parity:
  the pooled candidate remains 27.4% slower than native MemShare's 0.25505
  s/epoch. One candidate repeat reached 0.31349 s/epoch, but the paired runs
  show enough jitter that it is not used as the parity claim.

## 2026-09-22: Batch static supervision collectives

Latest state:

- The common Prepare -> owner task slice -> Snapshot accessor -> Batch ->
  dependency access -> coupled scan -> task -> state update path is unchanged.
  DCRNN's exact previous-state and gate-to-candidate Routes, and its bounded
  stale cache channels, are unchanged.
- Built-in full-snapshot `NodePredictionTask` with static prepared supervision
  now reduces its per-window activity vector once and caches the global boolean
  schedule. Optimizer steps consume that schedule instead of issuing one scalar
  activity collective per window. Custom tasks, callbacks, window-mean,
  sampled/chunk execution and non-node tasks retain the existing per-step guard.
- This is one runtime scheduling fact, not a new cache, model adapter, loader,
  execution stack or public option. Every rank enters the same scheduled
  collective and globally empty windows still skip Adam exactly.

Efficiency alternatives considered:

- The Torch profile attributed about 58--60% of DCRNN GPU time to required DGL
  diffusion GSpMM, while 28 scalar optimizer-activity reductions cost about
  70 ms on four GPUs. One vectorized Torch tensor plus the existing
  `CommScheduler` collective was the smallest reusable removal of that overhead.
- Replacing prepared DGL diffusion with Torch CSR preserved focused math but
  measured 4.0248 s/epoch on four GPUs and was removed. DGL bipartite blocks
  also preserved math, but combined with the retained schedule measured 3.2316
  versus 3.0895 s/epoch and was removed. A custom C++/CUDA diffusion operator is
  not justified before DGL/native profiling identifies a narrower kernel gap.

Files modified:

- `src/starrygl/runtime/{loop.py,epoch.py}`
- `tests/test_empty_supervision_step.py`
- `docs/design/current/snapshot_boundary_history.md`

Verification:

- Focused snapshot/runtime checks: 39 passed, 2 skipped. Two-rank CPU/Gloo
  empty-owner and gradient synchronization: 5 passed per rank. The full suite
  completed with 542 passed, 45 skipped and the same 9 pre-existing failures
  in stale-default expectations and retired smoothing helpers.
- A consecutive four-A40 control/candidate comparison over epochs 2--5 was
  3.3079 -> 3.0895 s/epoch (-6.60%), with the same epoch-5 loss to normal
  floating-point variation.
- Two-node/eight-A40 W=1 exact DCRNN repeated ten epochs at 1.9975 s/epoch over
  epochs 2--10, versus the prior 2.0838 s short control. A first repeat was
  2.0217 because epochs 3--4 spiked; its stable epochs 5--10 averaged 1.9302.
  The exact final source then repeated epochs 2--3 at 1.8375 s/epoch. Final
  epoch-10 loss was 0.0613518368 versus prior 0.0613518357.
- Outputs are `/tmp/starrygl_dcrnn_{aba66ec_control,static_activity}*` and
  `/mnt/nfs/zlj/starrygl_static_activity_candidate_20260922/results/`.

Unresolved risks:

- The ten-epoch mean clears 2 s by only 0.12%; inter-node jitter remains visible,
  so this is a reached gate, not a robust throughput margin. Test-set quality
  and longer convergence are not newly measured because model math, inference
  and state/cache behavior did not change.
- TGN still has the separately recorded throughput gap to MemShare.

## 2026-09-22: Reject unpaired TGN micro-optimizations

Latest state:

- The common window -> native accessor -> Batch -> dependency access -> model
  -> task -> state-update path remains unchanged. No production candidate in
  this pass cleared both the WIKI throughput and accuracy gates.
- A fresh unchanged-baseline train -> validation -> test run ended at test
  AP/AUC 0.91586/0.91064. Relative to native MemShare
  0.91886/0.91527, the remaining gaps are 0.300/0.463 percentage points.
- The retained baseline remains commit `dfc53a7`: two no-evaluation repeats
  take 0.32322 and 0.32533 s/epoch over rank-max epochs 2--10. Native
  MemShare remains 0.25505 s/epoch.

Efficiency alternatives considered:

- Moving the compact-MFG destination-prefix proof out of the attention hot
  path measured 0.33511 s/epoch when revalidated on CPU and 0.32542 when
  carried as metadata, versus the 0.32533 repeat baseline. Both were removed.
- Reusing target-route rows in TGN state update reduced isolated Python time
  but two end-to-end runs measured 0.33266 and about 0.336 s/epoch. A float32
  optimizer-supervision collective measured 0.33343 s/epoch. Both were removed.
- Fusing K/V projection reduced peak allocated/reserved memory from about
  1.334/5.362 GB to 1.241/4.167 GB. Its two timing runs were 0.31274 and
  0.32533 s/epoch, but final test AP/AUC fell to 0.90874/0.90349. Splitting the
  projection by compact node/edge/time components reduced memory further to
  1.142/3.506 GB and measured 0.32455 s/epoch, but ended at
  0.90901/0.90367. Both floating-point reorderings failed the accuracy gate
  and were removed.
- `torch.compile(dynamic=True)` was tested only in the thin diagnostic wrapper
  and could not start because the installed PyTorch/Triton APIs are
  incompatible (`get_cuda_stream` is missing). Dependencies were not changed.
  DGL attention, dense gradient bucketing, eager feature finish, delayed `col`
  materialization, native root-row export, and compact raw edges were already
  measured and rejected; no custom C++/CUDA operator is justified by these
  results.

Files modified:

- `docs/design/memshare_temporal_target_identity_20260913.md`

Verification:

- Exact native-attention output/gradient and temporal-model checks passed
  before each projection benchmark. The final production tree is unchanged
  and `git diff --check` passes.
- Outputs are `/tmp/starrygl_tgn_{prefix_hint_candidate,native_prefix_marker,
  float_active_flag,fused_kv_projection,compact_kv_components}*` and
  `/tmp/starrygl_tgn_dfc53a7_baseline_repeat_eval_e10`.

Unresolved risks:

- The 26--28% WIKI throughput gap is now dominated by the aggregate
  accessor/communication and many small model kernels, not one safe Python
  lookup. Further work needs either a verified communication reduction or a
  numerically stable fused operator with a full training accuracy gate.
- DCRNN's 2 s/epoch gate remains unmet at 2.0838 s/epoch.

## 2026-09-22: Preserve native sampled edge multiplicity

Latest state:

- The common Prepare -> task slice -> negatives -> native accessor -> Batch ->
  dependencies -> model -> task -> state update path is unchanged. Native MFG
  conversion is the specialization shared by Event and Snapshot-neighbor access.
- Runtime native blocks now retain sampler edge multiplicity by default, matching
  MemShare. `deduplicate_edges=True` remains an explicit opt-in; feature fetches
  still use the existing vectorized unique physical edge rows.
- No cache, adapter, model branch, loader, training loop, or native operator was
  added.

Efficiency alternatives considered:

- Profiling showed Python/Torch GraphBlock conversion, not native neighbor
  selection, was dominant. Retaining sampler output deletes the redundant native
  CSC dedup pass. DGL reconstruction and a new C++/CUDA kernel would add work.
- Delaying `col` materialization was also measured, but did not improve the
  10-epoch run and is not retained.

Files modified:

- `src/starrygl/runtime/sample/{__init__.py,CONTRACT.md}`
- `tests/test_runtime_graph_blocks.py`
- `docs/design/memshare_temporal_target_identity_20260913.md`

Verification:

- Focused runtime/model checks: 20 passed, 1 native-library skip. Full suite:
  541 passed, 45 skipped and the same nine pre-existing plan/default and retired
  smoothing-helper failures. Production compile-check and diff check passed.
- Four-A40 WIKI profiling over three epochs reduced rank-max GraphBlock build
  time from 0.20205 to 0.01507 s/epoch; retained edges increased only 0.058%.
- The final default 10-epoch no-evaluation run reduced epochs 2--10 from 0.39365
  to 0.32322 s/epoch (17.9%). Native MemShare remains 0.25505 s/epoch, so the
  remaining gap is 26.7% and performance parity is not yet reached.
- The aligned train -> validation -> test run ends at test AP/AUC
  0.91508/0.91062, within 0.379/0.465 percentage points of MemShare. Outputs are
  `/tmp/starrygl_tgn_native_multiplicity_default_{e10,eval_e10}`.

Unresolved risks:

- Other datasets may expose larger sampler multiplicity and need an explicit
  choice before cross-system accuracy claims.
- DCRNN uses the Snapshot path and is unaffected; its 2 s/epoch gate remains
  unmet at 2.0838 s/epoch.

## 2026-09-22: Make negative loss weighting final-ID aware

Latest state:

- Prepare -> task slice -> negative materialization -> graph accessor -> Batch
  -> dependencies -> model -> task -> state update remains the shared path.
- `NegativeSamplePool` has one optional vectorized
  `loss_weight_fn(sampled_ids, pool)`. It runs after destination sampling, so an
  overlapping global-pool draw can receive the same weight as a local-pool draw
  of the same final ID.
- Removed the internal string-selected inverse/MemShare formulas. Unit weight
  remains the default; a nonstandard target distribution must provide its
  explicit callable at the task/sampler boundary.

Efficiency alternatives considered:

- The existing Torch sampling and one batched callable were selected. Branch
  tags cannot express final-ID locality when candidate sets overlap; another
  sampler, model branch, DGL graph, or native kernel is unnecessary.

Files modified:

- `src/starrygl/task/{target.py,negative.py,CONTRACT.md}`
- `src/starrygl/runtime/event/target.py`
- `tests/test_starrygl_task.py`
- `docs/design/memshare_temporal_target_identity_20260913.md`

Verification:

- Focused task/runtime regression: 20 passed. TGN/MemShare math and model
  regression: 51 passed, 7 device/distributed skips. Production compile-check
  passed; touched modules remain below 500 lines.
- The new check draws from overlapping local/global candidate sets and verifies
  that equal final IDs receive equal callable weights regardless of branch.
- Fresh four-A40 WIKI, 10 epochs, one shared train -> validation -> test
  lifecycle per epoch: final StarryGL/MemShare validation AP is
  0.92619/0.93367 and test AP is 0.91321/0.91886; test AUC is
  0.90853/0.91527. The test AP/AUC gaps are 0.565/0.674 percentage points.
  Random IDs are not paired, but candidate distributions, final-ID weights,
  splits, metric implementation, and lifecycle are aligned.
- In the no-evaluation timing run, StarryGL final rank-mean BCE is 0.88886
  versus the fresh native 0.89238 (0.4% difference). Epochs 2--10 take
  0.39365 s/epoch versus native 0.25505 s/epoch: 54.3% slower. The callable
  adds only 0.6% over the prior StarryGL timing, so it is not the performance
  bottleneck.
- Outputs: `/tmp/starrygl_tgn_62e25df_{prepare,weighted_e10,weighted_eval_e10}`
  and `/tmp/memshare_native_eval_20260922`. The existing thin diagnostic
  wrapper under `../paper_method/rebuttal_20260913/` only injects the callable
  and records evaluation; it does not replace `Trainer.fit/evaluate`.

Unresolved risks:

- WIKI is one pilot and does not prove exact state-trace or multi-dataset
  parity. Fixed negative-ID replay is unnecessary for the requested stochastic
  tolerance but remains useful for a stricter numerical diagnostic.
- TGN performance is still not at MemShare parity; profiling must target the
  common accessor/dependency/model path rather than negative weighting.

## 2026-09-22: Align event negative pools with node replicas

Latest state:

- The shared path remains Prepare window row -> owner task slice -> negatives ->
  graph accessor -> Batch -> dependencies -> model -> task -> state update. The
  only specialization is Prepare's event negative candidate set.
- `event_view.dst_pool` now means destination IDs readable from the rank's
  authoritative nodes or shared-hot replicas. It no longer follows locally
  owned edges, which are the task/output ownership plane rather than the node
  replica plane.
- The implementation reuses packed `node_dist_index`, `node_is_hot`, and one
  vectorized Torch mask. No cache class, fallback artifact reader, task/model
  branch, communication, or Python entity loop was added.

Efficiency alternatives considered:

- The selected Torch filter runs once during Prepare. DGL graph construction
  and a C++/CUDA operator add overhead to a non-hot-path ownership lookup.
- Runtime reconstruction would preserve stale prepared artifacts but repeat
  invariant work and maintain two meanings for `dst_pool`; artifacts should be
  regenerated instead.

Files modified:

- `src/starrygl/prepare/event.py`
- `tests/test_partition_assignment.py`
- `docs/design/memshare_temporal_target_identity_20260913.md`

Verification:

- Focused regression: 41 passed, 1 CUDA skip. The larger first run was 47
  passed, 1 skipped plus the known unrelated snapshot-hot plan assertion.
- Fresh WIKI audit over 1,000 global destinations produced rank-local pool
  counts 663/675/674/671, exactly matching the native MemShare ownership and
  hot-replica mapping. The production module compile-check passed.

Unresolved risks:

- MemShare's final-ID loss weighting and fixed evaluation-negative protocol
  remain to be aligned. This change fixes candidate locality only.

## 2026-09-22: Reuse snapshot cache for DCRNN candidate input

Latest state:

- The shared path remains Prepare -> owner task slice -> Snapshot accessor ->
  Batch -> dependency access -> coupled scan -> task -> state commit. DCRNN's
  unavoidable specialization is its gate-produced candidate input.
- Bounded-stale DCRNN now lowers
  `neighbor_recurrent.candidate_input` at `before_candidate`. The runtime caches
  `reset * previous_state` with the existing cumulative-increment
  `SnapshotHistory`; the model remains one direct DCRNN implementation.
- Exact DCRNN exchanges the same candidate input with the existing autograd
  Route. Stale execution overlays current local rows on extrapolated detached
  remote rows. Other models create no empty channel.
- Coupled runtime no longer recognizes `candidate_input`: it predicts declared
  channels and calls the cell's optional `materialize_cached(...)` hook. DCRNN
  owns its gate/candidate formula and local-row overlay. Future R-GraphSAGE can
  use the same storage/communication hook once its model equations are verified.
- Channel allocation is limited to the fixed producer/consumer boundary union.
  State and channel packets share one final-boundary Route payload and global
  collective order. No full-node channel replica, Python entity loop, new
  public state kind, cache class, loader or training loop was added.

Efficiency alternatives considered:

- Reusing `SnapshotHistory`, Torch indexing and the prepared Route was selected.
  A new cache hierarchy, per-layer full-node allocation, DGL kernel change or
  C++/CUDA operator adds no value to this versioned tensor storage path.
- Publishing immediately after the gate could expose more overlap, but requires
  another globally scheduled collective epoch. It is deferred until profiling
  shows the final-boundary publication is insufficient.

Files modified:

- `src/starrygl/{plan.py,store/snapshot_history.py}`
- `src/starrygl/model/{dcrnn.py,gconv_gru.py}`
- `src/starrygl/runtime/{memory/__init__.py,memory/snapshot.py,state/build.py}`
- `src/starrygl/runtime/snapshot/{coupled.py,layerwise.py,scan.py}`
- `tests/{test_snapshot_history.py,test_open_compile_plan.py}`
- `docs/design/current/snapshot_boundary_history.md`

Verification:

- Focused CPU after the generic cell hook: 74 passed, 11 skipped; one unrelated
  pre-existing stale plan assertion deselected.
- Two-rank CPU/Gloo: 4 passed per rank, covering exact GConvGRU/DCRNN output and
  parameter-gradient parity plus bounded-stale compile/Prepare/fit/evaluate/
  predict. State/cache versions advance continuously through train/val/test and
  all pending boundary pushes drain.
- The first lifecycle assertion incorrectly counted two validation snapshots as
  two evaluation steps and one prediction callback. The real window contract is
  one two-snapshot evaluation step and two per-window prediction outputs; only
  the test expectations changed.
- Random initialization and sampled RNG identity are deliberately not parity
  gates here. The gate is one training/inference spine, matching state/cache
  lifecycle, exact-mode math, collective order and successful task execution.
- Full CPU regression: 533 passed, 10 failed, 49 skipped. Nine failures exactly
  match the pre-existing main-branch list; the tenth was the sandbox blocking
  Gloo address resolution. That two-process test passed separately outside the
  sandbox (1 passed), so this change introduces no new regression failure.

Unresolved risks:

- No GPU memory/throughput measurement or convergence result yet.
- Publication is still final-Batch-boundary, not gate-stage overlap.
- Only the current single-stage DCRNN is implemented; future stacked recurrent
  stages should bind one channel each only when the model exists.
- R-GraphSAGE model math remains unavailable; this version validates its cache
  integration point, not the named model or its quality/performance.

## 2026-09-12: Promote verified rebuttal V3 into the main package

Latest state before implementation:

- Main production hashes match all 106 files of the archived original Flickr
  window-slots manifest; no unique production edits need merging. V3 matches
  its frozen `empty_step_source_verification.json`. Promote the 19 changed
  production modules and seven corresponding tests, leaving all isolated
  sources and historical experiment records untouched.
- Common path and specialization were traced before edits and recorded in
  `docs/design/current/rebuttal_v3_integration.md`: Prepare/task rows -> optional
  negatives -> bound graph accessor -> Batch -> dependencies -> model -> task
  -> state update. Event/Snapshot and node/edge retain the same loader/loop;
  only static graph reuse and snapshot-history dependency access specialize.
- Reuse existing SnapshotHistory, GraphBlock caches and CommScheduler; no new
  public abstraction. Reused optimizer helper covers empty/nonempty branches.
  Torch/DGL operators implement tensor/graph work. DGL kernel-inspired or
  custom C++/CUDA code would not improve deletion of an unconsumed exchange or
  static metadata reuse; neither is added. No per-entity Python loops.
- Attempted alternatives: reviewed partial promotion, but model smoothing,
  split local/shared history and exact W>1 exchange form an already-validated
  chain. Static Prepare target flags cannot replace post-hook supervision.
  Keep the one scalar MAX collective required for the correct Adam step.
- Files to promote: `cli/coupled_ablation.py`; `model/{dcrnn,gconv_gru}.py`;
  `plan.py`; `runtime/dataloader/{blocks,loader,pipeline}.py`;
  `runtime/{epoch,loop}.py`; `runtime/memory/{__init__,shared,snapshot}.py`;
  `runtime/snapshot/{cache,coupled,layerwise,materialize,scan}.py`;
  `runtime/state/build.py`; `store/state.py`. Corresponding tests:
  coupled_ablation, snapshot_history, runtime_graph_blocks,
  snapshot_unused_hot_sync, snapshot_window_graph_reuse,
  empty_supervision_step and rebuttal_smoothing.
- Before promotion: focused V3 CPU suite 50 passed / 10 distributed skipped in
  2.98 s. Archived V3 NCCL seven tests/rank cover Adam, exact outputs/gradients
  and public fit. Main-tree checks remain pending.
- Limits: no new GPU performance claim, V1/V2 accuracy remains on the old Adam
  schedule, scalar supervision all-reduce cost unmeasured, publication skip
  cap still does not enforce a version-age bound. DCRNN reset-gate exchange
  remains required, and R-GraphSAGE is not introduced.

## 2026-08-03: Canonical Compile Semantic Arguments

Latest state:

- Narrowed the public Python `compile()` surface to the canonical semantic
  segments:
  - `data_source`
  - `backbone`
  - `task_segment`
- Removed `graph`, `model`, and `task` from the explicit `compile()`
  signature. These names are now treated as configuration-file section names,
  not Python API names.
- Added an explicit guard so legacy `compile(graph=..., model=..., task=...)`
  keywords are rejected instead of being silently captured by metadata.
- `from_config()` now translates config sections into the canonical compile
  arguments:
  - canonical `data` or legacy `graph` -> `data_source`
  - canonical `backbone` or legacy `model` -> `backbone`
  - `task` -> `task_segment`
- Mixed canonical and legacy data/model sections are rejected.
- Fixed config spec derivation so `execution.exact` is applied after
  model-name default spec inference instead of replacing temporal/state/scope
  with a consistency-only spec.

Abstraction introduced:

- No runtime abstraction introduced.
- This step introduces a narrow config translation boundary: config section
  names may remain legacy-compatible, but the Python API has one canonical
  semantic entry style.

Efficiency alternatives considered:

- This step changes config/API normalization and tests only.
- No tensor gather, sampler, DGL, communication, state hydration, model, or
  runtime hot path was changed.

Files written or modified:

- `src/starrygl/interface/api.py`
- `tests/test_open_compile_semantics.py`
- `tests/test_open_compile_plan.py`
- `tests/test_executor_state_request.py`
- `docs/migration/status_log.md`

Verification:

- Python compile check: `src/starrygl/interface/api.py` passed.
- Open compile semantics tests: 7 passed.
- Open compile plan tests: 10 passed.
- Executor state request tests: 5 passed.

Unresolved risks:

- `tests/test_starrygl_compile_plan.py` still contains old/stale plan
  assertions beyond the argument-name migration. After updating its call sites
  to `data_source/backbone/task_segment`, 2 tests passed and 21 failed due to
  pre-existing expectation drift, including assertions for old dependency
  lists, `wait_queue`, old distributed snapshot helpers, and legacy runtime
  await/sync resolver behavior. This file needs a separate old-plan assertion
  audit rather than being folded into this API-surface change.

Follow-up cleanup:

- Shortened the long `compile()` docstring to the public boundary statement.
- Removed duplicate model-name default spec construction by introducing one
  `_default_spec_for_model()` helper shared by direct compile inference and
  config spec inference.
- Consolidated duplicate negative-sampling service normalization in
  `interface/spec.py`.

Additional verification:

- Python compile check: `src/starrygl/interface/api.py` and
  `src/starrygl/interface/spec.py` passed.
- Open compile semantics, open compile plan, and executor state request tests:
  22 passed.

Naming cleanup:

- Removed config section aliasing from `interface/api.py`.
- `from_config()` now accepts only `data`, `backbone`, and `task` for the
  semantic segments. Legacy `graph`, `model`, and `gnn` sections are rejected
  instead of translated.
- Removed `gnn` to `runtime` compatibility merging from `api.py`.
- Tightened semantic value parsing in `api.py`: model names are read from
  `name`, temporal representation from `event_stream` or `snapshot_sequence`,
  and aggregation from `sampled_neighbor` or `full_neighbor`.

Naming cleanup verification:

- Python compile check: `src/starrygl/interface/api.py` passed.
- Open compile semantics, open compile plan, and executor state request tests:
  22 passed.

## 2026-08-02: State Read Request Template Lowering

Latest state:

- Added executor-internal `StateReadRequest` templates.
- Added `state_read_requests_from_plan(plan)` to lower
  `ExecutionPlan.state_dependencies` into runtime state-read templates.
- The request template carries model-facing state key, state kind, internal
  stage, freshness/cache/wait policy, owner policy, max staleness, and
  approximation metadata.
- Concrete batch node ids are intentionally not part of this template yet.
  They will be bound later from sampler/view materialization output.
- Exported the helper from `starrygl.executor.runtime` for internal executor
  use.

Abstraction introduced:

- `StateReadRequest` is the narrow runtime boundary between plan state
  dependencies and future state hydration scheduling.
- It is not a user API and does not change model execution; models still
  consume `Batch.state`.

Efficiency alternatives considered:

- This step performs compile/runtime-template lowering only.
- No tensor gather, sampler, DGL, communication, state manager, or hot-path
  hydration logic was changed.
- The next implementation step must bind requests to concrete node tensors
  with vectorized tensor scopes from sampling/view materialization; per-node
  Python loops remain unacceptable for the hot path.

Files written or modified:

- `src/starrygl/executor/runtime/state_request.py`
- `src/starrygl/executor/runtime/__init__.py`
- `tests/test_executor_state_request.py`
- `docs/migration/status_log.md`

Verification:

- Python compile check: `src/starrygl/executor/runtime/state_request.py` and
  `src/starrygl/executor/runtime/__init__.py` passed.
- State request and open compile plan tests: 15 passed.
- Default open-source test suite: 145 passed, 1 skipped.

Unresolved performance risks:

- None introduced.
- Concrete request binding and actual state hydration still need hot-path
  review before they consume `StateReadRequest` in training.

## 2026-07-28: Model Backbone State-Key Semantics

Latest state:

- Added `ModelBackbone.state_key`, `aggregate_key`, and
  `requires_temporal_state`.
- `normalize_backbone()` preserves the new semantic fields for explicit
  backbone declarations.
- `sg.compile()` now derives default state semantics from explicit backbone
  fields first, then from model-name defaults:
  - TGN/JODIE/APAN -> event persistent state
  - DCRNN -> coupled snapshot recurrent state
  - TGCN/MPNN-LSTM/EvolveGCN -> snapshot recurrent state
  - TGAT -> stateless
- `nn.Module` models may provide a `backbone_semantics` attribute; compile
  uses it when present.
- `ExecutionPlan` now exposes `model_state_key`, `model_aggregate_key`, and
  `requires_temporal_state` as semantic observability fields.
- No `state_read_stage` public field was added.

Abstraction introduced:

- The new backbone fields are semantic declarations for compile lowering. They
  do not create a second model execution API; models still consume `Batch`.
- Model-name defaults remain migration conveniences and are not the long-term
  source of execution behavior.

Efficiency alternatives considered:

- This step changes compile-time semantics and tests only.
- No tensor, sampler, DGL, communication, model, or runtime hot path was
  touched.

Files written or modified:

- `src/starrygl/interface/spec.py`
- `src/starrygl/interface/api.py`
- `src/starrygl/interface/plan.py`
- `docs/STARRYGL_INTERFACE.md`
- `docs/design/current/execution_spine.md`
- `tests/test_open_compile_plan.py`
- `docs/migration/status_log.md`

Verification:

- Python compile check: `src/starrygl/interface/spec.py`,
  `src/starrygl/interface/api.py`, and `src/starrygl/interface/plan.py`
  passed.
- Open public API, compile plan, and compile semantics tests: 22 passed.
- Default open-source test suite: 140 passed, 1 skipped.

Unresolved performance risks:

- None introduced.
- Runtime still needs a later lowering step from plan state dependencies to
  concrete state hydration and access scheduling.

## 2026-07-27: User Execution Exact Mapping

Latest state:

- Added user-facing `execution.exact` config lowering:
  - `execution.exact=true` -> internal `spec.consistency=exact`
  - `execution.exact=false` -> internal `spec.consistency=bounded_stale`
- `execution.max_staleness` and `execution.approximation` are copied into the
  internal spec for approximate execution.
- Direct spec mappings may also use `exact` as a boolean alias.
- Conflicting `execution.exact` and `spec.consistency` values are rejected at
  config load time.
- Open public API tests now assert that `execution.exact=false` makes only
  state dependencies bounded-stale while feature and task dependencies remain
  exact.

Abstraction introduced:

- No runtime abstraction introduced.
- This is a narrow config normalization layer that keeps the public interface
  on `exact` while preserving the existing internal `StarrySpec` fields.

Efficiency alternatives considered:

- This step changes config parsing and tests only.
- No tensor, sampler, DGL, communication, model, or runtime hot path was
  touched.

Files written or modified:

- `src/starrygl/interface/spec.py`
- `src/starrygl/interface/api.py`
- `docs/STARRYGL_INTERFACE.md`
- `tests/test_open_public_api.py`
- `docs/migration/status_log.md`

Verification:

- Python compile check: `src/starrygl/interface/spec.py` and
  `src/starrygl/interface/api.py` passed.
- Open public API, compile plan, and compile semantics tests: 20 passed.
- Default open-source test suite: 138 passed, 1 skipped.

Unresolved performance risks:

- None introduced.
- Runtime still consumes existing plan/runtime fields; later work must lower
  structured dependencies into actual fetch/await scheduling.

## 2026-07-27: Compile Dependency Lowering Split

Latest state:

- Split compile-time await dependency generation into feature, state, and task
  groups.
- Feature dependencies now include exact `x` reads and exact `edge_feat` reads
  for edge tasks.
- State dependencies still lower from temporal state semantics:
  - `persistent` -> `node_memory` and `mailbox`
  - coupled snapshot recurrent -> `neighbor_recurrent`
  - decoupled snapshot recurrent -> `node_recurrent`
- Task dependencies now include exact endpoint embeddings for edge tasks, exact
  labels for supervised tasks, and exact negative targets when negative
  sampling is configured.
- Approximate consistency now affects state dependencies only. Feature and task
  dependencies remain exact even when state freshness is bounded.
- Added `ExecutionPlan.feature_dependencies`, `state_dependencies`,
  `task_dependencies`, and `remote_dependency_sources` observability helpers.

Abstraction introduced:

- No runtime abstraction introduced.
- The new helpers are plan observability filters over the existing
  `await_dependencies` tuple. They keep the current executor boundary stable
  while making dependency classes explicit for later lowering.

Efficiency alternatives considered:

- This step changes compile-time plan construction and tests only.
- No tensor, sampler, DGL, communication, model, or runtime hot path was
  touched.
- The next runtime lowering step must still choose between torch/DGL operators,
  DGL-kernel-inspired implementations, and custom C++/CUDA for actual
  materialization, packing, and communication.

Files written or modified:

- `src/starrygl/interface/plan.py`
- `tests/test_open_compile_plan.py`
- `tests/test_open_compile_semantics.py`
- `docs/migration/status_log.md`

Verification:

- Python compile check: `src/starrygl/interface/plan.py` passed.
- Open compile plan and semantics tests: 13 passed.
- Default open-source test suite: 136 passed, 1 skipped.

Unresolved performance risks:

- None introduced.
- The plan now observes feature/task dependencies, but runtime fetch/materialize
  paths still need a separate hot-path review before consuming those structured
  groups directly.

## 2026-07-27: Public Consistency And Remote Dependency Contract

Latest state:

- Documented the open consistency boundary:
  - users see exact versus approximate consistency
  - stale state compensation and layer-wise exchange stay internal runtime
    strategies
  - `D_remote` is derived from plan dependencies, not user-authored config
- Documented `s` as the temporal state dependency key and `h` as the
  aggregation hidden key. Both are compile-time semantics and observability
  aids; models still consume unified `Batch`.
- Added open plan assertions that:
  - exact state dependencies lower to exact freshness and collective awaits
  - approximate state dependencies lower to bounded freshness with the selected
    approximation metadata
  - `D_remote` is derived from await dependency fulfillment
  - `plan.explain()` does not expose internal strategy names such as
    `stale_increment`, `timestamp_increment`, `layerwise`, or `memshare`

Abstraction introduced:

- No runtime abstraction introduced.
- This step tightens the public interface contract around existing
  `ExecutionPlan.await_dependencies`, `dependency_sources`, and `D_remote`
  observability.

Efficiency alternatives considered:

- This step changes documentation and tests only.
- No tensor, DGL, sampler, communication, model, or runtime hot path was
  touched.
- Runtime implementation choices remain open for later review: torch/DGL
  operators, DGL-kernel-inspired kernels, or C++/CUDA operators depending on
  the hot-path profile.

Files written or modified:

- `docs/design/current/execution_spine.md`
- `docs/STARRYGL_INTERFACE.md`
- `tests/test_open_compile_plan.py`
- `docs/migration/status_log.md`

Verification:

- Open compile plan test: 8 passed.
- Default open-source test suite: 136 passed, 1 skipped.

Unresolved performance risks:

- None introduced.
- The physical stale compensation and graph communication schedules still need
  separate implementation review before becoming runtime defaults.

## 2026-07-27: Open Compile Plan Gating

Latest state:

- Added `tests/test_open_compile_plan.py` as the open-source compile/plan
  semantic gating test.
- The new test covers retained open plan semantics:
  - event sampled lowering to `temporal_sampling_view` and `Batch.blocks`
  - snapshot full-graph lowering to `snapshot_block_view` and `Batch.graph`
  - sampled snapshot lowering to `temporal_sampling_view` and `Batch.blocks`
  - observable await dependencies and dependency sources
  - `cache_policy=none + wait_policy=reschedule`
  - shared-hot cache preserving owner-plane task ownership
  - canonical cache/freshness/wait policy lowering through
    `ExecutionOverride`
- Legacy-only assertions from `tests/test_starrygl_compile_plan.py` were not
  migrated into open gating.

Abstraction introduced:

- No runtime abstraction introduced.
- The new test file is a gating boundary that defines the currently accepted
  open compile/plan contract.

Efficiency alternatives considered:

- This step only adds tests.
- No tensor, sampler, DGL, communication, model, or runtime hot path was
  touched.

Files written or modified:

- `tests/test_open_compile_plan.py`
- `docs/migration/status_log.md`

Verification:

- Open compile plan test: 6 passed.
- Default open-source test suite: 134 passed, 1 skipped.

Unresolved performance risks:

- None introduced.
- Snapshot layout remains asserted only as a semantic `snapshot_block_view`
  surface; CSR-vs-CSC physical layout still needs a separate design decision
  before being fixed as public open behavior.

## 2026-07-24: Legacy Plan Assertion Audit

Latest state:

- Reviewed all assertions in `tests/test_starrygl_compile_plan.py`.
- Added `docs/migration/legacy_plan_assertion_audit.md` to classify each old
  plan/config test as:
  - keep as open semantics
  - keep only after design decision
  - quarantine as legacy compatibility
  - remove or replace
  - keep as validation rules if still desired
- No runtime behavior was changed in this step.

Abstraction introduced:

- No code abstraction introduced.
- The audit is a migration decision document for converting legacy plan tests
  into focused open plan tests.

Efficiency alternatives considered:

- This step only reviews tests and documentation.
- No tensor, DGL, communication, sampler, or model hot path was touched.

Files written or modified:

- `docs/migration/legacy_plan_assertion_audit.md`
- `docs/migration/status_log.md`

Verification:

- No test run was needed for documentation-only classification.
- Previous default open gating remains: 128 passed, 1 skipped.

Unresolved performance risks:

- None introduced.
- Follow-up implementation work must still review any retained plan semantics
  that affect hot paths, especially snapshot CSR/CSC layout selection and
  shared-hot await scheduling.

## 2026-07-24: Open Public API Gating And Legacy API Quarantine

Latest state:

- Added `tests/test_open_public_api.py` as the open-source public API gating
  test for the retained entry/config surface.
- `tests/test_starrygl_public_api.py` and
  `tests/test_starrygl_compile_plan.py` are now treated as legacy migration
  reference tests and are skipped by default collection.
- Legacy API reference tests can still be run explicitly with
  `STARRYGL_RUN_LEGACY_API_TESTS=1`.
- `starrygl.runtime.ArtifactBundle` and `artifact_bundle(...)` provide a
  native artifact inspection surface without importing legacy packages.
- `starrygl.spec.MemShareNegativeSample` is a lazy module-level alias to
  `SharedHotNegativeSample` for migration compatibility, while the top-level
  `starrygl` package still does not expose `MemShareNegativeSample`.

Abstraction introduced:

- The pytest collection hook is a temporary migration boundary between open
  public API gating and legacy API reference tests. It does not remove the
  legacy tests; it prevents them from defining the open-source completion bar.
- `ArtifactBundle` is a small file-indexing helper for open artifacts. It is
  not a runtime execution object.

Efficiency alternatives considered:

- This step changes import/test boundaries and adds artifact file indexing
  only. No tensor training hot path was touched.
- DGL operators, DGL-kernel-inspired implementations, and custom C++/CUDA are
  not relevant for this gating boundary.

Files written or modified:

- `src/starrygl/runtime/artifact.py`
- `src/starrygl/runtime/__init__.py`
- `src/starrygl/spec.py`
- `tests/conftest.py`
- `tests/test_open_public_api.py`
- `docs/migration/status_log.md`

Verification:

- Modified runtime/spec/test modules pass Python compile checks.
- Open public API test: 5 passed.
- Default open-source test suite: 128 passed, 1 skipped.
- Explicit legacy reference run with
  `STARRYGL_RUN_LEGACY_API_TESTS=1` still fails: 89 failed, 28 passed. The
  failures remain concentrated in legacy plan assertions, missing old config
  files, model zoo compatibility, registry API, private trainer batch methods,
  and broader legacy evaluate/fit behavior.

Unresolved performance risks:

- None introduced by this step.
- The main unresolved migration risk is architectural, not performance: decide
  which legacy plan/config assertions become open semantics and which remain
  quarantined migration references.

## 2026-07-24: Public Data Loader And Snapshot Artifact Boundary

Latest state:

- `starrygl.data` now exists and exports `DatasetLoader`, `NodeOwnerIndex`,
  `load_dataset`, and `flatten_tgb_node_label_dict`.
- `DatasetLoader` reads lightweight open artifacts using `graph.pt`,
  `dist.pt`, `rank_XXX.pt`, and `feature_XXX.pt`.
- TGB/TGM-style `node_label_dict` is flattened into node-label tensors with
  split ids inferred from edge timestamp/split ranges.
- `GraphStore` can now carry lightweight artifact fields: `edge_ids`,
  `timestamps`, `snapshot_ptr`, and `edge_weight`, plus `snapshot(index)`.
- `starrygl.view.snapshot` now provides `SnapshotBlockView`,
  `SnapshotWindow`, and `SnapshotWindowBlob` on top of the existing
  `GraphBlock` tensor-buffer contract.
- `Trainer.prepare_artifacts(...)` has a minimal open artifact writer for
  raw DTDG snapshot data and a narrow legacy preprocess hook point for
  CTDG/event compatibility tests.

Abstraction introduced:

- `DatasetLoader` is a small artifact reader for the current `graph.pt` style
  tests and examples. It is not a new execution runtime and does not own
  distributed scheduling.
- `SnapshotBlockView` is a view/materialization helper for snapshot graph
  tensors. It reuses `GraphBlock` directly and does not introduce a separate
  model-facing batch API.
- `NodeOwnerIndex` is a small structured return value for owner/local-row
  lookup. It avoids importing old runtime fetch modules.

Efficiency alternatives considered:

- Data loading uses tensor indexing and row-map tensors. No training hot-path
  per-node/per-edge loop was added.
- Snapshot degree artifact construction uses torch `index_add_` per snapshot
  window. This is acceptable for artifact preparation, not runtime training.
- DGL operators are not needed for the loader boundary; snapshot view still
  materializes direct CSR/COO tensor buffers.
- A custom C++/CUDA operator is not justified for this preparation-only degree
  feature path without profiling evidence.

Files written or modified:

- `src/starrygl/data/__init__.py`
- `src/starrygl/data/labels.py`
- `src/starrygl/data/loader.py`
- `src/starrygl/store/graph.py`
- `src/starrygl/view/snapshot.py`
- `src/starrygl/view/__init__.py`
- `src/starrygl/executor/runtime/trainer.py`
- `src/starrygl/__init__.py`
- `docs/migration/status_log.md`

Verification:

- Modified data/view/trainer modules pass Python compile checks.
- Migration/runtime focused sweep: 116 passed.
- `test_starrygl_public_api.py` now collects and runs, but the full file still
  fails because many old public surfaces are not yet migrated: runtime
  artifact bundle exports, model zoo compatibility module, registry API,
  several missing config files, `gnn_config`, old private trainer batch
  methods, and broader evaluate/fit legacy behavior.

Unresolved performance risks:

- `Trainer.prepare_artifacts` snapshot degree construction is preparation-only
  and not benchmarked.
- `SnapshotBlockView.window_from_store` has a simple local cache per call. A
  runtime-owned cache should be reviewed later if this path becomes hot.
- The legacy preprocess hook remains a documented compatibility point for
  tests; it should be removed once CTDG raw-source preparation is fully moved
  to the open prepare path.

## 2026-07-24: Executor GraphBlock And MFG Boundary

Latest state:

- `starrygl.executor.runtime.graph_blocks` owns executor-side `GraphBlock`
  construction, sampled MFG normalization, sampled feature graph construction,
  and sampled edge id aggregation.
- Event runtime now delegates event/root graph construction and sampled MFG
  edge/source-node extraction to `graph_blocks.py`.
- Snapshot runtime now delegates CSC snapshot `GraphBlock` construction,
  reverse-direction buffers, graph-block device moves, store world-size
  detection, and sampled edge id aggregation to `graph_blocks.py`.
- Sampling runtime now delegates native sampled block normalization and
  feature/target block selection to `graph_blocks.py`.
- Old private runtime names such as `_snapshot_graph_block`,
  `_move_snapshot_graph_block`, and `_sampled_edge_ids` are retained only as
  transitional aliases to the single implementation.

Abstraction introduced:

- `graph_blocks.py` is a narrow executor tensor-layout boundary for existing
  `GraphBlock` objects. It is not a new public block API and does not add a
  `BlockView`/`BlockLayout` split.
- The abstraction is necessary now because event, snapshot, and sampling paths
  were maintaining duplicate GraphBlock/MFG helpers while the model-facing
  contract is supposed to consume one unified `Batch.blocks`/`Batch.mfgs`
  representation.

Efficiency alternatives considered:

- The selected path keeps the existing torch tensor operations and moves them
  behind one implementation; no new per-node/per-edge Python loops were added.
- DGL operators are not introduced here because this boundary preserves direct
  tensor-buffer `GraphBlock` layouts and avoids making DGL block construction
  the hot path.
- DGL-kernel-inspired or custom C++/CUDA work remains relevant for future
  sampler output compaction and CSC/native materialization if profiling shows
  these tensor operations dominate.
- `sampled_edge_ids` preserves the previous unique/sorted aggregation behavior
  so edge feature fetch scope is not widened by duplicate sampled edges.

Files written or modified:

- `src/starrygl/executor/runtime/graph_blocks.py`
- `src/starrygl/executor/runtime/sampling.py`
- `src/starrygl/executor/runtime/event.py`
- `src/starrygl/executor/runtime/snapshot.py`
- `src/starrygl/executor/runtime/__init__.py`
- `tests/test_executor_graph_blocks.py`
- `docs/migration/status_log.md`

Verification:

- Modified graph-block/event/snapshot/sampling modules pass Python compile
  checks.
- Focused graph-block/model tests: 47 passed.
- Runtime/API/model/store/task sweep: 115 passed, 1 skipped.
- Full `python -m pytest tests -q` currently fails during collection because
  `tests/test_starrygl_public_api.py` imports `starrygl.data.DatasetLoader`,
  which is not present in the current open tree.
- Coverage shows duplicate local helper definitions for event/snapshot
  sampled edge ids, snapshot graph block construction, graph-block moves, and
  store world-size detection have been removed from event/snapshot runtime.

Unresolved performance risks:

- This step does not benchmark GraphBlock materialization or MFG normalization.
- Reverse-direction buffer construction still uses torch operations on the
  runtime path; if snapshot materialization profiling shows this is hot, it
  should be reviewed for DGL-kernel-inspired or native implementation.
- Native sampler output compaction and multi-layer block shape semantics still
  need a later pass before changing the current `native_feature_block` and
  `native_target_block` behavior.

## 2026-07-24: Executor Target Materialization Boundary

Latest state:

- `starrygl.executor.runtime.targets` owns executor-side supervision target
  construction.
- Event runtime no longer directly constructs model-facing `TaskTarget` or
  event runtime target dictionaries; it delegates to target helpers.
- Snapshot runtime no longer directly constructs model-facing `TaskTarget`,
  snapshot runtime target dictionaries, or snapshot negative-destination pool
  probability normalization.
- Sampling-root targets for event and snapshot neighbor sampling are also
  constructed through executor target helpers.
- Negative sample materialization is wrapped at the executor target boundary so
  event/snapshot materialization does not directly call the task negative
  sampling function.

Abstraction introduced:

- `targets.py` is a narrow executor construction boundary for existing
  `TaskTarget`, `TargetRoute`, and `NegativeSamplePool` objects. It is not a
  public target API and does not introduce a registry or adapter layer.
- This boundary is necessary now because event and snapshot materialization
  both build the same model/task-facing supervision objects while keeping
  route computation and graph sampling local to their existing modules.

Efficiency alternatives considered:

- The selected helpers only assemble existing dataclasses and normalize scalar
  probabilities. No tensor hot path was rewritten.
- Torch negative sampling remains the existing vectorized implementation in
  `starrygl.task.negative`.
- DGL operators and custom C++/CUDA are not relevant for this construction
  boundary.
- Route row lookup and GraphBlock sampling remain in event/snapshot modules to
  avoid premature abstraction around graph-specific hot paths.

Files written or modified:

- `src/starrygl/executor/runtime/targets.py`
- `src/starrygl/executor/runtime/event.py`
- `src/starrygl/executor/runtime/snapshot.py`
- `src/starrygl/executor/runtime/__init__.py`
- `tests/test_executor_targets.py`
- `docs/migration/status_log.md`

Verification:

- Modified target/event/snapshot modules pass Python compile checks.
- Focused target/task tests: 30 passed.
- Runtime/model/store/task sweep: 110 passed, 1 skipped.
- Coverage shows direct `TaskTarget(...)` construction in event/snapshot has
  been removed; construction is centralized in executor `targets.py`.

Unresolved performance risks:

- This step does not benchmark target materialization cost.
- Snapshot edge route row lookup and event target row lookup still live in
  their graph-specific modules; those should be reviewed with the sampling
  output to GraphBlock/MFG boundary before moving any hot-path route code.

## 2026-07-24: Executor Materialization Boundary

Latest state:

- `starrygl.executor.runtime.materialize` owns the shared Batch construction
  boundary for executor materialization.
- Event and snapshot runtime paths now build model-facing `Batch` objects
  through `materialized_batch(...)` instead of directly constructing
  `Batch(...)`.
- Materialized batches record an internal `meta.values["dependency_sources"]`
  manifest covering node features, remote node features, edge features,
  remote edge features, positive-edge features, state, and task targets.
- Deferred event/snapshot feature launches refresh the same dependency
  manifest after pending feature handles are attached.

Abstraction introduced:

- `materialized_batch(...)` is a narrow executor helper, not a new public Batch
  API. It is necessary now because event and snapshot paths already share the
  same model-facing Batch contract and dependency-source metadata.
- `dependency_sources` is internal runtime/debug metadata. Model math still
  consumes `Batch.features`, `Batch.state`, `Batch.targets`, and
  `Batch.mfgs`/`Batch.graph`.

Efficiency alternatives considered:

- The helper only assembles Python metadata and delegates to the existing
  `Batch` dataclass. It does not alter tensor gathering, sampler output, DGL
  block construction, or communication launch paths.
- Torch/DGL hot-path operations in event and snapshot materialization are
  unchanged.
- DGL-kernel-inspired or custom C++/CUDA work is not relevant for this metadata
  consolidation.
- No per-node/per-edge Python loop was added; dependency-source detection is
  based on mapping keys and pending-handle presence.

Files written or modified:

- `src/starrygl/executor/runtime/materialize.py`
- `src/starrygl/executor/runtime/event.py`
- `src/starrygl/executor/runtime/snapshot.py`
- `src/starrygl/executor/runtime/__init__.py`
- `tests/test_executor_materialize.py`
- `docs/migration/status_log.md`

Verification:

- Modified materialization modules and tests pass Python compile checks.
- Focused materialization/import tests: 8 passed.
- Runtime/model/store/task sweep: 104 passed, 1 skipped.
- Import coverage shows event/snapshot runtime paths no longer directly call
  `Batch(...)`; Batch construction is centralized in `materialize.py`.

Unresolved performance risks:

- This step does not benchmark materialization throughput.
- Repeated DGL block assembly and feature gather/scatter costs still need
  representative profiling before deciding whether to move specific hot paths
  into DGL-kernel-inspired or native operators.
- The dependency manifest is descriptive metadata only; compile-time lowering
  is not yet driven by it.

## 2026-07-24: Executor Memory Commit Boundary

Latest state:

- Batch-level state write-side boundaries now live in
  `starrygl.executor.runtime.memory`.
- `runtime.loop` no longer owns duplicate commit/flush/poll helper
  implementations. It imports executor memory functions for:
  `commit_state_delta`, `commit_empty_state_delta`, `flush_pending_state`,
  `launch_pending_shared_state`, `finish_shared_state`, and
  `poll_shared_state`.
- `state_access` reuses the same shared-hot polling helper from memory during
  read hydration.
- The lower-level `_commit_state_local` tensor update remains in memory as the
  internal authoritative local commit implementation.

Abstraction introduced:

- The added functions are batch-level executor boundaries, not a new scheduler
  abstraction. They are necessary now because fit/evaluate, read hydration, and
  async memory commit all need the same owner/shared-hot queue semantics.

Efficiency alternatives considered:

- Torch tensor commit paths inside `AsyncMemoryCommitter` and
  `_commit_state_local` are unchanged.
- DGL operators do not cover state-owner commit, mailbox append, or shared-hot
  refresh queues.
- A custom C++/CUDA commit kernel is deferred until profiling shows local
  commit/scatter dominates; this migration only removes duplicated Python
  orchestration.
- Queue operations remain batch-granular. No per-node/per-edge Python loop was
  added.

Files written or modified:

- `src/starrygl/executor/runtime/memory.py`
- `src/starrygl/executor/runtime/loop.py`
- `src/starrygl/executor/runtime/state_access.py`
- `tests/test_executor_memory.py`
- `docs/migration/status_log.md`

Verification:

- Modified runtime modules and the new test pass Python compile checks.
- Focused memory/access/state tests: 12 passed.
- Runtime/model/store/task sweep: 101 passed, 1 skipped.
- Import coverage shows no duplicate loop-level state commit helper remains;
  only memory's internal `_commit_state_local` tensor update remains.

Unresolved performance risks:

- Multi-rank owner commit and shared-hot refresh were not benchmarked in this
  step.
- The public runtime lowering still needs to decide when shared-hot bounded
  reads are enabled; this step only centralizes execution ownership.

## 2026-07-24: Public Model And Task Semantics

Latest state:

- `NodePredictionTask.compute_metrics()` reports normalized `accuracy` and
  `f1_micro`, while preserving `num_examples` for aggregation.
- `TGNModel` state and mailbox updates use event timestamps from runtime
  targets or `TaskTarget.target_ts` before falling back to block timestamps.
- `TGNModel` no longer treats positive-edge supervision features as sampled
  block edge features. Positive edge features remain mailbox-message inputs;
  missing sampled block edge features use zero edge context when positive edge
  features are present.
- `TGATModel` uses `out_dim` as the default node prediction head dimension and
  defaults to one attention head, so common hidden dimensions do not fail
  construction unless the caller explicitly requests an incompatible head
  count.

Abstraction introduced:

- No new abstraction was introduced. This step tightens existing public
  model/task semantics before further directory migration.

Efficiency alternatives considered:

- The selected changes are tensor-level arithmetic and indexing only; no
  Python per-node or per-edge hot-path loop was added.
- DGL operators are not relevant for metric normalization or mailbox timestamp
  selection.
- A DGL-kernel-inspired or custom C++/CUDA path is not justified for these
  model/task semantic fixes.
- TGN positive-edge and block-edge features remain separate tensor flows to
  avoid extra scatter/gather work on the sampled block hot path.

Files written or modified:

- `src/starrygl/task/prediction.py`
- `src/starrygl/model/tgn.py`
- `src/starrygl/model/tgat.py`
- `docs/migration/status_log.md`

Verification:

- Focused failing tests: 8 passed.
- Runtime/model/store/task sweep: 98 passed, 1 skipped.
- Modified Python modules pass compile checks.

Unresolved performance risks:

- Zero edge context for sampled TGN blocks is a semantic fallback when only
  positive-edge features are available. Representative training should verify
  whether configs with `edge_dim > 0` also materialize block edge features for
  the intended model path.
- Multi-rank runtime performance was not measured in this step.

## 2026-07-24: Executor Runtime Chain

Latest state:

- `starrygl.executor.runtime` now owns the main epoch execution chain:
  batches, event/snapshot materialization, model execution, loop,
  snapshot-chain replay, trainer, memory helpers, window scan, plan, prepare,
  and compile facade.
- `starrygl.runtime.*` modules for the migrated chain are now transitional
  re-exports. They remain import-compatible, but new implementation work should
  land in `starrygl.executor.runtime`.
- Model-side internal helper imports for TGCN, MPNN-LSTM, and DCRNN now point
  at executor runtime helpers.
- The temporary public `access_pipeline_double_buffer` option was removed.
  The existing `access_pipeline_launched_queue` bridge remains as a
  migration-only experimental feature-lookahead switch.

Abstraction introduced:

- No new abstraction was introduced. This step changes implementation
  ownership and import direction so the existing runtime chain has one concrete
  home under the executor package.
- Runtime compatibility modules are narrow re-exports only; they are not a new
  public API layer.

Efficiency alternatives considered:

- Torch/DGL tensor and graph operators remain unchanged in this step; no hot
  path algorithm was rewritten.
- DGL-kernel-inspired or custom C++/CUDA work is not justified by a directory
  migration alone.
- The selected path preserves existing optimized exchange, sampler,
  materialization, and memory code while removing executor-to-runtime import
  cycles.
- Deeper feature prefetch/double buffering was not promoted to public API
  without representative profiling.

Files written or modified:

- `src/starrygl/executor/runtime/__init__.py`
- `src/starrygl/executor/runtime/batches.py`
- `src/starrygl/executor/runtime/compile.py`
- `src/starrygl/executor/runtime/event.py`
- `src/starrygl/executor/runtime/loop.py`
- `src/starrygl/executor/runtime/memory.py`
- `src/starrygl/executor/runtime/model_exec.py`
- `src/starrygl/executor/runtime/plan.py`
- `src/starrygl/executor/runtime/prepare.py`
- `src/starrygl/executor/runtime/snapshot.py`
- `src/starrygl/executor/runtime/snapshot_chain.py`
- `src/starrygl/executor/runtime/trainer.py`
- `src/starrygl/executor/runtime/window_scan.py`
- `src/starrygl/runtime/batches.py`
- `src/starrygl/runtime/event.py`
- `src/starrygl/runtime/loop.py`
- `src/starrygl/runtime/memory.py`
- `src/starrygl/runtime/model_exec.py`
- `src/starrygl/runtime/plan.py`
- `src/starrygl/runtime/prepare.py`
- `src/starrygl/runtime/snapshot.py`
- `src/starrygl/runtime/snapshot_chain.py`
- `src/starrygl/runtime/trainer.py`
- `src/starrygl/runtime/window_scan.py`
- `src/starrygl/model/dcrnn.py`
- `src/starrygl/model/mpnn_lstm.py`
- `src/starrygl/model/tgcn.py`
- `tests/test_access_double_buffer.py`
- `tests/test_starrygl_public_api.py`

Verification:

- Executor/runtime and compatibility runtime modules pass Python compile
  checks.
- Focused import/access tests: 11 passed, 1 skipped.
- Import coverage shows no executor runtime implementation imports
  `starrygl.runtime`.
- Wider model/store/task test sweep: 90 passed, 1 skipped, 8 failed. The
  failures are existing model/task behavior mismatches outside this directory
  migration unit: TGAT head divisibility, TGN state timestamp/edge-feature
  expectations, and NodePrediction accuracy scaling.

Unresolved performance risks:

- This migration does not improve throughput by itself; it only consolidates
  ownership so future scheduling/profile changes have one implementation path.
- No representative multi-rank benchmark was run for event/snapshot epoch
  execution after the move.
- `access_pipeline_launched_queue` remains an experimental migration bridge.
  It should not become a documented public knob until profiling proves feature
  fetch wait is coverable by current-batch compute.

## 2026-07-23: Executor Access Scheduler

Latest state:

- `starrygl.executor.runtime.access.AccessScheduler` owns launched, pending,
  and ready batch read-dependency queues.
- One `AccessTicket` combines pending node/edge feature reads with state
  hydration reads while retaining separate feature and state Await boundaries.
- `runtime.loop` keeps window sampling/materialization orchestration but routes
  training and evaluation read completion through the executor scheduler.
- Snapshot replay, chunked snapshot evaluation, and the prepacked snapshot
  chain use the same access lifecycle.
- `CommScheduler` remains the only owner of collective launch and ticket order.

Abstraction introduced:

- `AccessScheduler` is necessary now because feature and state pending handles
  already have multiple real runtime call sites, and queue ownership was
  duplicated between `Batch.meta` and local loop control.
- `AccessTicket` is batch-scoped and intentionally does not add generic
  request adapters or a registry.
- Only `wait_policy=block` is implemented. `reschedule` raises explicitly
  rather than silently behaving as block at this scheduler boundary.

Efficiency alternatives considered:

- Torch vectorized `searchsorted` and `index_select` remain the selected
  feature restoration path; no per-node or per-edge Python loop was added.
- DGL does not provide owner-routed feature/state Await scheduling.
- A DGL-kernel-inspired or custom C++/CUDA restoration kernel could fuse edge
  lookup and scatter, but representative profiling is required before adding a
  native operator.
- The scheduler records existing asynchronous handles without adding threads,
  polling, or extra tensor copies. Window sampling/materialization queues stay
  in `runtime.loop` because they are a separate scheduling responsibility.

Files written or modified:

- `src/starrygl/executor/runtime/access.py`
- `src/starrygl/executor/runtime/__init__.py`
- `src/starrygl/runtime/loop.py`
- `src/starrygl/runtime/snapshot_chain.py`
- `tests/test_executor_access.py`
- `tests/test_open_compile_semantics.py`
- `docs/design/current/state_await_cache.md`
- `docs/STAGING_STATUS.md`
- `docs/migration/status_log.md`

Verification:

- Access, loop, snapshot-chain, executor export, and tests pass Python compile
  checks.
- Access/exchange/state/runtime/public-entry/store tests: 38 passed.
- Pending feature completion coverage points to the executor access module;
  event/snapshot direct finishes are immediate local-read fast paths, not
  queued pending handles.
- The broader `test_starrygl_public_api.py` cannot collect in this staging tree
  because the copied suite imports the intentionally absent `starrygl.data`
  compatibility package.

Unresolved performance risks:

- No real multi-rank feature/state overlap or early-window drain run has been
  executed.
- `reschedule` needs nonblocking handle readiness plus a local ready-work queue;
  it must not reorder or skip collective epochs.
- Queue operations are Python-side and batch-granular. This avoids
  per-node/per-edge loops, but representative profiling is still needed before
  claiming overlap or throughput gains.
- Immediate local feature reads in event/snapshot materialization remain
  synchronous fast paths and intentionally bypass the pending queue.

## 2026-07-23: Executor State Hydration

Latest state:

- `starrygl.executor.runtime.state_access` owns state read submission,
  completion, mailbox hydration, node/time compaction, and Batch installation.
- `runtime.loop` imports the executor implementation and only orchestrates
  when submit and finish occur.
- `runtime.trainer` imports prediction-time hydration from the executor path.
- Memory commit, shared synchronization, and refresh queues remain write-side
  responsibilities of `runtime.memory`.

Abstraction introduced:

- `PendingHydrateRead` is the concrete read-side pending handle exposed by the
  executor module. No generic Fetch/Await queue or adapter hierarchy was added
  in this step.

Efficiency alternatives considered:

- Existing vectorized Torch compaction and StateManager asynchronous handles
  were retained.
- DGL does not provide state freshness/version hydration primitives.
- A custom native state gather is deferred until representative profiling
  shows compact gather or mailbox restoration remains material.
- Feature and state pending handles remain distinct physical types; their
  scheduling can converge later under one executor access scheduler.

Files written or modified:

- `src/starrygl/executor/runtime/state_access.py`
- `src/starrygl/executor/runtime/__init__.py`
- `src/starrygl/runtime/loop.py`
- `src/starrygl/runtime/trainer.py`
- `tests/test_compact_runtime_layout.py`
- `tests/test_open_compile_semantics.py`

Verification:

- State access, loop, trainer, and test modules pass Python compile checks.
- Exchange/state/runtime/public-entry/store tests: 34 passed.
- Coverage confirms the hydration implementation exists only under the
  executor path; loop compatibility names resolve to the same objects.

Unresolved performance risks:

- Local asynchronous submit/finish is covered, but distributed memory/mailbox
  all-to-all hydration has not been run.
- Batch access queues and wait-policy behavior still live in `runtime.loop`;
  `wait_policy=reschedule` is not yet a real executor scheduling policy.
- Write-side memory commit and shared-cache queues have intentionally not been
  migrated or merged with read-side Fetch/Await.

## 2026-07-23: Executor Feature Exchange

Latest state:

- `starrygl.executor.runtime.exchange` owns node/edge feature access,
  staged reads, remote fetch scheduling, and output restoration.
- `starrygl.runtime.exchange` is a transitional re-export.
- Event, snapshot, and loop callers import exchange operations through the
  executor namespace.

Abstraction introduced:

- No new request hierarchy was introduced. Existing
  `PendingNodeFeatureFetch`, `StagedFeatureFetch`, and
  `NodeAccessRequestContext` remain the concrete interfaces because they
  already serve real asynchronous call sites.

Efficiency alternatives considered:

- Torch collectives and tensor packing remain the selected implementation.
- DGL operators do not cover feature-owner request/response exchange.
- A custom C++/CUDA packing path could reduce launch overhead, but requires
  representative distributed profiling before adding a native operator.
- Feature and state access were not forced into one generic adapter. Their
  freshness and version semantics differ and should be unified only at the
  scheduler contract.

Files written or modified:

- `src/starrygl/executor/runtime/exchange.py`
- `src/starrygl/runtime/exchange.py`
- `src/starrygl/runtime/event.py`
- `src/starrygl/runtime/snapshot.py`
- `src/starrygl/runtime/loop.py`
- `tests/test_executor_exchange.py`
- `tests/test_open_compile_semantics.py`

Verification:

- Modified exchange and caller modules pass Python compile checks.
- Exchange/runtime/public-entry/store tests: 33 passed.
- Import coverage points to the executor implementation except for the
  explicit compatibility re-export and identity test.

Unresolved performance risks:

- Local, staged, duplicate-id, and empty fetch behavior is covered, but a real
  multi-rank all-to-all run has not been executed.
- CUDA overlap and communication packing cost require representative workload
  profiling; phase profiling alone is not an end-to-end throughput result.

## 2026-07-23: Compact Runtime Indexing

Latest state:

- Batch-local lookup no longer allocates by `max(global_id) + 1`.
- Node and edge lookup uses `starrygl.utils.index.compact_lookup_rows`.
- `node_memory` and `mailbox` hydration keeps compact values and a
  `*_layout_inverse` mapping until model consumption.
- Node-time state compaction keeps node ids as `torch.long`; it no longer
  converts encoded ids to float64.

Abstraction introduced:

- `compact_lookup_rows(source_ids, query_ids)` is one shared tensor lookup
  helper. It is necessary now because the same sparse-id mapping was
  implemented independently in model, task-target, snapshot, and state paths.

Efficiency alternatives considered:

- Torch: stable `argsort` plus `searchsorted` is the selected vectorized path.
  It avoids Python element loops and global-id-sized temporary tensors.
- DGL: no graph operator directly represents arbitrary global-id table lookup.
- DGL-kernel-inspired/custom C++ or CUDA: a fused hash lookup or
  sampler-emitted inverse can reduce sorting, but is deferred until a real
  workload shows this operation remains material.
- Compact state values are gathered only when TGN/JODIE/APAN request their
  model layout. Other recurrent state kinds preserve the existing layout to
  avoid changing coupled execution semantics.

Files written or modified:

- `src/starrygl/utils/index.py`
- `src/starrygl/model/_graph_ops.py`
- `src/starrygl/model/layers/temporal.py`
- `src/starrygl/model/tgat.py`
- `src/starrygl/model/tgn.py`
- `src/starrygl/model/jodie.py`
- `src/starrygl/model/apan.py`
- `src/starrygl/model/dcrnn.py`
- `src/starrygl/runtime/event.py`
- `src/starrygl/runtime/snapshot.py`
- `src/starrygl/runtime/memory.py`
- `src/starrygl/runtime/loop.py`
- `tests/test_compact_runtime_layout.py`

Verification:

- Python compile check passed for every modified Python module.
- Compact/runtime/public-entry tests: 17 passed.
- Broader model tests: 45 passed, 5 existing semantic/API drift failures.

Unresolved performance risks:

- No representative distributed workload benchmark has been run yet, so no
  end-to-end speedup is claimed.
- Sort/search is `O(N log N)` and may be slower than dense indexing for small,
  truly dense local ids. Encoded sparse global ids are the intended case.
- Sampler-emitted node/edge inverse maps could remove repeated sort/search.
- Compact memory is still gathered into the model layout before memory update;
  fusing this gather requires a measured kernel-level justification.

## 2026-08-03: Interface Surface Simplification

Latest state:

- `compile()` keeps the canonical semantic inputs:
  `data_source`, `backbone`, `task_segment`, and optional `temporal_state`.
- `from_config()` accepts canonical `data`, `backbone`, `task`, and optional
  `temporal_state` sections.
- Explicit `temporal_state` is the only config surface for exact/stale
  temporal-state consistency; `memory`, `spec`, and `execution` are rejected to
  avoid duplicate ways to express freshness.
- Default temporal/state/scope inference is centralized in the compile path.
- Public semantic dataclasses no longer carry fields that do not lower into the
  execution plan yet: `dependency_sources`, `remote_dependency`,
  `sample_fn`, `filter_positive`, and generic `options`.
- `interface/api.py` was reduced to public entry orchestration. Config loading
  moved to `interface/config.py`; semantic default lowering moved to
  `interface/lowering.py`.
- Added `interface/compile_example.py` as a local example of the canonical
  `compile()` and `from_config()` surfaces. It is not exported as public API.
- Added `TemporalStateSemantics` so Python callers can express temporal-state
  exact/stale semantics without writing internal `StarrySpec`.
- `temporal_state` exposes the supported semantics directly:
  update mode (`gru`, `rnn`, `transformer`), bounded stale reads, change-rate
  filtering, and smooth aggregation compensation. It no longer exposes
  `time_decay`/`snapshot_decay` approximation labels.
- Added top-level `sampling` config semantics. `sampling.window` expresses
  event-window/chunk-decay/full-snapshot materialization, and
  `sampling.neighbor` expresses neighbor sampler fanouts/policy/seed/workers.
  The public section lowers into the existing runtime `sampling` options.

Abstraction introduced:

- No new abstraction was introduced. This step removes premature API surface
  and keeps dependency sources as an `ExecutionPlan` observation, not a user
  declaration.

Efficiency alternatives considered:

- No hot path changed. The cleanup only affects public-boundary normalization
  and compile-time semantic lowering.
- Plan dependency construction remains owned by `ChunkBindingPlanner`.

Files written or modified:

- `src/starrygl/interface/api.py`
- `src/starrygl/interface/compile_example.py`
- `src/starrygl/interface/config.py`
- `src/starrygl/interface/lowering.py`
- `src/starrygl/interface/spec.py`
- `src/starrygl/__init__.py`
- `src/starrygl/interface/plan.py`
- `tests/test_executor_state_request.py`
- `tests/test_open_compile_plan.py`
- `tests/test_open_compile_semantics.py`
- `docs/migration/status_log.md`

Verification:

- Python compile check passed for modified interface modules.
- Interface and state request tests: 29 passed.

Unresolved performance risks:

- None introduced by this cleanup.

## 2026-08-05: Compile Boundary And Lowering Consolidation

Latest state:

- `spec.py` now owns the canonical runtime defaults and public declaration
  normalization. The defaults match `interface/compile_example.py`, including
  bounded-stale temporal state, window/neighbor sampling, training, and
  preprocess values.
- `api.py` now only normalizes the four public semantic segments, resolves the
  model instance, invokes the planner, and constructs `Trainer`.
- `plan.py` now lowers normalized data, backbone, task, and runtime semantics
  into the internal `StarrySpec` and observable `ExecutionPlan`.
- The single-use `interface/lowering.py` module was removed.
- Public `compile()` no longer accepts physical `partition_plan`, `override`,
  or arbitrary metadata keyword arguments. Internal planner tests can still
  exercise `ExecutionOverride` without exposing it through the public entry.
- Normalized temporal-state configuration is retained under
  `Trainer.runtime_config["temporal_state"]` instead of being duplicated in
  `Trainer.metadata`.

Abstraction introduced:

- Added `DEFAULT_RUNTIME_CONFIG` and `normalize_runtime()` because one
  canonical source is now required to keep documented and executed defaults
  aligned. No registry, adapter, or compatibility branch was introduced.

Attempted approaches:

- Considered retaining a separate semantic-lowering module, but it had one
  caller and duplicated the planner boundary. The lowering functions were
  moved into `plan.py` as private implementation details.
- Considered keeping public execution overrides for existing tests. Those
  tests now call the internal planner directly so physical strategies do not
  expand the public compile surface.

Files written or modified:

- `src/starrygl/interface/api.py`
- `src/starrygl/interface/spec.py`
- `src/starrygl/interface/plan.py`
- `src/starrygl/interface/lowering.py` (removed)
- `tests/test_open_compile_semantics.py`
- `tests/test_open_compile_plan.py`
- `tests/test_executor_state_request.py`
- `docs/design/current/execution_spine.md`
- `docs/migration/status_log.md`

Efficiency alternatives considered:

- This change runs only at compile time. Runtime tensor, DGL, sampler,
  communication, materialization, and state-update hot paths are unchanged.
- Default merging uses nested mapping copies; a torch/DGL operator or native
  C++/CUDA implementation would provide no benefit for this cold path.

Verification:

- Python compile checks passed for `api.py`, `spec.py`, and `plan.py`.
- Interface, plan, and state-request tests were updated for the canonical
  defaults and public boundary.

Unresolved performance risks:

- None introduced. The prepare and Trainer stages still need separate review
  to verify that newly canonical preprocess names are consumed directly.

## 2026-08-05: Runtime Default Alignment Table

Latest state:

- Added a default-parameter alignment table for runtime config.
- Snapshot-window defaults are mapped against FlareDTDG-style chunk-window
  training.
- Event sampling, temporal state, training, execution, and preprocess defaults
  are mapped against MemShare-style event training and current open runtime
  fields.
- The document distinguishes covered config fields from derived physical
  metadata and internal-only runtime options.

Abstraction introduced:

- No code abstraction was introduced. This is a documentation-only alignment
  step.

Efficiency alternatives considered:

- No hot path changed.
- Physical chunk ids and chunk count remain derived from partition/preprocess
  artifacts rather than duplicated in public config.

Files written or modified:

- `docs/design/current/runtime_default_alignment.md`
- `docs/migration/status_log.md`

Verification:

- Documentation-only change; no tests were run.

Unresolved performance risks:

- None introduced by this documentation step.

## 2026-08-05: Compile Example Default Alignment

Latest state:

- `compile_example.py` now carries the current default runtime alignment.
- `runtime.temporal_state.filter.min_cosine_distance` defaults to `0.3`,
  matching the MemShare historical-cache cosine-distance threshold used in
  shared-memory experiments.
- Event neighbor sampling defaults now align with MemShare:
  `fanouts=[20]`, `workers=32`, and `policy="recent"`.
- Added `runtime.sampling.neighbor.boundary_sampling` defaults:
  `boundary_uniform` for uniform sampling and `boundary_decay_sampling` for
  recent sampling.
- Snapshot window defaults remain aligned with FlareDTDG:
  `snaps_count=8`, `num_full_snapshots=2`, `chunk_decay="half"`, and
  `chunk_order="rand"`.
- Training defaults in the example now align with MemShare TGN-style event
  training: `epochs=50`, `batch_size=3000`, `lr=0.0004`,
  `dropout=0.2`, and `att_dropout=0.2`.

Abstraction introduced:

- No abstraction was introduced. This step only updates example defaults and
  the default-alignment table.

Efficiency alternatives considered:

- No hot path changed.
- Boundary sampling policies are recorded as runtime config semantics but are
  not wired into a new execution path in this step.

Files written or modified:

- `src/starrygl/interface/compile_example.py`
- `docs/design/current/runtime_default_alignment.md`
- `docs/migration/status_log.md`

Verification:

- Python compile check passed for `src/starrygl/interface/compile_example.py`.

Unresolved performance risks:

- None introduced by this default-configuration cleanup.

## 2026-08-05: Stale And Sampling Default Update

Latest state:

- `compile_example.py` now defaults temporal-state consistency to
  `bounded_stale` with `max_staleness=1`.
- Temporal-state filter and smooth aggregation are enabled by default in the
  example.
- `gamma_init=0.5` is retained after checking MemShare learnable gamma
  initialization.
- Negative sampling defaults now record train-time local/remote probabilities:
  local `0.9`, remote `0.1`; eval and test remain random.
- Negative sampling policy is now expressed per stage (`train`, `eval`,
  `test`) instead of keeping a duplicate global `policy` field.
- Boundary sampling default probability is now `0.1`.
- Preprocess defaults now include post-partition chunking and hot-node ratio:
  `chunks_per_rank=1` and `hot_node_ratio=0.1`.
- The runtime default alignment table was updated to match these defaults.

Abstraction introduced:

- No abstraction was introduced. This step only updates default config examples
  and documentation.

Efficiency alternatives considered:

- No hot path changed.
- Boundary sampling and negative sampling defaults are recorded as config
  semantics but are not wired into a new internal execution path in this step.

Files written or modified:

- `src/starrygl/interface/compile_example.py`
- `docs/design/current/runtime_default_alignment.md`
- `docs/migration/status_log.md`

Verification:

- Python compile check passed for `src/starrygl/interface/compile_example.py`.
- Python 3.10 import check confirmed the updated example values.

Unresolved performance risks:

- None introduced by this default-configuration cleanup.

## 2026-08-05: Align Chunk-Decay Window Semantics

Latest state:

- `runtime.sampling.window` now includes the FlareDTDG-style window fields
  needed for chunk-decay parsing:
  `snaps_count`, `num_full_snapshots`, `chunk_decay`, and `chunk_order`.
- `Trainer._chunk_decay()` uses `runtime.sampling.window.snaps_count` when
  parsing string schedules such as `half` and `auto:<ratio>`.
- `Trainer._sampler_options()` forwards `runtime.sampling.window.chunk_order`
  to snapshot materialization options.
- `chunks_per_rank` and concrete chunk ids remain derived from partition data,
  not user-facing window semantics.

Abstraction introduced:

- No new abstraction was introduced. This step extends the existing sampling
  window config and reuses existing Trainer helper methods.

Efficiency alternatives considered:

- No hot path changed. The existing chunk-decay parser and snapshot execution
  path are reused.
- Chunk count still comes from partition/preprocess artifacts so users do not
  have to duplicate physical chunk metadata in config.

Files written or modified:

- `src/starrygl/executor/runtime/trainer.py`
- `src/starrygl/interface/compile_example.py`
- `tests/test_open_compile_semantics.py`
- `docs/migration/status_log.md`

Verification:

- Python compile check passed for modified interface/runtime modules.
- Interface, plan, and state request tests: 30 passed.

Unresolved performance risks:

- None introduced by this cleanup.

## 2026-08-05: Preserve Runtime Sampling Hierarchy

Latest state:

- `runtime.sampling` now preserves the public semantic hierarchy instead of
  flattening `window` and `neighbor` into one config level.
- `runtime.sampling.window` owns snapshot/window materialization semantics:
  `policy`, `chunk_decay`, and `num_full_snapshots`.
- `runtime.sampling.neighbor` owns neighbor sampler semantics: `fanouts`,
  `policy`, `seed`, and `workers`.
- Trainer runtime helpers now read `window` and `neighbor` separately:
  `_snapshot_policy()`, `_chunk_decay()`, `_num_full_snapshots()`,
  `_fanouts()`, and `_sampler_options()`.
- `window.chunk_decay` string schedules such as `half` and `auto:<ratio>` still
  use the existing chunk-decay parser.

Abstraction introduced:

- Added small Trainer helper methods for sampling section access. They exist to
  keep public config structure stable while avoiding duplicated dictionary
  traversal at each runtime read site.

Efficiency alternatives considered:

- No hot path changed. The update only changes compile-time config
  normalization and runtime option lookup before batch execution.
- The existing chunk-decay parser and snapshot materialization path are reused.

Files written or modified:

- `src/starrygl/interface/api.py`
- `src/starrygl/executor/runtime/trainer.py`
- `tests/test_open_compile_semantics.py`
- `docs/migration/status_log.md`

Verification:

- Python compile check passed for modified interface/runtime modules.
- Interface, plan, and state request tests: 30 passed.

Unresolved performance risks:

- None introduced by this cleanup.

## 2026-08-05: Public Config Runtime Consolidation

Latest state:

- Public config is now one canonical line: top-level `data`, `backbone`,
  `task`, and optional `runtime`.
- `compile()` now accepts `data_source`, `backbone`, `task_segment`, and
  optional `runtime`. It rejects legacy or duplicate public arguments such as
  `graph`, `model`, `task`, `spec`, `temporal_state`, `sampling`, `train`, and
  `preprocess`.
- `from_config()` maps only `data`, `backbone`, `task`, and `runtime` into
  `compile()`. Top-level `temporal_state`, `sampling`, `train`, and
  `preprocess` are rejected and must live under `runtime`.
- `runtime.temporal_state` remains the single user-facing place for exact
  versus bounded-stale temporal state reads. Internal `StarrySpec` still exists
  as compile lowering output, not as a public config section.
- `runtime.sampling` keeps semantic window and neighbor sampler structure, then
  lowers into the existing runtime `sampling` fields for the current executor.
- Custom Python model backbones are still supported by passing an `nn.Module`
  as `backbone`; if it defines `backbone_semantics`, those semantics are used
  for plan lowering.

Abstraction introduced:

- No new runtime abstraction was introduced. This is a public-boundary cleanup
  that keeps existing Trainer internals split into train/runtime/preprocess
  fields while exposing only one public `runtime` section.

Efficiency alternatives considered:

- No hot path changed. The only new work is compile-time normalization of
  nested `runtime` sections.
- Sampling semantic lowering is dictionary normalization only; sampler
  execution, route construction, feature/state fetch, and model compute are
  unchanged.

Files written or modified:

- `src/starrygl/interface/api.py`
- `src/starrygl/interface/config.py`
- `src/starrygl/interface/compile_example.py`
- `src/starrygl/interface/plan.py`
- `tests/test_executor_state_request.py`
- `tests/test_open_compile_plan.py`
- `tests/test_open_compile_semantics.py`
- `docs/migration/status_log.md`

Verification:

- Python compile check passed for modified interface modules.
- Interface, plan, and state request tests: 29 passed.

Unresolved performance risks:

- None introduced by this cleanup.

## 2026-08-06: Partition Ownership Implementation Extraction

Latest state:

- Added `starrygl.partition` as the single implementation location for graph
  ownership and intra-partition chunk assignment.
- `partition_graph()` now computes node masters, edge masters, shared-hot
  membership, node chunk ids, and edge chunk ids in one result.
- Both active prepare paths call the partition package instead of carrying
  duplicate `_resolve_partition`, speed-partition, and METIS helper code.
- Prepare still returns the existing artifact layout; conversion to the formal
  `PartitionPlan` is intentionally deferred to the next review unit.

Abstraction introduced:

- `PartitionConfig` isolates the inputs required by graph partitioning from
  unrelated view/materialization options in `PrepareConfig`.
- `PartitionAssignment` is a temporary typed result needed to preserve both
  node and edge ownership while the formal `PartitionPlan` bridge is built.

Attempted approaches:

- Considered moving only the private helper functions, but that would leave
  prepare responsible for coordinating owner, hot-node, and chunk outputs.
  A single `partition_graph()` call keeps that operation atomic.
- The direct `starrygl.native` speed-partition import is preferred. A narrow
  deprecated native import remains temporarily because the open package does
  not yet ship the speed-partition extension; removing it now would silently
  change multi-rank partitioning to round-robin.

Files written or modified:

- `src/starrygl/partition/__init__.py`
- `src/starrygl/partition/build.py`
- `src/starrygl/prepare/build_part_graph.py`
- `src/starrygl/stc/prepare/build_part_graph.py`
- `tests/test_partition_assignment.py`
- `docs/migration/status_log.md`

Efficiency alternatives considered:

- Existing native speed partition remains the preferred owner assignment.
- Existing DGL METIS remains the preferred intra-owner chunk assignment, with
  deterministic tensor round-robin only when DGL/native support is absent.
- The per-rank Python loop is preprocessing-only and was migrated unchanged;
  it is not accepted as a training hot path.

Verification:

- Python compile checks passed for the partition and prepare modules.
- Partition and store/prepare tests: 14 passed.

Unresolved performance risks:

- The native speed-partition extension still needs a direct
  `starrygl.native` build/install path before the deprecated import can be
  removed.
- `PartitionAssignment` has not yet been converted into the public/internal
  `PartitionPlan` contract.

## 2026-08-06: Remove Duplicate STC Prepare Package

Latest state:

- Removed the complete `src/starrygl/stc/` package.
- Its prepare sources were byte-for-byte duplicates of
  `src/starrygl/prepare/` after partition extraction.
- No source, test, or documentation entry imported `starrygl.stc`; the
  package was not a compatibility surface.
- `src/starrygl/prepare/` is now the only graph preparation implementation.

Abstraction introduced:

- None. This removes a parallel implementation stack.

Attempted approaches:

- Initially retained the STC copy and routed it through the new partition
  implementation. Reference and source comparison showed that the whole
  package was unreferenced duplication, so retaining it had no migration
  value.

Files removed or modified:

- `src/starrygl/stc/__init__.py` (removed)
- `src/starrygl/stc/prepare/__init__.py` (removed)
- `src/starrygl/stc/prepare/build_part_graph.py` (removed)
- `src/starrygl/stc/prepare/data.py` (removed)
- `docs/migration/status_log.md`

Efficiency alternatives considered:

- No runtime or preprocessing algorithm changed. Removing the duplicate code
  avoids future divergence without adding a forwarding module or import
  branch.

Verification:

- Confirmed no runtime/test imports of `starrygl.stc` remain.
- Partition, prepare/store, and interface tests were rerun after removal.

Unresolved performance risks:

- None introduced by this deletion.

## 2026-08-06: Formal PartitionPlan Output

Latest state:

- Partition contracts now have one implementation under
  `src/starrygl/partition/plan.py`; store no longer owns placement types.
- `partition_graph()` returns `PartitionPlan` directly with authoritative
  `node_master`, `edge_master`, `shared_nodes`, and `ChunkTable` values.
- `ChunkTable.chunk_ids` stores node-to-chunk assignment and
  `ChunkTable.values["edge_chunk"]` preserves the existing edge chunk artifact.
- `RouteTable` remains empty at partition time. Dependency-specific routes are
  deferred until execution planning has backbone/task requirements.
- Trainer prepare now maps canonical `num_parts`, `chunks_per_rank`, and
  `hot_node_ratio` fields into the existing prepare implementation.

Abstraction introduced:

- No additional layer was added. Temporary `PartitionAssignment` was removed
  and replaced by the already established `PartitionPlan` contract.

Attempted approaches:

- Considered retaining store-level re-exports for placement types, but no
  internal callers required them. Root exports now import the contracts from
  `starrygl.partition` directly, leaving one ownership location.

Files written, modified, or removed:

- `src/starrygl/partition/plan.py`
- `src/starrygl/partition/__init__.py`
- `src/starrygl/partition/build.py`
- `src/starrygl/prepare/build_part_graph.py`
- `src/starrygl/executor/runtime/trainer.py`
- `src/starrygl/interface/plan.py`
- `src/starrygl/store/__init__.py`
- `src/starrygl/store/placement.py` (removed)
- `src/starrygl/store/core/placement.py` (removed)
- `src/starrygl/__init__.py`
- `tests/test_partition_assignment.py`
- `docs/migration/status_log.md`

Efficiency alternatives considered:

- Partition algorithms and tensor layouts are unchanged. The new plan object
  stores references to existing owner/chunk tensors without per-node or
  per-edge conversion loops.
- `node_is_hot` is temporarily reconstructed with one vectorized indexed write
  in prepare for legacy artifact compatibility.

Verification:

- Python compile checks passed for partition, prepare, Trainer, interface plan,
  and package exports.
- Partition, prepare/store, compile semantics, and compile plan tests passed.
- A focused test verifies canonical preprocess values reach `PrepareConfig`.

Unresolved performance risks:

- Prepare still creates the partition internally and immediately materializes
  all enabled views. The next execution-order change must expose PartitionPlan
  before view/index materialization.

## 2026-08-06: Separate Partition And View Materialization

Latest state:

- `prepare_graph_data()` now calls `partition_graph()` first and passes the
  resulting `PartitionPlan` into `materialize_graph_views()`.
- The old `prepare_graph_views()` entry was removed rather than retained as a
  second naming path.
- Generic owner-to-local-row and chunk lookups moved into
  `prepare/partition_index.py` as `PartitionIndices`.
- `materialize_graph_views()` no longer accepts node/edge owner tensors and
  cannot perform graph partitioning itself.
- Existing event, temporal CSR, snapshot CSC, and artifact outputs remain
  unchanged in this review unit.

Abstraction introduced:

- `PartitionIndices` groups the generic tensor indices derived only from a
  `PartitionPlan`. This boundary is required by both view materialization and
  future dependency route construction.

Attempted approaches:

- Considered keeping an optional PartitionPlan argument on the old combined
  function. That would preserve two execution orders and allow partitioning to
  drift back into materialization, so the combined entry was removed.

Files written or modified:

- `src/starrygl/prepare/partition_index.py`
- `src/starrygl/prepare/build_part_graph.py`
- `src/starrygl/prepare/data.py`
- `src/starrygl/prepare/__init__.py`
- `src/starrygl/executor/runtime/prepare.py`
- `src/starrygl/executor/runtime/__init__.py`
- `src/starrygl/runtime/__init__.py`
- `src/starrygl/__init__.py`
- `tests/test_partition_assignment.py`
- `tests/test_starrygl_store.py`
- `docs/migration/status_log.md`

Efficiency alternatives considered:

- Dist-index construction remains vectorized torch code. No per-node or
  per-edge Python loop was introduced.
- Partition tensors are passed by reference into materialization; no full
  owner/chunk copies are created by the new boundary.
- TCSR/CSC construction is unchanged. Selecting only required views is deferred
  until the corrected ExecutionPlan contract is available.

Verification:

- Compile checks passed for prepare, runtime forwarding modules, and package
  exports.
- Partition, prepare/store, compile, plan, and state-request tests: 47 passed.
- A focused test directly supplies a PartitionPlan to materialization with all
  optional views disabled.

Unresolved performance risks:

- `prepare_graph_data()` still materializes views according to boolean
  `PrepareConfig` flags rather than a corrected ExecutionPlan.
- State-write routes are still embedded in event view construction and must be
  separated after backbone state requirements are defined.

## 2026-08-06: Lower Physical Views From Execution Semantics

Latest state:

- Added `ViewPlan` to `ExecutionPlan`; it records the execution kind, model-facing
  batch surface, and the exact physical graph layouts required by the path.
- Event sampled execution requires `event_view + temporal_csr`; event full-graph
  execution requires only `event_view`; snapshot execution requires only
  `snapshot_csc`, including sampled snapshot execution.
- Removed `include_event_view`, `include_temporal_csr`, and
  `include_snapshot_csc` from `PrepareConfig`.
- `Trainer.prepare()` now passes `trainer.plan.view` through
  `prepare_graph_data()` to `materialize_graph_views()`.

Abstraction introduced:

- `ViewPlan` is the single necessary contract between semantic lowering and
  physical graph-index construction. It prevents prepare code and user config
  from independently selecting layouts.

Attempted approaches:

- Considered retaining optional prepare booleans as overrides. This would leave
  two authorities for physical layout selection, so the booleans were removed.

Files written or modified:

- `src/starrygl/interface/plan.py`
- `src/starrygl/prepare/build_part_graph.py`
- `src/starrygl/prepare/data.py`
- `src/starrygl/executor/runtime/trainer.py`
- `src/starrygl/__init__.py`
- `tests/test_partition_assignment.py`
- `tests/test_starrygl_store.py`
- `docs/migration/status_log.md`

Efficiency alternatives considered:

- Selection is a compile-time tuple membership check; no Python loop was added
  to a training hot path.
- Existing vectorized torch TCSR/CSC builders are unchanged, but unused layouts
  are no longer built. No custom C++/CUDA operator is needed for selection.
- DGL construction remains outside this selection boundary; the selected view
  continues to expose direct tensor buffers.

Verification:

- Python compile checks passed for plan, prepare, Trainer, and package exports.
- Partition, store, compile semantics, compile plan, and state-request tests:
  48 passed.
- Path coverage confirms the removed prepare view switches no longer exist.

Unresolved performance risks:

- Snapshot sampled execution now selects snapshot CSC, but its loader-to-block
  hot path still needs parity and performance validation against FlareDTDG.
- Event-view state-write routes remain materialized before backbone state
  requirements are lowered; that boundary is intentionally deferred.

## 2026-08-06: Bind ExecutionPlan After Graph Partition

Latest state:

- `Trainer.prepare()` now executes the physical preparation order explicitly:
  load `GraphData`, build `PartitionPlan`, lower the bound `ExecutionPlan`, and
  materialize only the layouts selected by `ExecutionPlan.view`.
- The plan returned by `compile()` remains an unbound semantic plan with
  `partition_plan=None`; after `prepare()`, `trainer.plan.partition_plan` is the
  exact object produced by graph partitioning.
- Removed the combined `prepare_graph_data()` entry so callers cannot hide or
  reverse the partition-to-plan boundary.

Abstraction introduced:

- `partition_graph_data()` and `materialize_graph_data()` are the two stable
  graph-level operations needed around plan lowering. Each owns one operation
  and neither calls the other.

Attempted approaches:

- Considered retaining `prepare_graph_data()` as a convenience wrapper. It
  would necessarily choose a view before the partition-bound plan was lowered,
  so it was removed from package and runtime exports.

Files written or modified:

- `src/starrygl/prepare/data.py`
- `src/starrygl/prepare/__init__.py`
- `src/starrygl/executor/runtime/prepare.py`
- `src/starrygl/executor/runtime/trainer.py`
- `src/starrygl/__init__.py`
- `tests/test_partition_assignment.py`
- `tests/test_starrygl_store.py`
- `docs/migration/status_log.md`

Efficiency alternatives considered:

- Partition and view builders still receive the same loaded tensor objects;
  the new boundary introduces no graph tensor copy.
- `dataclasses.replace()` is used only for the small prepare config when a
  dataset supplies `time_ptr_2`.
- Existing vectorized torch/DGL/native partition and index construction paths
  are unchanged; no new hot-path Python loop or native operator is needed.

Verification:

- Python compile checks passed for prepare, Trainer, forwarding modules, and
  package exports.
- Partition, store, compile semantics, compile plan, and state-request tests:
  48 passed.
- Tests verify that the plan is unbound before prepare and contains real
  partition metadata after prepare.

Unresolved performance risks:

- `ExecutionPlan` now carries the real `PartitionPlan`, but dependency routes
  are not yet lowered from owner tensors; current await dependency declarations
  remain semantic placeholders.
- `PartitionPlan.route_table` remains empty until dependency/root information
  is available in the subsequent route-lowering review unit.

## 2026-08-11: Remove Partition RouteTable Placeholder

Latest state:

- Removed the unused `RouteTable` type and `PartitionPlan.route_table` field.
- `PartitionPlan` now contains only ownership, replica, shared-node, chunk, and
  partition metadata that can be determined during graph partitioning.
- Current partition design documentation assigns batch communication routing to
  future runtime `CommPlan` lowering.

Abstraction introduced:

- None. This review unit removes an unimplemented abstraction with no runtime
  consumer.

Attempted approaches:

- Considered retaining an empty route placeholder for future compatibility.
  It had no fixed ID, row, stage, or collective semantics and would constrain
  the later batch-level communication contract, so it was removed.

Files written or modified:

- `src/starrygl/partition/plan.py`
- `src/starrygl/partition/__init__.py`
- `src/starrygl/__init__.py`
- `tests/test_partition_assignment.py`
- `docs/design/current/partition.md`
- `docs/migration/status_log.md`

Efficiency alternatives considered:

- No tensor operation or hot path changed. Removing empty route serialization
  slightly reduces plan metadata and requires no torch/DGL/native alternative.

Verification:

- Path coverage found no remaining `RouteTable` or `route_table` references in
  target source, tests, or current design documents.
- Python compile checks passed for partition contracts and package exports.
- Partition, compile plan, compile semantics, and store tests: 43 passed.

Unresolved performance risks:

- Batch-level communication route lowering remains undefined and must be based
  on actual materialized dependencies rather than restored to partitioning.

## 2026-08-13: Converge Interface Plan Responsibilities

Latest state:

- Reduced `interface/plan.py` from 750 to 465 lines and reduced total plan plus
  dependency implementation to 569 lines.
- `ExecutionPlan` stores semantic state once in `StarrySpec`; temporal, state,
  scope, freshness, staleness, approximation, storage view, and execution spine
  are derived properties rather than duplicated constructor fields.
- Removed the internal `ExecutionOverride` planner branch, which was no longer
  reachable from the canonical public `compile()` API.
- Removed compatibility aliases and premature communication observations:
  `store_view`, `placement_policy`, `runtime_execution_plan`, `D_remote`,
  `remote_dependency_sources`, `comm_scheduler`, and `collective_ordering`.
- Moved the still-consumed feature/state/task request lowering into
  `interface/dependency.py` behind one `lower_dependencies()` call.

Abstraction introduced:

- `lower_dependencies()` is the single boundary needed by ExecutionPlan and
  runtime state-request construction. It replaces several plan-local helpers;
  no registry or compatibility layer was added.

Attempted approaches:

- Removing all await dependencies was considered, but runtime
  `state_request.py` consumes state dependencies today. Keeping that behavior
  in `plan.py` would continue mixing view and request lowering, so it was
  retained in a focused module.

Files written or modified:

- `src/starrygl/interface/plan.py`
- `src/starrygl/interface/dependency.py`
- `tests/test_open_compile_plan.py`
- `tests/test_open_compile_semantics.py`
- `docs/migration/status_log.md`

Efficiency alternatives considered:

- This is construction-time control-plane code. No graph tensor, DGL kernel,
  communication buffer, or training hot path changed.
- Dependency construction remains a fixed small list per compiled model/task;
  a torch/DGL/native implementation would add overhead without a hot-path gain.

Verification:

- Python compile checks passed for plan and dependency modules.
- Partition, compile semantics, compile plan, state-request, and store tests:
  46 passed.

Unresolved performance risks:

- Feature/task dependencies remain semantic declarations, not batch-resolved
  local/remote routes. Communication lowering must not infer remote status from
  the removed compatibility properties.

## 2026-08-13: Unify Batch Graph Inputs

Latest state:

- Removed the descriptive `ViewPlan.batch_surface` string; view planning now
  records only required physical layouts.
- Removed the compatibility `Batch.blocks` field. Sampled graph inputs use
  `Batch.mfgs[window][layer]`; full-graph inputs use `Batch.graph`.
- Native sampling now wraps one sampled block sequence as a single-window MFG.
- `GraphBlock` remains the only graph tensor container; no graph/block base
  classes or plan-time instances were added.
- `interface/plan.py` is now 271 lines.

Abstraction introduced:

- None. This removes a duplicate Batch representation and a descriptive plan
  field that had no scheduling behavior.

Files written or modified:

- `src/starrygl/interface/plan.py`
- `src/starrygl/batch/source.py`
- `src/starrygl/native/sampling.py`
- `src/starrygl/executor/runtime/loop.py`
- `src/starrygl/model/_graph_ops.py`
- focused tests using the current interface
- `docs/migration/status_log.md`

Efficiency alternatives considered:

- Reuses the existing tuple-based MFG representation and `GraphBlock`; no
  conversion, graph copy, DGL rebuild, or native operator is introduced.
- One sampled batch adds only an outer one-element tuple to represent its
  window dimension.

Verification:

- Source coverage found no remaining `batch_surface` or `batch.blocks` use.
- Plan, partition, dependency, store, model, access, and compact-layout tests:
  99 passed, 1 skipped.

Unresolved performance risks:

- Legacy test modules still contain assertions for removed compatibility APIs;
  they are migration-source tests and should be retired rather than restored.

## 2026-08-13: Remove Duplicate Prepare Route Type

Latest state:

- Removed the prepare-only `Route` class and its package exports.
- Snapshot materialization writes its artifact route dictionary directly.
- Runtime communication continues to use the sole executable
  `executor.runtime.comm.Route` contract.
- View materialization explicitly documents that it builds only layouts listed
  in `ViewPlan.required_layouts`.

Abstraction introduced:

- None. A duplicate type was deleted.

Efficiency alternatives considered:

- No tensor construction or hot path changed. Reusing runtime `Route` was not
  appropriate because the snapshot artifact has a distinct serialized
  `recv_src_row`; a plain dictionary is the existing artifact contract.
- Time splitting, event state-write metadata, and snapshot CSC construction
  were retained because each has active runtime consumers.

Files written or modified:

- `src/starrygl/prepare/build_part_graph.py`
- prepare/runtime/package exports
- `docs/migration/status_log.md`

Verification:

- Python compile checks passed for prepare and forwarding exports.
- Partition, store, plan, semantics, and compact-layout tests: 46 passed.

Unresolved performance risks:

- Snapshot CSC materialization still contains rank/snapshot Python loops; this
  requires profiling before replacing working tensor logic with a native op.

## 2026-08-14: Preserve Dataset Split Masks

Latest state:

- `PreparedViews` and persisted prepare artifacts now retain boolean
  `train/val/test` edge masks in `split_masks`.
- Explicit `split_labels` produce exact masks; ratio-based splits produce masks
  from the resolved ranges.
- `GraphStore.split_masks` exposes the persisted table.
- Existing split ranges, batch-size-derived `time_ptr_2`, and snapshot CSC route
  artifacts are unchanged.

Abstraction introduced:

- None. One helper converts the already-resolved split declaration to masks.

Efficiency alternatives considered:

- Masks use three vectorized boolean comparisons or contiguous slice writes;
  no per-edge Python loop or native operator is needed.

Files written or modified:

- `src/starrygl/prepare/build_part_graph.py`
- `src/starrygl/store/graph.py`
- `tests/test_partition_assignment.py`
- `tests/test_starrygl_store.py`
- `docs/migration/status_log.md`

Verification:

- Python compile checks passed for prepare and graph store modules.
- Partition, store, plan, semantics, and compact-layout tests: 47 passed.

Unresolved performance risks:

- Split labels remain constrained to contiguous train/val/test order because
  split time pointers still represent contiguous event ranges.

## 2026-08-14: Extract Split And Time Pointer Materialization

Latest state:

- Moved train/val/test ranges, masks, equal-timestamp boundaries, and
  batch/adaptive `time_ptr_2` construction to `prepare/split.py`.
- `build_part_graph.py` now calls five split operations and contains no split
  implementation.
- Removed unused `_clip_time_ptr_2()` and `_select_time_ptr_2_rows()` helpers.
- `build_part_graph.py` decreased from 881 to 635 lines; combined implementation
  is 814 lines.

Abstraction introduced:

- None. `split.py` is a direct functional module with no class, registry, or
  compatibility re-export.

Efficiency alternatives considered:

- Existing tensorized mask/range operations were retained. Adaptive batching
  keeps its timestamp-run loop because it executes during preprocessing and no
  measured native bottleneck exists.

Files written or modified:

- `src/starrygl/prepare/split.py`
- `src/starrygl/prepare/build_part_graph.py`
- `docs/migration/status_log.md`

Verification:

- Python compile checks passed for split and view materialization modules.
- Partition, store, compile semantics, and compact-layout tests: 39 passed.

Unresolved performance risks:

- Adaptive batch splitting remains CPU/Python preprocessing code; profile it
  before considering a tensor or native rewrite.

## 2026-08-14: Extract Temporal CSR Materialization

Latest state:

- Moved temporal CSR construction to `prepare/temporal_csr.py`.
- Moved the shared compressed-layout tensor helper with it; snapshot CSC
  temporarily imports that helper until its own migration unit.
- Removed the unused `view_edge_master` tensor construction.
- `build_part_graph.py` decreased from 635 to 553 lines; combined implementation
  remains 635 lines after removing dead work.

Abstraction introduced:

- None. The module exports one view builder and one tensor layout helper.

Efficiency alternatives considered:

- Existing vectorized torch stack/index/select/sort/bincount operations remain
  the hot path. No DGL rebuild or custom native operator was added.

Files written or modified:

- `src/starrygl/prepare/temporal_csr.py`
- `src/starrygl/prepare/build_part_graph.py`
- `docs/migration/status_log.md`

Verification:

- Python compile checks passed for temporal CSR and view orchestration.
- Partition, store, compile semantics, and compact-layout tests: 39 passed.

Unresolved performance risks:

- CSR sorting remains CPU torch preprocessing; profile before replacing it.

## 2026-08-14: Extract Snapshot CSC Materialization

Latest state:

- Moved snapshot CSC, GCN normalization, and required collective route artifact
  construction to `prepare/snapshot_csc.py`.
- Removed the unused `edge_master` parameter from snapshot materialization.
- Snapshot route fields remain part of every snapshot CSC slice.

Abstraction introduced:

- None. One functional module owns the existing snapshot artifact.

Efficiency alternatives considered:

- Preserved vectorized tensor indexing and the necessary rank/snapshot loops.
  No unmeasured native rewrite was introduced.

Files written or modified:

- `src/starrygl/prepare/snapshot_csc.py`
- `src/starrygl/prepare/build_part_graph.py`
- `docs/migration/status_log.md`

Verification:

- Python compile checks passed; focused prepare/store/plan tests passed.

Unresolved performance risks:

- Rank/snapshot loops remain preprocessing work pending profiling.

## 2026-08-14: Extract Event View Materialization

Latest state:

- Moved rank-local event view and local time-pointer mapping to
  `prepare/event.py`.
- State-write masks and normal/hot state-write routes remain intact.
- Removed the unused `local_pos` argument from time-pointer localization.
- `build_part_graph.py` now contains only `PrepareConfig`, `PreparedViews`, and
  `materialize_graph_views()` and is 270 lines.

Abstraction introduced:

- None. Event materialization remains a pair of direct functions.

Efficiency alternatives considered:

- Existing vectorized search/index operations remain unchanged; no graph or
  tensor copies were added by the module boundary.

Files written or modified:

- `src/starrygl/prepare/event.py`
- `src/starrygl/prepare/build_part_graph.py`
- `docs/migration/status_log.md`

Verification:

- Prepare module compile checks passed.
- Partition, store, compact-layout, compile semantics, and plan tests: 47 passed.

Unresolved performance risks:

- Event state-write metadata remains unconditional for event views; lowering it
  from backbone state requirements is a later semantic review.

## 2026-08-14: Default Temporal CSR To Bidirectional

- `ViewPlan.temporal_csr_bidirectional` now defaults to `True`.
- Runtime may still explicitly set `sampling.neighbor.bidirectional=False`.
- No new abstraction or hot-path operation was introduced; the existing
  vectorized reverse-edge construction is reused.
- Focused plan/partition tests cover the default.

## 2026-08-14: Unify Trainer Artifact Preparation

Latest state:

- `Trainer.prepare_artifacts()` now reuses `Trainer.prepare(save=True)`.
- Partitioning binds the real `PartitionPlan` before `ExecutionPlan.view`
  materialization for both event and snapshot inputs.
- Artifact readiness now checks the StarryGL store contract:
  `prepare.pt`, `graph_XXX.pt`, and `feature_XXX.pt`.
- Removed the separate event/snapshot artifact writers and their unused helper
  functions from the trainer.

Abstraction introduced:

- None. The existing prepare path and artifact writer are reused directly.

Efficiency alternatives considered:

- Reusing the materialized views avoids running a second preprocess pipeline
  and preserves the current vectorized partition/view builders.

Files written or modified:

- `src/starrygl/executor/runtime/trainer.py`
- `tests/test_starrygl_store.py`
- `docs/migration/status_log.md`

Verification:

- Python compile check passed.
- Focused store/prepare tests: 13 passed.
- Current compile/partition set: 30 passed; 23 failures remain in the legacy
  `test_starrygl_compile_plan.py` assertions for removed `spec`/`override`, old
  config aliases, and the removed `starrygl.distributed` module.

Unresolved performance risks:

- Artifact rebuild invalidation currently uses `force=True`; content-addressed
  prepare signatures can be added when configuration caching is required.
- Legacy artifact-layout tests still expect `graph.pt`, `dist.pt`, and
  `partition_data_XXX.pt`; those tests must move to the StarryGL store contract.

## 2026-08-14: Reduce Trainer Construction And Config Logic

Latest state:

- Moved direct model/task construction into `executor/runtime/compile.py`;
  `trainer.py` keeps only imported compatibility exports.
- Model construction now accepts the canonical backbone names and dimensions
  documented by the open interface instead of translating legacy aliases.
- Removed cross-layer preprocess/runtime/metadata lookup helpers. Prepare
  options now come from canonical `runtime.preprocess` configuration.
- Reduced `trainer.py` from 1692 to 1528 lines.

Abstraction introduced:

- None. Three existing construction functions were moved into the existing
  compile module; no registry or factory class was added.

Efficiency alternatives considered:

- Runtime execution and tensor hot paths are unchanged. This is ownership and
  parsing cleanup only.

Files written or modified:

- `src/starrygl/executor/runtime/compile.py`
- `src/starrygl/executor/runtime/trainer.py`
- `docs/migration/status_log.md`

Verification:

- Python compile checks passed.
- Open compile/plan and store tests: 37 passed.
- Model/task tests: 62 passed; four existing `NegativeSampler` tests still use
  removed `sample_fn` and `filter_positive` constructor fields.

Unresolved performance risks:

- None introduced. Distributed orchestration and fit/evaluate composition are
  still large and remain the next Trainer ownership review.

## 2026-08-14: Split Trainer Runtime Responsibilities

Latest state:

- Reduced public `executor/runtime/trainer.py` from 1528 to under 250 lines.
- `trainer.py` now owns the compiled trainer data, prepare/artifact entry, and
  observable config export.
- Moved train/evaluate/predict and distributed orchestration unchanged to
  module-level functions in `train.py`.
- Added `starrygl/main.py` as the Python task entry point. It creates the
  Trainer from config and explicitly runs prepare, fit, evaluate, and optional
  predict stages; `runtime/train.py` contains execution implementation only.
- Moved canonical runtime option resolution to `trainer_options.py` and pure
  orchestration helpers to `trainer_support.py`.
- `Trainer` remains the concrete public dataclass. Prepare, train, evaluate,
  predict, distributed execution, and config export remain visible public
  methods; their long implementations delegate to the execution module.

Abstraction introduced:

- One internal option helper base groups configuration resolution. Execution
  is a plain module, not a second Trainer/Runner/Engine class.

Efficiency alternatives considered:

- The change mechanically relocates Python orchestration only. Tensor, DGL,
  communication, sampling, and state hot paths are unchanged.

Files written or modified:

- `src/starrygl/executor/runtime/trainer.py`
- `src/starrygl/executor/runtime/train.py`
- `src/starrygl/main.py`
- `src/starrygl/executor/runtime/trainer_options.py`
- `src/starrygl/executor/runtime/trainer_support.py`
- `tests/test_open_compile_semantics.py`
- `docs/migration/status_log.md`

Verification:

- Python compile checks passed for all trainer modules.
- Open compile/plan and store tests, including ownership assertions: 38 passed.

Unresolved performance risks:

- None introduced; method bodies and execution order are unchanged.

## 2026-08-14: Reduce Prepare Data Loading To Canonical Input

Latest state:

- Rewrote `prepare/data.py` from 802 to 252 lines.
- The core loader now accepts canonical mappings, `.pt` mappings, and CSV files
  with `src`/`dst` headers.
- Removed Flare-specific overlapping-window generation, automatic degree
  features, next-snapshot labels, field aliases, sidecar guessing, optional
  pandas fallback, and duplicate table parsers.
- Partition and view materialization contracts are unchanged.

Abstraction introduced:

- None. One direct loader validates and normalizes the canonical tensors.

Efficiency alternatives considered:

- Tensor ordering and edge-aligned reindexing remain vectorized torch
  operations. CSV parsing is preprocessing-only and uses the standard library.
- Flare conversion must happen before the StarryGL core boundary and supply
  explicit `time_ptr_2`, features, and labels.

Files written or modified:

- `src/starrygl/prepare/data.py`
- `tests/test_starrygl_store.py`
- `docs/migration/status_log.md`

Verification:

- Python compile check passed.
- Partition, store, open compile, and plan tests: 46 passed.

Unresolved performance risks:

- Large CSV ingestion is not a training hot path; users with large sources
  should convert once to the canonical `.pt` mapping.

## 2026-08-17: Pack Temporal Features into Snapshot-CSC

Latest state:

- Prepare keeps one graph contract: event views, temporal CSR, and packed
  rank-local Snapshot-CSC slices with Route metadata.
- `runtime.preprocess.feature_layout` has two physical layouts: `separate`
  keeps all features in `feature_XXX.pt`; `snapshot_csc` embeds only temporal
  node features shaped `[S, N, F]` in the rank Snapshot-CSC artifact.
- Embedded values are gathered in each snapshot's `src_nodes` order and stored
  as `node_data.x = {data, ptr}`, matching FlareDTDG's compressed row-group
  form without importing its PartitionData/TensorData class hierarchy.
- Static node features and edge features remain in `feature_XXX.pt`. Runtime
  materialization reads embedded `x` directly; the public Batch/model contract
  is unchanged.

Abstraction introduced:

- None. Feature placement remains one serialization option on the existing
  rank artifact.

Efficiency alternatives considered:

- Snapshot packing uses vectorized `index_select` and `torch.cat` during
  preprocessing. The runtime consumes direct CSC buffers and tensor slices;
  it does not rebuild DGL graphs or loop over nodes on the training hot path.
- Keeping all temporal features in `FeatureManager` would avoid snapshot-local
  duplication but add indexed gathers at every batch. Custom C++/CUDA packing
  is unnecessary because packing runs once during preprocessing.
- A direct C++ temporal-CSR constructor was not added. The current native
  sampler still receives sorted `src`, `dst`, `ts`, and `edge_ids`, then builds
  its neighbor table in C++; prepared `indptr` and `indices` are not yet
  consumed directly by that constructor.

Files written or modified:

- `src/starrygl/store/feature.py`
- `src/starrygl/store/artifact.py`
- `src/starrygl/executor/runtime/graph_blocks.py`
- `src/starrygl/executor/runtime/snapshot.py`
- `src/starrygl/executor/runtime/trainer.py`
- `src/starrygl/interface/spec.py`
- `src/starrygl/interface/compile_example.py`
- `tests/test_starrygl_store.py`
- `docs/migration/status_log.md`

Verification:

- Python compile checks passed for the modified runtime, store, interface, and
  test modules.
- Store, partition, compile semantics, GraphBlock, materialization, compact
  layout, and model tests: 94 passed.
- Six selected legacy public-API prepare tests still fail before reaching this
  path because they call removed `compile(graph/model/task/spec/preprocess)`
  arguments or monkeypatch the removed legacy preprocess function. They need
  replacement with canonical `data_source/backbone/task_segment/runtime`
  tests; the compatibility surface was not restored.

Unresolved performance risks:

- Embedded mmap tensors are not yet explicitly pinned for asynchronous H2D
  transfer; this needs profiling before adding a pinning stage.
- Direct construction of the native sampler from prepared temporal-CSR
  buffers needs a measured C++ interface change before the redundant
  COO-to-neighbor-table build can be removed.

## 2026-08-17: Load Snapshot-CSC Rows Lazily

Latest state:

- Packed Snapshot-CSC artifacts remain columnar after `load_starrygl_store()`.
  An internal sequence restores one row only when its `snapshot_id` is read.
- Snapshot execution uses that sequence directly instead of constructing an
  eager `{snapshot_id: row}` dictionary. Event materialization and the shared
  execution loop are unchanged.
- The lazy row exposes tensor views into packed topology, Route, and embedded
  node-feature buffers; it does not copy those packed field slices.

Abstraction introduced:

- One private `_LazySnapshotSlices` sequence is required to preserve the
  existing `view["slices"]` access shape while deferring row restoration. It is
  not public API and does not introduce a second Store/View/Batch hierarchy.

Efficiency alternatives considered:

- Keeping eager row dictionaries was rejected because it scales Python object
  construction with all snapshots at store load time.
- Migrating FlareDTDG `TensorData`/`PartitionData` was unnecessary: the stored
  buffers already use `data + ptr`, and only lazy row access was missing.
- A C++/CUDA operator is not justified for metadata slicing. Training tensor
  operations remain torch views, `index_select`, and direct CSC buffers.

Files written or modified:

- `src/starrygl/store/artifact.py`
- `src/starrygl/executor/runtime/snapshot.py`
- `tests/test_starrygl_store.py`
- `docs/migration/status_log.md`

Verification:

- Python compile checks passed.
- Store, partition, compile semantics, GraphBlock, materialization, compact
  layout, and model tests: 94 passed.
- The focused artifact test also reaches public `iter_batches(mode="snapshot")`
  through the lazy sequence.

Unresolved performance risks:

- Restoring the slim format still concatenates the shared destination prefix
  with per-snapshot extra source nodes. Profile this before choosing between
  the smaller artifact and Flare-style full `src_ids + ptr` storage.
- Explicit pinned-memory and asynchronous H2D staging remain unmeasured.

## 2026-08-17: Split Snapshot Runtime Responsibilities

Latest state:

- Reduced `executor/runtime/snapshot.py` from 2027 to 329 lines. It now owns
  Snapshot window planning and iteration only, plus the existing public
  Snapshot exports.
- Moved each existing implementation once, without wrappers or fallback
  branches:
  - `snapshot_materialize.py`: Batch and window materialization, 488 lines.
  - `snapshot_cache.py`: Snapshot entry/blob caches, 349 lines.
  - `snapshot_rows.py`: CSC row slicing and chunk transforms, 450 lines.
  - `snapshot_features.py`: feature reads and deferred fetch, 319 lines.
  - `snapshot_targets.py`: supervision and endpoint targets, 207 lines.
- `loop.py` imports private runtime operations from their owning modules rather
  than using `snapshot.py` as a private re-export layer.

Abstractions introduced:

- No new class hierarchy or public API. The split follows five existing
  function clusters with independent dependencies and multiple real callers.

Efficiency alternatives considered:

- A single `snapshot_utils.py` was rejected because it only relocates the
  original monolith and preserves mixed ownership.
- Runtime tensor operations are unchanged. No new Python loop was added to a
  materialization, communication, or CSC hot path.
- Moving these operations into C++/CUDA would not reduce Python orchestration
  cost and is not justified without a measured tensor-kernel bottleneck.

Files written or modified:

- `src/starrygl/executor/runtime/snapshot.py`
- `src/starrygl/executor/runtime/snapshot_materialize.py`
- `src/starrygl/executor/runtime/snapshot_cache.py`
- `src/starrygl/executor/runtime/snapshot_rows.py`
- `src/starrygl/executor/runtime/snapshot_features.py`
- `src/starrygl/executor/runtime/snapshot_targets.py`
- `src/starrygl/executor/runtime/loop.py`
- `tests/test_starrygl_store.py`
- `docs/migration/status_log.md`

Verification:

- Python compile checks passed for all split modules and direct callers.
- Snapshot/store/partition/compile/GraphBlock/materialization/model tests:
  94 passed.
- Full test suite: 153 passed, 1 skipped, 12 failed. All 12 failures exercise
  previously removed legacy config/API names or removed legacy
  `NegativeSampler` arguments; no failure enters the Snapshot runtime.

Unresolved performance risks:

- This was a responsibility-only refactor. Snapshot throughput and memory use
  still require the planned pinned-memory/H2D profile.

## 2026-08-17: Separate Window And Neighbor-Sampling Policies

Latest state:

- `ExecutionPlan` and `RuntimeBatchPlan` now carry independent
  `window_policy` and `sampling_policy` fields.
- Event execution always lowers to `event_window`. Snapshot execution lowers
  to `full_snapshot` or `chunk_decay`. Both modes can select `full` or
  `neighbor` graph materialization without overloading one policy string.
- `runtime.sampling.mode=null` derives `full/neighbor` from
  `backbone.spatial_aggregation`; an explicit `full` or `neighbor` remains an
  override. Trainer defaults consume the compiled plan rather than reparsing
  the runtime mapping.
- Removed the runtime `SnapshotPolicy`/`SnapshotWindowPlan` aliases and the
  `neighbor_sample` materialization-policy spelling.

Abstraction introduced:

- No new class hierarchy. Two fields were added to the existing plan objects
  because `chunk_decay + neighbor` is a real composition that one enum could
  not represent.

Efficiency alternatives considered:

- Merging Event and Snapshot materializers was rejected: they use different
  physical layouts and only need to return the same `Batch` contract.
- No Python node/edge loop or extra tensor conversion was added. Existing
  torch/native sampling and CSC materialization paths remain in place.
- Building a Python/DGL sampled graph from every Snapshot-CSC row was rejected
  as a hot-path regression. A window-aware sampled-Snapshot implementation
  should reuse native CSC/T-CSR buffers or add a measured C++/CUDA operator.

Files written or modified:

- `src/starrygl/interface/{plan.py,spec.py,compile_example.py}`
- `src/starrygl/executor/runtime/{plan.py,batches.py,event.py,snapshot.py,snapshot_materialize.py,loop.py,train.py,trainer_options.py,snapshot_chain.py}`
- runtime/public re-exports and the three CLI entry modules
- `tests/test_open_compile_semantics.py`
- `tests/test_open_compile_plan.py`
- current interface/design notes and this status log

Verification:

- Python compile checks passed for all changed runtime, interface, CLI, and
  export modules.
- Interface/plan tests: 27 passed.
- Store, partition, compile semantics, GraphBlock, materialization, compact
  layout, and model tests: 95 passed.
- Access double-buffer tests: 2 passed, 1 skipped.
- Full suite: 154 passed, 1 skipped, 12 failed. The same 12 failures use
  removed legacy config/API names or removed legacy `NegativeSampler`
  arguments; no new failure was introduced.

Unresolved implementation and performance risks:

- Snapshot `sampling_policy=neighbor` still constructs the native sampler from
  Event/T-CSR topology. The new plan correctly represents window-local
  sampling, but physical `chunk_decay + neighbor` boundary enforcement is not
  yet validated. Do not replace it with per-window Python CSC sampling; add a
  native Snapshot-CSC/T-CSR path and benchmark it before claiming parity.

## 2026-08-17: Share Batch Materialization And Split Event Runtime

Latest state:

- Event and Snapshot iterators now use one `materialized_window()` path for
  `RuntimeBatchPlan` metadata, `WindowContext`, and `BatchWindow` construction.
- Block communication binding and non-empty feature-name selection are shared
  by Event and Snapshot materializers instead of being reimplemented locally.
- Target owner tensors and node-to-Batch row lookup use the shared target
  helpers in `targets.py`.
- Reduced `event.py` from 1691 to 436 lines. It now owns Event window planning
  and iteration. `event_materialize.py` owns Batch/target materialization (894
  lines), and `event_features.py` owns deferred feature launch/read operations
  (371 lines).
- Fixed the direct Snapshot batch path to preserve the pending node-feature
  fetch returned by `_read_snapshot_features()`.

Abstractions introduced:

- No new class hierarchy or public API. Two shared functions were added to the
  existing materialization module because both physical views already produce
  the same `RuntimeBatchPlan -> BatchWindow` contract.
- `event_features.py` is an ownership split of existing code, matching the
  existing Snapshot feature boundary; it does not wrap or duplicate the old
  Event implementation.

Efficiency alternatives considered:

- A callback-driven generic Event/Snapshot materializer was rejected because
  Event ranges and Snapshot-CSC/chunk caches have different hot paths.
- Feature gather and communication remain in the existing vectorized
  `exchange.py` torch/collective implementation. No Python node/edge loop or
  new DGL conversion was added.
- A custom C++/CUDA change is unnecessary for this responsibility-only
  refactor; native sampling and direct CSC/T-CSR buffers are unchanged.

Files written or modified:

- `src/starrygl/executor/runtime/event.py`
- `src/starrygl/executor/runtime/event_materialize.py`
- `src/starrygl/executor/runtime/event_features.py`
- `src/starrygl/executor/runtime/materialize.py`
- `src/starrygl/executor/runtime/{loop.py,snapshot.py,snapshot_cache.py}`
- `src/starrygl/executor/runtime/{snapshot_features.py,snapshot_materialize.py}`
- `src/starrygl/executor/runtime/{snapshot_targets.py,targets.py}`
- `tests/test_executor_materialize.py`
- `docs/migration/status_log.md`

Verification:

- Python compile and Pyflakes checks passed for the changed runtime modules.
- Focused materialization, target, compile, double-buffer, store, and compact
  layout tests: 51 passed, 1 skipped.
- Full suite: 155 passed, 1 skipped, 12 failed. The same 12 failures exercise
  removed legacy config/API names or removed legacy `NegativeSampler`
  arguments; no failure enters the changed runtime paths.

Unresolved performance risks:

- Background loader prefetch, sampled/launched double buffers, and dependency
  completion are still owned by nested queues in `loop.py` and `AccessScheduler`.
  Move them only when a standalone Loader/Scheduler contract is finalized;
  relocating them now would not change runtime cost.
- Snapshot neighbor sampling still needs native enforcement of Snapshot window
  boundaries, as recorded above.

## 2026-08-17: Make Safe Feature Double Buffer The Default

Latest state:

- Canonical runtime defaults now set `access_pipeline=true`. When the internal
  `access_pipeline_launched_queue` override is absent, it follows that public
  switch, so normal training uses two launched feature-access slots.
- `access_pipeline=false` disables both the sampled access pipeline and the
  launched double buffer unless the internal launched-queue override is
  explicitly enabled.
- The three CLI entry points use the same derived default. Their state-hydrate
  lookahead defaults are false because state collectives remain ordered on the
  main thread.
- `compile_example.py` documents the default without exposing a second public
  double-buffer policy.

Abstraction introduced:

- None. The existing `runtime.access_pipeline` boolean is the single public
  control; the existing launched queue remains an internal execution detail.

Efficiency alternatives considered:

- Defaulting `access_pipeline_launched_queue=true` independently was rejected:
  users would then need two flags to disable the pipeline safely.
- A new nested queue/double-buffer config was unnecessary. The existing two
  launched slots already implement the required overlap and passed collective
  ordering tests.
- State hydration was not moved into the lookahead slot because independently
  launched collectives can reorder ranks. Double-buffer lookahead remains
  feature-only.

Files written or modified:

- `src/starrygl/interface/{spec.py,compile_example.py}`
- `src/starrygl/executor/runtime/loop.py`
- `src/starrygl/cli/{gdelt_edge_predict.py,gdelt_node_predict.py,dtdg_benchmark.py}`
- `tests/{test_access_double_buffer.py,test_open_compile_semantics.py}`
- `docs/migration/status_log.md`

Verification:

- Python compile checks passed for runtime, interface, CLI, and tests.
- Focused access/materialization/compile/store tests: 50 passed, 1 skipped.
- Full suite: 156 passed, 1 skipped, 12 failed. The same 12 failures use
  removed legacy config/API names or removed legacy `NegativeSampler`
  arguments.
- Two-rank Gloo execution of `test_access_double_buffer.py`: 4 passed on each
  rank, including collective order with the default double buffer.

Unresolved performance risks:

- Correctness and ordering are covered, but end-to-end GPU throughput and peak
  memory with the new default have not yet been benchmarked. The default keeps
  two launched Batch windows and raises materialization prefetch capacity to
  three; profile representative Event and Snapshot workloads before changing
  those fixed depths.

## 2026-08-18: Ponytail Runtime And Source-Size Convergence

Latest state:

- Repository guidance now makes Ponytail `full` the default refactor mode and
  limits every production module under `src/starrygl` to 500 physical lines.
  `tests/test_source_size.py` enforces the limit.
- Production source size fell from the recorded 30,968-line baseline to 26,598
  lines. No production Python module exceeds 500 lines.
- `runtime/loop.py` is now batch-window orchestration only. Epoch execution,
  evaluation, Snapshot evaluation, and access-pipeline queues have independent
  implementation modules.
- Node and edge feature exchange use one tensor/collective implementation in
  `exchange.py`; cache lookup/write and partition-index device caching live in
  `feature_cache.py`. The prior padded-edge and profiling branches were
  removed.
- `memory.py` fell from 2,438 to 457 lines and is now the state-commit facade.
  Access, stale reads, shared-hot synchronization, filtering, and tensor
  operations are separate implementation responsibilities. Their combined
  size is 1,353 lines, so this is also a net deletion rather than a line-count
  split.
- Removed unconfigured shared-state experiments controlled by
  `STARRYGL_PACK_SHARED_STATE`, `STARRYGL_PACK_SHARED_UNION`, and
  `STARRYGL_SHARED_MULTI_COUNTS`, plus unconsumed fine-grained timing counters.
  The retained shared-hot path is one variable-length collective gather for
  memory and one for mailbox data.
- Bounded-stale misses now use a global need reduction before owner fetch, so
  ranks without local misses still enter the same collective epoch with empty
  requests.
- Negative-sampling UDFs and positive filtering are runtime constructor inputs
  of `EdgePredictionRootBuilder`; `NegativeSampler` remains a serializable
  semantic declaration.
- Tests that still asserted removed `graph/model/task/spec/execution` aliases
  now cover the canonical `data/backbone/task/runtime` API.

Attempted approaches:

- Pure line-number splitting was rejected because it would leave forwarding
  fragments and preserve the same coupling.
- Event/Snapshot execution was not forced into one callback framework. Shared
  Batch construction and target/feature steps were consolidated, while their
  different physical window planners remain separate.
- The feature exchange rewrite first kept cache code inline; it still exceeded
  the module limit. Cache ownership was moved to one reusable runtime cache
  module used by both node and edge exchange.
- Memory packed-union variants were considered for retention, but no config,
  test, or runtime decision selected them. Keeping one collective format is the
  smallest deadlock-auditable implementation.

Abstractions introduced:

- No public abstraction, registry, adapter, or compatibility framework was
  added. New modules are direct ownership boundaries for existing runtime
  responsibilities.
- `PendingMemoryMailboxRead` and shared-sync payloads remain internal batch-level
  handles. Models and tasks still consume the existing `Batch` contract.

Efficiency alternatives considered:

- Selected torch tensor operations (`unique`, `searchsorted`, `index_copy_`,
  `scatter_reduce_`) and the existing `CommScheduler` collectives. They keep
  request compaction, response restore, stale-cache fill, and latest-state
  selection out of per-node Python loops.
- DGL operators do not provide owner-state commit, mailbox append, variable
  feature exchange, or shared-hot state synchronization primitives, so they do
  not replace these paths.
- DGL-kernel-inspired or custom C++/CUDA operators remain candidates only if a
  representative profile identifies compaction/scatter or cache lookup as a
  dominant kernel. This refactor does not justify a native maintenance burden.
- End-to-end feature and state communication remains batch-granular and uses
  scheduled collectives; no free-form p2p path was introduced.

Files written or modified:

- `../AGENTS.md`
- `src/starrygl/executor/runtime/{loop.py,epoch.py,evaluation.py,snapshot_evaluation.py}`
- `src/starrygl/executor/runtime/{access_pipeline.py,batches.py,state_access.py,state_query.py}`
- `src/starrygl/executor/runtime/{exchange.py,feature_cache.py,comm.py,autograd_comm.py}`
- `src/starrygl/executor/runtime/{memory.py,memory_access.py,memory_filter.py,memory_ops.py,memory_shared.py,memory_stale.py}`
- `src/starrygl/executor/runtime/{event.py,event_materialize.py,event_targets.py,train.py,distributed_train.py}`
- `src/starrygl/executor/runtime/{trainer.py,trainer_support.py,trainer_options.py,window_scan.py,recurrent_state.py}`
- `src/starrygl/model/{tgn.py,tgn_mailbox.py,_graph_ops.py,graph_conv.py,tgat.py,base.py}`
- `src/starrygl/native/{sampling.py,sampling_output.py}`
- `src/starrygl/store/{state.py,mailbox.py,remote_fetch.py}`
- `src/starrygl/cli/{feature_cache.py,gdelt_edge_predict.py,gdelt_node_predict.py}`
- `src/starrygl/task/negative.py`
- `tests/{test_source_size.py,test_executor_memory.py,test_open_public_api.py,test_starrygl_core_skeleton.py,test_starrygl_task.py}`
- `docs/migration/status_log.md`

Verification:

- Python compile checks passed for all `src/starrygl` modules.
- Focused runtime, exchange, memory, model, store, and canonical API tests:
  74 passed, 1 skipped.
- Full suite: 170 passed, 2 skipped.
- Two-rank Gloo combined memory/mailbox owner fetch: 5 tests passed on each
  rank, including the distributed fetch assertion.
- Two-rank Gloo default access double buffer: 4 tests passed on each rank,
  including collective-order validation.

Unresolved performance risks:

- No representative GPU end-to-end benchmark was run, so the refactor claims
  code and correctness convergence, not a throughput improvement.
- Shared-hot synchronization now intentionally has one format. Reintroduce a
  packed format only after profiling shows collective launch/bytes dominate and
  after adding equivalent multi-rank ordering tests.
- The combined state/mailbox path can reuse the node-feature request context.
  A future optimization should make the shared compact-id contract explicit so
  mismatched feature/state cache-hit sets cannot silently disable reuse.

## 2026-08-18: Canonical Package Layout Cleanup

Latest state:

- `src/starrygl` now has one physical implementation path per responsibility.
  Public semantics live in top-level `api.py`, `spec.py`, and `plan.py`; runtime
  implementation lives only in `runtime/`.
- Removed transition namespaces under `interface/`, `executor/`,
  `model/backbone/`, `store/{core,loader,view,batch}/`, `task/segment/`,
  `utils/common/`, `cli/commands/`, `native/kernels/`, and the empty `loader/`.
- Python file count fell from 190 to 115. No production module exceeds 500
  lines.
- Removed the unconsumed `StateReadRequest` template and its existence-only
  tests. `ExecutionPlan.await_dependencies` remains the observable dependency
  declaration until runtime scheduling consumes it directly.
- Renamed runtime model/task construction from the ambiguous `compile.py` to
  `builders.py`; public compilation remains `starrygl.api.compile`.

Attempted approaches:

- Keeping one-line re-export modules was rejected because the open package no
  longer needs two import paths for one implementation.
- Large runtime modules were not merged merely to reduce file count. Existing
  files with real independent callers remain direct implementation boundaries.
- Historical entries in this log retain their original paths. Only current
  layout documentation was replaced.

Abstractions introduced:

- None. Config helpers were folded into `api.py`, and dependency lowering was
  folded into `plan.py` because each had one caller.

Efficiency alternatives considered:

- This is an import/layout-only migration. Torch/DGL/native hot paths and the
  collective communication protocol are unchanged, so no new kernel is
  justified.
- Removing re-export imports slightly reduces module-loading work but makes no
  training-throughput claim.

Files written or modified:

- `src/starrygl/{api.py,spec.py,plan.py,config_example.py,main.py}`
- `src/starrygl/runtime/` and all runtime/model/store/CLI callers
- `tests/` imports and canonical-layout assertions
- `docs/OPEN_SOURCE_LAYOUT.md`
- `docs/STAGING_STATUS.md` (removed)
- `docs/migration/status_log.md`

Verification:

- Compile check: `python -m compileall -q src/starrygl tests` passed.
- Canonical API/config/plan tests: 34 passed.
- Full suite: 165 passed, 2 skipped.
- Import coverage found no remaining references to removed package paths.

Unresolved performance risks:

- No performance benchmark was rerun because executable tensor, sampler,
  communication, and state-update code was not changed.
- `ExecutionPlan.wait_policy` and `await_dependencies` are still not the direct
  source of `AccessScheduler` behavior; that semantic/runtime binding remains
  separate work.

## 2026-08-18: Paper-Method Functional Convergence

Latest state:

- The canonical `data/backbone/task/runtime` configuration now lowers training
  batch limits, device, dropout, boundary neighbor sampling, and phase-specific
  negative sampling into the existing runtime. Explicit
  `negative_sampling: null` disables both plan and runtime negative generation.
- `ExecutionPlan` covers the paper's Event/Snapshot, sampled/full-neighbor, and
  coupled/decoupled dimensions through one planner. Persistent decoupled state
  now retains state fetch/commit semantics, and coupled Snapshot declarations
  lower to `neighbor_recurrent` dependencies.
- Plan state/cache observability now matches runtime construction: exact state
  uses owner reads, bounded state may use local/shared-hot reads, and a bound
  `PartitionPlan.shared_nodes` determines whether shared-hot placement exists.
- The sampled model input has one public spelling:
  `Batch.blocks[window][layer]`. The transitional `Batch.mfgs` field and method
  name were removed without a compatibility alias.
- Event and Snapshot train/evaluate/predict continue through one `run_epoch`
  lifecycle after their physical materialization fork.
- The current interface and execution documents were replaced with concise
  current-state contracts. A paper-method coverage matrix records implemented
  paths and remaining validation boundaries.

Attempted approaches:

- A second Event/Snapshot executor, public strategy registry, hook pipeline,
  and `reschedule` scheduler were rejected. Existing materializers feed the one
  tested blocking access scheduler.
- Strict global version-distance enforcement was not fabricated from local
  cache versions. The current bound applies to skipped shared-hot refreshes;
  an authoritative global bound needs a collective owner-version watermark.
- Sampled Snapshot was not patched with Python per-node or per-edge filtering.
  Window-local enforcement belongs in the native sampler hot path.

Abstractions introduced:

- No runtime or public abstraction was added. Existing `ExecutionPlan`,
  `PartitionPlan`, `Batch`, `StateManager`, and task contracts were tightened.
- `docs/design/current/paper_method_coverage.md` is documentation only.

Efficiency alternatives considered:

- Reused existing torch tensor compaction/scatter operations, native temporal
  sampling, Snapshot-CSC buffers, and globally ordered collectives.
- DGL does not replace owner-state commit, shared-hot synchronization, or
  Snapshot-CSC route handling. No repeated DGL block assembly was introduced.
- No custom C++/CUDA operator was added because this pass corrected lowering
  and contracts rather than identifying a profiled kernel bottleneck.
- No Python per-node, per-edge, per-neighbor, or per-state-update loop was added
  to a runtime hot path.

Files written or modified:

- `src/starrygl/{api.py,spec.py,plan.py,config_example.py}`
- `src/starrygl/batch/source.py`
- `src/starrygl/model/{_graph_ops.py,tgn.py,tgat.py,jodie.py,apan.py}`
- `src/starrygl/model/layers/temporal.py`
- `src/starrygl/native/sampling_output.py`
- `src/starrygl/runtime/{builders.py,trainer.py,trainer_options.py,train.py}`
- `src/starrygl/runtime/{access.py,batches.py,materialize.py,state_access.py}`
- `src/starrygl/runtime/{event_features.py,event_materialize.py,event_targets.py}`
- `src/starrygl/runtime/{snapshot_cache.py,snapshot_evaluation.py,snapshot_features.py,snapshot_materialize.py}`
- `src/starrygl/runtime/{memory.py,memory_filter.py,window_scan.py}`
- `src/starrygl/task/negative.py`
- `tests/{test_open_compile_plan.py,test_open_compile_semantics.py}`
- `tests/{test_runtime_materialize.py,test_runtime_memory.py}`
- `tests/{test_starrygl_core_skeleton.py,test_starrygl_model.py,test_starrygl_task.py}`
- `docs/STARRYGL_INTERFACE.md`
- `docs/design/current/{execution_spine.md,paper_method_coverage.md,runtime_default_alignment.md}`
- `docs/migration/status_log.md`

Verification:

- Compile check: `python -m compileall -q src/starrygl tests` passed.
- Focused plan, config, model, state, materialization, and Batch contract tests:
  92 passed, 1 skipped.
- Full suite: 173 passed, 2 skipped.
- Two-rank Gloo access/state suite: 10 tests passed on each rank, including
  default double-buffer collective ordering and remote memory/mailbox fetch.
- Source-size enforcement passed; all production Python files remain at or
  below 500 physical lines.
- Identity scan of changed source/tests/current docs found no author, team, or
  email disclosure.

Unresolved performance risks:

- Sampled Snapshot still needs native enforcement of selected snapshot/chunk
  boundaries before optimized parity can be claimed.
- `max_staleness=K` currently caps filtered shared-hot refresh skips. A strict
  global owner-version distance requires a scheduled version watermark.
- Representative multi-GPU end-to-end correctness, throughput, peak-memory,
  and scaling benchmarks were not run in this functional convergence pass.

## 2026-08-18: Three-Stage DataLoader Convergence

Latest state:

- Event and Snapshot now enter one loader spine:
  `WindowPlan -> TaskTarget roots -> GraphSampler -> materialize -> dependency launch`.
- Stage C emits an internal `SampledWindow`; Stage B materializes its `BatchWindow`
  and launches feature/state dependencies; Stage A awaits those dependencies,
  executes forward/loss/backward, then commits the model's `StateDelta`.
- The pipeline has two bounded handoff queues only: one sampled-window slot and
  two access slots representing the current window plus one lookahead window.
  The first sample is produced on the main thread so distributed window-count
  alignment is not initialized by the sampling thread.
- Bounded-stale state reads may launch in Stage B. Exact state reads launch in
  Stage A after the preceding window's state commit, preserving freshness.
- Sampled Snapshot lowering now prepares both Snapshot-CSC and TemporalCSR;
  Snapshot-CSC remains the window view and TemporalCSR is the native neighbor
  sampler input.
- `model.state_update()` now runs after backward and the optimizer step for a
  non-empty training batch.

Attempted approaches:

- A third materialization worker queue was not added. Materialization remains
  ordered on the consumer thread while already-launched asynchronous dependency
  handles overlap with model execution.
- MemShare stale compensation and Flare layerwise kernels were left unchanged.
- Legacy synchronous Event/Snapshot iterators remain thin compatibility calls
  over the same sampled-window implementation; no second execution path was
  retained.

Abstractions introduced:

- `SampledWindow`, `EventSample`, and `SnapshotSample` are internal data records
  that separate sampler output from materialization. They exist because both
  Event and Snapshot now use the same two queue boundaries; none is public API.

Efficiency alternatives considered:

- The bounded queue carries one coarse window descriptor, never per-node or
  per-edge work. Native temporal sampling and vectorized torch/DGL
  materialization remain the hot paths.
- A new C++/CUDA operator is not justified for coarse pipeline orchestration.
  Sampled Snapshot window-bound filtering still belongs in the native sampler,
  not a Python loop.
- Collective dependency launch remains globally ordered by `CommScheduler`;
  the sampler thread does not independently launch feature/state collectives.

Files written or modified:

- `src/starrygl/plan.py`
- `src/starrygl/runtime/{access.py,batches.py,event.py,event_materialize.py}`
- `src/starrygl/runtime/{loop.py,snapshot.py,snapshot_materialize.py}`
- `tests/{test_access_double_buffer.py,test_open_compile_semantics.py}`
- `tests/test_partition_assignment.py`
- `docs/migration/status_log.md`

Verification:

- Compile check: `python -m py_compile $(find src/starrygl -name '*.py')` passed.
- Focused loader/access/materialization/store/plan suite: 53 passed, 1 skipped.
- Full suite: 177 passed, 2 skipped.
- All modified production Python modules remain below 500 physical lines.

Unresolved performance risks:

- Sampled Snapshot still needs native enforcement of selected snapshot/chunk
  boundaries; requiring TemporalCSR fixes layout availability but does not prove
  window-local sampling parity.
- Mixed exact and bounded-stale state managers conservatively disable Stage-B
  state prefetch for the whole batch.
- The two-rank distributed access test is present but skipped outside `torchrun`;
  this pass did not rerun an end-to-end multi-GPU benchmark.

## 2026-08-18: MemShare-Style State Writeback

Latest state:

- State writeback now follows three explicit runtime boundaries:
  `finish_state_update`, `poll_state_update`, and `launch_state_update`.
  `model.state_update` still only returns `StateDelta`; models do not mutate
  memory, mailbox, or distributed stores.
- After backward and the optimizer step, owner state/mailbox all-to-all and
  shared-hot all-gather are launched from the same delta. The next batch polls
  completed shared updates and waits for owner completion only at its configured
  dependency boundary.
- State writeback remains a handle lane and does not add a third DataLoader
  queue. The sampled-window and access queues remain the only bounded handoffs.
- Shared handles now record whether communication was launched, preventing the
  same staged delta from being submitted again while its first sync is pending.
- Variable-length shared all-gather keeps the tensor-based NCCL path and uses
  `dist.all_gather` on Gloo, where `all_gather_into_tensor` is unsupported.

Attempted approaches:

- The reference training loop's explicit finish/launch organization was kept.
  Its global executor, busy-wait queue, direct model-side mailbox mutation, and
  untyped batch tuples were not copied because they conflict with the current
  `Batch`, `StateDelta`, and globally ordered communication contracts.
- A new state-update worker or queue was not introduced. Existing asynchronous
  collective handles provide the required overlap.

Abstractions introduced:

- No new class was added. Six narrow state-commit helpers were replaced by
  three lifecycle operations matching the actual execution boundaries.

Efficiency alternatives considered:

- Reused PyTorch asynchronous all-to-all/all-gather through `CommScheduler`.
  This orchestration does not justify a new DGL or C++/CUDA operator.
- Owner and shared-hot payloads are independent once `StateDelta` is built, so
  launching both before either wait exposes communication overlap without
  changing filtering, compensation, or timestamp conflict resolution.
- The Gloo list gather is a correctness/testing fallback; the NCCL hot path
  remains `all_gather_into_tensor`.

Files written or modified:

- `src/starrygl/runtime/{comm.py,loop.py,memory.py,state_access.py}`
- `tests/test_runtime_memory.py`
- `docs/design/current/execution_spine.md`
- `docs/migration/status_log.md`

Verification:

- Compile check: `python -m py_compile $(find src/starrygl -name '*.py')` passed.
- Focused state/access suite: 16 passed, 3 skipped outside `torchrun`.
- Full suite: 178 passed, 3 skipped.
- Two-rank Gloo suite: mailbox fetch, concurrent owner/shared state update, and
  double-buffer collective ordering each passed on both ranks.
- Modified production Python files remain below 500 physical lines.

Unresolved performance risks:

- Representative NCCL throughput, overlap percentage, and peak outstanding
  state-update handles have not been measured in this pass.
- Exact state intentionally waits for the authoritative owner update before a
  dependent read; only independent work and bounded-stale shared refresh can
  hide that communication.

## 2026-08-19: Single Runtime And Storage Spine

Latest state:

- Removed the second `starrygl.data` artifact API. Dataset loading, graph
  normalization, partitioning, and artifact reads now enter through
  `prepare.data` and `store.load_starrygl_store` only.
- Replaced workload-specific CLI modules with one `starrygl` command that runs
  `from_config -> prepare_artifacts -> fit -> evaluate -> predict`.
- Feature, state, mailbox, and state-writeback communication now share one
  owner-routed request/response/push protocol in `store.remote_fetch`.
- Removed dynamic feature-cache branches, snapshot-only evaluation loops,
  synchronous Event/Snapshot batch iterators, special evaluation GCN caches,
  and dead chunk-range materialization branches.
- `fit`, `evaluate`, and `predict` all call `runtime.loop.run_epoch`; the
  forwarding-only `runtime.evaluation` module was deleted.
- Bundled configs use only `data/backbone/task/runtime`. Snapshot configs now
  read converted unified directories instead of unsupported raw `.edges`
  files, and all bundled configs have a canonical compile test.
- Current user documentation and package layout no longer advertise removed
  `interface`, `executor`, `stc`, `data`, or legacy compile arguments.

Attempted approaches:

- Automatic raw `.edges` parsing was not added to `prepare`. The existing
  converter remains the single raw-data boundary and emits `graph.pt` plus
  feature/label sidecars; duplicating its snapshot split logic in runtime
  loading would recreate two preprocessing paths.
- Separate feature, memory, and mailbox communication implementations were
  deleted in favor of the existing collective scheduler and one owner route.
- Snapshot evaluation was not preserved as a specialized fast path because it
  bypassed `StateManager`, `StateDelta`, and the common DataLoader lifecycle.

Abstractions introduced:

- `OwnerRequest` and `OwnerPush` are internal records for the one collective
  owner protocol. They are needed by feature, state, and mailbox call sites and
  are not public API.
- No compatibility wrapper, registry, cache framework, or second execution
  plan was introduced.

Efficiency alternatives considered:

- Owner routing uses vectorized torch sorting, bincount, gather/scatter, and
  scheduled all-to-all. No Python per-node or per-message loop was added.
- Native TemporalCSR sampling, compressed Snapshot-CSC storage, rolling
  snapshot prefix caching, and asynchronous collective handles remain the hot
  paths; this orchestration pass did not justify a new DGL or C++/CUDA kernel.
- Deleting the duplicate evaluation materializer avoids repeated Python-side
  graph assembly and keeps one bounded double-buffer pipeline.

Files written or modified:

- `src/starrygl/{api.py,cli/main.py,prepare/data.py}`
- `src/starrygl/runtime/{batches.py,event.py,exchange.py,loop.py,memory_access.py}`
- `src/starrygl/runtime/{model_exec.py,snapshot.py,snapshot_cache.py}`
- `src/starrygl/runtime/{snapshot_materialize.py,snapshot_rows.py,train.py}`
- `src/starrygl/store/{artifact.py,remote_fetch.py,state.py,mailbox.py}`
- `configs/*.json`, `README.md`, `docs/OPEN_SOURCE_LAYOUT.md`, and
  `src/starrygl/tools/README.md`
- `tests/{test_open_compile_semantics.py,test_open_public_api.py}` and
  `tests/{test_runtime_exchange.py,test_runtime_memory.py,test_starrygl_store.py}`
- Removed legacy `src/starrygl/data/`, workload-specific CLI modules,
  `runtime/feature_cache.py`, `runtime/evaluation.py`,
  `runtime/snapshot_evaluation.py`, and the stale external compile example.

Verification:

- Compile check: `python -m compileall -q src/starrygl tests` passed.
- Full suite: 177 passed, 3 skipped.
- Focused canonical config, remote exchange, and memory suite: 36 passed,
  2 skipped outside `torchrun`.
- Two-rank Gloo remote feature/state/mailbox suite: 12 passed on each rank.
- All production Python modules remain at or below 500 physical lines; the
  largest is 497 lines.
- Identity/secret scan found no project author, team, email, credential, or
  local path disclosure. Vendored dependency copyright/author notices remain
  intact as required by their licenses.

Unresolved performance risks:

- Sampled Snapshot still needs native snapshot-window boundary enforcement
  before optimized window-local parity can be claimed.
- Strict global owner-version staleness needs a scheduled version watermark;
  current `max_staleness` bounds filtered shared-hot refresh skips.
- Representative NCCL throughput, overlap, peak memory, and multi-GPU scaling
  benchmarks remain to be run.

## 2026-08-19: Runtime Responsibility Convergence

Latest state:

- Grouped physical paths under `runtime/event/`, `runtime/snapshot/`, and
  `runtime/sample/`; no old module-path re-exports remain.
- Moved the bounded two-slot Fetch/Await scheduler to
  `runtime/sample/pipeline.py`, beside the loader that owns it.
- Grouped generic hydrate/update/recurrent behavior under `runtime/state/` and
  concrete node-memory/mailbox/shared-hot behavior under `runtime/memory/`.
- Removed the duplicated `Batch.targets["runtime"]` payload. `TaskTarget` now
  carries `state_write_mask` and `snapshot_id` directly.
- Removed the forwarding-only model execution module; Snapshot scan owns its
  decoupled recurrent dispatch.
- Merged the small shared refresh filter into the memory committer module.

Attempted approaches:

- Event and Snapshot target materializers were not merged: they consume
  different physical rows, while their common semantic result is already one
  `TaskTarget`.
- Memory local ops, shared collective sync, and historical reads were not
  collapsed into one large module. They call each other but implement distinct
  read/commit policies and combining them would obscure those boundaries.
- No compatibility aliases for old runtime module paths were retained.

Abstractions introduced:

- No public abstraction was added. The new directories only express existing
  physical responsibilities.
- `SharedStateRefreshFilter` names runtime cache-refresh admission;
  `MemoryIncrementEstimator` remains the separate model-side learnable-gamma
  compensation operation.

Efficiency alternatives considered:

- This migration preserves existing vectorized torch/DGL/native paths and
  scheduled collectives; it adds no Python per-node, per-edge, or per-message
  loop to a hot path.
- The work is ownership and deletion rather than a new kernel, so neither a DGL
  kernel mirror nor a custom C++/CUDA operator is justified.
- Layerwise communication remains in `runtime/snapshot/layerwise.py` and
  bounded-stale epoch caching remains in `runtime/memory/historical.py`.

Files written, moved, or removed:

- Moved Event, Snapshot, and Sample implementations into their runtime
  subpackages, including `sample/pipeline.py`.
- Moved state hydration/recurrent/build code into `runtime/state/` and memory
  access/filter/ops/shared/historical code into `runtime/memory/`.
- Updated `task/target.py`, Event/Snapshot materializers, temporal models,
  runtime tests, and current design documentation.
- Removed `runtime/targets.py`, `runtime/model_exec.py`, and the standalone
  memory filter module.

Verification:

- Compile check and focused runtime/model/API suites passed (78 passed,
  2 skipped outside distributed launch).
- Full suite: 177 passed, 3 skipped.
- Two-rank Gloo memory/exchange suite: 12 passed on each rank.
- Every production Python module remains below 500 physical lines.
- Identity scan found only required vendored dependency copyright notices; no
  project author, team, email, credential, or local path was exposed.

Unresolved performance risks:

- Sampled Snapshot still needs native snapshot-window boundary enforcement.
- Representative NCCL overlap and scaling benchmarks remain outstanding; this
  pass deliberately changed ownership and metadata only.

## 2026-08-19: Remove Trainer Support Grab Bag

Latest state:

- Deleted `runtime/trainer_support.py` and its unused `_preserve_state` path.
- Moved chunk-decay and output-dimension config lowering into
  `trainer_options.py`, epoch reset into `train.py`, and trainer-owned prepare,
  result, and distributed lifecycle helpers into `trainer.py`.
- Reused `Trainer._artifacts_ready()` for distributed artifact validation,
  removing the second artifact-world check.
- Distributed user entry remains `Trainer.fit_distributed()` or
  `Trainer.eval_distributed()`; `_initialize_distributed()` now lives beside
  those methods in `runtime/trainer.py`.

Attempted approaches:

- A separate distributed helper module was not created because the lifecycle
  has one owner and `comm.py` is already at the module size limit.
- The first direct `torchrun` smoke lacked an editable install and failed at
  package import; rerunning with the repository `src` path exercised the same
  initialization successfully.

Abstractions introduced:

- None. Existing helpers were moved to their sole callers and one dead helper
  was deleted.

Efficiency alternatives considered:

- No data, sampling, communication, or model hot path changed. This is static
  ownership cleanup and does not justify torch/DGL or custom C++/CUDA work.

Files written or removed:

- `src/starrygl/runtime/{trainer.py,trainer_options.py,train.py}`
- `tests/test_open_compile_semantics.py`
- `docs/design/current/execution_spine.md`
- Removed `src/starrygl/runtime/trainer_support.py`

Verification:

- Compile check passed.
- Full suite: 178 passed, 3 skipped.
- Direct two-rank Gloo initialization smoke passed with ranks 0 and 1.
- All production Python modules remain below 500 physical lines.

Unresolved risks:

- `starrygl.cli.main` still runs the local `fit/evaluate/predict` sequence and
  does not automatically select distributed Trainer methods under `torchrun`.

## 2026-08-20: Centralize Distributed Environment Context

Latest state:

- `utils.DistributedContext` is now the sole owner of process-group
  initialization, rank/world/local-rank identity, device binding, barrier, and
  shutdown.
- `Trainer` retains only distributed workflow responsibilities: rank-zero
  artifact preparation and rank-local store/model execution.
- Removed `_initialize_distributed`, `_resolve_dist_device`, `_dist_barrier`,
  and `_shutdown_dist` from `runtime/trainer.py`.
- Preserved the existing default-context, CPU/memory groups, host/rank topology,
  hybrid subgroup, CUDA stream, and scalar-reduction helpers.

Attempted approaches:

- A second distributed utility was not created; the existing context was
  extended for Trainer reuse.
- An initial 56-line reduction removed context capabilities solely because they
  had no current caller. That was too aggressive for this established utility
  surface; those capabilities were restored while Trainer-only duplicate
  helpers stayed deleted.

Abstraction introduced:

- No new abstraction. `DistributedContext` already existed; its surface now
  matches the one process lifecycle used by Trainer.

Efficiency alternatives considered:

- This changes initialization ownership only. Sampling, tensor packing,
  collective scheduling, and model kernels are unchanged, so no torch/DGL or
  native kernel alternative applies.

Files modified:

- `src/starrygl/utils/context.py`
- `src/starrygl/runtime/trainer.py`
- `tests/test_open_compile_semantics.py`
- Current layout, execution-spine, and migration documentation

Verification:

- Focused API/store/partition suite: 49 passed.
- Full suite: 178 passed, 3 skipped.
- Direct two-rank Gloo context initialization, CPU/memory-group creation,
  host/rank discovery, barrier, and shutdown passed.
- All production Python modules remain below 500 physical lines.

Unresolved risk:

- CLI automatic selection of distributed Trainer methods under `torchrun`
  remains a separate entry-policy decision.

## 2026-08-20: Separate Builder Policy Lowering

Latest state:

- `edge_prediction.loss` is now passed to `EdgePredictionTask` instead of
  being silently replaced by the task default.
- Model-side stale compensation remains controlled only by
  `temporal_state.smooth_aggregation`; runtime shared-refresh filtering remains
  controlled independently by `temporal_state.filter`.
- Removed the unused local `filter_cfg` that made the two policies appear to
  share model lowering.

Abstraction introduced:

- None. The existing direct builders remain the single construction path.

Efficiency alternatives considered:

- This only changes construction-time option lowering. It does not touch batch,
  sampling, communication, or model hot paths, so no torch/DGL/native kernel
  alternative applies.

Files modified:

- `src/starrygl/runtime/builders.py`
- `tests/test_runtime_builders.py`

Verification:

- Builder/task focused suite: 26 passed.
- Full suite: 179 passed, 3 skipped.

Unresolved risk:

- Model constructor names `memory_filter`, `filter_num_rows`, and
  `historical_mix` still describe the smooth-compensation implementation with
  legacy terminology. Public runtime config is canonical, but these internal
  names should only be renamed together with their model metadata consumers.
- `smooth_aggregation` currently lowers only into TGN/JODIE/APAN memory
  updaters. Coupled DTDG `neighbor_recurrent` reads have bounded-stale cache
  access but do not yet apply the same learnable historical-increment
  compensation.

## 2026-08-21: Add The Coupled GConvGRU State Path

Latest state:

- Replaced the incomplete DCRNN-specific model surface with one single-layer
  `GConvGRUModel`: graph convolution consumes current features and hydrated
  neighbor recurrent state, then a local GRU updates destination state.
- Snapshot recurrent semantics remain `snapshot_recurrent`; the plan now stores
  coupling only as `coupled` or `decoupled` and no longer emits
  `window_coupled`, `decoupled_recurrent`, or a duplicate `coupling_mode`.
- `gconv_gru` lowers to `neighbor_recurrent`, whose shape is declared by the
  model and whose placement remains owned by `PartitionPlan` and
  `StateManager`.
- Owner state and mailbox fetches now carry their existing version tensors.
  Loader hydration exposes kind-prefixed historical value, timestamp, and
  version fields for both `node_memory` and `neighbor_recurrent` shared-hot
  reads.
- Shared-hot candidates remain approximate timestamp-ordered cache entries;
  they do not claim an owner-authoritative version. Exact reads continue to use
  the owner plane.

Attempted approaches:

- Extending the diffusion-specific DCRNN implementation was rejected because
  it retained model-local freshness/delay controls and substantially more code
  than the required coupled-state contract.
- A second compensation payload or state request class was not added. Existing
  `StateRead` and `StateDelta` already provide the required read/write boundary.

Abstraction introduced:

- `GConvGRUCell` is the concrete model cell needed by the existing coupled
  window scan. No registry, adapter, or new state abstraction was introduced.

Efficiency alternatives considered:

- The model reuses the existing vectorized `GCNConv` path and PyTorch
  `GRUCell`; DGL/sparse fast paths remain available through `GraphBlock` cache
  policy. A custom C++/CUDA operator is not justified before profiling these
  installed operators.
- State gathering and version transfer reuse the scheduled collective fetch
  path. Packing mixed-dtype value/timestamp/version fields into a custom native
  message was not added without evidence that the small version transfer is a
  bottleneck.

Files written, modified, or removed:

- Added `src/starrygl/model/gconv_gru.py`; removed
  `src/starrygl/model/dcrnn.py` and its public exports.
- Updated model construction, semantic lowering, execution planning, state
  construction/hydration, the bundled Snapshot config, focused tests, and
  current design documentation.

Verification:

- Compile check passed for all changed production modules.
- Focused model/plan/state suite: 90 passed, 2 skipped.
- Full suite: 179 passed, 3 skipped.
- Two-rank Gloo memory/exchange suite: 12 passed on each rank.
- Every production Python module remains below 500 physical lines.

Unresolved performance risks:

- Propagating owner versions adds small integer collective payloads; a
  representative NCCL profile is still needed before deciding whether payload
  fusion is worthwhile.
- `neighbor_recurrent` now carries every input required for reusable historical
  compensation, but this step deliberately does not duplicate the TGN
  learnable compensation algorithm inside GConvGRU.

## 2026-08-21: Compensate Coupled State Before Graph Convolution

Latest state:

- Bounded state materialization now records which rows were actually loaded
  from shared-hot cache. Owner-local hot rows are no longer inferred stale from
  cache membership alone.
- GConvGRU replaces only those shared `neighbor_recurrent` rows with
  `historical + sigmoid(gamma) * estimated_increment` before `GCNConv`.
- The learnable gamma exists only for bounded-stale execution. Bounded ranks
  with no shared row retain a zero-gradient dependency so DDP observes the same
  parameter-use graph on every rank.
- The latest destination recurrent state still enters `GRUCell` unchanged.
  Increment statistics are updated from produced destination state during
  `state_update`, after backward in the normal runtime lifecycle.
- Renamed the existing model-side estimator to `StateIncrementEstimator` and
  reused it in TGN and GConvGRU; no second compensation implementation was
  introduced.

Abstraction introduced:

- None. The existing increment estimator and `StateRead.metadata` carry the two
  additional internal source tensors required by real call sites.

Efficiency alternatives considered:

- Shared-source tracking is generated while the existing vectorized bounded
  gather fills result rows. Compensation uses tensor selection and
  `index_copy_`; it adds no per-node Python loop or communication.
- No DGL or custom C++/CUDA operator is needed because the new work is a small
  dense elementwise update before the existing `GCNConv` hot path.

Files modified:

- `src/starrygl/model/gconv_gru.py`
- `src/starrygl/model/layers/{temporal.py,__init__.py}`
- `src/starrygl/runtime/{builders.py,memory/historical.py,state/access.py}`
- Focused runtime/model tests and current design documentation

Verification:

- Compile check passed for changed production modules.
- Focused model/state/builder suite: 57 passed, 2 skipped.
- Full suite: 182 passed, 3 skipped.
- Two-rank Gloo memory/exchange suite: 13 passed on each rank.
- Every production Python module remains below 500 physical lines.

Unresolved performance risks:

- The increment estimator is intentionally rank-local, matching the
  decentralized shared-hot cache plane. Representative NCCL profiling is still
  required to measure the complete bounded-stale path.

## 2026-08-21: Replace Access Tickets With A Ready-Batch Pipeline

Latest state:

- Training now uses two bounded queues: Stage C produces `SampledWindow`
  values, while Stage B materializes them, launches dependencies, awaits exact
  features and permitted bounded-stale state, and produces ready
  `BatchWindow` values.
- `AccessScheduler`, `AccessWindow`, and `AccessTicket` were removed. The
  training loop consumes `Batch` directly and no longer duplicates queue state
  in per-batch lifecycle flags.
- Exact temporal state remains in the execution thread after the previous
  state commit. Bounded-stale state may be completed by Stage B.
- With `access_pipeline=true`, deferred Event/Snapshot feature launch and
  deferred node-feature completion become internal defaults. Explicit sampler
  overrides still disable either optimization for diagnosis.
- The Stage-B slot handshake admits only one additional materialized/read
  window and confirms its collective launches before current computation
  proceeds. This preserves the global `CommScheduler` order without exposing a
  ticket to the trainer.

Attempted approaches:

- A plain background `map` feeding a size-one queue was rejected because the
  producer can begin a third window before blocking on `put`, violating the
  intended double-buffer bound.
- Letting Stage A state updates and Stage B reads freely race on
  `CommScheduler` was rejected because thread arrival order can differ by rank.
  The internal slot/launch handshake keeps the established batch-major
  collective order.
- Retaining `AccessTicket` solely as a future container was unnecessary once
  Stage B became the sole owner of feature and stale-state pending handles.

Abstraction introduced:

- No public abstraction. `_prefetch_ready` is the single internal Stage-B
  producer needed to coordinate the existing `SampledWindow` and `BatchWindow`
  contracts.

Efficiency alternatives considered:

- Sampling, compaction, feature exchange, state hydration, and state
  compensation continue to use the existing vectorized torch/native and
  scheduled collective implementations. No DGL or C++/CUDA operator change is
  justified for Python queue orchestration.
- Snapshot rolling row/blob materialization remains enabled when feature launch
  is deferred; mutable feature-bearing graph entries are not shared across
  concurrent windows.

Files modified:

- `src/starrygl/runtime/{loop.py,sample/loader.py,sample/pipeline.py}`
- `src/starrygl/runtime/snapshot/__init__.py`
- `tests/{test_access_double_buffer.py,test_runtime_access.py}`
- `docs/STARRYGL_INTERFACE.md`
- `docs/design/current/{execution_spine.md,state_await_cache.md}`
- `docs/migration/status_log.md`

Verification:

- Changed-module compile check passed.
- Full unit suite: 182 passed, 3 skipped.
- Two-rank Gloo queue/collective suite: 9 passed on each rank.
- `loop.py`, `sample/loader.py`, and `sample/pipeline.py` remain below 500 lines
  and total 790 lines, down from 934 before this migration.

Unresolved performance risks:

- The new queue establishes real CPU/communication overlap, but representative
  NCCL Event and Snapshot profiles are still required to quantify throughput
  and host-thread overhead.
- Stage B uses one Python worker. Native sampling and tensor operations release
  the GIL where applicable, but Python-heavy materialization would limit overlap
  until that work is vectorized or moved into the existing native path.

## 2026-08-21: Remove The Singleton BatchWindow Layer

Latest state:

- `SampledWindow.materialize()` now returns one model-facing `Batch` directly.
- Stage B launches and finishes dependencies for that `Batch`, then places it
  on the ready queue. `run_epoch()` consumes the queue with one `for batch`
  loop.
- `BatchWindow` and its public exports were removed. Snapshot history remains
  represented by `Batch.blocks[window][layer]`; no temporal information moved
  into the queue layer.

Attempted approaches:

- Retaining `BatchWindow` as a future multi-batch container was rejected: every
  producer constructed exactly one element, and no execution path consumed its
  separate context or metadata.
- Flattening only inside `run_epoch()` was rejected because it would preserve
  the unused wrapper and duplicate scheduling metadata upstream.

Abstraction introduced:

- None. This change removes a scheduling container and reuses `Batch.window`
  and `Batch.meta`, which already carry the required execution context.

Efficiency alternatives considered:

- The queue, tensor materialization, native sampling, feature/state fetches,
  and collective schedule are unchanged. Removing the wrapper avoids one
  allocation plus Python iteration per batch; no DGL or C++/CUDA change is
  justified for this control-path cleanup.

Files modified:

- `src/starrygl/{__init__.py,batch/,runtime/loop.py}`
- `src/starrygl/runtime/{event,snapshot,sample/}`
- Focused batch/materialization/pipeline tests and current design documents

Verification:

- Changed-module compile check and full-package `compileall` passed.
- Focused batch/public API/pipeline suite: 23 passed, 1 skipped.
- Full unit suite: 181 passed, 3 skipped.
- Two-rank Gloo ready-queue suite: 9 passed on each rank.
- No current source, test, or current design document references
  `BatchWindow`; every production Python module remains below 500 lines.

Unresolved performance risks:

- Representative NCCL Event and Snapshot profiling is still required to
  quantify overlap; this structural deletion does not alter communication or
  sampling kernels.

## 2026-08-21: Separate Dataloader From Sampling

Latest state:

- `runtime/sample/` now contains only native temporal sampling and sampled-MFG
  normalization.
- Batch request enumeration, Event/Snapshot `GraphBlock` construction, common
  materialization, feature/state dependency handling, device movement, and the
  two bounded queues now live under `runtime/dataloader/`.
- The old `SampledWindow` name and wrapper are gone. Stage C places a callable
  `BatchRequest` on the request queue; Stage B invokes it once to construct a
  `Batch`.
- `runtime.sample.loader`, `runtime.sample.pipeline`, and
  `runtime.sample.materialize` no longer exist or import.

Attempted approaches:

- Moving only `loader.py` was rejected because pipeline and common
  materialization would still leave dataloader behavior under `sample/`.
- Moving all block helpers was refined: sampled-MFG normalization remains in
  `sample/blocks.py`, while Event/Snapshot graph construction and device
  movement belong to `dataloader/blocks.py`.
- A one-field `BatchRequest` dataclass was removed in favor of its callable
  type, avoiding another queue wrapper.

Abstraction introduced:

- No runtime object. `BatchRequest` is only a callable type alias documenting
  the Stage-C-to-Stage-B handoff.

Efficiency alternatives considered:

- This is a module ownership change; native sampling, vectorized GraphBlock
  construction, scheduled collectives, and queue depths are unchanged.
- Direct callables remove one Python object allocation per batch request. No
  DGL or C++/CUDA work is justified for this control-plane migration.

Files modified:

- Added `src/starrygl/runtime/dataloader/` with `loader.py`, `pipeline.py`,
  `materialize.py`, and `blocks.py`.
- Reduced `src/starrygl/runtime/sample/` to sampler code and sampled-block
  normalization.
- Updated Event, Snapshot, trainer runtime, focused tests, and current design
  documents to use the new ownership boundary.

Verification:

- Full-package `compileall` passed.
- Focused dataloader/layout suite: 45 passed, 1 skipped.
- Full unit suite: 181 passed, 3 skipped.
- Two-rank Gloo ready-queue suite: 9 passed on each rank.
- New dataloader imports resolve; all three removed `runtime.sample` module
  paths fail resolution as intended.
- Every production module in both directories remains below 500 lines.

Unresolved performance risks:

- No new runtime risk was introduced. Representative NCCL Event/Snapshot
  profiling remains necessary to quantify pipeline overlap.

## 2026-08-21: Unify TaskTarget Construction

Latest state:

- Event and Snapshot targets now enter through `build_task_target`.
- Neighbor sampling constructs the semantic target before sampling, derives
  roots from it, and reuses the same target during materialization.
- Full Snapshot defers target construction until materialization because its
  graph path does not consume roots. Physical target rows/routes are attached
  only after the `GraphBlock` exists.
- Event materialization now enriches its existing immutable target instead of
  constructing a second target.

Attempted approaches:

- Building every Snapshot target in Stage C was rejected because Full Snapshot
  would scan labels/endpoints before graph materialization without using roots.
- Keeping separate Event/Snapshot target constructors on the runtime hot path
  was removed; the existing `TaskTarget` data contract is sufficient.

Abstraction introduced:

- `build_task_target` is the single semantic constructor required by the real
  Event and Snapshot call sites. No registry, adapter, or target stage was added.

Efficiency alternatives considered:

- Root and route construction remains vectorized with torch tensor operators.
  A DGL or C++/CUDA operator is not justified for the one-per-batch dataclass
  construction; Full Snapshot avoids that work in the sampling stage entirely.
- Snapshot CSC endpoints are read directly from tensor buffers instead of
  constructing a temporary `GraphBlock` solely for target extraction.

Files modified:

- `src/starrygl/task/{__init__.py,target.py}`
- `src/starrygl/runtime/{event/materialize.py,snapshot/materialize.py,snapshot/target.py}`
- `tests/test_runtime_targets.py`
- `docs/{STARRYGL_INTERFACE.md,design/current/execution_spine.md,migration/status_log.md}`

Verification:

- Changed-module compile check passed.
- Focused target/materialization/queue suite: 20 passed, 1 skipped.
- Full unit suite: 184 passed, 3 skipped.
- Two-rank Gloo ready-queue suite: 9 passed on each rank.
- All modified production Python modules remain below 500 physical lines.

Unresolved performance risks:

- The construction timing is covered by unit tests, but representative
  Snapshot neighbor/full profiles are still needed to quantify target and
  negative-root overhead on production graph sizes.

## 2026-08-24: Unify Snapshot Materialization Path

Latest state:

- Every Snapshot plan now produces `SnapshotSample`, yields the same
  `_materialize_snapshot_sample` `BatchRequest`, and finishes through
  `materialize_snapshot_sample` plus `_batch_from_snapshot_entries`.
- The separate `_materialize_rolling_sample` and
  `_materialize_rolling_snapshot_window_batch` control paths were removed.
- Full Snapshot still constructs `TaskTarget` lazily; neighbor Snapshot still
  constructs it before sampling and reuses it during materialization.
- Rolling full-entry and chunk-prefix reuse remains available through the
  `_SnapshotEntry`/`_SnapshotGraphBlob` cache inside entry materialization.

Attempted approaches:

- Removing rolling cache entirely was rejected because repeated windows would
  rebuild Snapshot-CSC graph buffers and reread unchanged features.
- Keeping rolling as a separate `BatchRequest` was rejected because a physical
  cache policy should not bypass the common Snapshot execution spine.
- Full Snapshot rows now remain uncut in Stage C and are limited once during
  Stage B materialization; neighbor rows are limited before sampling as
  required by sampler semantics.

Abstraction introduced:

- `_materialize_snapshot_entries` is the single internal entry materializer
  for cached and uncached full Snapshot reads. It is needed now to keep cache
  ownership in `snapshot/cache.py` while deleting the alternate batch path.

Efficiency alternatives considered:

- Snapshot graph construction, feature gathers, native neighbor sampling, and
  scheduled communication remain unchanged. No new Python node/edge loop or
  DGL/C++/CUDA operator was introduced.
- Cache-on and cache-off use the same Batch path. A focused test confirms a
  repeated full snapshot reuses one physical `_SnapshotEntry`.

Files modified:

- `src/starrygl/runtime/snapshot/{__init__.py,cache.py,materialize.py}`
- `tests/test_runtime_targets.py`
- `docs/{STARRYGL_INTERFACE.md,design/current/execution_spine.md,migration/status_log.md}`

Verification:

- Full-package compile check passed.
- Focused target/cache suite: 11 passed.
- Snapshot/Store/Plan suite: 51 passed.
- Full unit suite: 186 passed, 3 skipped.
- Two-rank Gloo ready-queue suite: 9 passed on each rank.
- All Snapshot production modules remain below 500 physical lines.

Unresolved performance risks:

- Representative NCCL Full/Neighbor Snapshot profiles are still required to
  verify that moving cache lookup under the common materializer has no
  measurable host overhead and preserves rolling throughput.

## 2026-08-24: Add Paper Experiment Contracts And Coupled DTDG Case

Latest state:

- The Event/Snapshot execution document now identifies the exact Stage-C
  sampling, Stage-B materialization/fetch, and shared ready-Batch handoff.
- Four runnable paper-oriented configurations cover coupled/decoupled Event and
  full-Snapshot paths. The added GConvGRU case consumes
  `neighbor_recurrent` before graph convolution and enables bounded-stale
  shared-hot compensation.
- A parameterized test generates every model/dataset/task combination stated
  in the supplied experiments and locks the split, event batch, view,
  coupling, state-kind, and 4-snapshot `auto:0.1` chunk-decay semantics.
- The canonical CLI now detects a `torchrun` world and uses the existing
  distributed Trainer methods. A thin launcher exposes single- or multi-node
  torchrun environment settings without adding another runtime entry path.

Attempted approaches:

- Copying one JSON file for every paper Cartesian-product row was rejected as
  duplicated configuration. The tests generate those semantic combinations;
  JSON files are retained only for four representative runnable paths.
- Inventing missing per-workload epochs, learning rates, and Snapshot hidden
  dimensions was rejected. The supplied paper text does not state them, so the
  configs inherit current public defaults.
- The new DTDG coupled case cannot be attributed to `method.tex`, which is
  empty. It uses the coupling definition in `introduction.tex`, the runtime
  settings, and the already tested GConvGRU `neighbor_recurrent` contract.

Abstraction introduced:

- None. The launcher calls `starrygl.cli.main`; the CLI selects existing
  `fit_distributed`/`eval_distributed` methods from the torchrun environment.

Efficiency alternatives considered:

- No sampling, graph materialization, communication, feature gather, state
  update, or model hot-path code changed. PyTorch/native/DGL/C++/CUDA choices
  are therefore unchanged in this step.
- Parameterized compile tests replace duplicated config files and perform no
  per-node or per-edge Python work in the runtime.

Files modified:

- `src/starrygl/cli/main.py`
- `configs/paper/*.json`
- `scripts/run_paper_experiment.sh`
- `tests/test_{paper_experiment_contracts,cli_main}.py`
- `docs/design/current/{execution_spine,paper_method_coverage}.md`
- `docs/migration/status_log.md`

Verification:

- Changed-tree compile check passed.
- Paper/CLI focused suite: 44 passed.
- Full unit suite: 230 passed, 3 skipped.
- Two-rank Gloo queue/state suite: 18 passed on each rank.
- `bash -n scripts/run_paper_experiment.sh` passed.
- The production module line-limit test passed as part of the full suite.

Unresolved performance risks:

- No real eight-dataset training or NCCL throughput run was launched. The new
  configurations require prepared datasets and representative 4/8/12/16-GPU
  runs before paper-level quality or scaling claims are valid.
- The paper source still needs authoritative per-workload epochs, learning
  rates, Snapshot hidden dimensions, and a non-empty method description.
- GConvGRU bounded-stale coupled Snapshot execution still needs distributed
  quality and timing comparison against its exact-state alternative.

## 2026-08-24: Unify DataLoader Feature Launch

Latest state:

- Event full/neighbor and Snapshot full/neighbor batches now use one Stage-B
  `launch_batch_features` entry and one `_deferred_feature_launch` handoff.
- Mode-specific deferred callbacks and config names were removed. Event and
  Snapshot retain only their necessary physical node-feature readers.
- Node and edge launch/await metadata is shared. Empty neighbor-padding ranks no
  longer skip feature launch and enter the same scheduled collective epochs.
- Sampled Snapshot blocks carry their source snapshot id before feature access,
  so temporal feature selection follows the selected window.

Attempted approaches:

- A loader adapter hierarchy was rejected because `Batch.mode`, `GraphBlock`,
  and the existing physical readers already contain the required distinction.
- Keeping two callbacks behind a generic dispatcher was rejected because it
  retained duplicate deferred metadata and edge-fetch orchestration.
- The distributed empty-rank test exposed a shared exchange bug: empty remote
  requests used an unshaped output dictionary. `_launch` now lets the existing
  `_empty` path allocate correctly shaped outputs.

Abstraction introduced:

- `runtime/dataloader/features.py` owns the one feature dependency launch and
  shared block-edge gather. It is required by all four real Batch paths; no
  public API, adapter, registry, or additional queue was added.

Efficiency alternatives considered:

- Node/edge gathers and scatter remain vectorized torch operations; distributed
  reads continue through the global `CommScheduler` all-to-all protocol.
- Rebuilding DGL blocks was rejected because direct GraphBlock tensor buffers
  already expose feature ids. C++/CUDA work is not justified for the remaining
  per-window/per-layer control loop; there is no Python node or edge loop.
- Existing Event/Snapshot sampling, chunk decay, rolling Snapshot cache,
  historical compensation, and layerwise kernels were not changed.

Files modified:

- `src/starrygl/runtime/dataloader/{features.py,loader.py,pipeline.py}`
- `src/starrygl/runtime/{loop.py,exchange.py,trainer_options.py}`
- `src/starrygl/runtime/event/{features.py,materialize.py}`
- `src/starrygl/runtime/snapshot/{__init__.py,cache.py,features.py,materialize.py}`
- `tests/test_{access_double_buffer,dataloader_features}.py`
- `docs/{STARRYGL_INTERFACE.md,design/current/execution_spine.md,migration/status_log.md}`

Verification:

- Changed-module compile check passed.
- Four-path feature test: 4 passed, 1 distributed test skipped in one process.
- Full unit suite: 234 passed, 4 skipped.
- Two-rank Gloo feature test: 5 passed on each rank, including a fully empty
  rank participating in a non-empty peer's feature request/response epoch.
- Two-rank Gloo queue/feature suite: 14 passed on each rank.

Unresolved performance risks:

- Representative NCCL profiling is still required to compare overlap and host
  launch overhead before and after this control-path consolidation.
- Temporal `[S, N, F]` remote reads outside embedded Snapshot-CSC layout still
  require a dedicated distributed correctness and throughput audit.

## 2026-08-25: Unify DatasetBatch Sampling And Materialization

Latest state:

- Event and Snapshot window modules now emit `DatasetBatch` values only.
- `runtime/dataloader/loader.py` owns the shared `RuntimeBatchPlan ->
  DatasetBatch -> sample -> BatchRequest -> materialize -> Batch` lifecycle.
- Existing Event/TCSR and Snapshot-CSC sampling/materialization kernels remain
  the only physical specializations. Snapshot rolling caches remain local to
  Snapshot materialization.
- Event distributed batch-count and empty-padding metadata is merged into the
  materialized `Batch.meta`, preserving collective alignment decisions.
- Full Snapshot no longer touches TemporalCSR unless neighbor sampling is
  selected.

Attempted approaches:

- A sampler/materializer registry or mode adapter hierarchy was rejected; two
  direct mode branches around the existing physical kernels are sufficient.
- Moving full-Snapshot target construction into Stage C was rejected because
  it adds work without supplying sampling roots. Its target remains deferred to
  materialization.
- Combining Event and Snapshot physical row formats was rejected; the common
  lifecycle ends at the largest stable interface, `DatasetBatch` and `Batch`.

Abstraction introduced:

- None. Existing `DatasetBatch`, `RuntimeBatchPlan`, and callable
  `BatchRequest` contracts are reused. The old mode-owned request generators
  were deleted.

Efficiency alternatives considered:

- Native temporal sampling, vectorized Snapshot-CSC slicing, feature gathers,
  rolling caches, and scheduled all-to-all communication are unchanged.
- The shared path adds no Python node/edge loop and no DGL graph rebuild.
  C++/CUDA changes are not justified for this control-flow-only merge.

Files modified:

- `src/starrygl/runtime/{event,snapshot}/__init__.py`
- `src/starrygl/runtime/dataloader/loader.py`
- `tests/test_runtime_targets.py`
- `docs/{STARRYGL_INTERFACE.md,design/current/execution_spine.md,migration/status_log.md}`

Verification:

- Focused target/lifecycle suite: 12 passed.
- Full unit suite: 235 passed, 4 skipped.
- Two-rank Gloo queue/feature suite: 14 passed on each rank.
- Full-package compile check and changed-file Pyflakes check passed.
- Event, Snapshot, and DataLoader modules remain below 500 physical lines.

Unresolved performance risks:

- Representative NCCL Event-neighbor and Snapshot full/neighbor profiles are
  still required to measure Stage-C/Stage-B host overlap after consolidation.
- Sampled Snapshot still reuses the native temporal sampler substrate; native
  enforcement of selected Snapshot/chunk boundaries remains a separate audit.

## 2026-08-27: Remove Edge Label Prediction Task

Latest state:

- `edge_label_prediction` is no longer a public task, builder target, or edge
  ownership alias. Task normalization accepts only the four implemented task
  names: `edge_prediction`, `node_prediction`, `node_classification`, and
  `node_regression`.
- Snapshot edge-label behavior was removed instead of adding a second target
  and window interpretation outside the current paper-critical paths.

Attempted approaches:

- Special-casing `edge_label_prediction` in Snapshot runtime was rejected; it
  would preserve a task whose temporal supervision contract is not defined.
- Leaving the task class available but failing during execution was rejected;
  invalid task names now fail at semantic normalization.

Abstraction introduced:

- None. Existing task normalization is the single public validation boundary.

Efficiency alternatives considered:

- This is control-path deletion and does not change sampling, materialization,
  communication, or model hot paths. No torch/DGL or native kernel alternative
  is applicable.

Files modified:

- `src/starrygl/{__init__.py,plan.py,spec.py}`
- `src/starrygl/task/{__init__.py,prediction.py}`
- `src/starrygl/runtime/builders.py`
- `tests/{test_runtime_builders.py,test_starrygl_task.py}`
- `docs/migration/status_log.md`

Verification:

- Changed-module compile check passed.
- Task, builder, public API, and compile semantic suite: 57 passed.
- Full unit suite: 234 passed, 4 skipped.

Unresolved performance risks:

- None; no runtime hot path changed.

## 2026-08-27: Enforce Paper Task Ownership Contract

Latest state:

- Task ownership is derived only from the canonical task name. Public config,
  typed `TaskSegment`, and `Trainer.to_config()` no longer expose an ownership
  choice. Runtime task construction preserves `node_classification` and
  `node_regression` instead of collapsing both names to `node_prediction`.
- Event edge targets remain physically selected by the edge-master EventView.
  Snapshot-CSC continues to partition compute edges by destination
  `node_master`, but each slice now carries an independent edge-target surface
  selected by `edge_master`.
- Snapshot edge targets preserve logical edge ids and separate physical edge
  rows for owner/label lookup. Existing `EndpointCollectRoute` gathers local and
  remote node outputs back to the edge owner before loss and metrics.
- Distributed legacy Snapshot artifacts without the edge-master target surface
  fail with a re-prepare error; the one-rank compatibility read remains narrow.

Attempted approaches:

- Reusing local Snapshot-CSC edges as supervision was rejected because CSC
  placement follows `node_master(dst)`, not the paper's `edge_master` task owner.
- Repartitioning Snapshot-CSC itself by `edge_master` was rejected because it
  would break destination-owned full-neighbor computation and boundary routes.
- A second task route type and user-selectable ownership policy were rejected;
  the existing `TaskTarget` and `EndpointCollectRoute` cover the required path.
- Two-rank validation exposed a stale import of `_all_to_all_counts`; the
  existing public `all_to_all_counts` helper is now reused.

Abstraction introduced:

- None. The only new physical data is the edge-master target tensors embedded
  in each existing Snapshot slice: logical ids, physical rows, endpoints, and
  timestamps.

Efficiency alternatives considered:

- Prepare uses vectorized torch masks and gathers. Across ranks, each snapshot
  target edge appears once; no Python node/edge loop or duplicate CSC is added.
- DGL reconstruction was rejected because target selection needs only tensor
  rows. A custom C++/CUDA operator is not justified for this offline prepare
  step; the execution hot path reuses scheduled all-to-all endpoint exchange.
- Shared-hot cache placement remains an input optimization and does not enter
  target, output, loss, or metric ownership decisions.

Files modified:

- `src/starrygl/{config_example.py,plan.py,spec.py}`
- `src/starrygl/prepare/{build_part_graph.py,snapshot_csc.py}`
- `src/starrygl/store/feature.py`
- `src/starrygl/task/{prediction.py,target.py}`
- `src/starrygl/runtime/{builders.py,epoch.py,trainer.py}`
- `src/starrygl/runtime/event/{materialize.py,target.py}`
- `src/starrygl/runtime/snapshot/target.py`
- `src/starrygl/model/_graph_ops.py`
- `configs/` task declarations and focused tests under `tests/`
- `docs/{STARRYGL_INTERFACE.md,design/current/{execution_spine,paper_method_coverage,partition}.md}`

Verification:

- Focused task/plan/prepare/store/route suite: 134 passed.
- Non-contiguous logical edge-id prepare/store/runtime suite: 40 passed.
- Full unit suite: 239 passed, 5 skipped outside distributed launch.
- Two-rank Gloo Snapshot endpoint collect: 1 passed on each rank, including an
  empty-target rank serving its node-master output.
- Two-rank Gloo double-buffer/feature suite: 14 passed on each rank.
- Full-package compile check passed; all production Python modules remain at or
  below 500 physical lines.

Unresolved performance risks:

- Representative NCCL profiling is still required to measure Snapshot
  edge-owner endpoint exchange and the small target-surface storage overhead.
- Existing multi-rank Snapshot artifacts must be rebuilt before edge prediction;
  no broad legacy reconstruction fallback was added.

## 2026-08-27: Bind Runtime Batch Semantics Explicitly

Latest state:

- `RuntimeBatchPlan` is carried by `DatasetBatch.plan`; runtime no longer hides
  or recovers the plan through `BatchMeta.values`.
- At this migration stage, the canonical task name was carried by
  `RuntimeBatchPlan`; the 2026-08-30 task/window decoupling below supersedes
  that field. The `_task_name` sampler option and the unused runtime
  `storage_view` duplicate remain removed.
- `TaskTarget` now contains supervision only. Event endpoints, event ids,
  event timestamps, and the prepared state-write mask are carried by one
  `EventContext` in `Batch.targets["events"]`.
- TGN, JODIE, and APAN state updates consume `EventContext`. Node-label
  timestamps no longer stand in for event timestamps during sampling, memory
  commit, or mailbox commit.

Attempted approaches:

- Keeping `RuntimeBatchPlan` in debug metadata was rejected because it made
  scheduler correctness depend on an untyped observability surface.
- Keeping task identity in sampler options was rejected because it duplicated
  the canonical task and allowed Event/Snapshot target lowering to diverge.
- Adding a new state-read stage was rejected. The existing Batch target surface
  is sufficient to separate supervision from event state-update inputs.

Abstraction introduced:

- `EventContext` is one frozen tensor container shared by Event sampling and
  the existing TGN/JODIE/APAN state-update paths. It replaces duplicated event
  fields; it does not introduce another execution path or manager.

Efficiency alternatives considered:

- Existing native sampling, vectorized graph materialization, feature/state
  gathers, and scheduled collectives are unchanged. Event sampling roots use
  one vectorized tensor concatenation with the real event timestamps.
- `EventContext` references tensors already materialized for the batch and does
  not copy per-node or per-edge payloads. No Python node/edge loop was added.
- A torch/DGL or custom C++/CUDA operator is not justified for this control and
  ownership binding change; GPU hot-path parity remains a benchmark question.

Files modified:

- `src/starrygl/{__init__.py,batch/{__init__.py,source.py},task/{negative.py,target.py}}`
- `src/starrygl/runtime/{loop.py,epoch.py,dataloader/{loader.py,pipeline.py}}`
- `src/starrygl/runtime/{event/{__init__.py,target.py,materialize.py},snapshot/{__init__.py,target.py,materialize.py}}`
- `src/starrygl/model/{tgn.py,tgn_mailbox.py,apan.py}`
- Focused tests under `tests/` and interface/execution design notes under
  `docs/`.

Verification:

- Focused runtime/target/model/store/plan suite: 102 passed, 1 skipped.
- Full unit suite: 239 passed, 5 skipped.
- Two-rank Gloo queue, feature, memory, and model suite: 65 passed on each rank.
- Full-package compile check passed. Every production Python module remains at
  or below 500 physical lines (maximum: 497).

Unresolved performance risks:

- This pass proves contract and collective-order parity, not paper throughput
  or metric parity. Representative NCCL Event-neighbor and Snapshot full/
  neighbor runs still need stage timings, peak memory, and end metrics compared
  with the preserved MemShare-public and FlareDTDG baselines.
- Sampled Snapshot still needs native enforcement of selected Snapshot/chunk
  boundaries before optimized parity can be claimed.

## 2026-08-28: Bind Native Sampling To The Public Extension

Latest state:

- Event sampling and speed partitioning no longer load the deprecated private
  `atc_starrygl_lib` binary.
- A missing public sampler now fails with the repository build command instead
  of silently loading an ABI-incompatible extension.
- `scripts/build_native.sh` constructs a real 12-argument `ParallelSampler`
  after compilation, covering the `node_is_hot` ABI addition.

Attempted approaches:

- Retaining the private binary as a compatibility fallback was rejected because
  it hides missing public build artifacts and can load an older constructor ABI.
- A second Python adapter for old and new constructor signatures was rejected;
  the C++ source and Python wrapper now have one required ABI.

Abstraction introduced:

- None. The existing CMake target and Python wrapper are reused.

Efficiency alternatives considered:

- Sampling kernels and tensor layouts are unchanged. The ABI smoke runs once at
  build time and adds no training hot-path work.
- No torch/DGL replacement or new C++/CUDA operator is needed for this packaging
  and module-loading correction.

Files modified:

- `src/starrygl/native/sampling.py`
- `src/starrygl/partition/build.py`
- `CMakeLists.txt`
- `scripts/build_native.sh`
- `README.md`

Verification:

- Public native CMake build completed for Python 3.10 and Torch 2.1.1+cu118.
- A clean CMake configure/build uses Torch's bundled pybind11 headers and does
  not require an undeclared standalone pybind11 package.
- Build-time 12-argument sampler ABI smoke passed.
- Public `NativeTemporalSampler.from_graph()` loaded
  `starrygl.native.lib.libstarrygl_sampler` and constructed successfully.
- Minimal CPU `compile -> prepare -> fit` Event/TGAT smoke completed one step.
- Partition tests: 6 passed. Full unit suite: 239 passed, 5 skipped.

Unresolved performance risks:

- Native kernel throughput is unchanged and was not benchmarked in this pass.
- Source and editable installs use `scripts/build_native.sh`; producing tagged
  platform wheels that build and bundle the extension remains release work.

## 2026-08-28: Preserve Snapshot Axes And Physical Edge Rows

Latest state:

- Dense Snapshot node features and labels preserve the `[S, N, ...]` layout;
  node partitioning selects axis 1 and scalar labels `[S, N]` remain temporal.
- Sampled blocks keep logical edge ids for graph semantics and carry a separate
  internal feature-row map. Bidirectional sampler rows share the original edge
  feature row instead of indexing features with user edge ids.
- Regression targets are reshaped to prediction shape only when their element
  counts agree, preventing accidental `[N, 1]` versus `[N]` broadcasting.

Attempted approaches:

- Inferring every rank-2 label as temporal was rejected because ordinary
  `[N, C]` labels have the same rank. Prepare infers the temporal contract only
  from matching Snapshot and node axes, then stores the result explicitly.
- Replacing logical edge ids with feature rows was rejected because routes,
  observability, and task ownership still require stable user edge ids.

Abstraction introduced:

- `node_label_temporal` records the otherwise ambiguous dense-label axis
  contract. `GraphBlock.cache["edge_feature_ids"]` reuses the existing physical
  cache surface for sampler edge rows; no new block or execution path was added.

Efficiency alternatives considered:

- Partitioning and materialization use tensor `index_select` and indexed
  assignment. Reverse-edge feature rows are duplicated as integer indices, not
  feature tensors. No Python node/edge loop was added.
- Existing native sampling remains the hot path. A new C++/CUDA operator is not
  justified for these axis and identifier corrections.

Files modified:

- `src/starrygl/prepare/{data.py,temporal_csr.py}`
- `src/starrygl/store/{feature.py,graph.py}`
- `src/starrygl/native/{sampling.py,sampling_output.py}`
- `src/starrygl/runtime/{trainer.py,trainer_options.py}`
- `src/starrygl/runtime/{sample/{__init__.py,blocks.py},event/materialize.py,snapshot/target.py}`
- `src/starrygl/task/prediction.py` and focused tests under `tests/`

Verification:

- Full unit suite: 243 passed, 5 skipped.
- Snapshot/TGCN `[S, N, F]` feature plus `[S, N]` regression smoke completed
  three steps without broadcasting warnings.
- Event/TGAT native-neighbor smoke with non-contiguous logical edge ids and edge
  features completed successfully.
- Compile check passed; every touched production module remains below 500
  physical lines.

Unresolved performance risks:

- Native sampler timestamps still use the existing signed 64-bit integer ABI.
  Non-integral timestamp datasets require an explicit encoding decision and
  kernel-level parity tests before changing that ABI.
- Correctness is established, but representative paper throughput and peak
  memory still require comparison with FlareDTDG and MemShare-public.

## 2026-08-28: Separate Authoritative State From Feature Replicas

Latest state:

- Runtime state and mailbox storage derives its authoritative rows from
  `node_dist_index` and the local rank. A replicated feature row can no longer
  shadow a newer shared-hot state value or accept an owner commit.
- Distributed state/mailbox reads enter one global miss decision on every rank,
  including ranks whose requested rows are all local or empty.
- State kinds that use owner collectives commit in stable manager order; ranks
  without an update submit an empty delta. Empty recurrent deltas follow the
  existing timestamp contract of each state kind.

Attempted approaches:

- Reusing `FeatureManager.node_row_map` was removed because feature replicas are
  read optimization, not state authority.
- Keeping an outer bounded-stale miss all-reduce was rejected after state fetch
  gained symmetric global arbitration; it caused two collectives for one miss.

Abstraction introduced:

- None. `_owner_layout()` is a construction helper for the existing
  `StateManager`/`MailboxManager`; owner and shared-hot managers remain separate
  physical planes behind the existing runtime manager.

Efficiency alternatives considered:

- Owner maps are built once with vectorized packed-index operations. Hydration,
  remote packing, state commit, and shared cache updates retain tensor and
  scheduled collective paths with no Python node/update loops.
- The bounded-stale owner fallback now performs one global miss decision rather
  than two. No new C++/CUDA operator is needed for construction/control logic.

Files modified:

- `src/starrygl/runtime/state/{build.py,__init__.py}`
- `src/starrygl/runtime/memory/historical.py`
- `src/starrygl/store/{state.py,mailbox.py}`
- `tests/test_runtime_memory.py`

Verification:

- Full unit suite: 244 passed, 7 skipped outside distributed launch.
- Full two-rank Gloo runtime-memory suite: 12 passed on each rank.
- The distributed suite covers one rank serving an all-local request while the
  peer has a remote miss, plus a real recurrent update paired with an empty
  owner-collective update.
- Compile and line-count checks passed; touched production modules are 357
  physical lines or fewer.

Unresolved performance risks:

- NCCL stage timings are still required; Gloo validates collective order, not
  GPU overlap or paper throughput.
- State artifact checkpoint/restore and evaluation reset semantics are handled
  in the next review unit.

## 2026-08-28: Bind Prepared Artifacts And Temporal Phase State

Latest state:

- Prepared artifacts carry a stable signature over data, effective prepare
  configuration, physical view requirements, partition inputs, and feature
  layout. Training/device options do not invalidate graph artifacts.
- Artifact loading rejects a mismatched or unsigned prepare bundle before
  binding its `PartitionPlan` into the current `ExecutionPlan`.
- One trainer caches its validated rank-local `StoreBundle`, so the state built
  during the final train epoch continues into validation and test.
- Evaluation and prediction advance temporal state by default. This matches the
  MemShare sequence: reset at epoch start, train, stateful val, then stateful
  test.
- Native `workers=0` now selects one synchronous C++ worker instead of creating
  an invalid zero-thread sampler.

Attempted approaches:

- Hashing all runtime/train settings was rejected because learning rate, epoch
  count, and device do not affect prepared graph layouts.
- Rebuilding state for each phase was removed because it discarded train/val
  history. A new public phase-state policy was not added; the canonical
  chronological flow is sufficient for the current paper path.

Abstraction introduced:

- `artifact_fingerprint()` is a small storage-boundary helper. It hashes
  in-memory tensors and uses path/size/mtime for file sources; it does not add a
  manifest framework or a second plan type.

Efficiency alternatives considered:

- Source files are not reread or content-hashed during cache checks. Reusing the
  StoreBundle also avoids repeated artifact deserialization between train, val,
  and test.
- State advancement uses the existing queued hydration/commit path. Native
  sampling and communication kernels are unchanged; no Python edge/update loop
  was introduced.

Files modified:

- `src/starrygl/runtime/{trainer.py,trainer_options.py,train.py}`
- `src/starrygl/store/artifact.py`
- `src/starrygl/native/sampling.py`
- `src/starrygl/runtime/sample/__init__.py`
- Focused store, lifecycle, and native-construction tests under `tests/`

Verification:

- Full unit suite: 247 passed, 7 skipped outside distributed launch.
- Real CPU Event/TGN `fit -> evaluate -> predict` smoke completed one step per
  split; state commit counts advanced `[1, 2, 3]` on one cached StoreBundle.
- Changed training options reused an artifact; changed graph timestamps were
  rejected before runtime execution.
- Compile and line-count checks passed; touched production modules are 444
  physical lines or fewer.

Unresolved performance risks:

- File signatures use metadata rather than full file content. Content changes
  that preserve both size and nanosecond mtime require `force=True` to rebuild.
- Checkpointing model plus temporal owner/shared state for a standalone
  evaluation process is not implemented; current parity covers the canonical
  same-process paper workflow.

## 2026-08-28: Align Snapshot Windows With Flare Execution

Latest state:

- Snapshot-CSC preserves explicit scalar edge weights and precomputes the same
  weighted symmetric GCN normalization as the Flare preprocessing path,
  including the unit self-loop contribution.
- `chunk_order` is interpreted as `chunk -> priority`; prefix order is its
  stable argsort. Chunk-limited full graphs are induced subgraphs over the
  selected prefix, so old windows no longer retain prefix-external neighbors.
- Decoupled T-GCN and MPNN-LSTM training keeps recurrent state inside each
  rolling Batch. It does not inject the previous overlapping Batch's final
  state, which would advance repeated snapshots twice.
- Snapshot scans retain one embedding/prediction per window. Node loss and MSE
  use Flare's equal-weight window mean rather than supervising only the final
  window or concatenating unequal node counts.
- Decoupled full-Snapshot evaluation and prediction reset recurrent state,
  replay preceding non-empty splits one full snapshot at a time, and score only
  the requested split. `evaluate()` selects test when val is explicitly empty,
  matching the paper's 40/0/60 Snapshot split.

Attempted approaches:

- Persisting one latest `node_recurrent` table through chunk-decay training was
  removed from the active training path: the table has no snapshot axis and
  cannot represent the state before each overlapping historical window.
- Concatenating all window predictions was rejected because truncated windows
  have different node counts and Flare weights windows, not nodes, equally.
- Keeping remote or later-chunk neighbors in a truncated full graph was
  rejected after checking Flare's `node_subgraph(prefix)` materialization.

Abstraction introduced:

- `WindowScanResult.window_embeddings` extends the existing scan result so the
  model and task can preserve window supervision without a second Batch/model
  API. Two small private train helpers share exact replay between evaluate and
  predict; no public execution policy was added.

Efficiency alternatives considered:

- Edge normalization, chunk priority ordering, induced-subgraph filtering,
  feature selection, and target routing remain vectorized torch operations.
  The only added Python iteration is over the short Snapshot window list,
  matching the existing model scan and avoiding node/edge loops.
- Existing layerwise boundary exchange remains the distributed GCN hot path;
  truncated windows intentionally avoid communication. A new C++/CUDA operator
  is not justified until profiling shows prefix CSC filtering is material.

Files modified:

- `src/starrygl/prepare/{data.py,build_part_graph.py,snapshot_csc.py}`
- `src/starrygl/runtime/{loop.py,train.py}`
- `src/starrygl/runtime/snapshot/{materialize.py,rows.py,scan.py,target.py}`
- `src/starrygl/model/{base.py,tgcn.py,mpnn_lstm.py,gconv_gru.py}`
- `src/starrygl/task/prediction.py` and focused tests under `tests/`

Verification:

- Full unit suite: 251 passed, 7 skipped outside distributed launch.
- A real CUDA T-GCN smoke with temporal `[S,N,F]` features, weighted edges,
  chunk decay, and next-snapshot labels completed three train steps and exact
  chronological test replay.
- Focused checks distinguish equal window mean from concatenated MSE, lock
  Flare chunk-priority order and induced prefixes, and verify evaluation and
  prediction replay order.
- Compile and source-size checks pass; touched production modules remain below
  500 physical lines.

Unresolved performance risks:

- Multi-rank NCCL parity for layerwise autograd exchange and endpoint embedding
  exchange is reviewed in the next unit; current Snapshot smoke is single-rank.
- Persistent GPU Snapshot entry caching can trade transfer time for peak memory
  and still needs representative Flare dataset profiling before changing its
  default.
- MPNN-LSTM uses PyTorch's fused `LSTMCell`, which is mathematically compatible
  with Flare's four-gate cell but does not reproduce its random initialization
  stream exactly.

## 2026-08-28: Preserve Gradients Across Owner And Layerwise Routes

Latest state:

- Snapshot endpoint embeddings return from `node_master` to `edge_master`
  through the existing autograd all-to-all operation. Edge loss gradients now
  reach the authoritative node embedding owner instead of stopping at an
  ordinary response push.
- Empty edge-owner ranks retain an autograd-connected empty response and run a
  zero-loss backward step. They therefore enter the same response-backward and
  parameter all-reduce epochs as ranks with targets.
- Gradient synchronization materializes zero gradients for trainable parameters
  without a local gradient, preserving one identical all-reduce sequence on all
  ranks.
- Full-Snapshot layerwise exchange is scheduled from route world size, not
  local nonzero payload size. A rank with empty local splits still participates
  in the globally ordered collective.

Attempted approaches:

- Keeping endpoint responses on ordinary `CommScheduler.launch_push()` was
  rejected because it cannot propagate edge-owner loss to node owners.
- Returning a newly allocated empty response on ranks with no local request was
  rejected after a two-rank backward exposed that the rank lost its `grad_fn`
  and skipped `_Push.backward`.
- Skipping parameters with `grad is None` was removed because rank-dependent
  gradient presence changes all-reduce order and can deadlock.

Abstraction introduced:

- No new public abstraction. `CommScheduler` now owns the existing autograd
  pull/push launch points so forward collective ticket order remains observable.

Efficiency alternatives considered:

- Endpoint and layerwise gradients reuse the existing packed
  `torch.distributed.all_to_all_single` implementation. Payload construction is
  vectorized (`argsort`, `bincount`, `index_select`, `index_copy_`); no Python
  node/edge loop or additional graph materialization was introduced.
- A custom C++/CUDA communication operator is not justified until NCCL traces
  show overhead beyond the collective itself. DGL does not provide an
  owner-routed differentiable endpoint exchange that replaces this path.

Files modified:

- `src/starrygl/model/_graph_ops.py`
- `src/starrygl/runtime/{comm.py,epoch.py,loop.py}`
- `src/starrygl/runtime/snapshot/{layerwise.py,scan.py}`
- `tests/test_starrygl_model.py`

Verification:

- Full unit suite: 252 passed, 9 skipped outside distributed launch.
- Two-rank Gloo checks pass for endpoint forward/backward ownership, layerwise
  remote-gradient return, and parameter synchronization with a missing local
  gradient (three tests on each rank).
- Focused communication/model/runtime suite: 73 passed, 4 skipped.
- Compile and source-size checks pass; all production Python modules remain at
  or below 500 physical lines (`runtime/comm.py` is exactly 500).

Unresolved performance risks:

- Multi-GPU NCCL traces are still required to measure endpoint-response and
  layerwise overlap at paper scale; Gloo establishes correctness and ordering.
- Empty-owner parameter all-reduce creates zero gradient buffers. This is
  required for correctness with manual synchronization; native DDP may later
  replace the helper after an end-to-end parity check.

## 2026-08-28: Align Historical Memory And APAN Mailbox Semantics

Latest state:

- Model-side smooth compensation is named and stored as a state increment
  estimator; it no longer shares metadata or implementation names with the
  runtime shared-refresh filter.
- Bounded-stale combined hydration reads memory and mailbox from the same
  shared-hot row and falls back to the owner only for unresolved cold rows.
  The unused second epoch-cache path was deleted.
- TGN/APAN Transformer memory now uses each mailbox slot timestamp and matches
  MemShare attention, residual, normalization, MLP, dropout, and activation
  order. Historical transitions are normalized before learnable-gamma mixing.
- APAN starts from the common endpoint self-mail payload, then forwards those
  messages to sampled neighbors and keeps the latest timestamp per node.

Attempted approaches:

- Reusing model compensation metadata to select shared refreshes was removed:
  partial compensation rows could silently omit another authoritative hot-node
  update in the same state delta.
- Keeping independent stale memory and exact mailbox reads was rejected because
  it mixes versions before the model update and preserves an avoidable owner
  communication dependency.
- APAN sampled-edge messages were replaced with endpoint self-mail forwarding;
  the former did not match MemShare `deliver_to=neighbors` semantics.

Abstraction introduced:

- No public abstraction. The existing `TGNMemoryUpdater` and combined
  memory/mailbox hydrate path remain the single shared implementations.

Efficiency alternatives considered:

- Cache selection, transition normalization, mailbox propagation, and latest
  message selection use vectorized Torch indexing and scatter reductions. No
  Python node/edge loop or repeated DGL block construction was added.
- Existing owner collectives and shared-hot all-gathers were reused. A custom
  C++/CUDA operator is not justified until profiling shows these tensor kernels,
  rather than communication, dominate representative workloads.

Files modified:

- `src/starrygl/model/{tgn.py,jodie.py,apan.py,layers/temporal.py}`
- `src/starrygl/runtime/memory/{__init__.py,access.py,historical.py,shared.py}`
- `src/starrygl/runtime/{builders.py,train.py}` and focused tests

Verification:

- Full unit suite: 256 passed, 10 skipped.
- Two-rank Gloo memory suite: 16 passed on each rank, including exact owner
  hydration and a bounded shared-hot memory/mailbox read with no owner ticket.
- Deterministic checks match MemShare Transformer math, normalized historical
  blending, and APAN self-plus-neighbor mailbox delivery.
- Compile and source-size checks pass; all production Python modules remain at
  or below 500 physical lines.

Unresolved performance risks:

- NCCL traces on paper datasets are still needed to quantify cache-hit overlap
  and shared-hot refresh traffic.
- TGN's public paper configuration uses one mailbox slot. Multi-slot
  Transformer mailbox behavior is exercised through APAN and has not been
  promoted as an additional TGN configuration.

## 2026-08-28: Close Task, Event Chronology, And Input-Parity Gaps

Latest state:

- Epoch metrics are aggregated from all ranks, including ranks with no local
  targets. Macro-F1 is rebuilt from the global confusion matrix rather than
  averaged from rank-local F1 values.
- Event node supervision follows the same chronological windows as event edge
  execution. Empty target owners still run a zero-gradient synchronized step,
  so distributed parameter collectives stay rank-symmetric.
- TGL `time`/`ext_roll`, feature sidecars, and temporal node-label events convert
  into the canonical StarryGL tensors without changing edge identity or order.
  CSV parsing now uses pandas' C parser instead of one Python object per row.
- Representative GDELT Event and Soc-Bitcoin Snapshot configurations pin the
  matching MemShare/TGL and FlareDTDG model/training defaults. Train-time
  AP/AUC is disabled by default; evaluation retains Torch-native AP/AUC and
  Macro-F1 computation.
- Native temporal-sampler tests lock timestamp-aware `(node, timestamp)`
  compaction, preventing a later ID-only deduplication from merging historical
  instances.

Attempted approaches:

- Rank-local metric returns and scalar F1 averaging were removed because they
  either change collective order or produce the wrong global Macro-F1.
- Python `csv.DictReader` and per-line float lists were removed from paper-data
  conversion because their object overhead is not viable for GDELT-scale input.
- Scikit-learn batch metrics were removed from the runtime path because they
  were undeclared and copied GPU predictions to CPU every training batch.

Abstraction introduced:

- No new runtime abstraction. Existing Event windows, task targets, epoch
  reduction, and canonical `GraphData` remain the single execution contracts.
  Pandas is an explicit package dependency used only at CSV input boundaries.

Efficiency alternatives considered:

- The selected input path mirrors FlareDTDG's pandas C parser and converts
  columns directly to contiguous Torch tensors. A custom C++ parser is not
  justified for a one-time conversion before canonical `.pt` artifacts are
  reused.
- Torch-native sorting and reductions remain the metric implementation. A
  fused CUDA metric operator is unnecessary because train-time metrics are off
  and evaluation sorting has not been measured as a bottleneck.
- Event label assignment uses `searchsorted` and tensor indexing; no per-event
  or per-node Python loop was added to the runtime hot path.

Files modified:

- `src/starrygl/{spec.py,task/prediction.py,runtime/epoch.py,runtime/loop.py}`
- `src/starrygl/prepare/{data.py,event.py,build_part_graph.py}`
- `src/starrygl/runtime/{trainer.py,event/__init__.py}`
- `src/starrygl/tools/convert_to_starrygl_format.py`
- `configs/paper/*.json`, `pyproject.toml`, focused tests, and paper coverage
  documentation

Verification:

- Full unit suite: 264 passed, 12 skipped.
- Two-rank Gloo checks pass on both ranks for empty-owner synchronized training
  and metrics with a rank that has no targets.
- Data/store/partition/dataloader suite: 33 passed, 1 skipped.
- Full package compile and source-size checks pass; every production Python
  module remains at or below 500 physical lines.

Unresolved performance risks:

- Pandas removes per-row Python objects but still materializes a complete input
  table. Measure conversion peak memory on the full GDELT file before deciding
  whether chunked parsing is necessary.
- Paper-level metric and throughput parity still requires the real datasets,
  multi-GPU NCCL runs, identical seeds, and baseline result comparison. Unit and
  Gloo tests establish contracts and collective order, not published accuracy
  or speed.

## 2026-08-28: Close Snapshot, EvolveGCN, And State-Lowering Parity Gaps

Latest state:

- Sampled Snapshot now builds its native topology from the selected Snapshot
  rows and applies the selected chunk prefix to sampling roots. It no longer
  samples from the complete Event/T-CSR history for a window-local request.
- Full Snapshot chunk ordering preserves and remaps Route indices. An induced
  chunk-limited graph drops the stale Route and rebuilds the smaller local
  surface rather than using invalid offsets.
- Converted DTDG node-regression artifacts record next-Snapshot label semantics;
  runtime planning excludes the final Snapshot when no future label exists.
- EvolveGCN evolves and applies one weight for every Snapshot in the training
  window, exposes all window logits to the common task loss, and starts each
  overlapping training window from batch-local state. Evaluation still carries
  model state in chronological order.
- Learnable stale-state compensation is constructed only for
  `bounded_stale + smooth_aggregation`. Exact TGN/JODIE/APAN/GConvGRU execution
  no longer creates unused compensation parameters.

Attempted approaches:

- Reusing the complete temporal sampler for sampled Snapshot was removed because
  it admitted edges outside the selected Snapshot and chunk boundary.
- Carrying a chunk-reordered Route unchanged was rejected because its positions
  refer to the pre-reorder destination layout.
- Persisting EvolveGCN state across overlapping training windows was removed;
  that evolves the same historical Snapshot repeatedly from an already advanced
  weight and differs from FlareDTDG's window-local training blobs.
- Enabling smooth compensation in exact mode was removed because exact reads do
  not expose shared-history metadata and leave gamma permanently unused.

Abstraction introduced:

- No public abstraction. Existing `RuntimeBatchPlan`, native sampler,
  `GraphBlock`, common Snapshot scan, `ModelOutput`, and temporal-state lowering
  remain the only contracts used by these fixes.

Efficiency alternatives considered:

- Snapshot selection, chunk-root filtering, Route remapping, and label-window
  exclusion use tensor masks, indexing, and `searchsorted`; no Python
  node/edge/message loop was added to the training hot path.
- Rebuilding a native sampler from a selected Snapshot window is the smallest
  correct implementation. A fused Snapshot-CSC sampler is justified only if
  paper-scale profiling shows rebuild cost on the critical path.
- EvolveGCN reuses the existing window scan and graph-convolution operators.
  No repeated DGL graph construction or custom CUDA kernel was added.
- Filter and smooth semantics reuse the existing MemShare-derived refresh filter
  and increment estimator. DGL does not replace these state/cache operations;
  a custom C++/CUDA operator needs profile evidence.

Files modified:

- `src/starrygl/runtime/{builders.py,loop.py,dataloader/loader.py}`
- `src/starrygl/runtime/snapshot/{materialize.py,rows.py,target.py}`
- `src/starrygl/runtime/sample/__init__.py`
- `src/starrygl/model/evolve_gcn.py`
- `src/starrygl/tools/convert_to_starrygl_format.py`
- focused model/runtime/target/materialization tests and current design docs

Verification:

- Full unit suite: 269 passed, 12 skipped.
- Focused model/state/pipeline/paper-contract suite: 115 passed, 10 skipped.
- Two-rank Gloo suite: 110 passed on each rank, covering empty owners, ready-batch
  collective ordering, remote feature/state/mailbox access, endpoint exchange,
  layerwise gradients, and distributed task metrics.
- Full package/test compile checks pass. Every production Python module remains
  at or below 500 physical lines (`runtime/comm.py` is exactly 500).

Unresolved performance risks:

- `max_staleness` currently caps local filtered shared-hot refresh skips; a
  strict global owner-version distance still requires a scheduled collective
  watermark. The current cap also reduces MemShare's default `max_skip=10` to
  one when `max_staleness=1`; changing this requires an explicit consistency
  decision plus traffic/quality measurements.
- PyTorch LSTM-cell initialization is not bitwise identical to FlareDTDG's
  custom gate initialization even where the recurrent equations match.
- Real paper datasets, identical seeds, and multi-GPU NCCL traces are still
  required to establish final metric, throughput, overlap, peak-memory, and
  scaling parity. Gloo and unit tests prove contracts and collective order, not
  published performance.

## 2026-08-30: Carry Coupled State Across Snapshot Windows

Latest state:

- Coupled Snapshot execution hydrates the union state table once and now treats
  it as a mutable window working table. Every temporal unit reads the latest
  locally produced destination state rather than re-reading the immutable
  window-entry value.
- Local destination updates replace every duplicate hydrated row for that node,
  which covers repeated node ids contributed by multiple Snapshots. Remote rows
  that are never local destinations retain their hydrated exact/cache value.
- The paper's chunk-decay paragraph now follows FlareDTDG's implemented
  geometric `auto:s` schedule, including `F` full Snapshots and one shared chunk
  permutation.

Attempted approaches:

- Re-reading `Batch.state` for every Snapshot was removed because it discarded
  the immediately preceding local recurrent update.
- Hydrating one full state tensor per Snapshot was rejected because the union
  table already contains the required rows and repeated communication would
  increase memory and latency without fixing recurrence.
- The existing exact owner hydrate was not presented as complete exact coupled
  execution: remote `H_{k-1}` still needs an autograd-aware Route exchange
  between adjacent Snapshots.

Abstraction introduced:

- No public abstraction. `CoupledStateMaterialization` remains the state-layout
  boundary; one vectorized `scatter_state_table` helper updates repeated rows.

Efficiency alternatives considered:

- Duplicate-row updates reuse tensor `argsort/searchsorted` lookup plus
  `index_copy_`; no Python node, edge, neighbor, or state-update loop was added.
- Reusing Snapshot-CSC Route exchange is the intended exact path. It was not
  wired speculatively because collective ordering and the owned-row layout need
  a dedicated two-rank correctness test.
- A precomputed per-Snapshot state-row map may remove repeated table lookup, but
  should be added only if paper-scale profiling places lookup on the hot path.

Files modified:

- `src/starrygl/runtime/state/recurrent.py`
- `src/starrygl/runtime/snapshot/scan.py`
- `tests/test_starrygl_model.py`
- `docs/design/current/{execution_spine.md,paper_method_coverage.md}`
- `../paper_method/method.tex`

Verification:

- Focused coupled/state/model suite: 72 passed, 8 skipped.
- Full single-process suite: 269 passed, 12 skipped.
- Two-rank Gloo model/memory suite: 66 passed on each rank.
- Production package compile check passes; modified production modules remain
  below 500 physical lines.

Unresolved performance and correctness risks:

- Exact multi-rank coupled Snapshot execution still lacks the per-Snapshot
  autograd boundary-state exchange required for current remote `H_{k-1}`.
- External state-table lookup sorts node ids during each temporal-unit
  materialization. Measure this before adding prepared row maps or a native
  gather/scatter operator.
- FlareDTDG's optional `enable_states`/`mix` snapshot-slot fallback is not part
  of this coupled path. The published/default `pad` behavior does not require
  that optional cache, but an explicit parity experiment is still needed if
  `mix` is enabled.

## 2026-08-30: Remove State Versions From The Runtime Path

Latest state:

- State and mailbox storage, remote reads, shared-hot overlay, and `Batch.state`
  now carry values and semantic timestamps only. Per-row version tensors and
  their collective payloads were removed.
- Bounded staleness is defined by the existing publisher-local filter: accepted
  updates reset its skip count and `max_staleness` bounds consecutive skipped
  refreshes. Filtered updates are then gathered into every rank's shared-hot
  cache replica.
- Exact reads remain ordered by state-commit waits or Snapshot Route exchange;
  they do not use a revision number. Uncommitted `model_recurrent` state remains
  absent from the model batch so EvolveGCN uses its learnable initial weight.

Attempted approaches:

- A per-node owner revision watermark was removed because no production caller
  supplied the required revision and it did not implement the selected
  filter-skip semantics.
- Replacing versions with another batch-visible counter was rejected. The
  filter already owns the only synchronization counter required by the method.

Abstraction introduced:

- None. Existing state-read and pending-fetch records were shortened by
  deleting version fields.

Efficiency alternatives considered:

- No torch/DGL or native operator is needed for this deletion. Removing the
  integer gather/copy/all-to-all path reduces allocations and communication;
  timestamps continue to use vectorized tensor selection for competing updates.

Files modified:

- `src/starrygl/store/{state.py,mailbox.py,remote_fetch.py}`
- `src/starrygl/runtime/state/access.py`
- `src/starrygl/runtime/memory/{access.py,historical.py,ops.py}`
- `src/starrygl/model/evolve_gcn.py`
- `tests/{test_compact_runtime_layout.py,test_runtime_access.py,test_runtime_memory.py,test_starrygl_model.py,test_starrygl_store.py}`
- `docs/design/current/{execution_spine.md,state_await_cache.md,paper_method_coverage.md}`

Verification:

- Focused state/store/model suite: 86 passed, 8 skipped.
- Full single-process suite: 268 passed, 12 skipped.
- Two-rank Gloo state/store/access suite: 45 passed on each rank.
- Full source/test compile check passes; every production Python module remains
  at or below 500 physical lines.

Unresolved risks:

- `max_staleness` is intentionally a publisher-local skipped-refresh bound, not
  a global wall-clock or owner-update distance.
- Exact multi-rank coupled Snapshot execution still needs the previously noted
  per-Snapshot autograd Route wiring; removing versions neither adds nor removes
  that requirement.

## 2026-08-30: Use Prepared Snapshot Row Order For Split IDs

Latest state:

- Snapshot-CSC construction and Snapshot window planning now share the prepared
  `time_ptr_2` row number directly as `snapshot_id`.
- Runtime derives each split's contiguous id range from the train/validation/test
  row counts. It no longer scans and compares `[begin, end]` rows.
- `RuntimeBatchPlan` now describes only input windows and sampling policy;
  task names and target ids were removed from it. Event and Snapshot both emit
  this task-independent plan.
- Snapshot DatasetBatch construction independently binds the registered task to
  `target_snapshot_id`: node tasks use the input anchor, while edge prediction
  uses the next Snapshot and drops only the final anchor in that split.
- TaskTarget timing is unchanged: root-dependent targets precede neighbor
  sampling, while physical TargetRoute rows bind only after GraphBlock creation.

Attempted approaches:

- Persisting a duplicate `split_snapshot_ids` tensor was rejected because split
  pointers already define contiguous row counts in fixed order.
- Event-boundary `cumsum` was rejected because it produces event offsets, not
  Snapshot row ids; cumulative split row counts provide the required offsets.
- Keeping task-specific edge and node WindowPlan builders was rejected because
  task selection changes supervision, not the physical input window.

Abstraction introduced:

- None. `_snapshot_ids_for_split` replaces the old value-matching helper, and
  the existing DatasetBatch values carry the one scalar target reference.

Efficiency alternatives considered:

- The selected path uses one three-element torch cumulative sum per plan build.
  Search, DGL, and custom native operators are unnecessary for this control path.

Files modified:

- `src/starrygl/runtime/snapshot/__init__.py`
- `src/starrygl/{batch/source.py,runtime/event/__init__.py}`
- `src/starrygl/runtime/{event/materialize.py,snapshot/target.py,snapshot/materialize.py}`
- `src/starrygl/runtime/dataloader/loader.py`
- `tests/test_open_compile_semantics.py`
- `tests/{test_runtime_targets.py,test_starrygl_store.py}`
- `docs/STARRYGL_INTERFACE.md`
- `docs/design/current/execution_spine.md`
- `docs/migration/status_log.md`

Verification:

- Focused compile/target/store suite: 66 passed.
- Full unit suite: 270 passed, 12 skipped.
- Full source/test compile check passes; all touched production modules remain
  below 500 physical lines.

Unresolved risks:

- Runtime relies on the existing prepare invariant that Snapshot rows are stored
  in train/validation/test order; focused coverage now includes duplicate empty
  event ranges to protect row-identity semantics.
- End-to-end performance is unchanged by this control-path refactor and still
  requires the planned FlareDTDG/MemShare parity benchmarks.

## 2026-08-30: Share Global Window IDs Across Event And Snapshot

Latest state:

- Event and Snapshot now use the global row number of prepared `time_ptr_2` as
  the same `window_id`; each split is one half-open Python range derived from
  the fixed train/validation/test row counts.
- Event reads rank-local edge bounds from `event_view["time_ptr_2"][window_id]`.
  Snapshot uses the same integer to select its Snapshot-CSC row.
- Event runtime no longer performs a second `edge_batch_size` split. For raw
  Event input, prepare defaults to `time_split=batch` and lowers
  `runtime.train.batch_size` into `target_batch_size`.
- Empty rank-local Event rows remain in the shared range so distributed ranks
  preserve the same collective schedule.

Attempted approaches:

- Keeping split-local Event ids plus global Snapshot ids was rejected because
  every downstream handoff then needs an offset or lookup.
- Persisting another split-id tensor was rejected; three prepared row counts
  already define the ranges.
- Runtime Event re-splitting was removed because its generated batches cannot
  retain the invariant `window_id == time_ptr_2 row`.

Abstraction introduced:

- Internal `split_window_range` returns a built-in `range`; it is shared by the
  Event and Snapshot producers and is not exported as public API.

Efficiency alternatives considered:

- A torch prefix sum, row-value search, DGL operator, and native operator were
  unnecessary for three control-plane counts. Python computes the range once
  per split; all edge and graph reads remain direct tensor indexing.

Files modified:

- `src/starrygl/store/graph.py`
- `src/starrygl/runtime/{event/__init__.py,snapshot/__init__.py,trainer.py,trainer_options.py}`
- `src/starrygl/{config_example.py,store/__init__.py}`
- `tests/{test_open_compile_semantics.py,test_partition_assignment.py,test_runtime_targets.py,test_starrygl_core_skeleton.py}`
- `docs/design/current/execution_spine.md`
- `docs/migration/status_log.md`

Verification:

- Focused compile/target/store suite: 67 passed.
- Full unit suite: 272 passed, 12 skipped.
- Full source/test compile check passes; every touched production module is at
  or below 500 physical lines.

Unresolved risks:

- Existing artifacts must retain the prepared invariant that `time_ptr_2` is
  the train/validation/test concatenation used to build Snapshot-CSC rows.
- End-to-end distributed throughput still requires the planned parity
  benchmarks; this change removes Python planning work but does not benchmark
  native sampling or communication kernels.

## 2026-08-31: Remove Retained Per-Batch Runtime Plans

Latest state:

- Correction complete. Event and Snapshot loader paths iterate the prepared
  global `time_ptr_2` row id `i` directly.
- `RuntimeBatchPlan`, the full per-split plan list, `DatasetBatch.plan`, and
  `WindowContext` were deleted. Epoch-static policy remains in the loader;
  final Batch context is `BatchMeta(step=i, split=...)`.
- Snapshot history is derived on demand as `range(begin, i + 1)` plus a plain
  chunk-limit tuple. Edge supervision derives its target as `i + 1`; no target
  Snapshot id is stored in an intermediate batch object.
- `DatasetBatch` remains only as the small negative-sampling UDF value carrier
  and is no longer part of the training loader path.

Failure and corrective standard:

- The implementation treated removal of the redundant plan as optional follow-up
  work after the user had made it part of the accepted design. Future review
  units must verify the complete accepted invariant at every consumer, not stop
  when only its first visible symptom is fixed.
- This unit is complete only when the loader hot path passes integer window ids,
  epoch-static policy is not copied per batch, Snapshot history is derived from
  `i`, and no production consumer imports `RuntimeBatchPlan`.

Attempted approaches:

- Keeping the plan as a transitional compatibility object is rejected. There
  is no independent runtime information in it that cannot be derived from `i`
  and epoch-static loader options.
- Retaining `WindowContext` as debug metadata was rejected because it duplicated
  `BatchMeta.step` and split information on every final Batch.

Abstraction introduced:

- No replacement plan class. `snapshot_window` is one direct control helper
  returning only built-in `range` and tuple values required by materialization.

Efficiency alternatives considered:

- Prebuilding torch id/limit tensors, Python plan objects, or another dataclass
  was rejected. The selected path performs constant-size Python control work
  once per window; tensor graph slicing, native sampling, feature access, and
  communication remain on the existing torch/C++ paths.
- A DGL or custom C++/CUDA operator is unnecessary for deriving a short history
  range. Native work remains required only for the actual neighbor sampling and
  graph/data hot paths.

Files modified:

- `src/starrygl/batch/{source.py,__init__.py}`
- `src/starrygl/runtime/dataloader/{loader.py,materialize.py}`
- `src/starrygl/runtime/event/{__init__.py,materialize.py}`
- `src/starrygl/runtime/snapshot/{__init__.py,materialize.py,cache.py}`
- `src/starrygl/runtime/loop.py`
- `src/starrygl/{__init__.py,native/sampling_output.py}`
- `src/starrygl/view/snapshot.py`
- `src/starrygl/task/negative.py`
- `tests/{test_access_double_buffer.py,test_open_compile_semantics.py,test_runtime_materialize.py,test_runtime_targets.py,test_starrygl_task.py}`
- `docs/{STARRYGL_INTERFACE.md,OPEN_SOURCE_LAYOUT.md}`
- `docs/design/current/execution_spine.md`
- `docs/migration/status_log.md`

Verification:

- Repository search finds no current production/test/design reference to
  `RuntimeBatchPlan`, `DatasetBatch.plan`, or `WindowContext`.
- Full unit suite: 272 passed, 12 skipped.
- Full source/test compile check passes; every touched production module is at
  or below 500 physical lines.

Unresolved risks:

- This control-path deletion has not yet been measured in paper-scale
  distributed benchmarks. It removes allocations and indirection but does not
  by itself establish end-to-end speedup.

## 2026-08-31: Unify Event And Snapshot Window Entry

Latest state:

- Event and Snapshot now enter one loader spine:
  `split_window_range -> input_window -> TaskTarget -> optional native sample -> BatchRequest -> materialize`.
- Both modes use the prepared global `time_ptr_2` row as `window_id`. Event uses
  `range(i, i + 1)`; Snapshot derives its history range and chunk limits from
  the same `i` without a per-batch plan object.
- Every sampling policy constructs `TaskTarget` in Stage C. Stage B only
  attaches the physical `TargetRoute`; it no longer rebuilds full-Snapshot
  targets.
- Event rows and node-label rows are separate index spaces. Label timestamp
  assignment is cached by target construction and is absent from loader and
  sampler signatures.
- The legacy `no_sample` root-only path and the per-epoch Snapshot id dictionary
  were removed. `drop_last` now removes only a final Event batch smaller than
  the prepared `target_batch_size`.
- `ExecutionPlan.execution_order` exposes the shared prefix
  `select_input_window -> build_task_target` and the common native sampling
  step when enabled.

Attempted approaches:

- Keeping Event label-row preparation and Snapshot history preparation as two
  loader setup branches was rejected because it preserved different target
  timing behind a nominally shared loop.
- Adding another Window/Plan dataclass was rejected; `range` and a chunk-limit
  tuple contain all per-window control data.
- Rebuilding a `snapshot_id -> row` dictionary was rejected because prepare
  already guarantees Snapshot-CSC row order matches global `time_ptr_2` order.

Abstraction introduced:

- No public abstraction. The loader has small private pure functions for policy
  validation, supervised row selection, and deriving the built-in input range.

Efficiency alternatives considered:

- Window selection remains constant-size Python control work. Tensor label
  assignment uses `searchsorted`, sorting, and prefix sums once per split and is
  cached; graph neighbor sampling remains in the existing C++ native sampler.
- A DGL or custom C++/CUDA operator is unnecessary for selecting one prepared
  row or deriving a short history range. Snapshot CSC slicing, feature access,
  communication, and native sampling retain their tensor/native hot paths.
- Existing two bounded queues and launch-ahead semantics were moved unchanged
  from `loader.py` to the existing `dataloader/pipeline.py` responsibility.

Files modified:

- `src/starrygl/{plan.py,task/target.py}`
- `src/starrygl/runtime/dataloader/{loader.py,pipeline.py}`
- `src/starrygl/runtime/event/{__init__.py,materialize.py,target.py}`
- `src/starrygl/runtime/snapshot/{__init__.py,materialize.py,rows.py,target.py}`
- `tests/{test_open_compile_plan.py,test_open_compile_semantics.py,test_runtime_targets.py}`
- `docs/design/current/execution_spine.md`
- `docs/migration/status_log.md`

Verification:

- Full unit suite: 273 passed, 12 skipped.
- Regression coverage confirms complete Event tails are retained, incomplete
  tails are dropped, both plan modes share the same execution prefix, and full
  Snapshot targets are constructed before materialization.
- All touched production Python modules remain at or below 500 physical lines.

Unresolved risks:

- Paper-scale distributed throughput and native sampler parity still require
  the planned Event/Snapshot benchmark runs; this unit establishes control-flow
  and target-order parity but does not claim end-to-end performance parity.

## 2026-08-31: Bind Samplers Outside Batch Request Iteration

Latest state:

- `iter_batch_requests` is now a three-input loop over prepared windows, a bound
  sampler function, and a bound materializer function. It no longer receives or
  forwards graph, task, policy, sampling, feature, or cache configuration.
- Event creates its native sampler once before request iteration. Snapshot keeps
  its existing per-topology native sampler cache behind the bound Snapshot
  sampler because different Snapshot rows can carry different topology.
- Runtime directly consumes canonical `window_policy`, `sampling_policy`,
  `wait_policy`, and `gradient_sync` values. The duplicate loader policy
  default/validation and spelling normalization were removed.
- Snapshot rolling caches are bound once per loader and evicted from the sampled
  row itself; `min_active` is no longer passed through every request.

Attempted approaches:

- A loader configuration dataclass and a sampler adapter class were rejected;
  both would store the same epoch-static fields only to forward them again.
- Keeping generic mode dispatch helpers was rejected because each helper merely
  repeated the complete Event/Snapshot argument list.

Abstraction introduced:

- No class or public API was added. The only contract is a bound sampler callable
  and a bound materializer callable consumed by the existing request iterator.

Efficiency alternatives considered:

- The selected path uses Python closures once per epoch and keeps C++ native
  neighbor sampling unchanged. No DGL or custom C++/CUDA work is justified for
  binding fixed Python arguments.
- Snapshot sampler caching remains necessary because one global sampler cannot
  represent changing Snapshot topology. Event avoids per-window sampler setup.

Files modified:

- `src/starrygl/runtime/dataloader/loader.py`
- `src/starrygl/runtime/loop.py`
- `tests/test_runtime_targets.py`
- `tests/test_access_double_buffer.py`
- `docs/design/current/execution_spine.md`
- `docs/migration/status_log.md`

Verification:

- Targeted runtime, pipeline, and compile-semantics tests: 57 passed, 2 skipped.
- Python compile checks pass for the modified runtime and test modules.
- `runtime/dataloader/loader.py` decreased from 437 to 314 physical lines.

Unresolved risks:

- This removes Python argument forwarding and repeated Event sampler setup, but
  paper-scale throughput still requires the planned distributed benchmark.

## 2026-08-31: Confirm Prepare And Prepared-Task Contract

Latest state:

- Documented one flattened temporal input indexed by global `time_ptr_2` rows.
- Confirmed that Snapshot neighbor sampling stores TemporalCSR rather than
  SnapshotCSC and initializes one native sampler at task startup.
- Confirmed that Prepare stores the configured positive node or edge task as
  flat tensors plus one prefix pointer after partitioning.
- Edge-task identity keeps the canonical input name `edge_ids`; no separate
  `target_ids` alias is introduced.
- Training negatives remain online; deterministic evaluation/test negatives
  are cached per run and persisted only when the benchmark requires them.
- Recorded the intended straight-line interfaces from `GraphData` through
  PartitionPlan, selected graph views, flat task tensors, and StoreBundle.
- Made common-path-first lowering a repository rule: mode-specific behavior is
  bound once as a graph accessor or service and rejoins one execution spine.

Attempted approaches:

- Persisting every train/eval/test negative tensor was rejected because it adds
  storage and freezes training diversity without removing a material runtime
  cost.
- Per-window task objects and per-Snapshot sampler construction were rejected;
  direct tensor slices and one TemporalCSR sampler cover the required path.
- Separate Event/Snapshot loaders, queues, and positive-target constructors were
  rejected because only graph access differs after the prepared task slice.

Abstraction introduced:

- No runtime abstraction or implementation was added. This step records the
  agreed Prepare boundary before further code changes.

Efficiency alternatives considered:

- Task lookup is a prefix-pointer slice with no event scan or Python hot loop.
- Random training endpoints should use a GPU Torch/native operation before the
  native sampler; no custom CUDA operator is justified until profiling shows
  that operation is material.

Files modified:

- `../AGENTS.md`
- `docs/design/current/prepare_contract.md`
- `docs/design/current/execution_spine.md`
- `docs/migration/status_log.md`

Verification:

- Documentation review only; no implementation or tests changed.

Unresolved risks:

- Current runtime code has not yet been audited or changed to enforce this
  contract. That alignment remains a later, separately reviewed migration unit.

## 2026-09-01: Fix The Stage B To Stage A Contract

Latest state:

- The internal handoffs are now explicit and shared by Event/Snapshot and
  node/edge execution: Stage C emits
  `BatchRequest(CommScheduler) -> Batch | None`; Stage B keeps pending read
  handles private and emits only a dependency-ready `Batch` to Stage A.
- The bounded-stale state decision is made once in `run_epoch`. Stage B receives
  a bound state-read callable only when every active state manager is
  bounded-stale. Exact or mixed-freshness state stays entirely in Stage A.
- Stage A now always finishes the previous authoritative owner commit before an
  exact or mixed-freshness state read. The old internal
  `state_commit_wait_interval` branch was removed because it could bypass an
  exact dependency.
- Stage A retains one execution order: supervision, model encode, task loss,
  backward/gradient sync/optimizer step, model state delta, runtime commit, and
  metrics. State read/write handles do not enter `Batch` or add another queue.
- Queue depth, lookahead handshake, empty-rank participation, and the single
  global `CommScheduler` collective order are unchanged.

Attempted approaches:

- A `StageBatch`, access ticket, protocol class, and new stage package were
  rejected. The existing callable and model-facing `Batch` are the complete
  boundary values.
- Rechecking freshness independently in the loader and execution loop was
  rejected because the two stages could classify the same state dependency
  differently. One bound callable records the decision without a config
  wrapper.
- Retaining a configurable exact-state wait interval was rejected: exact means
  the previous owner commit is a mandatory dependency, not a throughput knob.

Abstraction introduced:

- No class or public API. `BatchRequest` now requires the epoch
  `CommScheduler`, and the existing Stage-B launch accepts one optional bound
  state-read callable instead of a manager, policy boolean, and generic
  launcher.

Efficiency alternatives considered:

- The change is constant-size Python orchestration at epoch/batch boundaries.
  Feature/state gathers, compaction, sampling, route packing, and collective
  communication remain in the existing Torch/native/CommScheduler paths.
- DGL or a custom C++/CUDA operator cannot improve a semantic owner-commit
  dependency. Bounded-stale reads retain Stage-B lookahead; exact reads retain
  the unavoidable Stage-A wait.

Files modified:

- `src/starrygl/runtime/dataloader/{loader.py,pipeline.py}`
- `src/starrygl/runtime/loop.py`
- `src/starrygl/config_example.py`
- `tests/{test_access_double_buffer.py,test_runtime_targets.py}`
- `docs/STARRYGL_INTERFACE.md`
- `docs/design/current/{execution_spine.md,state_await_cache.md}`
- `docs/migration/status_log.md`

Verification:

- Focused Stage/data-access/state suite: 43 passed, 8 skipped.
- Full unit suite: 274 passed, 12 skipped.
- Full source/test compile check passed.
- Modified production modules are 361, 308, 231, and 328 physical lines; all
  production Python modules remain at or below 500 lines.

Unresolved risks:

- The skipped distributed tests still require a two-rank `torchrun` run, and
  representative NCCL Event/Snapshot profiles are still required to quantify
  the Stage-B overlap.
- Mixed exact and bounded-stale managers intentionally use the conservative
  Stage-A path for every state kind until collective epochs can safely separate
  those reads.
- The prepared flat task-table alignment recorded in `prepare_contract.md`
  remains a separate runtime migration unit.

## 2026-09-01: Localize And Freeze Prepare-To-Stage-C Contracts

Latest state:

- Moved the authoritative design contracts beside their owning source modules.
  `docs/design/current` now keeps stable links and cross-module indexes only.
- Fixed one common execution contract:
  `window row -> prepared task slice -> negatives -> bound graph accessor ->
  request_queue -> materialize/await -> ready_queue -> model/task/state update`.
- Fixed the two-queue Stage protocol. Stage C owns task slicing and graph access;
  Stage B owns materialization and dependency handles; Stage A owns exact state,
  model/task execution, backward, and state commit.
- Defined complete Prepare output for ownership, packed locations, chunks,
  owner-filtered task tables, feature shards, Event, T-CSR, Snapshot-CSC, and
  their routes.
- Distinguished stable logical `edge_ids` from canonical physical `edge_rows`.
  Current `edge_feature_ids` uses are migration gaps, not another accepted
  spelling.
- Fixed the native target contract to one task-static
  `sample_neighbors(roots, scope)` operation. Event scope is per-root cutoff
  timestamp; Snapshot scope is a tensor of canonical edge-row ranges reused by
  every GNN layer.
- Replaced the speculative design-phase migration list with ordered parity
  units in `docs/migration/starrygl_migration_plan.md`.
- Confirmed Snapshot neighbor execution uses T-CSR only. Snapshot-CSC remains a
  full/chunk graph accessor with precomputed boundary Route, and rejoins the
  same Stage-B dependency path.

Attempted approaches:

- Duplicating full contract text in both `docs/design/current` and source
  directories was rejected because two authoritative copies would drift.
- Adding `GraphAccessResult`, Stage-specific Batch wrappers, sampler adapters,
  or per-window plan classes was rejected. A short
  `(blocks, node_ids, edge_rows)` tuple and the existing `BatchRequest`/`Batch`
  boundaries are sufficient.
- Keeping Event and Snapshot target construction in their physical view files
  was rejected. Prepare aligns one flat task table before either graph accessor.
- Reusing `edge_ids` for both dataset identity and feature row was rejected
  because reordered views need both meanings; the two canonical names represent
  distinct data, not aliases.

Abstraction introduced:

- No runtime class or public API was added. Module-local Markdown contracts are
  the only new structure, with one package-level contract index.

Efficiency alternatives considered:

- Prepared packed tensors and prefix pointers are retained from the Flare-style
  Snapshot path; Python object lists and repeated CSC construction are excluded
  from the target hot path.
- T-CSR neighbor/history loops, compaction, read grouping, and route construction
  are assigned to the existing native/Torch path following MemShare. Python may
  iterate windows only.
- Snapshot routes are precomputed; sampled routes are sample-driven. Both use
  one globally ordered NCCL `CommScheduler`, avoiding a second communication
  implementation.

Files written or modified:

- `src/starrygl/{CONTRACT.md,PAPER_METHOD_COVERAGE.md}`
- `src/starrygl/{partition,prepare,store,native,view,batch,task}/CONTRACT.md`
- `src/starrygl/runtime/{CONTRACT.md,CONFIG.md}`
- `src/starrygl/runtime/{dataloader,sample,snapshot,state}/CONTRACT.md`
- `docs/STARRYGL_INTERFACE.md`
- `docs/design/current/{execution_spine,prepare_contract,partition,negative_sampling,state_await_cache,runtime_default_alignment,paper_method_coverage}.md`
- `docs/design/current/migration_plan.md`
- `docs/migration/starrygl_migration_plan.md`
- `docs/migration/status_log.md`

Verification:

- Documentation-only migration; no production Python/C++ implementation was
  changed and no functional tests were required.
- Relative Markdown links and canonical term usage were checked from the source
  tree.

Unresolved risks:

- Native packed edge location decoding uses `>> 50` while Prepare uses the
  frozen 48-bit local-row contract; distributed native edge routing is blocked
  until corrected and tested.
- Python does not yet bind native edge-read packed indexes, and native Snapshot
  sampling still accepts timestamp values rather than edge-row ranges.
- Positive targets remain partly runtime-built and partly embedded in Snapshot
  slices. Prepare task-table alignment is the next implementation unit.
- Event/Snapshot loaders still branch into separate materializers. They have not
  yet been reduced to one bound graph-access tuple.
- Multi-rank correctness, throughput, memory, and convergence parity with the
  reference Event and Snapshot paths remain mandatory before paper-level
  performance claims.

## 2026-09-01: Translate Module Contracts To Chinese

Latest state:

- Rewrote all 13 module-local `CONTRACT.md` files in concise Chinese.
- Kept code identifiers in English, but explained each execution concept in
  plain language before using it as a contract term.
- Preserved the frozen Prepare, sampler, Stage A/B/C, Snapshot-CSC, task, and
  state semantics; this change does not claim implementation parity.

Attempted approaches:

- Rejected bilingual copies because two full versions would drift.
- Rejected literal line-by-line translation because it preserved repetition
  and dense terminology. Each file now follows responsibility, data flow,
  rules, and known gaps.

Abstraction introduced:

- None. This is a documentation-only rewrite.

Efficiency alternatives considered:

- No hot-path implementation changed. Existing Torch/DGL/native and globally
  ordered NCCL requirements remain unchanged.

Files written or modified:

- `src/starrygl/CONTRACT.md`
- `src/starrygl/{partition,prepare,store,native,view,batch,task}/CONTRACT.md`
- `src/starrygl/runtime/CONTRACT.md`
- `src/starrygl/runtime/{dataloader,sample,snapshot,state}/CONTRACT.md`
- `docs/migration/status_log.md`

Verification:

- Checked all 13 contract files for relative links, balanced Markdown fences,
  Chinese content, and identity-related terms.
- No production Python/C++ file changed, so functional tests were not run.

Unresolved risks:

- The implementation gaps listed in each contract remain open. The Chinese
  rewrite only makes the agreed target easier to review.

## 2026-09-01: Audit And Freeze The Paper-v1 Runtime Contract

Latest state:

- Audited the package, DataLoader, T-CSR, Snapshot-CSC, and state contracts
  against the local paper method/experiment sources, the current runtime, and
  the MemShare/Flare reference data flows. Release readiness remains `No-Go`
  until the implementation and performance gates below pass.
- Kept one Stage C/B/A spine, two depth-one queues, one `Batch`, one
  `GraphBlock`, and one `CommScheduler`. Replaced the target per-window
  `BatchRequest(CommScheduler)` callable boundary with the short tuple
  `(window_id, targets, blocks, node_ids, edge_rows)`; static services bind once.
- Fixed the bounded-state contract: StateManager/cache owns `run_generation`,
  `producer_window`, `cache_sync_window`, and `committed_through`. Filter
  `max_skip` no longer substitutes for `bounded(K)`, and versions remain out of
  `Batch`.
- Fixed one plan-driven collective order across Stage-B prefetch, exact state,
  layerwise forward/reverse, endpoint collection, gradient synchronization, and
  state commit. Empty ranks submit empty payloads; one dispatcher, not thread
  arrival, launches the next slot.
- Fixed the v1 physical pipeline: CPU/native negatives and sampling with pinned
  output, non-blocking Stage-B H2D/NCCL streams, Stage-A compute stream, and
  event-owned buffer reuse. No third queue or Stage wrapper was introduced.
- Fixed Event sampling identity to `(node_id, cutoff_ts)`, Snapshot versioned
  feature identity to `(snapshot_id,node_id)`, the dynamic
  counts/request/payload Route protocol, and the paper boundary-retention
  formulas.
- Fixed Snapshot execution ownership: runtime-owned shared graph operators hold
  layerwise Route/Await; models only provide tensor math. Overlapping Snapshot
  training state is Batch-local, EvolveGCN model state is replicated, and exact
  coupled state is satisfied per temporal unit rather than once per Batch.
- Chose the evaluated Flare chunk-decay behavior for paper-v1: one permutation
  per epoch, local induced/empty-Route decayed histories, and prepared Route for
  recent full Snapshots. The paper's per-iteration wording must be aligned or
  rebenchmarked.
- Unified layerwise and historical-cache control only at Route, planned slot,
  CUDA event/await, scatter, and profiling. Their storage, freshness, autograd,
  and lifetime remain intentionally distinct.
- Added explicit paper-v1 gates, including the published 4-to-16-GPU targets:
  TGN 3.23x, TGAT 4.17x, and T-GCN 2.84x, plus quality, convergence, peak memory,
  no-OOM, empty-rank, skewed-arrival, and trace-overlap evidence.

Attempted approaches:

- A universal CacheManager for layer embeddings and temporal state was rejected:
  it would either retain autograd graphs across windows or erase state freshness.
- A third queue, Stage-specific Batch classes, per-window request/plan objects,
  and a RouteBuilder hierarchy were rejected. Two queues, a short tuple, native
  read/scatter tensors, and existing Route/CommScheduler are sufficient.
- Receiver-only chunk Route cropping was rejected because peer send/recv sizes
  are not symmetric. Per-step peer-aware Route rebuilding was also rejected for
  v1; the evaluated Flare local-prefix behavior is simpler and benchmarked.
- GPU negative generation followed by a CPU sampler was rejected because it
  adds a synchronized D2H root transfer. Placement follows the bound sampler.
- Local ticket allocation was rejected as a global order guarantee; ranks can
  reach Stage A/B from different threads in different orders.

Abstraction introduced:

- No production class, registry, provider hierarchy, or public API was added.
  The only added internal semantics are a plan-ordered short slot tuple and
  StateManager-owned version/watermark tensors, both required by existing
  distributed correctness claims.

Efficiency alternatives considered:

- CPU native sampling plus pinned/non-blocking Torch transfer is selected for
  v1 because the existing sampler is CPU-native. DGL does not provide the
  causal multi-history sampler or requester/owner Route protocol; Python loops
  are excluded. A GPU sampler is deferred until it can replace the whole path.
- Request grouping, compaction, scatter, filter, version compare, and duplicate
  state reduction remain Torch/native tensor operations. A custom C++/CUDA
  operator is justified only if profiling shows those operators dominate.
- Full Snapshot reuses packed CSC, prepared Route, and Flare layerwise ordering;
  no repeated DGL block construction or runtime graph scan is accepted.
- NCCL collective launch remains globally scheduled. Free-form p2p/background
  cache broadcast was rejected because it cannot prove order or deadlock safety.

Files modified:

- `src/starrygl/{CONTRACT.md,PAPER_METHOD_COVERAGE.md}`
- `src/starrygl/{partition,view,batch,task}/CONTRACT.md`
- `src/starrygl/runtime/CONTRACT.md`
- `src/starrygl/runtime/{dataloader,sample,snapshot,state}/CONTRACT.md`
- `docs/migration/{starrygl_migration_plan.md,status_log.md}`

Verification:

- Paper experiment lowering contracts: 42 passed.
- DataLoader/target/state/compile-plan regression set: 55 passed, 7 skipped
  because distributed launch was not active.
- Full unit suite: 274 passed, 12 skipped; skipped cases require distributed
  launch and are still part of the release blockers below.
- Checked all 15 authoritative contract/coverage/migration documents: no broken
  relative Markdown links and no unbalanced fenced blocks.
- Documentation only; no production Python/C++ implementation was changed.

Unresolved risks:

- The current scheduler still allocates tickets by local arrival and lacks the
  plan-driven dispatcher; two-rank skew/empty-slot tests are mandatory.
- Historical reads still have no producer-age check, GPU filter storage is not
  guaranteed, and memory/mailbox/model-state coherence tests remain open.
- Native 48-bit edge decoding, Snapshot `[H,2]` sampling, Event cutoff identity,
  boundary-retention tau, and counts/request/payload Route parity remain open.
- Stage-B H2D is not yet pinned/non-blocking, scheduler/handles still leak through
  some block caches, and empty-target full Stage-A execution is not implemented.
- Paper runnable configs still lack JODIE/APAN/MPNN-LSTM/EvolveGCN representatives;
  current plan tests and four JSON files are not execution/performance evidence.
- No paper-scale NCCL, convergence, peak-memory, no-OOM, or throughput run was
  performed in this documentation step. Paper-v1 therefore remains `No-Go`.

## 2026-09-02: Record The Unified Runtime Interface Discussion

Latest state:

- Expanded `docs/STARRYGL_INTERFACE.md` from a public-entry stub into the
  readable Paper-v1 interface discussion summary. Module-local contracts remain
  authoritative, and production-code changes are still not authorized.
- Recorded the confirmed single-spine rule: model families reuse operator and
  dependency interfaces while `compile` generates their static execution order;
  they do not select another runtime.
- Recorded the confirmed Flare-compatible chunk rule: one reproducible
  permutation per epoch/rank, shared by every Batch in that epoch, with nested
  prefixes for older Snapshots and the recent `F` Snapshots kept full.
- Summarized the proposed `runtime_cell` operator boundary and the compiled
  `node_recurrent`, `neighbor_recurrent`, and `model_recurrent` orders for
  continued review. These signatures are explicitly marked under review.
- Added one source-to-target reference map for MemShare Event/native/cache work
  and FlareDTDG Snapshot/layerwise/EvolveGCN work without adding either legacy
  execution stack as a dependency.

Attempted approaches:

- Rejected a new monolithic authoritative contract because it would duplicate
  and drift from the module-local contracts. The existing interface overview is
  used as the readable summary and links to the detailed owners.
- Rejected writing the earlier per-Batch chunk proposal. The confirmed contract
  is per epoch and already matches the evaluated Flare reference behavior.

Abstraction introduced:

- None in production code. The document records a proposed minimal internal
  operator protocol for review; no class or public API was added.

Efficiency alternatives considered:

- Reuse MemShare native sampling, compact Route, pinned-buffer, mailbox, and
  historical-cache data flows before writing new hot-path code.
- Reuse FlareDTDG Snapshot-CSC, autograd Route, layerwise order, nested-prefix,
  and EvolveGCN tensor math without copying its Engine or Loader boundary.
- Keep one dispatcher and two queues; no registry, adapter, Stage wrapper,
  provider hierarchy, third queue, or universal cache abstraction is proposed.

Files written or modified:

- `docs/STARRYGL_INTERFACE.md`
- `docs/migration/status_log.md`

Verification:

- Documentation-only update; no production Python/C++ code changed and no
  functional tests were run.
- Markdown relative links, fenced blocks, and the per-epoch chunk wording were
  checked from the source tree.

Unresolved risks:

- The internal operator signatures and exact scan result remain discussion
  items until explicitly approved and written into their module contracts.
- All implementation and paper-scale blockers listed in the interface summary
  and module contracts remain open; release readiness is still `No-Go`.

## 2026-09-02: Translate The Unified Interface Summary To Chinese

Latest state:

- Translated `docs/STARRYGL_INTERFACE.md` in place so the complete unified
  runtime discussion can be reviewed in Chinese.
- Preserved code identifiers, formulas, signatures, confirmed decisions,
  under-review markers, reference paths, and release blockers without changing
  their semantics.
- Kept one document rather than adding an English/Chinese pair that would drift.

Attempted approaches:

- Rejected a second translated copy. The existing summary is now Chinese, while
  identifiers remain in their canonical code spelling.

Abstraction introduced:

- None. This is a documentation-only translation.

Efficiency alternatives considered:

- No hot-path choice changed. The summary still requires MemShare/Flare data
  flow reuse, one dispatcher, two queues, and native/vectorized hot paths.

Files written or modified:

- `docs/STARRYGL_INTERFACE.md`
- `docs/migration/status_log.md`

Verification:

- Documentation-only update; no production Python/C++ code changed and no
  functional tests were run.
- Relative links, fenced blocks, confirmed per-epoch wording, and Chinese
  coverage were checked from the source tree.

Unresolved risks:

- Internal operator signatures remain under review.
- All implementation and paper-scale `No-Go` items remain open.

## 2026-09-02: Replace Bounded State With Selective Refresh

Latest state:

- Recorded the final decision to remove the independent `bounded(K)` contract.
  The target now supports exact reads and an explicit `stale_cache`
  approximation with no maximum producer-age guarantee.
- Moved the only refresh limit to `runtime.temporal_state.filter.max_skip`:
  after that many consecutive candidate refreshes are filtered, the next
  candidate is sent unconditionally.
- Removed generation/producer-window validation from the target contract. The
  static dispatcher orders refreshes; reset/checkpoint/split/error paths must
  drain pending work and clear stale cache and skip counters.
- Updated the manuscript wording and equation so they no longer claim bounded
  staleness. Historical status entries remain unchanged as an audit trail and
  are superseded by this decision.

Attempted approaches:

- Rejected retaining `bounded_stale` as an alias because its name would promise
  a guarantee the runtime no longer checks.
- Rejected keeping unused generation/version fields. They become necessary only
  if a future design admits free-form or out-of-order refresh messages.

Abstraction introduced:

- None. The decision deletes a consistency mode and reuses the existing refresh
  filter, static slot order, cache, and reset lifecycle.

Efficiency alternatives considered:

- Selected the existing MemShare-style change filter plus forced refresh after
  `max_skip`; no per-read age check or version tensor is added to the hot path.
- Exact owner/watermark reads remain available when correctness is required or
  when stale cache misses.

Files written or modified:

- `docs/STARRYGL_INTERFACE.md`
- `src/starrygl/CONTRACT.md`
- `src/starrygl/batch/CONTRACT.md`
- `src/starrygl/PAPER_METHOD_COVERAGE.md`
- `src/starrygl/runtime/{CONTRACT.md,CONFIG.md}`
- `src/starrygl/runtime/{dataloader,snapshot,state}/CONTRACT.md`
- `docs/migration/{starrygl_migration_plan.md,status_log.md}`
- `../paper_method/method.tex`

Verification:

- Documentation-only change; no production Python/C++ code changed and no
  functional tests were run.
- Checked target/summary/manuscript wording for remaining active bounded-state
  claims, relative Markdown links, and balanced Markdown/LaTeX delimiters.

Unresolved risks:

- Production config and plan code still use `bounded_stale/max_staleness` and
  must migrate to `stale_cache/filter.max_skip` during implementation.
- The required drain-and-clear lifecycle and exact/stale-cache multi-rank quality
  comparison are not implemented; paper release readiness remains `No-Go`.

## 2026-09-02: Separate Snapshot Overlap, Split, Epoch, And Checkpoint Lifecycles

Latest state:

- Explained why overlapping Snapshot windows cannot carry their final recurrent
  state: `[1,2,3] -> [2,3,4]` would feed `state@3` into Snapshot 2, leak future
  information, and advance Snapshots 2 and 3 twice.
- Confirmed that non-overlapping chronological windows may carry state because
  the previous final state is the next window's exact predecessor.
- Corrected the earlier lifecycle wording: a split boundary drains required
  work but preserves temporal state and cache; only a new epoch or explicit
  `reset()` reinitializes temporal runtime state.
- Defined checkpoint save as drain-and-serialize without reset. Resume at the
  same cursor restores the complete runtime state; a weights-only or new-epoch
  load resets and replays from the epoch boundary.

Attempted approaches:

- Rejected treating every split as an independent sequence because split only
  selects supervision and does not cut the temporal history.
- Rejected carrying an overlapping Batch's final recurrent state because it is
  not the predecessor state of the next Batch's first temporal unit.
- Rejected clearing state on checkpoint save because that would make an
  observability/persistence action change training semantics.

Abstraction introduced:

- None. This documentation change separates four lifecycle boundaries using
  the existing state, dispatcher, drain, reset, and checkpoint concepts.

Efficiency alternatives considered:

- Reuse the already-required boundary drain and serialized runtime tensors; no
  generation protocol, second state manager, or per-window wrapper is added.
- Overlapping windows keep Batch-local recurrent tensors; non-overlapping
  execution reuses the predecessor state without recomputation.

Files modified:

- `docs/STARRYGL_INTERFACE.md`
- `src/starrygl/runtime/CONTRACT.md`
- `src/starrygl/runtime/snapshot/CONTRACT.md`
- `src/starrygl/runtime/state/CONTRACT.md`
- `docs/migration/starrygl_migration_plan.md`
- `docs/migration/status_log.md`

Verification:

- Documentation-only change; no production Python/C++ code changed and no
  functional tests were run.
- Checked the active summary and runtime contracts for consistent overlap,
  split, epoch, reset, and checkpoint wording.

Unresolved risks:

- Production reset/checkpoint code does not yet implement or test this complete
  lifecycle, especially mid-epoch resume of owner/cache/filter state.
- A standalone split evaluation must restore a complete split-boundary state or
  replay from the epoch start; starting it from empty state is invalid.

## 2026-09-02: Allow Target-Causal Chunk Boundary Carry

Latest state:

- Corrected the previous blanket prohibition on carry between overlapping
  Snapshot batches. Paper-v1 chunk decay may carry only the detached embedding
  or state of the previous Batch's final Snapshot.
- For `[1,2,3] -> [2,3,4]`, `E3` may seed or fill omitted chunks while 2 and 3
  are recomputed. Only Snapshot 4 is supervised/exposed/committed, so no value
  produced after target time 4 is consumed.
- Distinguished within-Batch layerwise exchange for recent full Snapshots from
  the single final-Snapshot payload carried across Batches.
- The carry reuses the existing state commit/cache-refresh Route and static
  slot; it is preserved across split and cleared across epoch reset.

Attempted approaches:

- Rejected requiring strict causality for unsupervised intermediate Snapshots;
  that would discard the history needed to fill chunk-decayed rows.
- Rejected broadcasting every intermediate embedding. Only the final Snapshot
  is needed as the next Batch boundary and the reference path detaches reused
  historical state.
- Kept the carry disabled when an intermediate Snapshot has loss, metric,
  external output, or state commit, because that would expose future-relative
  intermediate values.

Abstraction introduced:

- None. `chunk_decay + latest-target-only` lowers to the existing dependency,
  Route, state commit/cache-refresh slot, and StateManager storage.

Efficiency alternatives considered:

- Reusing one detached final-Snapshot payload avoids retaining the prior Batch
  autograd graph or communicating every historical Snapshot embedding.
- No new cache manager, cross-Batch wrapper, consistency mode, or runtime path
  is added.

Files modified:

- `docs/STARRYGL_INTERFACE.md`
- `src/starrygl/runtime/CONTRACT.md`
- `src/starrygl/runtime/snapshot/CONTRACT.md`
- `src/starrygl/runtime/state/CONTRACT.md`
- `docs/migration/starrygl_migration_plan.md`
- `docs/migration/status_log.md`

Verification:

- Compared the contract with FlareDTDG's rolling `RNNStateManager`, including
  detached state storage and missing-row mixing.
- Documentation-only change; no production Python/C++ code changed and no
  functional tests were run.

Unresolved risks:

- Compile does not yet enforce latest-Snapshot-only supervision before enabling
  overlapping chunk boundary carry.
- The runtime does not yet prove producer time `<` final target time or restrict
  cross-Batch communication to only the final Snapshot payload.

## 2026-09-02: Replace Per-Batch Requests With One Access Tuple

Latest state:

- Removed `BatchRequest`, the Event/Snapshot sample wrappers, their per-window
  `partial` callables, and the separate mode materializers from the production
  loader path.
- The epoch binds one Event or Snapshot graph accessor. Both now emit
  `(window_id, targets, blocks, node_ids, edge_rows)` and rejoin one common
  Batch materializer, feature/state dependency launch, model, task, backward,
  and state-update path.
- Event and Snapshot-neighbor reuse `access_native_graphs`; full/chunk Snapshot
  selects prepared CSC rows directly. This is one execution spine, not a
  CTDG/DTDG runtime switch.
- The request and ready queues remain depth one. Request capacity is reserved
  before the next accessor invocation, so a blocked producer cannot sample an
  extra window outside its bounded window.
- Snapshot access no longer builds `window_tasks`: Paper-v1 production batches
  supervise only the final Snapshot. `chunk_decay` training uses the existing
  persistent state manager and commits the model's detached final state;
  ordinary `full_snapshot` training remains Batch-local.

Attempted approaches:

- Rejected retaining compatibility aliases for the deleted request/sample
  wrappers; tests and callers were moved directly to the tuple boundary.
- Rejected forcing Snapshot-CSC through the temporal sampler. Prepared CSC and
  native T-CSR differ only inside the epoch-bound accessor.
- Rejected another accessor class, Stage object, registry, or carry manager.
  A bound function, one tuple, and the existing StateManager are sufficient.

Abstraction introduced:

- `AccessedWindow` is a type alias for the existing five-value tuple, not a
  runtime object. `access_native_graphs` is the shared native sampling operation
  used by the two real Event/Snapshot-neighbor call sites.

Efficiency alternatives considered:

- Reused Torch tensor operations and the existing native sampler instead of
  adding Python node/edge loops. Full Snapshot continues to reuse prepared CSC
  and its rolling topology cache.
- Kept exactly two depth-one queues and removed closure allocation per Batch.
  Native `[H,2]` sampling and pinned/non-blocking H2D remain C++/CUDA work; this
  change does not emulate either with Python loops.

Files modified:

- `src/starrygl/runtime/dataloader/loader.py`
- `src/starrygl/runtime/dataloader/materialize.py`
- `src/starrygl/runtime/dataloader/pipeline.py`
- `src/starrygl/runtime/event/materialize.py`
- `src/starrygl/runtime/snapshot/materialize.py`
- `src/starrygl/runtime/sample/__init__.py`
- `src/starrygl/runtime/loop.py`
- `tests/test_access_double_buffer.py`
- `tests/test_runtime_targets.py`
- `tests/test_starrygl_model.py`
- `docs/STARRYGL_INTERFACE.md`
- `src/starrygl/runtime/CONTRACT.md`
- `src/starrygl/runtime/dataloader/CONTRACT.md`
- `docs/migration/starrygl_migration_plan.md`
- `docs/migration/status_log.md`

Verification:

- `python -m py_compile` passed for all modified production and focused test
  modules.
- Focused queue/access/carry tests: `28 passed, 2 skipped`.
- Full suite after the final queue-capacity regression: `275 passed, 12
  skipped`.

Unresolved risks:

- Task targets still come from migration-era Event/Snapshot constructors rather
  than one prepared task table consumed before graph access.
- The native sampler still loops over Snapshot histories in Python and does not
  yet accept one native `[H,2]` range request or return the complete dynamic
  counts/request/payload Route tensors.
- H2D uses ordinary `.to(device)` and does not yet provide pinned buffers,
  non-blocking prefetch-stream ownership, CUDA events, or `record_stream`
  lifetime guarantees.
- `CommScheduler` is still arrival-ticket driven rather than plan-driven; full
  empty-rank forward/reverse/gradient/state slot parity and paper-scale
  performance gates remain open.

## 2026-09-03: Clean Batch Carriers And Bind The Unified DataLoader

Latest state:

- Replaced the function-style loader with one public-style `DataLoader` class.
  Its constructor binds the Store, native sampler, Event/Snapshot accessor,
  Snapshot rolling caches, communication scheduler, device and prefetch
  stream once. `__iter__` owns two depth-one queues and the Stage-C/Stage-B
  workers; queue values remain the short access tuple and the final `Batch`.
- Removed unused or forwarding-only carriers: `BatchMeta`, `DatasetBatch`,
  `BatchRequest`, old Event/negative-sampling wrappers, the forwarding
  `batch/source.py` module, test-only staged-feature APIs, the empty
  `CoupledRecurrentCell`, and unused negative-pool compatibility fields.
  The live raw Event carrier is now `EventRows`.
- Removed loader recognition of special node-label strings. Runtime uses the
  integer `node_label_horizon`; old `node_label_source` is translated only at
  graph/artifact read boundaries.
- Stage B now pins CPU tensor buffers and performs non-blocking H2D on one
  loader-owned CUDA prefetch stream. It launches feature payload collectives
  before waiting and hands only dependency-ready `Batch` values to Stage A.
- Fixed Event node-feature compaction: sampler rows are deduplicated by static
  node id, fetched once, and scattered back to the original layout. A compact
  local handle is no longer mistaken for a completed handle and forced to
  gather during launch.
- Reused one CUDA communication stream per `CommScheduler` for non-autograd
  layerwise Route. The Flare-style coroutine scheduler is covered by a test
  proving all same-layer routes launch before the first wait.
- Snapshot-CSC edge prediction remains on the same outer chain as Event and
  Snapshot-neighbor execution. Its target construction is still bound inside
  the Snapshot accessor and is a prepared-task-table migration gap, not a
  second loader/runtime.
- Confirmed state allocation occurs once before an epoch from
  `ExecutionPlan.state_dependencies`; models return `StateDelta`, and runtime
  validates and commits it after backward. Distributed all-to-all payloads use
  `async_op=True` handles and explicit finish/drain points; dynamic count and
  request-node discovery are not yet fully asynchronous.

Attempted approaches:

- Rejected adapting the MemShare or Flare loader classes directly because that
  would import their wrappers and create parallel Event/Snapshot execution
  stacks. Reused their launch-before-wait, pinned-transfer, compact-layout and
  layerwise coroutine patterns inside the existing StarryGL spine.
- Rejected keeping test-only staged feature fetch objects as a possible future
  optimization. They had no production caller and falsely described a second
  staging protocol.
- Rejected another Stage/request/result class. Standard-library queues,
  semaphores and a bound loader already express the required ownership.
- Kept dynamic count/request completion synchronous for now instead of adding
  a free-form background communication thread that could violate collective
  order. The static `CommPlan` must resolve this globally.

Abstraction introduced:

- `DataLoader` is the one necessary resource-owning boundary, analogous in
  shape to FlareDTDG `STGraphLoader` and MemShare `DistributedDataLoader`. No
  registry, adapter, provider hierarchy, Stage wrapper, cache base class or
  second runtime was added.

Efficiency alternatives considered:

- Selected vectorized `torch.unique(..., return_inverse=True)` for static node
  feature dedup and one `torch.unique`/`searchsorted` pass for edge features.
  State reads continue to compact `(node_id, cutoff_ts)` rather than
  incorrectly merging temporal identities.
- Kept the existing native sampler for per-neighbor work. Python only
  orchestrates windows and queues; it does not add per-node, per-edge or
  per-peer loops.
- Matched MemShare's feature launch/finish split: response payload communication
  is launched with an asynchronous distributed work handle and completed in
  Stage B while Stage A executes the prior batch.
- Matched FlareDTDG's layerwise send/yield/receive ordering. Routed windows are
  launched first, all coroutines yield before waits, and empty prepared routes
  still retain their collective epoch semantics.
- DGL/PyTorch operators were sufficient for current compaction and H2D. A
  custom C++/CUDA buffer manager was not added because reusable native pinned
  output and native `[H,2]` Route generation are not implemented yet.

Files written, modified, or removed:

- Batch/task/model: `src/starrygl/{__init__.py,batch/__init__.py}`,
  removed `src/starrygl/batch/source.py`,
  `src/starrygl/task/{__init__.py,negative.py,target.py}`,
  `src/starrygl/model/{tgat.py,tgn.py,tgn_mailbox.py}`.
- Loader/runtime: `src/starrygl/runtime/{comm.py,epoch.py,exchange.py,loop.py,trainer.py}`,
  `src/starrygl/runtime/dataloader/{__init__.py,blocks.py,features.py,loader.py,materialize.py,pipeline.py}`,
  `src/starrygl/runtime/event/{features.py,materialize.py}`,
  `src/starrygl/runtime/snapshot/{materialize.py,scan.py}` and
  `src/starrygl/runtime/state/access.py`.
- Artifact/store/view: `src/starrygl/native/sampling_output.py`,
  `src/starrygl/prepare/data.py`, `src/starrygl/store/{feature.py,graph.py}`,
  `src/starrygl/tools/convert_to_starrygl_format.py` and
  `src/starrygl/view/snapshot.py`.
- Tests: `tests/test_access_double_buffer.py`, `tests/test_data_conversion.py`,
  `tests/test_dataloader_features.py`, `tests/test_open_compile_semantics.py`,
  `tests/test_open_public_api.py`, `tests/test_runtime_access.py`,
  `tests/test_runtime_exchange.py`, `tests/test_runtime_materialize.py`,
  `tests/test_runtime_targets.py`, `tests/test_starrygl_model.py`,
  `tests/test_starrygl_store.py` and `tests/test_starrygl_task.py`.
- Contracts: `docs/STARRYGL_INTERFACE.md`,
  `docs/migration/starrygl_migration_plan.md`,
  `src/starrygl/{batch,runtime,runtime/dataloader,runtime/sample,store,view}/CONTRACT.md`
  and this status log.

Verification:

- Focused DataLoader, feature and model scheduling regression:
  `61 passed, 6 skipped`.
- Final full suite: `266 passed, 12 skipped`.
- Modified production modules pass `python -m py_compile`; every production
  Python module remains at or below 500 physical lines (`runtime/comm.py` is
  498 lines and `runtime/dataloader/loader.py` is 443 lines).
- Active production/tests contain no references to the removed carriers or
  staged-feature protocol. The only remaining `node_label_source` reads are
  the two intentional old-input/artifact translation boundaries.

Unresolved risks:

- `CommScheduler` still assigns arrival-order tickets rather than consuming a
  compile-generated static slot plan. `CommScheduler` is also temporarily read
  through `GraphBlock.cache` by layer/endpoint operators instead of being owned
  solely by a runtime executor.
- Dynamic feature/state Route still synchronously exchanges counts and waits
  for request nodes before it can launch response payloads. Native `[H,2]`
  history plus complete counts/request/payload Route output remains required.
- Pinning is currently per batch. Native reusable pinned buffers,
  `record_stream`-safe buffer recycling and a GPU overlap trace are still
  required before claiming optimal overlap.
- Prepared `task_ptr/payload`, full empty-rank static-slot parity, real state
  version/watermark lifecycle, coupled DTDG/EvolveGCN multi-rank parity and the
  paper-scale throughput gates remain open. Unit tests do not establish the
  TGN/TGAT/T-GCN speedups.

## 2026-09-03: Move All Supervision To The Prepared Task Table

Latest state:

- Prepare now writes one owner-local flat `task_ptr[K+1]` and columnar task
  payload into every `label_RRR.pt`. Node payload contains `node_ids`, optional
  labels and Event cutoff timestamps; edge payload contains `src`, `dst`,
  stable `edge_ids`, physical `edge_rows`, optional labels and Event cutoff
  timestamps.
- Snapshot successor targets and node-label horizon are encoded in task rows.
  The final unavailable row of every split is empty, so a train row cannot
  consume a validation target. Flare-style Web labels remain read at row `k`
  because conversion has already shifted `label[k+1]` into that row.
- Event, Snapshot-neighbor and Snapshot-CSC now use the same sequence:
  `LabelStore.task_slice -> build_window_task_target -> graph accessor ->
  attach_target_route`. `DataLoader` no longer receives or parses a task name.
- Removed Snapshot-CSC `target_*` tensors, its mode-specific target constructor,
  runtime Event label-window reconstruction, per-target owner tensors, and
  unused persisted label horizon/timestamp/split/edge-id metadata.
- Artifact signatures now include the task name and artifact readiness requires
  every rank's label shard, preventing a task from reusing another task's table.

Attempted approaches:

- Rejected keeping the task-specific Snapshot last-row branch as a migration
  fallback. Missing task tables fail once at `LabelStore.task_slice` and require
  rebuilding the artifact.
- Rejected a new task-batch/context wrapper. The existing `LabelStore`, flat
  tensors and `TaskTarget` are sufficient.
- Rejected copying target tensors into Snapshot-CSC. Task ownership and graph
  compute ownership are independent and now remain in their respective planes.

Abstraction introduced:

- Added only the distinct Prepare responsibility `prepare/task.py`; it builds
  flat owner task columns. `store/label.py` separates label artifact assembly
  from feature artifact assembly so both production modules stay below 500
  lines. No registry, adapter, loader, queue, runtime or cache abstraction was
  added.

Efficiency alternatives considered:

- Selected vectorized Torch owner masks, `searchsorted`, index selection and
  flat concatenation at Prepare time. Python only loops over ranks and window
  rows, which are Prepare/control boundaries rather than node/edge hot loops.
- DGL operators do not add value for columnar supervision packing. A custom
  C++/CUDA task-table operator would only move offline tensor indexing and is
  deferred unless Prepare profiling shows it material.
- Runtime reads two pointer values and tensor views; it performs no label scan,
  target graph lookup or task-name branch before graph access.

Files written, modified, or removed:

- Added `src/starrygl/prepare/task.py` and `src/starrygl/store/label.py`.
- Modified `src/starrygl/prepare/{build_part_graph.py,snapshot_csc.py}`,
  `src/starrygl/store/{__init__.py,feature.py,graph.py}`,
  `src/starrygl/task/{__init__.py,target.py}`,
  `src/starrygl/runtime/{loop.py,trainer.py}`,
  `src/starrygl/runtime/dataloader/loader.py`,
  `src/starrygl/runtime/event/{__init__.py,materialize.py,target.py}` and
  `src/starrygl/runtime/snapshot/materialize.py`; removed the now-empty
  mode-specific `src/starrygl/runtime/snapshot/target.py`.
- Updated focused tests, the interface overview, Prepare/Store/Batch/Task/
  Runtime/DataLoader/Snapshot contracts, and the migration plan.

Verification:

- Focused task/store/loader tests: `75 passed, 2 skipped`.
- Full suite: `264 passed, 12 skipped`.
- Direct prepared-artifact smokes confirm Event edge ids `[10,11]` and Snapshot
  windows `[0,1,2]` with successor target `[11]` followed by an empty final row.
- All production Python modules remain at or below 500 physical lines.

Unresolved risks:

- `TaskTarget` still carries transitional duplicate `target_ids/edge_ids` and a
  per-window negative-pool object; remove them only with the model/task call
  sites that still consume them.
- Event/GraphBlock edge feature paths still need complete stable `edge_ids`
  versus physical `edge_rows` naming parity.
- Plan-driven collective slots, native `[H,2]` history and dynamic Route,
  reusable pinned buffers, real state versions, empty-rank multi-GPU parity and
  paper-scale throughput gates remain open. This change does not claim paper
  performance parity.

## 2026-09-11: Flickr Coupled-State Ablation — 100-Epoch Pilot Complete

- User requires current starrygl-open, Flickr node regression, DCRNN and
  GConvGRU. Legacy logs are excluded from new experimental results.
- Traced shared Prepare/Store/DataLoader/Batch/state/scan/task/commit path;
  recorded the cell and dependency-materializer boundary in
  `docs/design/current/flickr_coupled_ablation.md` before implementation.
- Planned changes: direct DCRNN cell, common runtime materializer injection,
  package CLI for a controlled delay experiment using existing run_epoch.
- Reuse Torch/DGL operators and current increment estimator; no custom kernel
  or new training loop. Deliberate limitation: full single-rank snapshots for
  the first accuracy experiment, with explicit rejection of partial diffusion.
- Initial checks: three GConvGRU/coupled-delay/compensation tests pass. CUDA
  works outside the sandbox. Raw Flickr conversion started with current code.
- Risks: no convergence result yet; snapshot delay is distinct from the
  publisher-local skipped-refresh bound; history must persist across batches;
  exact checkpoint replay and disjoint evaluation splits must be verified.

Implementation/check update:

- Added direct `model/dcrnn.py`, reused GConvGRU output/state/compensation,
  and updated model exports, `runtime/builders.py`, `spec.py`, `plan.py`.
- Added `runtime/state/delay.py` and a bound materializer argument in the
  existing `runtime/{loop.py,snapshot/scan.py}`; no new batch training loop.
- Added `cli/coupled_ablation.py` with source hashes, exact validation replay,
  validation-selected checkpoints and separate exact/operational test scoring.
- Added `tests/test_coupled_ablation.py`. The end-to-end smoke found and fixed
  a common-loop bug: unlabeled hydrated snapshot boundaries must advance state
  even when there is no loss target.
- Full suite: `273 passed, 14 skipped`. AST and 500-line module checks pass.
  Tiny complete train/validation/test probe passes. Complete Flickr raw
  conversion (33,140,017 edges) and single-rank Prepare finish successfully.
- The user asked whether this is the existing actual stale path. Clarified:
  this CLI is fixed-delay injection, distinct from shared-hot stale reads.
  No full-Flickr training or new convergence results are claimed. Actual
  shared-hot ablation still requires multi-rank reads and DCRNN reverse-route
  support; local DCRNN explicitly rejects partial snapshots.
- Historical TGM plots created during initial inventory are excluded from
  this experiment, as recorded in their directory README.

Actual shared-hot follow-up (2026-09-11):

- User clarified that actual `bounded_stale/shared_hot` returns are primary.
  Replaced the experiment CLI's fixed-delay execution with the existing
  `build_state_managers -> AsyncMemoryCommitter -> materialize_bounded` path.
  Shared Prepare/window/task/Batch/hydrate/scan/loss/commit spine is retained;
  only DCRNN's gate-to-candidate stage needs an existing autograd Route/await.
  This boundary was recorded in the design note before implementation.
- Added bidirectional diffusion buffers in `prepare/snapshot_csc.py`, selected
  through `plan.py` and `prepare/build_part_graph.py`; preserve these buffers
  across `store/{feature,artifact}.py` and `runtime/dataloader/blocks.py`.
  Runtime `snapshot/layerwise.py` exchanges reset gates; DCRNN math uses DGL.
- Fixed root hydration ordering in `runtime/state/access.py`: unique queries
  still need inverse expansion when their node order changes. GConvGRU commits
  owned state-table rows even at unlabeled boundaries and stamps snapshot s+1
  as producer version. Temporal features use FeatureManager's shard node map,
  not snapshot-local feature-row offsets (`runtime/snapshot/features.py`).
- CLI uses two ranks, 10% hot nodes, full snapshots, chronological 40/20/40
  splits, actual-policy validation selection and exact/actual test replay.
  Added read-only Batch observation to the common loop. Audit counts actual
  shared-source rows and producer-version lags; it does not inject delay.
- Efficiency alternatives: Torch vectorized preparation/gathers and DGL sparse
  reductions suffice; no DGL custom kernel or C++/CUDA operator is necessary.
  Reuse the scheduled Route and common run_epoch. Feature replicas include
  the union of needed neighbors; no model-local manager or new queue exists.
- Checks so far: 275 passed, 15 skipped in the default suite; two-rank Gloo
  DCRNN output and parameter-gradient parity passes. Two-A40 complete tiny
  train/validation/test replay succeeds, shared reads are 1/3 of requested rows
  with measured lag 0/1, and exact owner reads have zero lag.
- Known limitation confirmed by the tiny run: current compensation updates are
  zero for remote-only shared reads because those nodes are not locally
  produced; gamma stays at 0.5. Preserve this behavior for the current-policy
  ablation and report it. No compensation improvement is claimed.
- Full Flickr preparation is in progress with the corrected two-rank layouts.
  No current full-Flickr convergence result is available at this checkpoint.

Full-data launch/check update:

- Removed the superseded synthetic delay provider and its CLI/runtime injection
  and tests. Existing generic coupled-scan materializer support remains.
  Default suite now passes: 272 passed, 15 skipped. Two-rank DCRNN parity and
  serialized directional-buffer round trip pass; all 103 production modules
  parse and remain within 500 lines.
- Both complete Flickr two-rank artifact sets are prepared. DCRNN artifacts
  occupy about 30 GiB under `/tmp/starrygl_open_flickr_ablation_20260911`;
  GConvGRU artifacts occupy about 20 GiB under the workspace's
  `.experiment_artifacts`, reached through the named /tmp artifact symlink.
- Launched six seed-42 runs: exact / shared_hot / shared_hot+compensation for
  DCRNN and GConvGRU, 100 epochs with validation every 5 epochs. Source hashes
  are identical across all arms. Jobs share four A40s; times are observational,
  not isolated throughput measurements.
- Full-data early checks confirm nonzero actual shared reads, lag 0/1,
  no future-state rows and no stale owner rows. DCRNN shared reads cover 3.96%
  of requested state rows; GConvGRU covers 3.07%. Compensation remains at zero
  updates, and with/without-compensation first-epoch losses match exactly.
- Live report and plots: `docs/experiments/flickr_shared_hot_20260911/`.
  Each result is marked pending until actual train/validation/test completes.
  Current limitations: one seed, serialized access (pipeline disabled), small
  hidden width and K=1 refresh bound; no statistical or throughput claim.

Mid-run validation and timing update (2026-09-11):

- Stopped two redundant compensation diagnostic jobs after four identical
  with/without-compensation training losses and zero increment updates. Their
  diagnostic JSON explicitly distinguishes them from the four 100-epoch main
  experiments, which continue without source changes.
- Exact end-to-end DCRNN single-rank versus two-rank two-epoch training parity
  passes: maximum parameter difference 7.45e-9; test MSE differs by 1.49e-8.
  Saved the check in the experiment directory as `end_to_end_parity.json`.
- Added matched-epoch timing tables, validation-MSE versus cumulative time
  figures and first-observed threshold-crossing CSV to the analysis report.
  Reused existing epoch timers; no runtime or model changes. Timers exclude
  setup/checkpoint overhead; validation includes warm replay. Shared devices,
  disabled access pipeline and additional exact final replay are documented.
- At 49 matched epochs, DCRNN averages 32.38/33.49 seconds per training epoch
  and GConvGRU 30.02/30.46 (exact/shared). These are concurrent-run observations,
  not isolated speedup evidence; final tables update when all main runs finish.

Completed full-data pilot (2026-09-11):

- All four main two-rank jobs finish 100 epochs, 21 validations and final
  test replay successfully (process exit code 0). DCRNN exact/shared select
  epoch 100; GConvGRU exact selects 100 and shared+comp selects 95.
- Operational test MSE, exact/shared: DCRNN 0.02334201667/0.02301401121;
  GConvGRU 0.04857231069/0.04616281059. Exact-state replay of the same shared
  checkpoints gives 0.02301963612 and 0.04617455712, respectively. Thus fixed
  checkpoint read-policy differences are -0.0244% and -0.0254%; independently
  trained-arm differences must not be attributed solely to the final read.
- Actual shared test reads cover 5.4746%/4.4205% of requested state rows for
  DCRNN/GConvGRU, with observed lag 0/1. All recorded train/validation/test
  reads have zero stale-owner and future rows. Compensation updates remain
  zero. Both ranks' model checkpoints match tensor-for-tensor in every run.
- Mean training seconds per epoch, exact/shared: 32.52/33.25 (DCRNN),
  29.98/30.37 (GConvGRU). Startup-to-final-training-log wall minutes are
  76.23/77.95 and 72.28/72.98. These are concurrent observations with auditing,
  disabled pipeline and an enabled no-effect compensation path for GConvGRU,
  not an isolated speedup benchmark. Shared final test timers include two
  replays and cannot be divided directly by exact's single-replay timer.
- Final artifacts include report, epoch/time/gap PNG+PDF figures, CSVs, raw
  manifests/metrics/results, source snapshot, algorithm/reproduction notes,
  standalone analysis, wall-time metadata and verification JSON under
  `docs/experiments/flickr_shared_hot_20260911/`. All 103 archived/current
  source hashes match the tested implementation; no core code changed during
  full-data training. Default tests remain 272 passed / 15 skipped; separate
  distributed output/gradient and end-to-end parity checks also passed.
- Limits: one seed, hidden=8 and K=1 publisher skip bound. Validation still
  falls 14%-21% over epochs 80-100; all model test errors exceed the persistence
  baseline (0.01507540513). Report fixed-budget precision, not fully converged
  accuracy, statistical significance, compensation benefit or production
  throughput. Formal read-side freshness validation and isolated performance
  profiling remain outside this current-return ablation.

## 2026-09-09: Move Coupled, Context, Endpoint, And H2D Await Into Runtime

Latest state:

- GConvGRU now declares a `neighbor_recurrent` runtime cell and delegates its
  coupled temporal scan to `runtime.snapshot.scan.encode_model`; the model no
  longer launches the scan itself.
- EvolveGCN now exposes local context math, model-state advance and spatial
  math. Runtime packs all window contexts and launches the global reduction
  through the shared `CommScheduler` before scanning the evolved weight.
- T-GCN and MPNN-LSTM cells now expose only local per-layer math; Runtime owns
  their Flare-style inter-layer embedding Route launch and await.
- Endpoint Route materialization and its autograd communication moved from
  model helper code to `runtime.endpoint`. The training loop invokes it after
  model embeddings and before task loss, passing the runtime-owned scheduler.
- DataLoader now sends the internal short tuple `(Batch, ready_event)` from
  Stage B. Stage A calls compute-stream `wait_event` and `record_stream`; the
  previous host `event.synchronize()` was removed.
- Explicit arrival `ticket` arguments were removed from `CommScheduler`.
  `launched` now records only a local ordinal for diagnostics. It is not a
  cross-rank order or future static slot.
- The unused prepared Snapshot Route `ticket` column was removed from new
  artifacts and unpacking. Older artifacts remain readable because the extra
  column is ignored.
- `ExecutionPlan.execution_order` remains a semantic lowering summary. Its
  explanation labels that scope, and represented task loss now precedes owner
  state commit. Snapshot recurrent carry remains inside the model scan.

Attempted approaches:

- Rejected a second coupled/model-recurrent executor API exposed to users.
  Existing `runtime_cell`, `WindowScanResult`, scan and model-output hooks were
  sufficient.
- Rejected retaining endpoint communication in TGN or `_graph_ops`; models may
  still perform local endpoint indexing, while an `EndpointCollectRoute` is
  fulfilled only by Runtime.
- Rejected treating local arrival order as a partial static dispatcher. A
  future internal `CommPlan` must carry compile-generated slots explicitly.
- Rejected another ready-batch wrapper. A two-item tuple carries the private
  CUDA event without adding data to `Batch`.

Abstraction introduced:

- Added only `runtime.endpoint`, a distinct Runtime responsibility shared by
  every edge model that consumes `EndpointCollectRoute`. No registry, adapter,
  Stage class, cache base, model family API or second runtime was added.

Efficiency alternatives considered:

- Reused the existing autograd Route and dynamic owner request/response flow
  for endpoint embeddings, preserving deduplication and gradient return rather
  than rebuilding a model-specific collective.
- Reused the Flare-style temporal scan and packed all EvolveGCN window context
  summaries into one reduction. Per-node context reduction remains vectorized
  Torch; no Python node/edge loop or new kernel was introduced.
- Used native CUDA stream event dependencies and allocator `record_stream`
  lifetime tracking. A reusable native pinned-buffer pool remains deferred
  until the native sampler can fill it directly and profiling justifies it.
- DGL/PyTorch operators cover the changed math and movement. No custom C++/CUDA
  operator is needed for this review unit.

Files written or modified:

- Models: `src/starrygl/model/{_graph_ops.py,evolve_gcn.py,gconv_gru.py,
  mpnn_lstm.py,tgcn.py}`.
- Runtime: `src/starrygl/runtime/{comm.py,endpoint.py,loop.py}` and
  `src/starrygl/runtime/{dataloader/{loader.py,pipeline.py},snapshot/{layerwise.py,scan.py}}`.
- Prepare/store: `src/starrygl/prepare/snapshot_csc.py` and
  `src/starrygl/store/{artifact.py,feature.py}`.
- Plan/docs: `src/starrygl/{CONTRACT.md,plan.py}`, runtime/DataLoader contracts,
  `docs/STARRYGL_INTERFACE.md`, the migration plan and this log.
- Tests: `tests/{test_access_double_buffer.py,test_open_compile_plan.py,
  test_starrygl_model.py}`.

Verification:

- Focused plan/model/DataLoader regression: `99 passed, 5 skipped`; the final
  layerwise-focused subset passed `68 passed, 7 skipped`.
- Final single-process suite: `265 passed, 14 skipped` (`279` collected).
- Two-rank Gloo verification passes dynamic endpoint Route, prepared endpoint
  Route, empty-rank forward/backward participation, remote gradient return and
  Runtime-owned EvolveGCN context reduction: `3 passed` per rank.
- Every modified production Python module is below 500 physical lines.
- `python -m compileall -q src/starrygl` passes.

Unresolved risks:

- Compile-generated `CommPlan` slots and the unique static dispatcher are not
  implemented. Local ordinals provide observability only.
- Dynamic counts/request-node exchange remains synchronous before asynchronous
  payload communication. Native `[H,2]` history and dynamic Route output remain
  open.
- Layer/state Route paths still partly obtain the scheduler through block
  cache. Endpoint and Evolve context now receive it from Runtime directly.
- Reusable native pinned buffers, empty-rank full-slot parity, paper-scale GPU
  overlap traces and the published throughput gates remain open. Functional
  unit tests do not establish paper performance.

## 2026-09-10: Reduce CommScheduler To A Thin Communication Context

Latest state:

- Removed the local launch ordinal and launch-history list from
  `CommScheduler`; no runtime behavior depended on either value.
- `CommScheduler` now only binds a process group, reuses per-device CUDA
  streams, and launches/finishes Route collectives. It does not own a global
  stage order, compile-time slots, backward, gradient synchronization, or state
  semantics.
- DataLoader double-buffer handshakes own feature-prefetch timing; the runtime
  Snapshot scan owns history/layer ordering; endpoint, autograd/DDP, and
  StateManager own their existing communication call sites.
- Removed the planned global `CommPlan`/dispatcher from the current contracts.
  When paths share one process group, every rank must still enter each
  collective required by a peer, using an empty payload where necessary.

Attempted approaches:

- Rejected compile-generated collective slots and a unique global dispatcher:
  the fixed two-queue/runtime call path already determines execution order.
- Rejected separate process groups for feature and layerwise communication
  without profiling evidence that independent communicators improve overlap.
- Kept the existing class name to avoid a behavior-free rename across runtime,
  store, and tests; its narrowed docstring is the authoritative meaning.

Abstraction introduced:

- None. This change deletes diagnostic scheduler state and narrows an existing
  object instead of adding another plan, dispatcher, or wrapper.

Efficiency alternatives considered:

- Reused the current double-buffer handshake and Flare-style runtime scan.
  Removing ordinal bookkeeping avoids Python list growth on the communication
  path without changing tensor packing, NCCL operations, or stream overlap.
- Separate NCCL communicators remain deferred until a GPU trace demonstrates a
  same-process-group launch bottleneck.

Files written or modified:

- `src/starrygl/runtime/comm.py`.
- `tests/test_starrygl_model.py` and `tests/test_runtime_memory.py`.
- The package, runtime, DataLoader, Snapshot, state, sample, view, batch, task,
  paper-coverage, interface-overview, and migration-plan documents.

Verification:

- Focused runtime/model/DataLoader tests: `92 passed, 13 skipped`.
- Full single-process suite: `265 passed, 14 skipped`.
- Two-rank Gloo endpoint, layerwise-gradient, gradient-sync, Evolve context,
  state/mailbox fetch, empty-participant commit, and shared-update tests:
  `10 passed` per rank.
- Every production Python module remains at or below 500 physical lines.

Unresolved risks:

- The complete same-process-group feature/layerwise/endpoint/gradient/state
  path still needs deliberately skewed multi-rank testing and a GPU overlap
  trace; unit tests do not establish paper performance.
- Dynamic counts/request exchange, native `[H,2]` Route output and reusable
  pinned buffers remain open performance work.

## 2026-09-11: Sliding snapshot boundary history implemented

Latest state and shared path:

- Traced Prepare/task/access/Batch/scan/loss/backward/StateDelta before editing;
  only hot compute layout and timestamped dependency storage/read specialize it.
  See `docs/design/current/snapshot_boundary_history.md`.
- User selected cumulative increments, learnable gamma, snapshot-indexed memory
  and existing hot filtering. This replaces the last-boundary-only approximate
  carry sketch. Same node/time recomputation replaces its slot and statistics.
- Existing AsyncMemoryCommitter retains owner authority. Each batch retains
  temporal outputs locally and publishes its final boundary: filtered hot
  all_gather followed by fixed-route cold owner push, with initial zero state
  as an explicit seed for missing received versions. No inner hidden-state fetch.
- Main-thread history gathers coexist with feature prefetch. Pending packets own
  immutable send buffers, at most two cold pushes remain in flight, and reset
  drains pending operations. plan.explain describes the policy.
- Full-snapshot artifacts require re-Prepare; all ranks check shared artifact
  metadata before communication. Sampled-neighbor reads retain their prior path.
  CLI compensation counters now read the temporal statistics.

Abstraction and efficiency alternatives:

- One SnapshotHistory tensor store holds node/time slots and versioned increment
  statistics. Coupled scan moved into its own module to keep production modules
  below 500 lines. No new loader, trainer or provider hierarchy.
- Reused Torch gathers/scatters, DGL sparse ops, StateManager owner commit,
  existing filter/all_gather and CommScheduler Route. Subscriber exchange occurs
  once at setup. Python loops are at snapshot/batch/peer-setup boundaries.
- Direct temporal validity lookup costs O(T * requested rows); a predecessor
  index or custom kernel is deferred until profiling justifies it. Full temporal
  storage follows the user's request; no ring cache or native kernel was added.

Files written or modified:

- `store/snapshot_history.py`; `runtime/memory/{snapshot.py,__init__.py,shared.py}`;
  `runtime/state/{build.py,access.py,recurrent.py,CONTRACT.md}`;
  `runtime/snapshot/{scan.py,coupled.py}`; `runtime/dataloader/pipeline.py`.
- `prepare/{snapshot_csc.py,build_part_graph.py}`, `plan.py`, `model/gconv_gru.py`,
  and `cli/coupled_ablation.py` (all source paths under `src/starrygl`).
- `tests/{test_snapshot_history.py,test_coupled_ablation.py,test_partition_assignment.py}`.
- `docs/STARRYGL_INTERFACE.md`, the design note above, and this log. Archived
  Flickr results and their source snapshot were not regenerated.

Validation:

- Full single-process suite: 275 passed, 20 skipped.
  Final focused history/ablation/partition/plan suite: 27 passed, 6 skipped.
- Two-rank Gloo history/filter/owner/sliding-backward: 2 passed per rank, covering
  both models. The same checks passed on two A40 GPUs with NCCL.
- Two-rank NCCL compile -> Prepare -> fit with two-snapshot sliding windows and
  feature pipeline: 2 passed per rank, including unlabeled boundary commit.
- DCRNN owner-only and replicated-hot complete-graph output/gradient reference:
  2 passed per rank under Gloo. Owner masks prevent duplicate hot supervision.
- Compileall passed; all production Python modules remain <= 500 physical lines.
- Corrected test setup to use move_graph_block and count three scored windows
  plus one unlabeled state update; no production fallback was added.

Remaining limits:

- No new Flickr convergence/timing run or demonstrated communication speedup.
  DCRNN reset-gate exchange still waits each snapshot; max_skip does not bound age.
- Cache memory scales with T*N*H. At Flickr h=8, 71 slots and all 2,302,925 nodes
  needed locally, packets plus validity consume about 11.12 GiB per rank,
  excluding graph/model/owner stores and temporary gathers.
- Full runtime checkpoint/resume and sampled-neighbor temporal history are not
  introduced by this step. Epoch replay/reset and pending-drain behavior are tested.


## 2026-09-11: Window-relative boundary slots

Latest state and common path:

- User replaced dataset-wide timestamp slots with sliding-window retention.
  SnapshotHistory now allocates W output slots plus one predecessor, where W
  follows full-snapshot count plus the normalized chunk-decay history length.
- Traced Prepare -> window/task slice -> graph accessor -> Batch -> state read
  -> model/scan -> task/backward -> StateDelta -> owner/cache commit before
  implementation. Specialization remains history storage and construction-time
  window capacity; fit/evaluate/predict use their existing window arguments.
- Slots retain actual producer versions for causal lookup and delta-t. Advancing
  preserves overlapping outputs and promotes the latest received predecessor;
  filtered seeds and cumulative counts survive reuse. Late messages cannot
  replace a newer predecessor or reused output; backward replay requires reset.
- Cumulative compensation, learnable gamma, hot filtering, fixed cold-owner
  Route, and task/state ownership retain the preceding implementation's behavior.
  Prepared hot-compute artifacts remain reusable; sampled-neighbor behavior is
  unchanged.

Abstraction and efficiency alternatives:

- Reused SnapshotHistory and the canonical runtime; no new abstraction or public
  cache-size configuration. Allocation is O(W*N*H), independent of dataset T.
- Torch masked gather/scatter implements slot selection and updates. A
  full-buffer roll copies retained states unnecessarily; Python per-node loops
  are excluded. DGL graph operators offer no benefit for this dense storage
  operation; DGL-inspired/custom native lookup is deferred until profiling
  shows O(W*requested nodes) selection to be a bottleneck.

Files modified:

- `src/starrygl/store/snapshot_history.py`;
  `src/starrygl/runtime/memory/{snapshot.py,shared.py}`;
  `src/starrygl/runtime/state/{build.py,CONTRACT.md}`;
  `src/starrygl/runtime/{trainer_options.py,train.py}`; `src/starrygl/plan.py`.
- `tests/test_snapshot_history.py`, `docs/STARRYGL_INTERFACE.md`,
  `docs/design/current/snapshot_boundary_history.md`, and this log.

Validation and attempted approaches:

- Full single-process suite: 278 passed, 20 skipped. New checks cover repeated
  wraparound, cumulative counts, filtered predecessors, late/future packets,
  immutable gathered tensors, constant allocation and trainer window overrides.
  Corrected the capacity test to use an instantiated model instead of the
  trainer's lazy model configuration; no production fallback was added.
- Two-rank A40/NCCL history/filter/owner/backward checks passed for both models.
  Compile -> Prepare -> sliding fit also passed for both models, with version 4
  committed into three slots (W=2) and all pending pushes drained.
- A combined distributed invocation passed the two history tests, then hit an
  NCCL bootstrap socket failure while reinitializing process groups, followed by
  an artifact-fingerprint mismatch. Running the two training tests separately
  passed on each rank (2 passed, 8 deselected); no production change was needed.
- Compileall passed. All 106 production Python modules are <= 500 physical lines.

Remaining limits:

- No new Flickr convergence/timing measurement. With h=8, W=2 and every Flickr
  node locally needed, packed slots plus validity are estimated at 0.47 GiB per
  rank, excluding graphs, other stores and transient gathers. This is not a
  measured peak. Lookup still scans W+1 slots per requested node.
- DCRNN still waits for its per-snapshot reset-gate exchange. max_skip controls
  publication skips rather than bounding state age. Runtime checkpoint/resume
  is not added. Distributed test groups are run separately because of the
  observed process-group recreation/bootstrap failure in the combined run.


## 2026-09-11: Flickr window-cache rerun (in progress)

- Reuse the CLI -> compile/Prepare -> Store -> canonical run_epoch -> Batch ->
  dependency hydration -> coupled scan -> loss/backward -> StateDelta commit
  path. Only experiment observability changes: distinguish hot/cold historical
  reads, verify nonzero increments on stale rows, and record wall/peak memory.
- Six 100-epoch, seed-42 arms: DCRNN/GConvGRU each exact, window cache without
  compensation, and window cache with compensation. Reuse previous W=1 input
  window, h=8, lr=0.001, two ranks, 10% hot and every-five-epoch validation.
  Each GPU runs at most one rank. New hot-compute Prepare artifacts use a fresh
  directory under /mnt/data/zlj/starrygl-experiments/flickr_window_slots_20260911.
- No new runtime abstraction. Use existing Torch tensor reductions for audit,
  Python/stdlib epoch timers and existing CUDA allocator counters; DGL or custom
  kernels are unnecessary for experiment instrumentation. Validation pending.

Rerun checkpoint (16:07 local): CLI instrumentation tests pass (full suite
279 passed, 20 skipped); a two-rank Gloo CLI hot-replica run completed training,
validation, operational test and exact replay. New/old partition ownership and
time splits are identical. Static hot replication adds 86.55% incoming edges
and 87.47% DCRNN outgoing edges. The six-arm queue is running; first exact arms
started at 15:44 on separate GPU pairs. Source frozen and results/analysis stored
under docs/experiments/flickr_window_slots_20260911. Completion validation and
full-data cache accuracy/timing conclusions remain pending.

Rerun steering (17:00 local): prioritize GConvGRU and show existing DCRNN
curves. Both new exact arms completed 100 epochs and final tests. GConvGRU
continues cache -> compensation on GPU 2/3. DCRNN cache was stopped before
its first epoch record; cache/compensation are deferred. Preserve its startup
log as an intentional interruption, not a completed measurement. The displayed
DCRNN paired curve is explicitly the previous archived implementation.
Full-snapshot DCRNN reuses one prepared graph for gates and candidate; there
is no second random neighbor sampler. Fresh remote reset gates still use the
autograd Route and wait between these stages, even with cached prior hidden.
New GConvGRU early cache reads have zero measured hot/cold lag; no nonzero
compensation benefit is claimed. Experiment-only analysis now records the
deferred DCRNN arms and supports GConvGRU-only completion verification; a thin
completion watcher reuses the existing analyzer and verifier. Training source
remains frozen. Remaining work: GConvGRU cache/comp full runs and measured
staleness/compensation effectiveness; DCRNN cache/comp deferred by request.

Runner audit (17:19 local): GConvGRU cache reached 32 epochs and continues.
Confirmed the cosine filter is connected, effective max_skip=1; it controls
publication only, after local hot history commits. Cold subscribers receive
unfiltered batch-end owner pushes. The current W=1 experiment disables the
data-access pipeline; optional graph/feature prefetch is supported, while
history reads intentionally stay on the main thread. No unconditional
batch-entry cold drain was found; bounded queue backpressure and all_gather
count synchronization remain possible exposed costs. Full-data filter skip
counts and communication/compute overlap have not been measured.

Added one assertion to `tests/test_snapshot_history.py` proving a filtered
hot publication leaves local hot history fresh; updated the boundary design
note and this log. Separate two-rank NCCL GConvGRU filter/history and W=2
pipeline compile/Prepare/fit groups each passed on both ranks (1 passed per
rank per group). A controlled cosine-filter probe observed publish/skip/publish,
archived as `filter_probe.json`; it is not a Flickr publication measurement.
The first probe used the shell's incompatible Python; rerunning with the
training environment passed. The completion watcher was also restarted under
that same environment. No new abstraction or production implementation;
reuse existing tests and Torch operations, no DGL/native changes required.

Design correction from the user's smooth-aggregation equations: gamma must
blend a local UPDATE output with a prediction from synchronized shared state,
independently for each retained snapshot slot. Alpha remains the publication
threshold. Current production code instead scales the increment with gamma,
shares local/hot received history, and compares cosine changes against a
rank-local last accepted candidate. These are implementation gaps relative to
the clarified mechanism. Recorded the multi-slot target and required checks in
`snapshot_boundary_history.md`; marked the existing experiment's generated
report as W=1 and the old increment formula. No production changes or new
smooth-aggregation results are claimed. Preserve the canonical runtime and
reuse tensor history operations when implementing the correction; no new
model family, state wrapper, or extra mixing parameter is proposed.

The clarified implementation is now available in the isolated checkout
`../.worktrees/snapshot_smoothing/starrygl-open` (relative to this repository),
with `verification.json` and `snapshot_smoothing.patch`. It uses cumulative
sum/count for both hot and cold nodes, a per-slot learnable blend only for hot
UPDATE outputs, and separate local/shared histories. Full suite: 282 passed,
20 skipped; two-rank NCCL history/backward and W=2 pipeline training passed for
both models. The original production source remains unchanged for the running
Flickr queue. GConvGRU cache reached epoch 71 at 17:50; the isolated correction
has not been used for a full-data convergence/timing measurement.

2026-09-11 corrected smoothing evaluation: isolated ../.worktrees/snapshot_smoothing/starrygl-open source now also fixes exact W>1 causal predecessor/inner-snapshot neighbor exchange, with both-model NCCL output/gradient parity and 283 passed/22 skipped single-process checks. First pilot stopped before a complete epoch to incorporate this correction. New queue PID 2561531 uses GPUs 0/1, W=3, pipeline enabled; original W=1 GConvGRU queue continues on 2/3 unchanged. New results and automatic convergence/time plots are separate in docs/experiments/flickr_smoothing_w3_20260911. Production code in this original checkout remains frozen.

2026-09-12 DCRNN evaluation continuation: read-only audit confirms corrected W=3 GConvGRU exact/cache/smooth all completed 100 epochs; DCRNN only old W=1 exact completed, old cache stopped before first epoch. Prepare a DCRNN thin CLI queue and report scripts under docs/experiments/flickr_dcrnn_smoothing_w3_20260912. Reuse the already verified corrected runtime spine and production hashes; no new abstraction or core change, no re-Prepare. Existing Torch/DGL/NCCL implementation retained. One-epoch DCRNN pilot followed by 100-epoch exact/cache/smooth on GPU 0/1; all-GPU idle at setup. Current sandbox permits workspace output and read-only old artifacts. Pending: signature checks, actual launch, pilot memory/stability and final DCRNN results.

DCRNN W=3 owner/hot artifact signatures passed. Queue PID 2609806 and pilot torchrun PID 2609807 launched successfully; host-process and GPU queries confirm two active workers on GPUs 0/1, first training 25 steps logged. A sandbox-local /proc check could not see host PIDs; this was process-namespace isolation, not a stopped experiment. Output/logs remain under workspace. Report scripts compile-checked; no core source edits or new tests required. Pilot completion and formal results pending.

2026-09-12 user changes evaluation budget to 10 epochs. Stop old DCRNN 100-epoch supervisor, preserve pilot train/val, and use existing CLI with --epochs 10. Thin launcher supports GPU pair/arm arguments for exact/cache on 0/1 and smooth on 2/3. Reuse GConvGRU first 10 epochs from same production hashes/config; plot only observed training/validation MSE. No production changes, new graph processing, or abstraction. Combined plot is a report artifact; no new training path. Verify launcher syntax, source/config parity and actual epoch records.

2026-09-12 ten-epoch comparison complete: all six model/arm training and epoch-10 validation records available; compare.py verifies common source/config parity. Exported PNG/PDF/CSV/report and verification.json in docs/experiments/flickr_models_10ep_20260912; checked 60 finite training points, 18 validation points and epochs 1–10 per arm, visually inspected final figure. DCRNN epoch-10 validation exact/cache/smooth: 0.15364307/0.15364831/0.16275131; GConvGRU: 0.25059717/0.25073050/0.28501802. GConvGRU uses identical-configuration first-ten-epoch prefix. DCRNN cache final test continues independently of the completed requested curves. Corrected report timing description for parallel GPU pairs. No core changes; single seed and shared-host timing remain limitations.


## 2026-09-12: Integrate owner boundary and snapshot alignment (V7)

Common path and specialization: docs/design/current/owner_boundary_flare_alignment.md. Main src/tests/configs matched frozen V3 before edits; applied only the audited 25-file set (17 production replacements, 4 test replacements, 4 new tests), then migrated the GC example from local/shared smooth_aggregation to explicit boundary_prediction with log(9). Main independent docs/CONTRACT content preserved. Backup: ../.experiment_artifacts/rebuttal_20260912/main_before_v7. Exact file hashes and modified-file inventory: ../paper_method/rebuttal_20260912/main_v7_integration.json and main_integration_audit.json.

Torch/DGL reuse and deletion selected over new kernels/frameworks: remove unused edge ID sorting/feature fetches and hot recomputation; reuse history/filter/collectives, static graph layouts and vectorized node-ID state alignment. Existing TaskTarget/window-loss consumer reused; no new execution spine. V6 NCCL and V7 CPU checks passed as recorded in the design note. All production bytes equal frozen V7; all modules<=500 lines. Flare-compatible training-profile GPU timing is queued after the independent V6 ten-epoch campaign. Remaining risks: partial-graph materialization and target-routing overhead, exact versus approximate model math, snapshot-version versus optimizer-version lag, and the documented Flare protocol differences. No final convergence or Flare speed parity claimed.

## 2026-09-12 V9 snapshot CPU hotpath integration

Hash-verified main V7 -> frozen V9. Shared Prepare owner table -> accessor -> Batch -> dependencies/model/task -> state path unchanged. Three source replacements: snapshot/materialize.py (one dst join), snapshot/rows.py (linear CSC edge permutation), utils/index.py (bounded CPU inverse lookup); three focused tests added. Existing Torch scatter/gather and DGL buffers suffice: no kernel, loader, registry or public API added. GPU helper path unchanged. Before images/receipt live in main_before_v9 and main_v9_integration.json.

Validation: V8 47 targeted passes, V9 182 passed/14 skipped plus final focused23/3; full Flickr four-A40 V8/V9 cases completed with27 steps each. V9 W8 F2 J128 train means TG6.0173, MP6.7661, EV6.6876s; source/config/signature/epoch audit in flare_lookup_results. Small recorded GPU MSE differences are preserved, not called bitwise parity. Feature cache alone is not uniformly faster, and was not enabled in main configs. Flare remains1.9896/2.8666/3.1782s under its native implementation; chunk/model/gradient/eval differences remain. Main GPU graph placement is still CPU; opt-in device prototype is isolated and unintegrated.


## 2026-09-12 V10 device snapshot materialization integration

Common source -> Batch -> dependency/model/task -> state spine preserved; see docs/design/snapshot_device_materialization.md. Five existing production modules changed, two tests added. The explicit snapshot_materialize_on_device option now reuses the existing prefetch stream, row/graph cache, target mover, events and stream lifetime handling for TGCN/MPNN-LSTM/EvolveGCN full/chunk node execution. Default source placement remains CPU. Torch/DGL tensor operations suffice; no custom CUDA operator, alternate loader or public API added. Independent strict-partial/global-J correctness fixes preserve full routes at s=1 and clear routes for partial induced graphs.

Main V9 hashes checked before copying frozen V10; all106production files now equal39cc5b44f21da5e63c439db04df3dbd38b7a476a664ccd88f93a3189d56b9b46. Backup/receipt: main_before_v10 and main_v10_integration.json. CPU94passed/17skipped, CUDA plus two-rankNCCL40passed. Three actual four-A40 Flickr runs passed494integrity checks; steady train means TG4.1183,MP4.0623,EV1.9628s. TG/MP three-epoch losses equal V9 CPU; small EV differences are retained in flare_device_results.md. GPU peak allocated6.244/9.387/4.301GiB. Risks: additional memory, remaining row/target copies, GPU scalar syncs, model/chunk/gradient/evaluation differences from nativeFlare. Native Flare parity is not achieved or claimed.


## 2026-09-12 V11/V12 candidates measured; retain V10 main

Two bounded common-path candidates were evaluated in isolated checkouts: V11 groups dense parameter gradients by dtype/device using Torch flatten/reduce/copy; V12 additionally builds a cached bipartite DGL graph directly from CSC buffers. Each changes one existing production module, adds no public API/loader/state manager, and uses installed Torch/DGL operators instead of custom C++/CUDA. Existing supervised/empty-row optimizer semantics and forward/norm order are retained. V11's gradient reduction is per parameter group, not a per-node Python loop. V12 still needs edge_rows and one graph construction per new cached topology, so it does not eliminate every graph conversion.

V11 final gradient CUDA/NCCL11passed, snapshot40passed and owner/empty each rank4passed; initial oracle failures and inherited old-hot tests are explicitly recorded in gradient_buckets_v11_audit.md. V12 CPU121passed20skipped2deselected; CUDA/NCCL39passed. Four-A403epoch clean means TG/MP/EV: V10 4.1183/4.0623/1.9628s, V11 4.0371/4.0856/2.0012s, V12 4.2474/4.1334/2.0805s. Neither candidate shows a consistent overall speed benefit here; do not infer significance from two steady epochs. Main remains frozen V10(39cc5...), verified106source hashes. V11/V12 src/tests/design notes and all first-failure/success/benchmark logs remain reviewable under .worktrees and paper_method/rebuttal_20260912. No candidate production/test files were copied to main.

Unresolved: duplicate row/target transfers, remaining synchronization and native Flare sampling/model/gradient/evaluation differences. No complete Flare parity claimed. Exact manifests and selection are in final_source_selection.json, flare_gradients_results and flare_csc_results.


## 2026-09-13 — Integrate verified Flare layout optimization (before copy)

Main source/tests still match the saved V10 backup. Common-path trace, scope, Torch/DGL versus native alternatives, memory tradeoff, numerical protocol, and unresolved performance/interface risks are recorded in docs/design/flare_parity_20260913.md before copying. Select layout_final; exclude dense lookup and int32 screens. Copy only eight changed production files and seven changed/new tests, preserving all other main files and documentation. No new runtime or model abstraction. Main regression and strict candidate-to-main source hashes follow integration.

### Integrated and verified

Eight production files copied: model/graph_conv.py, model/tgcn.py, partition/build.py, prepare/data.py, runtime/loop.py, runtime/trainer.py, runtime/snapshot/cache.py, task/target.py. Seven candidate test files copied; two further existing fixtures (test_access_double_buffer.py and test_empty_supervision_step.py) now supply the formal Store LabelStore rather than an incomplete SimpleNamespace. Assertions and production behavior were not weakened.

Main's 106 Python source files exactly match the selected timing/audit manifest; none exceed 500 lines, and all compile. Source/modified-file receipt: paper_method/rebuttal_20260913/flare_main_integration.json. All production changes are Torch/DGL reuse; native build not required.

Full main test command: GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src /home/zlj/.miniconda3/envs/tgnn_3.10/bin/python -m pytest tests -q. Result: 495 passed, 9 failed, 40 skipped (14.67 s). All nine failure IDs reproduce identically with the saved pre-integration source: old shared_hot/view/default-config expectations and removed coupled-model smoothing/increment members. They remain unresolved; this is not an all-green suite. Initial ten additional failures were incomplete store fixtures, corrected as described above. XML/logs: flare_main_final_tests and flare_main_preexisting_tests under the report directory.

Explicit CUDA/NCCL/device-rebinding command: CUDA_VISIBLE_DEVICES=0,1 NCCL_IB_DISABLE=1 GLOO_SOCKET_IFNAME=lo STARRYGL_TEST_CUDA_MATERIALIZE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src /home/zlj/.miniconda3/envs/tgnn_3.10/bin/python -m pytest tests/test_snapshot_device_materialization.py tests/test_epoch_task_payload.py -k 'cuda or nccl or rebinds' -q. Result: 5 passed, 17 deselected (15.15 s).

Matched performance remains 2.223034097 versus native 1.996627616 seconds per epoch, 11.34% slower. Numerical clean/audit checks pass their original tolerances with identical owners, initialization, priorities and explicit benchmark-only synchronized-gradient scaling .25. Remaining risks: one seed/training-only evidence; resident task-payload memory; current nine pre-existing regression failures; incomplete minimal model-interface migration; DCRNN/GConvGRU not reevaluated against native in this round. Rejected dense/int32/token/order/lerp candidates remain isolated. No performance-parity completion claim.


## 2026-09-13 — Round3 target-bound integration (before copy)

Common-path trace, exact row-range argument, Torch versus DGL/native alternatives, no-new-abstraction scope, attempted candidates and performance limits recorded in docs/design/flare_parity_round3_20260913.md before integration. Clean and independent audit passed. Main still matches its saved pre-round3 source. Copy only the three selected production files and one new test; retain all other source/docs and original tolerances. Main regression pending.

### Round3 integrated / checks completed

Copied runtime/snapshot/materialize.py, task/target.py, task/prediction.py and new tests/test_target_row_bound.py only. Main's 106 production Python files strictly match the selected timing/audit source manifest (a37c8fa77d09c703d8b24cb363e936c958fc3cf9734337dddfc6453c60d52d49); all compile and remain <=500 lines. No graph/sampling/model/optimizer/state/collective change.

Main full regression: GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src /home/zlj/.miniconda3/envs/tgnn_3.10/bin/python -m pytest tests -q: 502 passed, 9 failed, 40 skipped (13.63s). Failure IDs match both the previous-main run and original pre-round2 source reproduction exactly. These nine existing failures remain unresolved; suite is not all green.

Main CUDA/NCCL/device rebinding: CUDA_VISIBLE_DEVICES=0,1 NCCL_IB_DISABLE=1 GLOO_SOCKET_IFNAME=lo STARRYGL_TEST_CUDA_MATERIALIZE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src /home/zlj/.miniconda3/envs/tgnn_3.10/bin/python -m pytest tests/test_snapshot_device_materialization.py tests/test_epoch_task_payload.py -k 'cuda or nccl or rebinds' -q: 5 passed, 17 deselected (15.13s). Separate real four-rank clean trajectory and batch1/8/27 prediction/gradient audits pass unchanged rtol1e-4/atol2e-6 and exact ID/label/rank checks.

Fresh ten-epoch mean2–10: native 2.010964218s, old implementation control 2.278488272s, integrated 2.175101861s; slow rounds retained. Observed reduction versus fresh control 4.54%; remaining native gap 8.16%. Report prior 2.223034097s as a different run, not the fresh control. All screened alternatives (step, shape, grad buckets, pin, metadata, combined unblock, pageable async) remain unmerged. The trace's 179.8ms target scalar wait mostly absorbed preceding forward work; do not claim it all became wall-time savings.

Evidence: paper_method/rebuttal_20260913/round3_main_integration.json, round3_main_tests.xml/log, round3_main_cuda_tests.xml/log, tgcn_round3_final_verification.json, tgcn_round3_final_audit_verification.json, round3_transfer_analysis.json/md. Final report uses separate flare_performance_round3.* files; prior reports remain unchanged.

Remaining: timing/quality evidence is TGCN/Flickr four A40s and one seed only, with explicit benchmark-only gradient scale .25. No test-set/multiseed or DCRNN/GConvGRU parity claim; minimal model interface migration remains incomplete. Native performance is closer, not fully matched.


## 2026-09-13 round4 integration (before copy)

Shared Prepare-to-state-update spine and unavoidable boundaries documented in docs/design/flare_parity_round4_20260913.md. Selected two isolated changes: shared DataLoader prefetch-stream lifetime in existing per-store/device runtime_cache; exact snapshot-history ID union constructed with CPU Torch membership before one H2D transfer. No new runtime/loader/public API/kernel. Torch slicing candidate not selected (no speed improvement), fixed-dst transfer candidate not GPU tested/not selected.

Validation before integration: 24 stream CPU/CUDA/NCCL checks passed; history22 CPU passes and6 CPU/Gloo checks per rank; final combined6 coupled CUDA/NCCL exact/sliding/owner filtered cases per rank passed. Two fresh10epochcontrol/final clean pairs and separate prediction/gradient audit all pass originalrtol1e-4/atol2e-6. Meancontrol2.179795311s/final2.203320628s (+1.079%), native1.996583284s; all epochs retained, same seed42, not a multiseed claim. Independent default-SG h8 memory diagnostic reduces r1e10reserved35.100GB->9.007GB with same1.981GB live bytes. h64 completes3epochs; h128OOMr1epoch1, not a success. Constructor syntheticT128 peak1.890GB->.434GB with exactIDs and .0897->.1272s time tradeoff.

Backup: .experiment_artifacts/rebuttal_20260913/main_before_round4_integration. Main regression still pending. Known risks: all-snapshot feature/task residency, global row maps, wide-window activations and allocation fragmentation, current unequal imported partition, nine existing test failures, no actual larger-N full run, model interface migration still incomplete.


## 2026-09-13 round4 final validation complete

Main modifications: runtime/dataloader/loader.py, runtime/memory/snapshot.py; tests/test_loader_stream_reuse.py and tests/test_snapshot_history_initialization.py. Design: docs/design/flare_parity_round4_20260913.md plus per-candidate notes. Main all106 production module hashes strictly match final measured candidate; all compile and <=500 lines.

Main full regression:508 passed /9 failed /41 skipped,14.40s. Exactly the same9 failing IDs as round3_main_tests.xml; no new failure remains. CUDA/NCCL/lifecycle selection:7 passed /17 deselected,15.38s. Combined coupled GConvGRU/DCRNN exact/sliding/owner-filtered NCCL:6 passed /24 deselected per rank on the identical final source,18.97s. Isolated full-test fixture-copy error was resolved by copying existing configs and focused replay passed; it did not recur in main.

Two same-seed10epoch fresh timing pairs (all e2-10 retained):control2.172420/2.187171s,final2.211940/2.194701s; aggregate2.179795->2.203321s,+1.079%time,10.355% above this round's native1.996583s. No speedup claimed. All four clean trajectory/checkpoint audits and separate final prediction/gradient audit pass fixedrtol1e-4/atol2e-6. Native protocol still requires explicitSGgradientx.25 in benchmark only. Larger-width capacity diagnostics use defaultSGgradient normalization and must not be mixed with the native-parity MSE.

Artifact index:paper_method/rebuttal_20260913/round4_final_validation.json,round4_final_source.json,round4_history_*_memory_v2.json,round4_scale_audit.md,round4_main_*_tests.*,tgcn_round4_*_verification.json. Larger h128 remains an observed OOM, actual larger-N graph training remains unvalidated, nine pre-existing failures and minimal-interface migration remain open. Prior round3/layout/tuning/interface reports preserved unchanged.


## 2026-09-13 MemShare parity: temporal target identity (before implementation)

Prepare canonical event/task rows -> DataLoader prepared task slice -> training/evaluation negatives -> Event graph accessor / native MFG -> materialized Batch -> state dependency hydration -> encode -> task -> runtime-owned StateDelta commit. Snapshot uses the same loader, Batch, task and state commit; only graph access and recurrent scan specialize.

The native MFG already carries srcdata/dstdata ts. Target route construction nevertheless searches only node IDs; two occurrences of a node at different cutoffs select the same last row. Lazy routing repeats the same lookup in model graph helpers. This prevents a valid timestamp-aware deduplication claim and can contaminate loss/gradient comparisons with MemShare.

Reuse the existing vectorized state-query (node, timestamp) grouping in utils/index.py for state hydration, root deduplication and target row mapping. Bind a temporal target route before the lazy branch when the MFG has timestamp rows; batch endpoint gathers then consume it across all Event models. Snapshot paths without timestamped MFG rows retain their current mapping. No new model/runtime chain or configuration.

Efficiency: Torch stable sorting/scatter reuse is sufficient and preserves int64 node IDs and fractional cutoffs independently, without lossy float ID packing. DGL exposes graph gathers but not this composite identity join. Existing native sampler supplies timestamp columns; a custom C++ join is deferred until profiling justifies replacing the Torch operation. No per-node or per-edge Python loop. The unrelated native sampling integer-time conversion remains a separate parity check.

Validation planned: repeated node/different cutoff target rows, both negative modes and ratios, lazy routing, missing cutoff, large int64 IDs, state grouping regression, Event/model tests. Existing source/Flare benchmark files remain frozen. A correct row join alone does not establish complete temporal attention/state or MemShare performance parity.


## 2026-09-13 MemShare temporal target identity validation

Moved existing state query grouping to utils/index.py and reused it for Event root dedup and timestamp-aware task routes. Modified runtime/state/access.py, runtime/event/target.py, task/target.py; updated existing grouping test import and added tests/test_temporal_target_identity.py. Explicit endpoint collection retains the existing owner-plane path. Independent review found current Snapshot blocks do not carry dst cutoff columns, so their route/dataflow stays unchanged. Focused tests32 passed; first full CPU regression513 passed/9 failed/41 skipped, with exactly the prior9 failure IDs. No new native kernel, no per-node Python, no collectives changed. MemShare speed/convergence parity still unestablished; TGN reference math and fresh matched Prepare are being audited separately.


## 2026-09-13 MemShare owner response device mismatch (before repair)

## First four-rank execution: owner response device repair

The actual WIKI run stopped before epoch1: Stage B submits CPU request IDs/order, receives CUDA feature payload after feature cache moves to GPU, then remote_fetch._restore_order passes the CPU order into CUDA index_copy_. This is the common owner response restoration used by features, state and mailbox; repair the index device once there. Same Prepare -> task -> accessor -> Batch -> dependency -> model -> task -> commit spine as above. No collective order or value math changes. Selected Torch .to(value.device) at the restoration point; DGL/custom C++ cannot remove the requirement that CUDA scatter indices reside on device. Tests must include a CPU order with CUDA payload and empty response; rerun actual four-rank driver. Extra per-response small index H2D remains measurable overhead.


## 2026-09-13 — MemShare TGN component parity: pre-edit audit

Trace/design: ../../../paper_method/rebuttal_20260913/memshare_parity_audit.md (workspace paper folder). Shared path remains window → task → negatives → graph accessor → Batch → dependency access → model → task → state update. Event query identity is root-owned; this bounded change owns model/tgn.py, model/layers/temporal.py and new focused tests only. Planned corrections: native TGN unscaled attention and query-cutoff GRU input, preserving positive-event commit timestamps and other models default score convention. Existing Torch/DGL operators retained; no new kernel, per-node Python loop, wrapper model or runtime path. Native w_out concat order is handled by reference weight mapping, not another production layout. Status: pre-edit; numerical/reference tests and full-training parity pending. Thin benchmark/config live under paper_method; old Flare artifacts untouched.


## 2026-09-13 — Independent snapshot uniform sampling (pre-edit)

Shared Prepare/owner task slice -> common DataLoader/native sampler -> Batch -> dependency/model/task -> runtime state commit is traced in docs/design/snapshot_uniform_sampling.md. Only explicit snapshot_uniform sampling specializes task-static topology keys and root snapshot IDs; existing historical policies remain. Reuse vectorized Torch bucketize plus existing native equality-time sampling; no new loader/model/runtime stack or per-entity Python loop. Planned source: runtime/sample/__init__.py, runtime/snapshot/materialize.py, runtime/dataloader/loader.py. R-GraphSAGE itself remains unimplemented because primary Method equations/author code are unavailable; no SAGE+GRU substitute. CPU native and sampled forward/gradient checks pending; no GPU jobs.


### Independent snapshot uniform prerequisite — CPU complete


## Implemented and checked

Only runtime/sample/__init__.py, runtime/snapshot/materialize.py and
runtime/dataloader/loader.py changed. The cached per-device loader CUDA streams
remain intact. Use `runtime.sampling.neighbor.policy="snapshot_uniform"` with
`backbone.num_layers=1` (or `fit(sampler_options={"policy":"snapshot_uniform"})`
with the same one-layer model). The default historical policies are unchanged.
This selects the exact intra-snapshot **neighbor universe**, but finite fanout
is still a sampled-neighbor approximation; it does not add boundary decay or
make the model's sampling exact. No Prepare format/signature change is needed.

The initial real-native two-layer check exposed an existing MFG limitation:
compact and parallel outputs use the original roots in block 0 but only sampled
neighbors in block 1. Neither ordering produces the required destination-to-next-
source chain; isolated original roots disappear from the final block. The old
source-row target fallback can consequently exceed destination output rows.
The new policy therefore explicitly rejects `num_layers != 1` at construction.
It does not pretend this is repaired, alter generic MFG normalization, or add a
new sampler framework. Native equality sampling preserves snapshot keys across
hops, but that alone is insufficient for multi-layer model execution.

Final focused CPU command (OMP/MKL/OPENBLAS=1, CUDA_VISIBLE_DEVICES='', PYTHONPATH=src):
`/home/zlj/.miniconda3/envs/tgnn_3.10/bin/python -m pytest tests/test_snapshot_uniform_sampling.py tests/test_runtime_graph_blocks.py tests/test_runtime_targets.py tests/test_dataloader_features.py tests/test_access_double_buffer.py -q --tb=short`.
Result: **56 passed, 3 skipped**, 1.91s. Actual native tests executed. Coverage
includes nonidentity edge IDs/physical-row permutations/reverse duplicates,
empty snapshots, old/future exclusion, topology/native cache reuse, Event and
multi-layer rejection, independent dense-matrix mean-operator values and input/
parameter gradients, duplicate owner targets, differing task time vs native
snapshot key, and four real TGCN/Adam steps through the common loader/run_epoch.
No GPU or distributed performance run was executed. XML and per-file hashes:
`paper_method/rebuttal_20260913/snapshot_uniform_cpu_tests.xml` and
`snapshot_uniform_implementation.json`. All production modules remain <=500 lines.

R-GraphSAGE is still blocked on unavailable Method equations/verified author
code. This prerequisite must not be counted as named-model implementation,
reproduction, or evaluation coverage.


## 2026-09-13 MemShare TGN component math: CPU reference complete

Changed model/tgn.py and model/layers/temporal.py: TGN uses native unscaled attention and sampled query cutoff for GRU elapsed time. Other attention callers retain the default scale; JODIE/APAN retain the existing shared timestamp helper. Positive-event memory/mailbox commit timestamps remain unchanged. Existing Torch/DGL tensor operations reused; no new abstraction, kernel, model-local state manager, or runtime path. Production modules are 361/363 lines.

Added tests/test_tgn_memshare_math.py using unchanged native class/method bodies loaded in an isolated test namespace, with copied weights and optional native accelerators/dropout disabled. Native attention and GRU output, input-gradient and parameter-gradient tests pass fixed rtol1e-4/atol2e-6; positive commit timestamps are separately checked. Updated one query-time assertion in tests/test_starrygl_model.py. Command: python -m pytest tests/test_tgn_memshare_math.py tests/test_starrygl_model.py -k 'tgn or time' -q. Result:10 passed,1 skipped,44 deselected,8.34s. No GPU job run by this subtask.

Thin diagnostic benchmark/config and audit: paper_method/rebuttal_20260913/benchmark_memshare_main.py, memshare_main_wiki_recent_config.json, memshare_parity_audit.md. WIKI is a small development pilot, not WikiTalk/manuscript coverage. Initial weights, exact negative IDs, sampled order, dropout RNG, and persistent-state traces remain unpaired; component checks do not prove full training/convergence parity. Exact main owner state differs from native hot-replica execution. Event shared-cache reads still lack a verified version-age bound; filter max_skip alone does not establish one. Root owns temporal identity/device corrections and post-math four-GPU diagnostics; independent optimizer work is separate. Prior Flare scripts/reports remain unchanged.


## 2026-09-13 MemShare optimizer supervision: integration

Before integration, shared Prepare/task/Batch/dependency/model/task/state spine and optimization proof audited in .worktrees/memshare_optimizer_candidate. Only runtime/epoch.py changes: create the MAX flag using CUDA torch.full and use local True as proof that global MAX is True; local False still waits on active.item. All gradient reductions and flag collectives remain in the same order. Existing Torch primitive chosen; no DGL/custom kernel, no new abstraction/config/loop. Four-rank10epoch exact WIKI controlled sourcepair (onlythisfilediffers) gives e2-10 rankmaxmean .4962867923 -> .4643935349s,6.4263%less time. Native3epoch reference .2624930143s has differenthotcache/read semantics and unpairedweights/randomdraws; no crosssystempairedspeedup or convergenceclaim. CPU3 and two-rankGloo3/rankpassed; candidatewithsameTGNmath two-rankNCCL empty/nonempty/global-empty/Adam-state tests3/rankpassed. Mainfullregression pending. Sourcecandidatehash0becf178660a643c96da0c9f16b7a2d533b056ed5258ba54315ab44d865ed008.

MemShare math evidence rerun (OMP/MKL/OPENBLAS=1, CUDA_VISIBLE_DEVICES empty):10 passed,1 skipped,44 deselected,1.88s. Persisted paper_method/rebuttal_20260913/memshare_math_cpu_tests.log and .xml. Native all_update+hot replicas is a named protocol, not an established exact temporal-consistency guarantee. No source changed during this rerun.


## 2026-09-13 MemShare/R-GraphSAGE prerequisite: final verification

Main verification: all 106 production modules compile; the longest has 489 lines.
Full CPU regression: 530 passed, 9 failed, 41 skipped. The nine failing IDs are
identical to round4; no new failures. The first test launch used the workspace
root and returned exit 4; the verified rerun used starrygl-open as cwd.

The actual main package completed four-rank WIKI/TGN training for three epochs,
with ten batches each. All four recorded source manifests match the final main
hashes. The optimizer function passed three empty/nonempty supervision and Adam
state checks per rank under two-rank NCCL. The focused actual-native TGN math
selection passed ten tests with one skip. Independent snapshot sampling checks
passed 56 with three skips, including actual native execution and gradients.
Receipt: paper_method/rebuttal_20260913/memshare_final_validation.json.

Modified core responsibilities:
- utils/index.py and runtime/state/access.py: shared node/time grouping.
- runtime/event/target.py and task/target.py: temporal root and output identity.
- store/remote_fetch.py: restore response order on the payload device.
- model/tgn.py and model/layers/temporal.py: native TGN component math.
- runtime/epoch.py: avoid the redundant supervision-flag readback.
- runtime/sample/__init__.py, runtime/snapshot/materialize.py and
  runtime/dataloader/loader.py: explicit one-layer snapshot sampling.

Design notes, the method coverage record and focused tests were updated. No
legacy API or alternate model/runtime stack was added. Historical Flare results
remain unchanged.

Open issues: native hot/cache execution differs from SG exact owner reads;
initial weights, negative samples, dropout and state traces are not paired;
WIKI is a diagnostic dataset, not paper WikiTalk. Appendix W4 differs from
matched Flare W8, and threshold wording needs correction. The nine existing
failures remain. Native multi-layer MFGs do not preserve the root chain for the
new snapshot policy, so it explicitly supports one layer only. R-GraphSAGE's
Method equations or verified author code remain unavailable: the named model
is not implemented. These checks do not establish full MemShare performance or
convergence parity.

## 2026-09-13 MemShare full combination: rank-local CMA broadcast fix (pre-edit)

The actual shared-hot WIKI run completed epoch 1 then hit a confirmed epoch-2
broadcast shape mismatch: rank 0 CMA count [1,1], other ranks [875,1], sequence613.
Shared path: Prepare -> window/target/negatives -> accessor -> common Batch ->
hydrate -> encode -> task/backward -> optimizer -> StateDelta -> owner/shared
commit. Only the epoch-boundary model synchronization and Event estimator buffer
persistence change; no new runtime path. Design:
`docs/design/memshare_local_increment_state.md`.

Planned edits: model/layers/temporal.py makes observation count/sum nonpersistent
buffers; runtime/epoch.py broadcasts Tensor state_dict values (parameters and
persistent buffers). Gamma and recursive formulas remain unchanged. Reusing
Torch persistence is sufficient; DGL/custom kernels/new abstractions add nothing.
Tests: new focused checkpoint/sync tests and actual two-rank Gloo different-shape
reproduction; root owns NCCL/full run. Old checkpoints with CMA scratch keys need
those obsolete keys removed for strict loading. Tests/results pending.

## 2026-09-13 MemShare full combination: completed validation

Implemented the two buffer-persistence declarations in model/layers/temporal.py
and Tensor state_dict synchronization in runtime/epoch.py. The focused test is
tests/test_rank_local_increment_sync.py. No gamma/recurrence or owner-policy
change, no alternate runtime, no native kernel. Torch's buffer persistence
separates local observations from replicated model coefficients; DGL/custom
C++/CUDA would not address this orchestration bug.

Focused CPU: 8 passed, 1 skipped. The worker's initial Gloo launcher failed to
connect to its rendezvous store; root reran the actual two-rank test successfully,
3 passed per rank (memshare_local_increment_gloo_root.log). Full suite: 532 passed,
9 previously existing failures, 42 skipped; new failure IDs=[].

Actual four-A40 WIKI/TGN full combination: native clean10 and SG fixed clean10
both exited0; mean slowest-rank training time over epochs2-10 is respectively
.2596569061s and .3990370462s. Separate native/SG audit3 jobs exited0, SG with
distributed DEBUG enabled. Fixed clean/audit source hashes match the final main
tree. All106 Python production files compile, maximum489 physical lines.

New temporary benchmark artifacts in paper_method/rebuttal_20260913 include
memshare_full_main_config.json, run_memshare_full_main.sh,
run_memshare_full_native.sh, audit_memshare_main.py,
profile_memshare_full_native.py, memshare_full_results.{py,md,json}, and
memshare_full_validation.json. The existing thin benchmark_memshare_main.py adds
only plan, resolved sampler options and wrapper hash to its manifest. Input and
semantic audits are memshare_full_inputs.md, memshare_full_native_protocol.md and
memshare_full_main_semantics.md. Shell launchers passed syntax checks. Original
Flare reports and the old exact MemShare diagnostic remain unchanged.

The initial SG hang, DEBUG mismatch and first native instrumentation-alias
failure are retained and excluded from timing. Filtering and gamma updates are
observed, but native/main local-hot state, negative pools/weights, random inputs,
state-update timing and owner traffic are not yet paired. These are separate
full-configuration measurements, not performance/convergence parity or AP
evidence. Nominal hot10% selects923 candidates, of which875 cross partitions;
the same875 shared IDs are imported. Strict reader-age bounds remain unverified.


## 2026-09-15 Revision ablations: pre-implementation

Shared path and specialization recorded in docs/design/current/revision_ablation_20260915.md. Reuse Prepare/task/accessor/Batch/dependency/model/task/state and existing run_epoch for all four models. Extend the existing experiment CLI with regular configs and epoch-level rank-MAX timing; reuse native sampling, Torch/DGL graph operations, layerwise switch and chunk_decay. No new runtime abstraction or low-level kernel. Planned changes: cli/coupled_ablation.py, cli/experiment_metrics.py, focused experiment tests, design note and temporary external command/config launcher. TGN and three-view runs excluded. Tests and two-node smoke pending; outstanding risks include GPU-vs-host communication timing, stale-policy semantics, source/environment parity and storage capacity.


## 2026-09-15 Revision ablations: queue launched and DCRNN verified

Implemented config-driven reuse in cli/coupled_ablation.py and epoch-boundary measurements in cli/experiment_metrics.py. No new training runtime/model adapter/native operator. Prepare/task/negative/accessor/Batch/dependency/model/task/state remains the common spine; only existing sampler/view/dependency bindings vary. Torch collective reductions and task_ptr slices suffice; DGL/native hot paths are unchanged. CLI seeds are set before compile (which can instantiate TGAT), evaluation RNG is isolated, and CUDA-complete epoch time is reduced with MAX across all ranks. Experiments use a frozen copy with 131 Python/native/support files. All 107 production Python modules compile; maximum module length remains 489.

Modified/new files: cli/coupled_ablation.py, cli/experiment_metrics.py, tests/test_experiment_cli.py, docs/design/current/revision_ablation_20260915.md, this log; temporary declarations/launch/analysis under paper_method/rebuttal_20260915/{make_matrix,run_queue,summarize}.py, experiment_plan.md, matrix.json, jobs.json, launch.json, validation.json. No deprecated implementation changed.

Checks: original coupled tests 9 passed/2 skipped; combined coupled/config/runtime graph regression 22 passed/3 skipped. Reproducibility plus layerwise selection 5 passed/1 skipped. The actual two-rank Gloo timer check passed once per rank and confirmed identical slowest-rank time. gpu06/gpu07 both import the frozen package and native library with Torch 2.1.1+cu118 and DGL 1.1.3+cu118. Three real eight-rank two-epoch DCRNN smokes (exact/cache/mean) each exited [0,0], returned finite validation losses and matched source manifests. They are validation-only smokes, not final quality/performance estimates. Further model smokes are queued.

Active root: /mnt/nfs/zlj/starrygl_revision_ablation_20260915; supervisor PID 3316044 on gpu07. Queue: 9 smokes then 63 formal jobs, paired seeds 42/43/44, 100 epochs, one eight-GPU job at a time. TGN and three-view runs excluded. DConv provisionally follows plan C2 GConvGRU. Prepared DCRNN artifacts occupy about 35 GiB.

Attempts: NFS rejected cp -a permission preservation, but every copied input byte hash matched; no data recopy needed. gpu06 required explicit CUDA library paths and a working directory outside its legacy ~/starrygl import. The first supervisor waited on NFS flock; it was stopped before training and replaced with a local /tmp lock. Shared source hashes and all command/process receipts are retained.

Risks/limits: fixed budgets do not establish convergence; first-epoch cold graph I/O is substantial (DCRNN epoch1 about55s vs epoch2 about5s). Only boundary-state send bytes are presently measured, not total wire traffic or uncovered GPU communication wait. Event AP/AUC is the existing per-owner-batch mean. TGCN uses J=128 globally (16/rank), so requested s=0.1 rounds to oldest coverage2/16. Cache max_skip/max_staleness=10 is an explicit implementation condition. Least-edge partition, adaptive theta/s and further auxiliary controls remain outside this first launch batch, not silently approximated by another switch. Formal results are still pending.

## 2026-09-15 TGAT native two-layer gate: repair pending


## TGAT two-layer gate failure (before repair)

The actual two-layer native-sampler CPU experiment failed in TGAT's base feature
gather. The accessor fetches the unique union of sampled node IDs but the model
labels these rows with the first block's (possibly duplicated) src IDs. The
common feature handoff drops those IDs. Additionally, non-chain compact native
frontiers place task roots in block0, while native_target_block selects the last
block; TGAT's private node-only endpoint lookup hides this and loses query time.
The repair keeps feature node IDs in Batch.features, selects the actual root
block in the shared sampler helper, and reuses the existing target-route edge
scorer. Hidden frontier lookup must preserve (node, cutoff) using existing Torch
temporal_lookup_rows. No new model/runtime family or native kernel is needed.
Python orchestration remains per layer/window only. Test both actual native
two-layer training and different cutoffs for the same node before releasing it.
The queue supervisor is temporarily stopped at the GConvGRU preparation boundary;
its already-running CPU Prepare continues, and three completed DCRNN smokes are
preserved. A new frozen source revision is required before formal training.


## 2026-09-15 TGAT repair validated; frozen revision v2 and queue resumed

Implemented in model/tgat.py, runtime/dataloader/features.py,
runtime/sample/blocks.py and view/base.py. Preserve the fetched feature-table
node IDs; shared MFG-chain detection compares query times as well as node IDs.
Compact-frontier hidden gathers use existing temporal_lookup_rows; endpoint
scores use the common target route and correct root block. Removed duplicate
TGAT endpoint/chain lookup code. Torch tensor operations handle all rows; no
new per-node/edge loop or native operator. Existing layer-level orchestration
remains; temporal sorting adds GPU work and is included in measured training.

Tests: 84 passed, 7 skipped across test_experiment_cli.py,
test_starrygl_model.py, test_runtime_graph_blocks.py,
test_dataloader_features.py and test_runtime_targets.py. This includes native
2-layer train/eval/checkpoint repeatability, W3 TGCN layerwise/sequential
checkpoint equality, differing query cutoffs for one node, and backward gradient
routing. All 107 production modules compile; maximum 489 physical lines.

Frozen v1 and its manifest are archived under the experiment root as code_v1
and source_manifest_v1.json. Three DCRNN smokes remain explicitly v1 evidence.
No formal run existed before revision. Frozen v2 is code/, with four changed
files, all 131 file hashes verified; source_revision.json records provenance.
The GConvGRU CPU Prepare completed successfully while the supervisor was stopped.
Supervisor 3316044 resumed, releasing GConvGRU eight-rank smoke jobs. Further
model smokes and all formal results are pending. Existing experimental limits
in the launch plan remain in force.


## 2026-09-15 Eight-rank launch gates: GConvGRU and input checks

Both GConvGRU v2 smokes (exact and mean at similarity alpha0.7) completed with
node exit codes [0,0], finite validation losses, and source hashes matching v2.
TGAT H=0/.01/.1/.2 artifacts are prepared. hot_owner_audit.json verifies identical
node masters, edge masters and chunk assignments against frozen inputs in all
four artifacts; only hot sets differ (0/19/198/396 nodes). Shared-artifact
signatures match across TGCN s/executor arms and across TGAT theta arms.
The TGCN full preparation completed in about 452 seconds, and its two execution
smokes are running next. No further production changes; formal runs still pending.


## 2026-09-15 Sequential distributed GCN gate: before repair

Eight-rank TGCN layerwise passed; sequential failed before epoch1 because
cell.materialize() calls the full GCN stack without exchanging intermediate
owner embeddings. Layer2 then indexes a src layout containing remote rows with
an owner-only hidden tensor. The DGL error is a symptom; padding would silently
lose remote messages. The shared Prepare/task/accessor/Batch/dependency/model/
task/state spine is unchanged. In the existing sequential scan, cells exposing
per-layer methods will use the same compute_gcn_layer, finalize_gcn and existing
materialize_embedding_src_async as layerwise, with an immediate stream-aware
wait before the next layer. Collective order is global (window, layer), without
rank-local skipping. This fixes both TGCN and MPNN-LSTM; custom local cells retain
their current materialize contract. No new runtime/API/adapter is introduced.

Reuse Torch/DGL math and scheduled autograd all_to_all; custom/native kernels
cannot fix a missing dependency edge. Loops remain per window/layer, never per
node/edge. Validate two-rank Gloo forward and backward parity, then rerun the
failed eight-rank smoke under a new name. Source v2 and failed logs stay archived.
Also correct inherited coupled-only CLI manifest labels for config-driven models
and use setsid --wait in the temporary launcher so SSH returns torchrun's real
exit status. Formal jobs have not started; freeze a new revision before restart.


## 2026-09-15 Sequential repair v3 validated and queue restarted

runtime/snapshot/scan.py now reuses per-layer cell math and scheduled autograd
boundary exchange in its sequential path, waiting on the compute stream before
advancing each layer. The change applies to both TGCN and MPNN-LSTM; it preserves
custom local cell materialization. No new abstraction/native kernel. CLI
coupled_ablation.py removes inapplicable coupled-specific manifest fields for
config-driven noncoupled models. Tests/test_starrygl_model.py adds actual two-rank
forward, input-gradient and parameter-gradient parity; test_experiment_cli.py
checks manifest labels. Regression: 68 passed/8 skipped; two-rank Gloo parity:
2 passed per rank. All 107 production modules compile, maximum489 lines.

Archived source v2 and failed smoke_tgcn_sequential_s0.1_s42. Current v3 changes
only scan.py and the experiment CLI; all formal jobs are still pending and will
use v3. The retry job is smoke_tgcn_sequential_s0.1_s42_retry1. Supervisor3328623
restarted on gpu07. Temporary run_queue.py uses setsid --wait; an actual SSH
exit7 check returned7. The previous launcher could report zero for a detached
remote failure, but the local failure and required rank0 result gate stopped the
queue. No failed run is included in quality/performance statistics. Eight-rank
retry and TGAT smokes remain to be checked.


## 2026-09-15 Launch accepted: all four model gates passed

All nine planned eight-rank smoke gates completed with [0,0] node exit codes,
finite validation losses and the expected frozen source hashes. The initial
failed sequential attempt is preserved and excluded; its retry passed. Actual
TGCN layerwise/sequential checkpoints match across all eight ranks within
rtol1e-5/atol1e-6 (maximum absolute parameter difference2.9802322387695312e-08);
both best validation losses are0.3566224208244911 and owner workloads match.
TGAT recent and native theta0.1 smokes both passed, including two-layer feature
and time-aware root handling. These two-epoch numbers are gates, not final
ablation estimates.

Formal queue is active at dcrnn_exact_s42 on eight A40 GPUs; supervisor3328623.
All63 formal jobs use sourcev3, seeds42/43/44, 100 epochs, validation checkpoint
selection and a final test. TGN and three-view experiments are excluded.
launch_acceptance.json, validation.json and tgcn_executor_parity.json record
acceptance/provenance. Analysis script was run with zero completed formal jobs;
it lists the active job and62 queued jobs, and reports no aggregate estimates.
The source manifest's131 files were reverified. Full communication decomposition,
pooled AP and other explicitly deferred plan items remain unimplemented.


## 2026-09-22 Generic cache integration performance gate

Measured pushed `master@d5d2905` through the existing Prepare -> Store ->
`run_epoch` -> Batch -> state/cache -> model -> task -> state-update spine. No
benchmark executor, model adapter, registry, or cache implementation was added.
Flickr DCRNN used hidden size 8, 27 supervised windows, synchronized slowest-rank
CUDA epoch time, and one process per A40. Epoch 1 is cold and excluded.

| DCRNN mode | GPUs | history W | mean s/epoch |
|---|---:|---:|---:|
| exact | 4 | 3 | 6.9047 |
| bounded stale/cache | 4 | 3 | 6.9993 |
| exact | 4 | 1 | 2.6597 |
| bounded stale/cache | 4 | 1 | 2.7953 |
| exact | 8 | 1 | 2.0838 |

The requested 2.0 s/epoch gate is not met. W=3 recomputes three overlapping
snapshots per optimizer step; restoring the original W=1 DCRNN protocol removes
most of that cost. Stale cache does not hide the remaining common compute and is
slightly slower here because publication/history maintenance remains. A reused,
previously verified dense-gradient bucket candidate passed 10 CPU/Gloo tests
(one CUDA skip) plus 19 current focused tests, but its 8-GPU W=1 epochs 2--10
mean was 2.0967 s versus the unmodified 2.0838 s short control, so it was removed.

Fresh WIKI/TGN (the event-model interpretation of TGNN here), 4 A40s, 10 batches
per epoch and epochs 2--10: StarryGL 0.39127 s/epoch versus the current MemShare
checkout 0.25505 s/epoch, so StarryGL is 53.4% slower. A DGL
`edge_softmax`/`copy_e_sum` attention candidate passed three MemShare math/gradient
checks and eight current model checks but measured 0.39564 s/epoch; it too was
removed. Torch scatter, DGL operators, the prior gradient grouping, and a custom
kernel were considered; the retained Torch path is the smallest and fastest of
the measured current choices, while a custom kernel remains unjustified without
a narrower profile.

Fresh final training BCE was 0.72266 for StarryGL and 0.89238 for MemShare, but
these are not accuracy-parity evidence: the existing audit proves different
mixed-negative pools/loss weights, hot-replica state semantics, and no common
fixed evaluation negatives. Random draw identity is intentionally not required;
the distribution, loss, split and evaluation protocol still must be aligned
before AP/AUC or convergence parity can be claimed. Benchmark outputs are under
`/tmp/starrygl_dcrnn_*`, `/mnt/nfs/zlj/starrygl_d5d2905_bench/results`,
`/tmp/starrygl_tgn_d5d2905_e10`, and
`/tmp/memshare_native_fresh_20260922_e10`. Only this status log changed in this
measurement version; production returned exactly to pushed `d5d2905`.
