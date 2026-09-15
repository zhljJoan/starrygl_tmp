# Legacy Plan Assertion Audit

Date: 2026-07-24

Scope:

- `tests/test_starrygl_compile_plan.py`
- This file is now a legacy migration reference, not default open-source
  gating. Default open gating is `tests/test_open_public_api.py` plus the
  runtime/model/store/task focused tests.

## Classification

### Keep As Open Semantics

These assertions describe the target semantic-to-physical lowering and should
be migrated into an open plan test after aligning names with the current
implementation:

- `test_compile_returns_lightweight_trainer_with_observable_plan`
  - Keep: `trainer.plan`, `plan.explain()`, execution spine, storage view,
    batch surface, await dependency visibility, owner policy.
  - Rewrite: do not require `partition_plan is not None` until compile builds
    a real `PartitionPlan` for lightweight declarations.
- `test_cache_none_with_reschedule_remains_owner_or_collective_await`
  - Keep: `cache_policy=none + wait_policy=reschedule` remains valid and must
    map to owner/collective await, not local/shared-hot cache.
- `test_compile_lowers_sampled_snapshot_to_blocks_contract`
  - Keep: sampled snapshot uses `Batch.blocks`/`temporal_sampling_view` and
    does not route through full snapshot `Batch.graph`.
  - Rewrite: only keep export fields that are part of the open config
    contract.
- `test_from_config_accepts_ctdg_temporal_sampling_plan_pair`
  - Keep conceptually, but CTDG/DTDG names must not become user-facing open
    API. Rewrite around `temporal="event"` plus semantic graph declaration.
- `test_from_config_accepts_dtdg_sampled_snapshot_plan_pair`
  - Keep conceptually as sampled snapshot lowering, but do not expose DTDG as
    a public backend selector.
- `test_execution_override_uses_new_policy_names`
  - Keep: `ExecutionOverride` uses canonical policy names.
- `test_shared_hot_cache_keeps_task_ownership_on_owner_plane`
  - Keep: shared hot cache never changes task/output/loss owner plane.
- `test_from_config_lowers_runtime_await_policy_without_wait_mode`
  - Keep the canonical `runtime.await` semantics.
  - Rewrite legacy `wait_mode` and `commit_order=memshare` assertions into
    import/config normalization tests only if these aliases remain supported
    as migration inputs.
- `test_compile_lowers_runtime_await_policy_without_explicit_override`
  - Keep: explicit runtime await policy can lower plan behavior without using
    `ExecutionOverride`.
- `test_from_config_derives_plan_from_canonical_runtime_policy_fields`
  - Keep only if top-level `runtime.cache_policy/freshness_policy/wait_policy`
    remain part of the open config surface.
- `test_await_dependencies_make_cache_queue_and_strict_dependencies_observable`
  - Keep: strict exact dependencies and approximate shared-hot dependencies
    must be observable in `plan.await_dependencies`.

### Keep Only After Design Decision

These tests represent plausible semantics but currently conflict with open
API constraints or naming:

- `test_compile_separates_cache_freshness_and_wait_policy`
  - Good semantic separation.
  - Needs decision: snapshot full-graph hot path layout should be `csr` or
    current `csc`? If model/backbone requires CSR for snapshot GCN, keep and
    fix lowering; otherwise rewrite.
- `test_trainer_config_exposes_new_plan_fields`
  - Good observability goal.
  - Needs decision: whether `runtime.execution_plan`, `runtime.layout_format`,
    etc. are open export fields or only `plan.*` fields.
- `test_from_config_derives_canonical_shared_hot_selector_through_sync_resolver`
  - Good canonical selector idea.
  - Needs decision: retain `sync_selector` or express this through
    `runtime.await`/`ExecutionOverride` only.

### Quarantine As Legacy Compatibility

These tests are mostly old config/alias migration behavior. They should not
define the open API unless explicitly re-approved:

- `test_from_config_derives_shared_hot_plan_for_memshare_public_historical`
- `test_from_config_derives_legacy_sync_aliases_through_sync_resolver`
- `test_compile_ignores_bare_legacy_wait_mode_alias`
- `test_compile_ignores_bare_legacy_async_commit_order_alias`

Reason:

- They preserve `memshare`/old wait-mode vocabulary.
- The open API direction uses `shared_hot`, `freshness_policy`,
  `wait_policy`, and explicit await dependencies.

### Remove Or Replace

These tests should not be migrated directly:

- `test_distributed_snapshot_detection_keeps_sampled_snapshot_out_of_full_graph_path`
  - Depends on old `starrygl.distributed`.
  - Replace with executor/runtime plan validation if needed.
- `test_sampled_snapshot_runtime_allows_local_dispatch_but_rejects_distributed`
  - Depends on private `Trainer._validate_execution_config`,
    `eval_distributed`, and `predict_distributed` behavior.
  - Replace with public compile-time plan validation once distributed sampled
    snapshot support is decided.

### Keep As Validation Rules If Still Desired

These tests encode strict validation behavior. Keep only if open config should
reject these inputs at public boundaries:

- `test_from_config_rejects_stgraph_execution_plan_alias`
  - Recommended: keep rejection. `stgraph` is a backend/legacy name.
- `test_from_config_rejects_graph_mode_execution_plan_mismatch`
  - Recommended: rewrite without CTDG/DTDG backend names.
- `test_compile_rejects_graph_mode_spec_mismatch`
  - Recommended: keep validation, but express mismatch using semantic
    temporal representation instead of backend mode strings.

## Recommended Next Open Tests

Create or extend `tests/test_open_compile_plan.py` with focused tests for:

- event sampled plan exposes:
  - `execution_spine`
  - `storage_view`
  - `batch_surface`
  - `await_dependencies`
  - `owner_policy`
  - `plan.explain()`
- snapshot full-graph plan exposes:
  - `snapshot_block_view`
  - `Batch.graph`
  - `neighbor_recurrent` await dependency
  - cache/freshness/wait policies
- sampled snapshot plan exposes:
  - `temporal_sampling_view`
  - `Batch.blocks`
  - no full-graph snapshot materialization step
- owner plane invariants:
  - edge task owner remains edge owner
  - node task owner remains node owner
  - shared hot cache does not change task ownership
- canonical await config:
  - exact/block
  - shared_hot/bounded/reschedule
  - cache none/reschedule owner-or-collective await

## Do Not Migrate As Public API

- `starrygl.distributed`
- private trainer methods used as tests
- `memshare` vocabulary as stable public output
- old benchmark config file paths
- `runtime.execution_plan` export fields unless explicitly accepted as open
  config contract

