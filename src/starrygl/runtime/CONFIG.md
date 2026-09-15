# Runtime Configuration Alignment

This module-local note is the authoritative default-alignment record. It does
not make these defaults stable public API and does not imply the internal
execution path has already been changed.

## Scope

The paths below name normalized runtime fields after public semantic lowering.
They do not define alternate top-level API sections; the public entry contract
lives in `docs/STARRYGL_INTERFACE.md`.

`runtime.sampling.window` follows the FlareDTDG snapshot-window defaults.
Other event/runtime defaults are aligned with MemShare-style event training.

## Snapshot Window Defaults

| StarryGL field | Target default | Source default | Current coverage | Notes |
| --- | ---: | --- | --- | --- |
| `runtime.sampling.mode` | derived from backbone when null | StarryGL semantic lowering | Covered | `full` and `neighbor` are window-local graph policies; `chunk_decay` is not a sampling mode. |
| `runtime.sampling.window.policy` | `chunk_decay` | FlareDTDG `--chunk-decay` path | Covered | `event_window` and `full_snapshot` remain valid policies. |
| `runtime.sampling.window.snaps_count` | `8` | FlareDTDG `--snaps-count=8` | Covered | Number of snapshots in one training window. |
| `runtime.sampling.window.num_full_snapshots` | `2` | FlareDTDG `--fulls-count=2` | Covered | Full snapshots appended after decayed chunk snapshots. |
| `runtime.sampling.window.chunk_decay` | `"half"` | FlareDTDG `--chunk-decay=half` | Covered | Existing parser also supports `auto:<ratio>` and explicit schedules. |
| `runtime.sampling.window.chunk_order` | `"rand"` | FlareDTDG `--chunk-order=rand` | Covered | `loss` and `perb` are source-side choices but not selected as open defaults. |
| Physical chunk count | Derived | FlareDTDG derives from `chunk_index` | Not user config | Should come from `PartitionPlan.chunk_table` or prepared snapshot metadata. |
| Node chunk ids | Derived | FlareDTDG reads node chunk index `c` | Not user config | Should stay physical artifact metadata, not public runtime config. |

## Event Sampling Defaults

| StarryGL field | Target default | Source default | Current coverage | Notes |
| --- | ---: | --- | --- | --- |
| `runtime.sampling.mode` | `neighbor` when backbone uses `sampled_neighbor` | MemShare event sampling | Covered | Event window construction remains `event_window`. |
| `runtime.sampling.neighbor.fanouts` | `[20]` | MemShare `neighbor: [20]` | Covered | Default event neighbor fanout. |
| `runtime.sampling.neighbor.policy` | `"recent"` | MemShare `strategy: recent` | Covered | Native sampler internal fallback may use a more specific recent-decay policy. |
| `runtime.sampling.neighbor.workers` | `32` | MemShare `num_thread: 32` | Covered | Used when native CPU worker sampling is enabled. |
| `runtime.sampling.neighbor.boundary_sampling.enabled` | `True` | MemShare boundary sampling experiments | Covered in example | Public spelling uses `boundary`, not the legacy source misspelling. |
| `runtime.sampling.neighbor.boundary_sampling.uniform_policy` | `"boundary_uniform"` | MemShare boundary uniform sampling path | Covered in example | Used when `neighbor.policy="uniform"`. |
| `runtime.sampling.neighbor.boundary_sampling.recent_policy` | `"boundary_decay_sampling"` | MemShare boundary recent-decay sampling path | Covered in example | Used when `neighbor.policy="recent"`. |
| `runtime.sampling.neighbor.boundary_sampling.probability` | `0.1` | MemShare shared boundary experiments use top-k/shared paths with low boundary probability | Covered in example | Default for `boundary_decay_sampling`. |

## Temporal State Defaults

| StarryGL field | Target default | Source default | Current coverage | Notes |
| --- | ---: | --- | --- | --- |
| `runtime.temporal_state.update` | Backbone-specific | MemShare TGN=`gru`, JODIE=`rnn`, APAN=`transformer` | Covered | Usually inferred from `backbone.name`; explicit override stays possible. |
| `runtime.temporal_state.consistency` | `stale_cache` | MemShare-style historical/shared-cache execution | Migration required | Current code still accepts the misleading legacy name `bounded_stale`; target semantics provide no maximum-age guarantee. |
| `runtime.temporal_state.filter.max_skip` | `1` | Existing publisher-local skip behavior | Migration required | Maximum consecutive candidate refreshes suppressed by the change filter; not a freshness bound. |
| `runtime.temporal_state.filter.enabled` | `True` | MemShare historical cache filter | Covered | Used with stale/shared-hot state reads. |
| `runtime.temporal_state.filter.min_cosine_distance` | `0.3` | MemShare historical cache `shared_memory_ssim=0.3` experiments | Covered in example | MemShare computes `1 - cosine_similarity`; this maps to cosine distance. |
| `runtime.temporal_state.smooth_aggregation.enabled` | `True` | MemShare historical cache compensation | Covered | Gamma is learnable in model/runtime implementation; config only initializes behavior if used. |
| `runtime.temporal_state.smooth_aggregation.gamma_init` | `0.5` | MemShare learnable gamma initialized from `torch.tensor([0.5])` | Covered | Non-learnable source branches use constants, but learnable mode initializes at `0.5`. |
| State kind | Backbone-derived | MemShare `memory.type=node` | Plan-derived | User should not configure `node_memory/mailbox` directly. |
| Mailbox size | Model-specific | TGN/JODIE=`1`, APAN=`10` | Model config, not runtime default | Belongs to backbone/model construction, not runtime scheduling. |

## Training Defaults

| StarryGL field | Target default | Source default | Current coverage | Notes |
| --- | ---: | --- | --- | --- |
| `runtime.train.epochs` | `50` for MemShare event defaults; `200` for FlareDTDG snapshot scripts | Source-dependent | Covered | Open example follows MemShare event defaults; benchmark configs can override. |
| `runtime.train.batch_size` | `3000` for MemShare event defaults | MemShare `batch_size: 3000` | Covered | FlareDTDG snapshot windows do not use this in the same way. |
| `runtime.train.optimizer` | `"adam"` | Both sources use Adam | Covered | |
| `runtime.train.lr` | TGN=`0.0004`, JODIE/APAN=`0.0002`; FlareDTDG=`0.001` | Source configs | Covered | Should be model/backbone profile default, not one global value. |
| `runtime.train.weight_decay` | `0.0` | FlareDTDG `--weight-decay=0.0` | Covered | |
| `runtime.train.dropout` | TGN=`0.2`, JODIE/APAN=`0.1` | MemShare configs | Covered in example | Model/profile-specific. |
| `runtime.train.att_dropout` | TGN=`0.2`, APAN=`0.1` | MemShare configs | Covered in example | Model/profile-specific. |
| `runtime.train.patience` | `20` | MemShare `--patience=20`, FlareDTDG `--early-stop-patience=20` | Not in public example | Training-loop feature; not part of compile lowering. |
| `runtime.train.max_batches_per_epoch` | `None` | Debug-only in open runtime | Covered | `None` executes every materialized batch; a positive value is a debug limit. |

## Negative Sampling Defaults

| StarryGL field | Target default | Source default | Current coverage | Notes |
| --- | ---: | --- | --- | --- |
| `task.services.negative_sampling.ratio` | `1` | MemShare `--neg_samples=1` | Covered | One negative per positive by default. |
| `task.services.negative_sampling.mode` | `"dst"` | MemShare local negative destination sampling path | Covered | Positive src is kept, negative dst is sampled. |
| `task.services.negative_sampling.train.policy` | `"random"` | MemShare random negative endpoint generation | Covered in example | Train-stage policy. |
| `task.services.negative_sampling.train.local_probability` | `0.9` | Requested open default | Covered in example | Train-time local negative source probability. |
| `task.services.negative_sampling.train.remote_probability` | `0.1` | Requested open default | Covered in example | Train-time remote negative source probability. |
| `task.services.negative_sampling.eval.policy` | `"random"` | Requested open default | Covered in example | Evaluation uses random negatives. |
| `task.services.negative_sampling.test.policy` | `"random"` | Requested open default | Covered in example | Test uses random negatives. |

## Runtime Execution Defaults

| StarryGL field | Target default | Source default | Current coverage | Notes |
| --- | ---: | --- | --- | --- |
| `runtime.seed` | `6773` for MemShare-style runs | MemShare `--seed=6773` | Runtime option exists | Keep deterministic seed as runtime option. |
| `runtime.pipeline_mode` or equivalent | `"on"` | MemShare `--pipeline_mode=on` | Internal options exist | Current open runtime uses lower-level access/prefetch flags; public default should remain semantic. |
| `runtime.profile_runtime` | `False` | Profiling flags are opt-in | Internal option exists | Do not enable by default. |
| `runtime.collect_phase_timing` | `False` | MemShare opt-in | Not public example | Profiling/debug only. |
| `runtime.skip_eval` | `False` | MemShare opt-in | Not public example | Training command behavior. |
| `runtime.device` | `"cuda"` when available | Both sources run GPU-first | Covered | Open example uses `"cuda"`. |
| `runtime.gradient_sync` | Default distributed behavior | Open runtime option | Internal option exists | Should be hidden unless required by distributed tests. |

## Preprocess And Partition Defaults

| StarryGL field | Target default | Source default | Current coverage | Notes |
| --- | ---: | --- | --- | --- |
| `runtime.preprocess.num_parts` | Derived from world size | Both sources use distributed partition artifacts | Covered only as simple example | Prefer `world_size/chunks_per_rank` through prepare path. |
| `runtime.preprocess.chunks_per_rank` | `1` unless snapshot chunking asks for more | Open prepare default | Covered in example | Splits each owner partition into lower-level chunks after partitioning. |
| `runtime.preprocess.hot_node_ratio` | `0.1` | MemShare partition `topk_ratio=0.1` and shared-hot runs | Covered in example | Public semantic name for shared-hot node ratio; internal bridges may map it to `hot_ratio` or `speed_partition_topk_ratio`. |
| Shared hot nodes | Derived | MemShare top-k shared memory path | Plan/runtime-derived | Should not change owner plane semantics. |

## Alignment Decisions

- Covered fields should stay in `runtime`; do not add top-level sections.
- Snapshot window defaults should be declared under
  `runtime.sampling.window`, not as flat runtime fields.
- Event neighbor defaults should be declared under
  `runtime.sampling.neighbor`, not as model/backbone fields.
- Exact versus stale remains under `runtime.temporal_state`; the target stale
  policy is `stale_cache`, with refresh frequency controlled separately by
  `filter.max_skip`.
- Physical partition metadata, including concrete chunk ids and chunk counts,
  should be derived from preprocess artifacts and `PartitionPlan`.
