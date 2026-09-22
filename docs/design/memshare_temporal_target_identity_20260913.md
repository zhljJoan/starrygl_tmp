# Event row identity for MemShare parity (2026-09-13)

## TGN parity uses the clean MemShare/master reference (2026-09-22)

The TGN model remains on the unified StarryGL path and reuses the existing
prefetch/CommScheduler/StateManager pipeline.  A clean archive of MemShare's
`master` commit, rather than the dirty local `dual_dedup` checkout, passed the
existing three checks for attention output and gradients, GRU memory update,
and timestamp/mailbox commit semantics.  No model or runtime fork is needed;
the unresolved difference is end-to-end scheduling cost.

## Retain the existing native operator chain (2026-09-22)

Kernel work is outside the current cache migration.  The considered
compact-sampler dense-index replacement was cancelled before implementation;
the existing PyTorch/DGL/native graph path remains intact.  Cache work stays at
the shared dependency boundary: `StateManager`/`AsyncMemoryCommitter` serves
both DCRNN `neighbor_recurrent` and Event `node_memory`/`mailbox`, while
Snapshot history is the unavoidable materialization specialization.  No new
operator, parallel runtime path or public cache type is introduced.

## Reject Torch CSC segment attention (2026-09-22)

Event MFGs already expose CSC `indptr`, so attention softmax and aggregation
could replace destination scatter operations with public `torch.segment_reduce`.
At WIKI-like 100k edges, 8k destinations, two heads and 50 values/head, the
complete forward/backward screen took 1.290 ms versus 1.006 for the retained
scatter path, a 28.2% regression; outputs also differed by up to 5.96e-7 from
reduction reordering.  No production branch, metadata or test was added.

## Reject public coalesced framework gradients (2026-09-22)

The canonical model -> task -> backward -> optimizer path currently reduces
every parameter separately after backward.  A same-source 30-epoch diagnostic
without framework gradient synchronization bounds its cost at 0.31017 versus
0.32146 s/epoch; this is only 3.5% of the TGN time, not the parity gap.

The candidate used PyTorch's existing coalesced all-reduce at that one shared
optimizer boundary and kept the per-parameter path for mixed dtype/device
models.  TGN timing improved from 0.32146 to 0.31449 s/epoch and adjacent
four-A40 DCRNN from 3.24579 to 3.19674, but TGN test AP/AUC fell to
0.90751/0.90139.  The public operator also emits a deprecation warning in the
installed Torch.  The candidate was removed: no DDP wrapper, hook schedule,
bucket object, configuration or custom kernel remains.

## Reject packed combined owner-state responses (2026-09-22)

The shared path remains dependency access -> Batch -> model.  The existing
combined memory/mailbox hydrate already reuses one owner request and globally
ordered `CommScheduler`, but launches memory values, memory timestamps, mailbox
values and mailbox timestamps separately on its response route.  The candidate
grouped tensors by dtype/device, flattened and concatenated them once, then
split/reshaped them after the same route.

Mixed-shape/dtype and two-rank empty/nonempty checks passed.  On four A40s, WIKI
epochs 2--30 measured 0.31871 s/epoch versus 0.32146 for the adjacent control;
medians were 0.31699 and 0.31815.  The 0.85% mean and 0.37% median changes do
not justify a packet-layout contract and extra cat/split copies.  The candidate
was removed, retaining the current Torch collectives and no new cache policy,
owner semantics, p2p path, model branch, DGL work, or native kernel.  DCRNN's
Snapshot cache/history path remains unchanged.

## Reject prepared unique state writes (2026-09-22)

The shared path remains Prepare event rows -> accessor -> Batch -> model ->
runtime state commit.  `build_state_write_mask` selects the final occurrence of
each endpoint node in a window, so task-side row filtering can only remove
writes, never introduce duplicates.  A self-loop is one node even though it has
both endpoint roles; selecting its source role preserves the same embedding,
timestamp and message while making the invariant exact for new and old
artifacts.

The candidate let TGN/JODIE/APAN consume the prepared selection directly instead
of launching CUDA `torch.unique` and a latest-position scatter in every batch.
It used the existing mask/index operators; DGL and a custom kernel add no missing
operation.  It changed no cache freshness, owner/shared plane, communication
schedule, StateDelta contract, or Snapshot/DCRNN path.

The unique-only screen measured 0.33263 s/epoch.  Its bounded follow-up also
consumed the existing `TargetRoute.pos_{src,dst}_rows`, but measured 0.33964;
the retained pooled baseline is 0.32495 s/epoch.  These operations are hidden by
the asynchronous commit/next-batch pipeline.  All implementation and tests were
removed, retaining self-loop compatibility and the existing defensive dedup.

## Reject finite-only attention operator alignment (2026-09-22)

The common path remains Batch -> `StarryModel.encode` -> task.  TGN's one
specialized temporal-attention layer already matches MemShare's unscaled score,
LeakyReLU, softmax, message sum, output projection and layer normalization.
StarryGL alone applies three `nan_to_num` passes around this finite path.  The
candidate removed those extra kernels while retaining max-shifted grouped
softmax and its denominator clamp.  Native-reference output and gradient checks
passed, but four-A40 WIKI/TGN measured 0.32506 s/epoch versus the retained
0.32495 pooled baseline and adjacent 0.32248 control.  The implementation was
removed.  Existing Torch scatter and numerical guards remain; no C++/CUDA
operator or second model path is introduced.

## Reject parallel native sampler output (2026-09-22)

The existing native sampler output selector was tested without a source change
under the same accessor -> Batch -> dependency -> model path.  `parallel`
reduced four-A40 WIKI/TGN peak allocated/reserved memory to about 1.266/5.146 GB
from 1.33/5.36 GB, but epochs 2--3 averaged 0.32943 s/epoch versus the default's
roughly 0.322--0.325 range.  It remains an opt-in memory tradeoff.  No second
materializer, Torch/DGL reconstruction, Python hot-path loop, or native output
format is added.

## Reject unconditional bounded-state routes (2026-09-22)

The canonical path remains accessor -> Batch -> dependency access -> model ->
task -> state update.  Bounded TGN state hydration already uses one globally
ordered owner route with empty payloads for ranks whose owner/shared-hot rows
are complete.  Its preceding `collective_needed` all-reduce only decides
whether that route is globally empty; on WIKI every batch enters the route, so
the probe is redundant.

The candidate made distributed combined memory/mailbox reads always enter the
existing owner route, while single-rank reads retained the local fast path.  It
added no cache, route, model path, DGL graph, or native operator.  Focused and
two-rank empty-payload checks passed, but four-A40 WIKI measured 0.33174
s/epoch versus the adjacent 0.32248 control.  The small readiness reduction was
already hidden while unconditional empty routes exposed more work, so the
candidate was removed.  DCRNN's Snapshot path was unchanged.

## Reject device-resident dependency routes (2026-09-22)

The common path remains prepared window -> accessor -> Batch -> dependency
access -> model -> task -> state update.  Event and Snapshot already share the
same feature launcher; the only boundary here is whether the configured feature
cache is resident on CUDA.  In that case sampled node and edge dependency IDs
should enter the existing launcher on CUDA too.  Keeping CPU IDs currently
forces GPU row-map lookup results back through CPU before NCCL copies them to
CUDA again.

The candidate used one public Torch transfer on the existing prefetch stream
before the existing node/edge launchers.  CPU caches stayed unchanged, and it
added no route/cache interface, model branch, collective, DGL reconstruction,
or custom operator.  Focused checks passed, but two four-A40 WIKI runs averaged
0.31957 and 0.32772 s/epoch; pooled 0.32365 was slower than the adjacent
0.32248 control.  The implementation and its test were removed.

## Static event supervision schedule (2026-09-22)

The shared path remains Prepare task rows -> event accessor -> Batch ->
dependencies -> model -> task -> state update. For the built-in event-edge task,
prepared `task_ptr` fixes each rank's positive supervision activity before the
epoch, just as it does for the full-snapshot node task. The common runtime can
therefore reuse its one-vector `CommScheduler` reduction and cached boolean
schedule; event graph access, negatives, endpoint exchange, TGN math and state
commit remain the specialization boundary. Custom tasks and callbacks retain
the per-step guard. This is an extension of the existing scheduling fact, not a
new cache, communication API, loader or event execution path.

Four-rank WIKI validation retains the extension. Two adjacent ten-epoch pairs
reduce pooled rank-max time from 0.32979 to 0.32495 s/epoch; the small 1.47%
benefit is explicitly within visible run jitter. The passing accuracy repeat
ends at test AP/AUC 0.91328/0.90875, within 0.558/0.652 percentage points of
native MemShare. The remaining pooled throughput gap is 27.4%, so this is reuse
of a proven control optimization, not TGN performance parity.

Before implementation: Prepare canonical event/task rows -> DataLoader prepared task slice -> training/evaluation negatives -> Event graph accessor / native MFG -> materialized Batch -> state dependency hydration -> encode -> task -> runtime-owned StateDelta commit. Snapshot uses the same loader, Batch, task and state commit; only graph access and recurrent scan specialize.

The native MFG already carries srcdata/dstdata ts. Target route construction nevertheless searches only node IDs; two occurrences of a node at different cutoffs select the same last row. Lazy routing repeats the same lookup in model graph helpers. This prevents a valid timestamp-aware deduplication claim and can contaminate loss/gradient comparisons with MemShare.

Reuse the existing vectorized state-query (node, timestamp) grouping in utils/index.py for state hydration, root deduplication and target row mapping. Bind a temporal target route before the lazy branch when the MFG has timestamp rows; batch endpoint gathers then consume it across all Event models. Snapshot paths without timestamped MFG rows retain their current mapping. No new model/runtime chain or configuration.

Efficiency: Torch stable sorting/scatter reuse is sufficient and preserves int64 node IDs and fractional cutoffs independently, without lossy float ID packing. DGL exposes graph gathers but not this composite identity join. Existing native sampler supplies timestamp columns; a custom C++ join is deferred until profiling justifies replacing the Torch operation. No per-node or per-edge Python loop. The unrelated native sampling integer-time conversion remains a separate parity check.

Validation planned: repeated node/different cutoff target rows, both negative modes and ratios, lazy routing, missing cutoff, large int64 IDs, state grouping regression, Event/model tests. Existing source/Flare benchmark files remain frozen. A correct row join alone does not establish complete temporal attention/state or MemShare performance parity.

## First four-rank execution: owner response device repair

The actual WIKI run stopped before epoch1: Stage B submits CPU request IDs/order, receives CUDA feature payload after feature cache moves to GPU, then remote_fetch._restore_order passes the CPU order into CUDA index_copy_. This is the common owner response restoration used by features, state and mailbox; repair the index device once there. Same Prepare -> task -> accessor -> Batch -> dependency -> model -> task -> commit spine as above. No collective order or value math changes. Selected Torch .to(value.device) at the restoration point; DGL/custom C++ cannot remove the requirement that CUDA scatter indices reside on device. Tests must include a CPU order with CUDA payload and empty response; rerun actual four-rank driver. Extra per-response small index H2D remains measurable overhead.

## Replica-local negative pool alignment (2026-09-22)

The common spine remains Prepare window row -> task slice -> negatives -> graph accessor -> Batch -> dependencies -> model -> task -> state update. Only Prepare's rank-local negative candidate set changes: it is the global destination set intersected with authoritative nodes plus shared-hot read replicas. Edge ownership continues to own positive targets and loss; it does not define node-read locality.

The existing packed `node_dist_index`, `node_is_hot`, and Torch boolean indexing are sufficient. Computing the pool once in Prepare replaces the prior edge-owner destination pool plus hot merge; no cache, loader, task, model, communication, or per-entity Python path is added. DGL and a native kernel offer no useful work reduction for this one-time vectorized filter. Existing prepared artifacts retain their old pool and must be regenerated for protocol comparisons.

MemShare's correction is based on the final sampled ID's replica locality, including global-pool draws that land locally. `NegativeSamplePool` therefore accepts one optional vectorized `loss_weight_fn(sampled_ids, pool)` after random sampling. This replaces string-named correction formulas; default sampling still returns unit weights. The protocol-specific formula stays in the benchmark/task caller rather than the runtime or model, and both Event and Snapshot continue through the same task materialization function.

Fresh four-A40 WIKI validation uses the same per-epoch train -> validation -> test lifecycle on both systems, global evaluation destination pools, and persistent but separately seeded random streams. Exact random IDs are intentionally not paired. At epoch 10 StarryGL/MemShare test AP is 0.91321/0.91886 and test AUC is 0.90853/0.91527; the remaining gap is below 0.7 percentage points. This establishes protocol-level accuracy proximity for the WIKI pilot, not exact state-trace equivalence or broader dataset parity.

## Native MFG edge multiplicity (2026-09-22)

The common spine is unchanged through task roots -> native accessor -> Batch ->
dependencies -> model -> task -> state update. Profiling isolates native MFG
conversion, shared by Event and Snapshot-neighbor access. MemShare keeps sampled
block edge multiplicity and only uniques IDs for feature reads. StarryGL now does
the same by default; explicit edge deduplication remains opt-in. This removes a
second CSC traversal while the existing `sampled_edge_ids` keeps one vectorized
unique feature request. Torch/native buffers are reused; DGL block reconstruction
and a custom kernel add no missing operation. The final WIKI test AP/AUC is
0.91508/0.91062, within 0.379/0.465 percentage points of MemShare.

## Performance parity guardrail (2026-09-22)

An optimization is retained only when the full common runtime improves without
regressing the aligned train -> validation -> test protocol. Isolated savings
did not satisfy this rule: destination-prefix metadata and target-route reuse
were hidden by the pipeline; changing the optimizer flag collective did not
improve end-to-end time; fused and compact-component K/V projections reduced
memory but their reordered floating-point operations lost about 0.7 percentage
points of test AP/AUC against the unchanged repeat. All were removed. The fresh
unchanged run reaches test AP/AUC 0.91586/0.91064, within 0.300/0.463 percentage
points of native MemShare. Future fusion must keep this full-training gate; a
component or single-batch equality check is insufficient.
