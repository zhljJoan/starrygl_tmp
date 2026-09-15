# Sliding-window boundary cache

User decision (2026-09-11): retain cumulative increments, learnable gamma and
selective hot refresh; retain slots by sliding-window position instead of
allocating one slot for every dataset timestamp. Archived Flickr results still
refer to their archived implementation.

Current implementation (2026-09-12): the corrected V3 local/shared histories,
per-slot UPDATE blend, exact W>1 predecessor exchange, static graph reuse and
globally empty optimizer guard are now in the main package. See
[integration and validation](rebuttal_v3_integration.md). Earlier timing and
validation paragraphs below describe their dated source versions.

## Shared execution and storage boundary

The path traced before implementation is
`Prepare -> window row -> task slice -> graph accessor -> Batch -> state access ->
encode/scan -> loss/backward -> StateDelta -> owner commit/cache refresh`.
Event, Snapshot, node and edge work keep this runtime spine. This correction only
changes the existing history buffer's capacity, advancement and physical lookup.
No new loader, trainer, provider hierarchy or public cache-size setting.

## Window-relative slots

- Let W be the actual number of full snapshots plus the nonnegative chunk-decay
  prefix entries. Resolve it from the existing trainer window arguments when
  constructing state managers, including fit/evaluate/predict overrides.
- Allocate W circular output slots plus one predecessor slot, each covering
  locally needed nodes. A packet contains h, cumulative mean increment, count
  and producer version. Version zero denotes the explicit zero initial state;
  output of snapshot s records version s+1. Versions are metadata for causal
  lookup and delta-t, never indices into a dataset-sized allocation.
- For input snapshots `[1,2,3]`, retain E1/E2/E3 and a predecessor seed. When the
  window becomes `[2,3,4]`, promote the latest received observation <= E1 to the
  predecessor slot, retain E2/E3, and reuse the expired output slot for E4.
  Filtered nodes can keep an older predecessor and its cumulative statistics.
- Read the latest valid packet whose version <= the consumer snapshot id.
  Current-Batch immediately preceding local/hot outputs take priority over cache.
  Cold rows use `h + age * cumulative_mean_increment`. Local and received hot
  histories remain separate; optional gamma blends each local UPDATE with the
  corresponding shared prediction as specified below.
- Normalize actual observed changes by elapsed snapshots. A recomputed node/time
  derives its statistics from earlier observations and replaces its output slot,
  so repeatedly scanning the overlap does not count the same observation twice.
  The predecessor carries the running count across window slides.
- Late packets at/before window entry can improve the predecessor only if newer
  than its existing observation. They cannot overwrite newer reused output slots.
  Packets beyond the active window are ignored. A backward window jump requires
  reset/replay; resetting also restores the initial predecessor.

## Ownership and communication

Prepared destinations include owner nodes and hot compute replicas. Loss,
metrics and authoritative state commits remain on node_master. Models return
StateDelta; runtime writes the history after backward.

At batch commit all computed temporal outputs are retained locally, while only
the final boundary is published: hot candidates pass the existing cosine/norm
filter and max_skip, then use all_gather; cold boundary owners use the fixed
all_to_all subscriber Route bound once at setup. State, increments, count and
version travel together. Empty hot participants keep the collective order.
For audited GC/DC cells without hot fusion, received hot state is not consumed;
construction omits those hot epochs identically on all ranks, keeping cold pushes.

Main-thread batch entry polls completed receives and materializes independent
Batch tensors. Feature prefetch remains enabled; Stage B does not mutate/read
this temporal cache concurrently. In-flight packets own their send buffers.
Approximate mode adds no per-snapshot hidden-state pull. Exact W>1 exchanges live
previous states for every inner snapshot after reading its causal predecessor.
DCRNN's reset-gate exchange and wait remain per snapshot. max_skip bounds
publication skips, not snapshot age.

The prepared `snapshot_hot_compute` declaration is unchanged by this retention
correction: artifacts from the immediately preceding hot-compute implementation
can be reused. Older owner-only artifacts still require re-Prepare. Sampled
neighbor execution retains its prior state path.

## Efficiency and validation

Torch masked max/gather/scatter replaces dataset-wide temporal lookup. Reuse the
existing SnapshotHistory, StateManager, committer, model compensation and filter.
No full-buffer roll, per-node Python loop or custom kernel. Storage is O(W*N*H)
and read selection O(W*requested nodes); native/predecessor-index optimization
remains unnecessary without a measured bottleneck.

For h=8 and all 2,302,925 Flickr nodes locally needed, W=2 uses about 0.47 GiB for
three packed slots plus validity, versus the previous 71-slot estimate of
11.12 GiB. These are allocation estimates excluding graphs, other stores and
transient gathers, not measured GPU peaks.

Checks cover repeated wraparound, preserved cumulative counts, filtered old
seeds, late/future arrivals, immutable gathered tensors, fixed allocation,
window overrides, and two-rank GPU sliding training beyond cache capacity.

```bash
python -m pytest -q tests
STARRYGL_TEST_NCCL=1 python -m torch.distributed.run --standalone --nproc_per_node=2 -m pytest -q tests/test_snapshot_history.py -k history_push
STARRYGL_TEST_NCCL=1 python -m torch.distributed.run --standalone --nproc_per_node=2 -m pytest -q tests/test_snapshot_history.py -k compile_prepare
python -m compileall -q src/starrygl
```

The window-sized correction passed 278 single-process tests (20 skipped).
Two-rank NCCL history exchange/backward and compile/Prepare/sliding-fit checks
passed for DCRNN and GConvGRU. Run the distributed groups separately with
`-k history_push` and `-k compile_prepare`: a combined invocation encountered an
NCCL bootstrap failure when recreating process groups; the isolated training
group passed. Compileall and the 500-line module limit also passed.
No new Flickr convergence or speedup is claimed.

## Flickr runner audit (2026-09-11)

The current rerun uses W=1 and explicitly disables `access_pipeline`. Hot
publication calls `shared_delta` -> `change_filter.allow/update` before
all_gather. Its cosine-distance threshold is 0.3, with effective max_skip=1
(`min(configured 10, max_staleness 1)`). Local computed hot outputs are written
to SnapshotHistory before that publication filter, so skipping publication
does not make the locally computed hot replica's version stale.

Cold nodes use a fixed, rank-wide subscriber Route and push the latest boundary
after each batch, without the hot filter. Batch entry polls completed hot/cold
receives before hydration; it does not unconditionally drain cold pushes.
Two pending cold pushes trigger backpressure; final epoch flush drains all.
Recorded Flickr cold lag is zero so far: completed packets are available by
hydration. This does not measure exposed communication time: all_gather count
exchange and queue backpressure can synchronize, and this run does not profile
the overlap between transfers and batch preparation.

With `access_pipeline=True`, the existing loader can prepare/fetch the next
batch's graph/features. SnapshotHistory deliberately stays out of the worker's
state-prefetch callback; causal history hydration runs on the main thread at
batch entry. This differs from batch-end owner push. The two-rank GConvGRU W=2
compile/Prepare/fit check with the pipeline enabled passed on GPU 0/1, as did
the filtered-publication/local-freshness check. No production code or settings
of the running Flickr queue were changed.

## Confirmed smooth-aggregation target for multiple slots

The user clarified that smooth aggregation weights the locally UPDATED state
against the predicted synchronized shared state. The earlier implementation
`cached + sigmoid(gamma) * age * mean_increment` does not implement this rule.
The following is the agreed mathematical direction, not a claim that the
running experiment already implements it.

For every snapshot j in the sliding window, distinguish the local cell output
U_j from the final hot state S_j. Select a synchronized shared observation
with producer snapshot tau_j <= j, independently for that node and slot:

```
P_j = shared_state[tau_j] + (j - tau_j) * mean_increment[tau_j]
S_j = sigmoid(gamma) * U_j + (1 - sigmoid(gamma)) * P_j
```

Retain the previously requested cumulative increment statistics. Gamma is one
learnable model parameter shared across slots; alpha denotes the cosine
publication threshold, not another mixing parameter. Local compute history
and synchronized shared history must remain distinguishable. In particular,
writing a new local output must not refresh the recorded shared version.
Never apply one globally newest shared state to every slot, blend future
observations into earlier snapshots, or discard older retained slots by
deduplicating the entire window on node id alone.

Both histories retain window-relative slots plus a predecessor; producer
versions remain lookup metadata. Batch-end communication still publishes the
filtered final boundary. An arriving packet updates its corresponding retained
slot or a useful predecessor; other slots are retained and predicted as needed.
Filtering must compare against the synchronized shared observation available
before the local update, not merely this rank's previous accepted candidate.

For the existing coupled DGNN cells, the port is `cell_j -> U_j -> smooth S_j`
before output/state carry. Their graph operators already occur inside cell_j;
S_j feeds subsequent snapshot message passing. This preserves the existing
GConvGRU/DCRNN architecture, rather than claiming TGN's exact updater-before-GNN
operator ordering. The model computes the differentiable blend; runtime owns
all cache writes and owner commits. Cold nodes without a local candidate use
causal cached state/prediction rather than a fabricated two-branch blend.

Required checks before a new experiment: per-slot local/shared ages, same-age
but different-state gamma gradients, overlap carry after smoothing, shared
reference changes caused by other ranks, late packets across slot reuse, and
W>1 runs with graph/feature prefetch enabled. The earlier W=1 Flickr rerun used
gamma-scaled increments and did not validate this smooth-aggregation target.
The corrected V3 implementation and its parity tests have since been promoted;
accuracy/timing records must still identify the source and optimizer schedule.
