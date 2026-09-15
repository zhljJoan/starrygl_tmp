# StarryGL Migration Plan

Status: active review sequence. Each unit is implemented and confirmed
separately; do not combine units to hide parity failures.

The behavioral source of truth is the module contract index at
[`src/starrygl/CONTRACT.md`](../../src/starrygl/CONTRACT.md). This plan names
order and gates only.

## Unit 1: Canonical Prepare Artifacts

Status: owner-local `task_ptr/payload` is implemented in `label_RRR.pt` and
Snapshot-CSC no longer stores task targets. Canonical `edge_rows` naming and
the full artifact parity gate remain open.

Implement and test:

```text
logical edge_ids versus physical edge_rows
48-bit packed node/edge locations in Python and C++
fixed artifact file ownership
flat task_ptr and owner-filtered task payload in label_RRR.pt
EventView, global T-CSR, rank Snapshot-CSC, features, and Routes
```

Gate: tiny deterministic artifacts round-trip with exact tensors, ordering,
owners, chunks, feature rows, and task rows. No runtime target fallback is used.

## Unit 2: Native T-CSR Contract

Implement one task-static `sample_neighbors(roots, scope)`:

```text
Event scope       per-root cutoff timestamp
Snapshot scope    H canonical edge-row ranges
output            compact [history][layer] MFGs plus read/scatter indexes
```

Bind partition arrays and packed edge-read locations once. Remove per-epoch
topology reconstruction and Python per-Snapshot sampling.

Gate: sampled edge/root/feature rows match the reference sampler for recent,
uniform, and boundary policies; two-rank edge-read grouping uses 48-bit ranks.

## Unit 3: Prepared Task And Graph Access

Status: implemented. Event, Snapshot-neighbor and Snapshot-CSC all slice the
same prepared task table before graph access and call the same post-graph
`attach_target_route`; Snapshot horizon/successor semantics are encoded as
empty or shifted task rows rather than loader branches.

Bind task slice, negative service, and one graph accessor at task setup.
Both native T-CSR and Snapshot-CSC return:

```text
(blocks, node_ids, edge_rows)
```

Remove `target_snapshot_id`, per-task last-window branches, Snapshot-embedded
targets, and mode-specific positive target construction.

Gate: Event, sampled Snapshot, and full Snapshot produce identical TaskTarget
semantics and correct roots/routes for node and edge tasks.

## Unit 4: Stage C/B Queue Convergence

Status: bound `DataLoader`, two queues, per-batch pinning, non-blocking prefetch
stream, compute-stream event wait and `record_stream` lifetime are implemented;
native reusable pinned buffers and multi-rank empty-participant coverage remain
part of the gate.

Keep exactly two depth-one queues:

```text
Stage C -> (window_id, targets, blocks, node_ids, edge_rows)
        -> Stage B -> (Batch, ready_event) -> Stage A
```

The event is an internal queue value, not a Batch field. Stage A turns it into
a stream dependency without host synchronization.

Replace Event/Snapshot loader closures and per-window callables with one bound
request loop and one materializer. Empty-target ranks still produce graph work,
zero loss and every collective call required by a peer.

Gate: single-rank ordering and two-rank empty-participant tests pass without a
cross-rank ticket, pending handle in Batch, per-batch plan, or third queue; a
GPU trace confirms H2D/feature payload overlap and safe buffer lifetime.

## Unit 5: Unified Dependency Access

Consume `D_remote` through one Stage-B provider path:

```text
exact node/edge features
explicit approximate stale-cache state
exact endpoint embedding collect
```

Snapshot precomputed Route and sample-driven Route share the same launch, await,
and scatter code. Dynamic reads execute counts/request/payload in fixed function order.
Exact state remains Stage A and waits its producer temporal-unit watermark.

Gate: local/remote feature and state tensors match direct owner reads, and all
ranks enter matching collective call sites despite deliberately skewed Stage-A/B arrival.

## Unit 6: Snapshot-CSC Schedule

Retain packed Flare-style Snapshot rows, epoch-static chunk permutation, local
induced prefix graphs with empty Route, and precomputed Route for recent full
Snapshots behind the common graph accessor. Implement runtime-owned layerwise
overlap and ordered coupled/model recurrent state without a second outer loop.

Gate: full and chunk-decay outputs match the reference Snapshot path per layer,
window, task loss, recurrent state, and gradient.

## Unit 7: State Parity

Align owner memory/mailbox and recurrent state with the state contract:

```text
exact owner dependency
approximate shared-hot stale-cache read
committed_through / refresh skip counter
filter and forced refresh
historical increment compensation
asynchronous commit and final drain
```

Gate: TGN-style persistent state and coupled Snapshot neighbor state match the
reference path under exact mode; stale-cache mode has an explicit quality
comparison and never claims a maximum state age. Split transitions drain and
preserve runtime state; epoch reset drains and clears temporal replicas;
checkpoint save drains and serializes without reset, and same-cursor load
restores the complete runtime state. Overlapping chunk-decay keeps live
intermediate state Batch-local but carries the detached final-Snapshot boundary
embedding only for latest-target supervision; replicated EvolveGCN state has a
multi-rank reduction test.

## Unit 8: Paper Parity And Cleanup

Run artifact, per-batch, convergence, throughput, peak-memory, and scaling
parity for representative Event, decoupled Snapshot, and coupled Snapshot
workloads. Only after the gates pass, remove transitional target/materializer,
wrapper, fallback, and legacy modules.

The 4 -> 16 GPU release targets are TGN `3.23x`, TGAT `4.17x`, and T-GCN
`2.84x`, plus paper quality/convergence and no OOM on the published setup.

Every unit updates `status_log.md` with modified files, checks, performance
alternatives, and unresolved risks. No unit claims performance from functional
unit tests alone.
