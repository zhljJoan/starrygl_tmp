# Paper Method Coverage

Status: design and release-gate mapping; it does not claim current performance
parity. The method source used in this workspace is
[method.tex](../../../paper_method/method.tex), and the checked-in evaluation
source is [experiments.tex](../../docs/paper_method/source/experiments.tex).
Module behavior is owned by the adjacent [contract index](CONTRACT.md).

## Paper-v1 Evaluated Paths

| Temporal representation | Aggregation / coupling | Models | Required physical path |
| --- | --- | --- | --- |
| Event | sampled / decoupled | TGAT | EventView + T-CSR + Stage-B exact fetch |
| Event | sampled / coupled | JODIE, TGN, APAN | T-CSR + memory/mailbox version group + historical cache |
| Snapshot | full/chunk / decoupled | T-GCN, MPNN-LSTM | Snapshot-CSC + runtime-owned layerwise scan |
| Snapshot | full/chunk / coupled | EvolveGCN | Snapshot-CSC + chronological replicated model state |

Node/edge tasks are variations of the same task table and ownership path, not
new runtime paths. Event full graph is not evaluated and must not create a
second runtime without a concrete model and benchmark.

Snapshot-neighbor and coupled GConvGRU/`neighbor_recurrent` are supported
contract extensions. They require correctness tests but cannot substitute for
the four evaluated rows above when claiming paper-v1 parity.

## Method To Module

| Paper capability | Contract owner |
| --- | --- |
| owners, hot nodes, chunks | `partition/CONTRACT.md` |
| canonical table and multi-view artifacts | `prepare/CONTRACT.md` |
| T-CSR causal/history sampling and boundary retention | `native/CONTRACT.md`, `runtime/sample/CONTRACT.md` |
| Snapshot-CSC, chunk decay and layerwise Route | `runtime/snapshot/CONTRACT.md` |
| two-queue CPU/H2D/NCCL/GPU overlap | `runtime/CONTRACT.md`, `runtime/dataloader/CONTRACT.md` |
| exact/selective-refresh stale cache and compensation | `runtime/state/CONTRACT.md` |
| node/edge ownership and random negatives | `task/CONTRACT.md` |
| unified model input | `view/CONTRACT.md`, `batch/CONTRACT.md` |

## Paper-Critical Invariants

- Event, T-CSR and Snapshot-CSC are secondary indexes over one canonical edge
  table and global temporal-unit index.
- Event MFG identity is `(node_id, cutoff_ts)`; Snapshot versioned-feature
  identity is `(snapshot_id, node_id)`.
- Sampled reads use counts/request/payload Route phases; full Snapshot consumes
  prepared boundary Route. Both reuse the same communication operations and
  keep empty ranks in every collective required by a peer.
- Decoupled full Snapshot uses runtime-owned layerwise scheduling; chronological
  recurrent update starts only after spatial layers finish.
- Coupled stale-cache reads are an explicit approximation without a maximum-age
  guarantee. `max_skip` limits consecutive filtered refreshes only.
- Owner and shared-hot planes remain separate.
- Layer embedding and historical state share Route/launch/events only; they
  do not share cache storage or lifetime.

## Reproducible Release Gates

No paper-performance claim is valid until one benchmark record fixes git/code
revision, artifacts, config, seed, hardware, warmup, measured steps and units,
then records median steady-state throughput, peak CPU/GPU memory, communication
bytes, quality and Stage overlap.

Required gates are:

1. artifact parity for owners, chunks, T-CSR, Snapshot-CSC, features and Route;
2. per-Batch parity for targets, sampled edges, feature reads, exact/stale state
   values, loss, gradients and updates;
3. two-rank and paper-scale correctness with empty collective participants,
   deliberately skewed Stage-A/B arrival, and clean early exit;
4. paper AP/Macro-F1/MSE and convergence protocol for every evaluated model;
5. 4 -> 16 GPU normalized throughput at least TGN `3.23x`, TGAT `4.17x` and
   T-GCN `2.84x`, with no OOM on the published A40/100-Gbps setup;
6. a GPU trace showing Stage-C sampling overlaps A(k), and Stage-B H2D/NCCL
   overlaps A(k) without hidden global synchronization.

Plan-lowering tests and four representative JSON files prove neither execution
coverage nor performance.

## Manuscript/Reference Alignment Before Claim

The evaluated Flare reference generates one chunk permutation per epoch and
uses local induced graphs with empty Route for decayed history; only the recent
full Snapshots perform boundary exchange. The current method prose says
“each training iteration”. Paper-v1 follows the evaluated reference contract in
`runtime/snapshot/CONTRACT.md`; the manuscript must adopt that wording, or the
per-iteration variant must be rerun through all quality/throughput gates.

Known implementation blockers remain in each module contract and
`docs/migration/status_log.md`; until they close, release readiness is `No-Go`.


## 2026-09-13 incremental verification

The [current manuscript/config/module audit](../../../paper_method/rebuttal_20260913/current_parameter_module_audit.md)
separates reference matching from appendix matching: submitted appendix W4 and
matched Flare W8 differ. Current Event work has verified timestamp-aware target
routes, CUDA owner-response ordering, and two TGN math components against actual
MemShare source. WIKI four-rank diagnostic execution is not a paper WikiTalk run,
nor a paired MemShare convergence/performance reproduction. Native all_update hot
replicas and SG exact owner reads still differ; initial weights, negative arrays,
dropout and state traces remain unpaired.

`neighbor.policy="snapshot_uniform"` now provides tested one-layer independent
snapshot sampling through the existing native sampler and common loader. Multiple
layers are explicitly rejected because the native MFG chain loses target roots.
R-GraphSAGE itself is not implemented: its full recurrence/aggregation/sampling
definition or verified author code is still needed. This prerequisite does not
close named-model coverage or the multi-layer TGAT/native sampler audit.
