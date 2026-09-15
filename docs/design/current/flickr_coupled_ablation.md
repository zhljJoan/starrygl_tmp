# Flickr: actual bounded_stale/shared_hot convergence ablation

Current rerun (2026-09-11): the historical protocol/results below describe the
archived owner-only experiment. The new run uses the window-slot implementation
in [snapshot_boundary_history.md](snapshot_boundary_history.md), with W=1 plus
one predecessor, hot compute replicas, filtered hot all_gather and cold-owner
push. It runs all six model/policy/compensation arms for 100 epochs. Each GPU
hosts one rank; two model jobs can share host CPU/storage but never a GPU.
The shared execution path remains Prepare -> Store -> run_epoch -> Batch ->
state hydration -> scan/task -> state commit. Only the existing CLI read audit
and epoch-level timing metadata are extended; no alternate training loop.
New output: `/mnt/data/zlj/starrygl-experiments/flickr_window_slots_20260911`.


Use the local current `starrygl-open/src/starrygl` implementation. Models:
DCRNN and GConvGRU. Task: next-snapshot log-in-degree node regression with
raw in/out-degree inputs, using the current Flickr converter (33,140,017 raw
edges, 2,302,925 nodes, 70 prepared snapshots). Historical TGM logs are excluded.

## Shared execution and model boundary

The path traced before implementation is:
`convert -> compile -> Prepare -> Store -> window row -> prepared task slice ->
graph accessor -> Batch -> state hydration -> coupled scan -> task -> state commit`.
The common `run_epoch` owns every training/evaluation batch. DCRNN's two local
DGL diffusion operators require a runtime-owned reset-gate exchange between
them through the existing scheduled autograd Route. Prepare includes incoming
and outgoing remote neighbors with global directional degrees. No second loader,
state manager, communication stack or training loop is added.

DCRNN uses a size-2 filter (zero-hop and one-hop terms in each direction),
unit self-loops and an input projection. Its random-walk supports use the
transposed orientation of the [authors' DCGRU implementation](https://github.com/liyaguang/DCRNN/blob/master/model/dcrnn_cell.py).
The unit-loop convention follows the local DynaHB-derived reference; this is
a node-regression cell experiment, not a reproduction of the published traffic
sequence-to-sequence architecture. States are detached between batch windows.
Only prepared full-snapshot execution is evaluated; chunk-limited directional
diffusion and higher-order cross-rank diffusion are outside this experiment.

Exact uses StateManager. Approximate runs use the existing AsyncMemoryCommitter
and materialize_bounded: owner-local reads first, shared-hot reads for remote hot
nodes, owner fetch for misses. No fixed delay is injected. The earlier synthetic
delay provider and CLI injection were removed once the user clarified scope.
The read audit observes the hydrated Batch before model math and does not modify it.

## Paired protocol

- Two ranks, 10% hot nodes, one full snapshot per batch, hidden dimension 8,
  Adam learning rate 0.001. Same ownership, labels, features and initialization
  within each model. Feature replicas contain needed cross-partition neighbors.
- Chronological 40/20/40 train/validation/test windows. No test selection.
- Initial complete-data pilot: seed 42, 100 epochs, validation at epoch 1 and
  every 5 epochs. The epoch and split boundaries preserve recurrent semantics:
  reset before each epoch/replay and warm up all earlier splits before scoring.
  Unlabeled boundary snapshots advance state without contributing loss.
- Compare exact, bounded_stale with max_staleness=1, and the same policy with
  current smooth aggregation enabled. Check the current compensation actually
  updates before attributing any benefit to it.
- Choose best checkpoint by validation under the actual policy. Report MSE and
  RMSE on actual-policy test replay and exact-state replay of that checkpoint.
  Epoch curves show train/validation MSE. Report timings as observed runtime;
  parallel jobs and auditing mean these are not isolated throughput benchmarks.
  Compare both per-epoch training and cumulative training plus validation at
  matched epoch budgets; also plot validation MSE against each time measure.
  Report the first observed validation threshold crossing (five-epoch cadence).
  Timers are synchronized rank-0 intervals, not full wall time: training excludes
  the preceding state reset and following audit reduction; validation includes
  warm replay. Setup, checkpoint writes and final tests are outside these sums.
  Approximate final evaluation additionally replays exact state, so its test
  timer cannot be compared directly with the exact arm's single replay.

## State-age semantics and checks

A state produced by snapshot s carries version s+1; initial state is version 0.
At snapshot t, exact predecessor version is t. Measured lag is t minus the
returned timestamp. Report actual shared-source count, shared fraction, lag
histogram, initial shared rows, future rows and stale owner rows.
`max_staleness` limits skipped publisher refreshes; it is not asserted to be a
universal snapshot-age bound. The cache policy and read audit are separate.

The tiny two-A40 run shows shared reads with lag 0/1 and zero stale owner reads.
Current compensation receives only remote shared reads, while its increment
update requires those nodes to be locally produced. Consequently its updates
are zero and gamma stays 0.5 in this owner-only layout. The DCRNN tiny run's
train/validation/test results match with and without compensation. Preserve and
report this current behavior, rather than substituting a different algorithm.
Both full-data paired models also matched training MSE exactly over their first
four epochs with/without compensation. The two redundant diagnostic jobs were
then stopped; four main exact/shared-hot runs continue. Diagnostic output is
explicitly marked as an effectiveness check, not a completed convergence run.

Correctness gates include dense DCRNN diffusion/gradient reference, two-rank
complete-graph output/gradient parity, serialized directional-buffer round trip,
permuted state-read ordering, owned-node state commits across unlabeled windows,
temporal feature shard mapping and a complete node-regression replay smoke.

Efficiency alternatives considered: vectorized Torch preparation and gathers,
DGL sparse reductions (avoiding edge-by-feature message temporaries), and reuse
of the current Route. No custom C++/CUDA operator or DGL kernel rewrite is needed
for this accuracy experiment. Python operates at snapshot/epoch boundaries only.

Artifacts: `/tmp/starrygl_open_flickr_ablation_20260911`; GConvGRU prepared data
is stored under `/home/zlj/starrygl-undate/.experiment_artifacts` to fit local disks.
Each run records source SHA256 values, compile config, plan, epoch JSONL, rank
checkpoints and result JSON. The four main runs completed 100 epochs and final
test replay; both ranks' model checkpoints match exactly. Results, algorithm,
timing definitions, source snapshot and reproducible analysis are archived in
`docs/experiments/flickr_shared_hot_20260911/`.

The fixed shared checkpoints' operational/exact test MSE differences are about
-0.0244% for DCRNN and -0.0254% for GConvGRU. The independently trained arms
differ more, so those gaps also reflect training and checkpoint selection.
There is no measured training speedup in this concurrent run. GConvGRU shared
timing includes compensation-path work despite its zero increment. Validation
still decreases 14%-21% over the final 20 epochs, and the persistence baseline
outperforms both models on test. These are fixed-budget, single-seed results,
not evidence of fully converged accuracy or isolated throughput.
