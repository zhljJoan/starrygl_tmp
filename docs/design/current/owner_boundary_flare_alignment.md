# Owner boundary and snapshot training alignment (2026-09-12)

The shared path remains Prepare/PartitionPlan -> window row -> owner task
slice/optional negatives -> graph accessor -> Batch -> dependency access ->
model -> task/backward -> authoritative state commit and cache publication.
Event models retain their shared-hot memory policy and local/shared smoothing.
No new loader, queue, training loop, state authority or model API is introduced.

Full-snapshot coupled execution computes only owner destination nodes. Remote
boundary inputs read sliding history slots and use
`cached_h + sigmoid(gamma_boundary) * snapshot_age * mean_increment`.
The increment is the cumulative arithmetic mean. This gamma controls trend
prediction; it is distinct from Event local/shared output mixing. Cache-only
and fixed-one increment remain available for attribution.

All owner state commits update history before filtering. Each unique owner
boundary candidate is checked against its last published state, then retained
rows expand to subscribers. Counts, node IDs and state/increment/version
packets use globally ordered collectives, including empty payloads. There is
one publication per batch and a batch-boundary wait for pending publication;
there is no per-snapshot boundary owner pull or hot all_gather. DCRNN still
requires its per-snapshot reset-gate exchange.

The K bound applies to snapshot-output version age under consecutive batches
whose newest owner snapshot is complete; other starts require warmup or are
rejected. It is not a bound on optimizer-parameter versions. An overlapping
snapshot can be recomputed after an Adam step, so zero temporal age does not
imply numerical equivalence to the current exact batch.

Snapshot training can declare task `train_loss_mode="window_mean"` (default
`last_only`). Runtime creates each slot's owner/destination targets and reuses
the existing task loss. Empty slots contribute zero with the same window
denominator for loss and MSE. An empty newest slot does not discard earlier
valid supervision. Event/edge/neighbor paths reject this option. Non-coupled
training uses window-local state for full and chunk-decay windows; evaluation
retains chronological carry. Coupled state semantics are unchanged.

Recurrent state crosses a chunk-prefix/full-layout boundary by node ID, using
existing vectorized lookup/gather. Known matching owner or epoch-prefix
orders retain the pad path; manual layouts use node IDs. Static graph entries
are reused by (snapshot, chunk limit), and built-in snapshot models fetch
only consumed edge fields (including weights). Batch inputs and autograd
state are never cached. DGL/PyTorch operators suffice; no custom native
kernel, registry or compatibility layer is needed.

The integrated source is byte-identical to frozen V7. V3 main files were
hash-checked and backed up before replacing 17 production files and 8 tests;
independent main docs and CONTRACT updates were preserved. The GC example
now explicitly uses `boundary_prediction` with initial logit log(9); the old
local-mixing gamma=.5 has no equivalent interpretation under this formula.

Validation/evidence live in `paper_method/rebuttal_20260912`: V6 filtered
NCCL checks, exact output/gradient and strict-artifact fit checks; V7 hotpath
CPU 119 passed/12 skipped, supervision 91 passed/6 skipped, combined 39
passed. All 106 production modules are at most 500 lines. V6 four-A40 W3
three-epoch learnable timings are 3.135s (GC) and 6.785s (DC), training only.
These are preliminary timing results, not final convergence or Flare parity.

Flare alignment reuses its owner partition and identical first 27 training
snapshots, W8/F2/J128/s=.1 and window-mean supervision. Keep horizon=1 and
limit training to 27 batches to avoid another historical-window update at
the unlabeled boundary. Chunk assignments, random ordering, native DDP
gradient scaling and evaluation ranges still differ. EvolveGCN global state
pooling also differs from Flare's local pooling. Actual matched-profile GPU
measurements remain required; do not infer parity from the W3 numbers.

V9 follow-up: default snapshot source now uses one destination join, linear CSC column permutation and a CPU inverse table only when its span fits4*max(source,query) entries. Sparse IDs and GPU lookups retain compact sorting; duplicate matches preserve last-row semantics. All source modules equal frozen V9 after the checked integration. Four-GPU W8 results are6.017/6.766/6.688s (TG/MP/EV); this closes the original44s CPU bottleneck but does not achieve native Flare parity. Feature placement/GPU-window work remains a separately measured physical optimization. See main_v9_integration.json and flare_lookup_results.md.

V10 integrated after CUDA/NCCL and full-Flickr verification: explicit device materialization reuses the common loader/stream/cache (see ../snapshot_device_materialization.md), while default CPU placement and coupled/Event paths stay as before. W8 measured train means4.1183/4.0623/1.9628s; native Flare remains a performance reference with documented protocol/math differences. See main_v10_integration.json and flare_device_results.md.
