# Flare parity integration — 2026-09-13

## Scope recorded before integration

The selected source is `.worktrees/rebuttal_flare_layout_final/starrygl-open`,
validated on four A40 GPUs with the native Flare TGCN/Flickr protocol. Main's
original source/tests/status are preserved in
`.experiment_artifacts/rebuttal_20260913/main_before_layout_integration/`.

The canonical spine stays Prepare -> window row -> prepared task slice ->
optional negatives -> graph accessor -> Batch -> dependency access -> model ->
task -> state update. Task/output/loss ownership and scheduled collectives do not
change. Full snapshot reordering is confined to device materialization with an
epoch chunk order and window_mean supervision; last_only keeps its owner layout.
Task payload placement binds once per epoch to the accessor device, including a
return to CPU when the option is disabled. Gate and shared GCN operator changes
also apply on CPU; these are not all GPU-only changes.

## Selected changes and alternatives

- Import an existing validated node_to_chunk assignment through Prepare. Preserve
  default partition behavior and legacy fingerprints when this argument is absent.
- Reuse cat + Linear for TGCN gates, removing split matrix operations.
- Reuse the existing epoch chunk permutation for full snapshots, preserving the
  same node/snapshot sets and native recurrent prefix correspondence.
- Put existing GCN self-loop weights into the cached DGL adjacency. This removes
  sparse backward long tails without a new kernel; trainable norms remain live.
- Apply intermediate GCN ReLU before row materialization in the shared GCN layer;
  row gather/copy/zero padding commutes with this operation.
- Place prepared node IDs/labels once per epoch. task_ptr remains on CPU. The
  largest Flickr rank payload is 799.4 MiB of additional resident storage, not a
  measured training peak. Changing mode/device may transfer the whole table.

These reuse Torch/DGL tensor operations and existing storage/graph caches. DGL
kernel behavior was profiled; custom CUDA, a new loader/cache abstraction, and a
second model/runtime chain were unnecessary. No native source/build changes.
Per-node/per-edge work stays tensorized. All production modules remain <=500 lines.
The minimal encode(Batch)/state_update interface migration is still incomplete;
runtime_cell/runtime_prepare_scan/runtime_output_from_scan remain in use.

## Evidence and limits

Clean ten-epoch runs, mean rank-maximum training time over epochs 2–10:
native Flare 1.996627616 s, selected StarryGL 2.223034097 s, earlier matched
StarryGL 3.826178991 s. This is a 41.9% time reduction; StarryGL is still 11.34%
slower than native. Separate clean and instrumented numerical audits pass the
unchanged rtol=1e-4, atol=2e-6 criteria. Clean final active-parameter max absolute
difference is 8.3446503e-7. Reordering reductions is not bitwise equality.

Protocol: TGCN, full soc-flickr-growth, W8/F2/J128/s=.1, hidden8, two GCN layers,
27 batches per epoch, Adam .001, seed42, identical native owner/chunk assignments,
initial parameters, and per-rank/per-epoch chunk priorities. The benchmark uses
an explicit optimizer pre-step synchronized-gradient multiplier .25 to reproduce
native DDP normalization; this is not a change to StarryGL's default objective.
Only training convergence for one seed was checked. This result does not establish
DCRNN/GConvGRU parity, test-set quality, or complete paper performance parity.

int32, CPU autograd token, same-width transform-order <, lerp, and known-bound
dense target lookup are excluded: their isolated screens did not improve the
selected implementation consistently. Their artifacts remain for review.

Before integration, focused CPU checks: 84 passed, 4 skipped, 2 deselected;
CUDA/NCCL/device rebinding checks: 5 passed, 17 deselected. Main full regression
and strict source hash verification will be recorded in the migration status log.
