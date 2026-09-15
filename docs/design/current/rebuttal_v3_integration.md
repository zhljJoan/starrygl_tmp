# Integrate the verified rebuttal runtime

2026-09-12: integration planned before production edits. This promotes frozen
`.worktrees/rebuttal_empty_step/starrygl-open`; it adds no execution policy.
The source manifest is `paper_method/rebuttal_20260912/empty_step_source_verification.json`
in the workspace. All 106 main-package Python files still match the archived
`flickr_window_slots_20260911/raw/gconv_gru_exact_s42/manifest.json`; no main-only
production edits will be overwritten. Promotion covers 19 production modules
and seven tests. Existing documentation, experiment archives and isolated
experiment versions are preserved.

## Shared path and specialization

Prepare fixes node/edge ownership, task rows, buffers and routes. The common
path remains `window row -> prepared task slice -> optional negatives -> graph
accessor -> Batch -> dependency access -> model -> task -> state update`.
Event/Snapshot and node/edge tasks share run_epoch and its bound DataLoader.
Graph access and state-kind hydration remain the physical specialization;
there is no second loader, loop, target constructor or public model entry.

Full-snapshot GConvGRU/DCRNN share static topology buffers and DGL layouts across
overlapping windows. Each batch keeps fresh feature dictionaries, state,
targets and autograd values. The device memo retains W graph entries; consumer
and existing depth-one prefetch references additionally retain in-flight
windows. Neighbor sampling, chunk-limited layouts and custom model caches do
not opt into this audited reuse.

Exact W>1 reads the causal predecessor and exchanges live previous state at
every inner snapshot. DCRNN still exchanges same-snapshot reset gates;
graph/feature prefetch cannot remove this dependency. Approximate history
separates local outputs from received hot observations. Cold prediction is
cache + age * cumulative mean increment. Optional hot smoothing blends a local
UPDATE with the corresponding shared-slot prediction; an unobserved shared row
preserves the local UPDATE. Gamma is a learned logit, not an increment scale.
The CLI exposes independent increment/publication controls. `max_staleness`
still bounds publication skips, not actual state age.

When these two snapshot models have no fusion, hot shared packets have no
consumer. Construction disables this exchange identically on every rank while
preserving owner commits and cold pushes. The internal attribution switch can
retain it; ranks cannot independently omit collective epochs.

The common optimizer helper preserves backward and gradient synchronization,
then reduces actual post-hook supervision with one scalar MAX through
CommScheduler. Synchronized optimizers step iff any rank has targets. Globally
empty windows still update model state without moving Adam moments or its
clock. Prepared counts cannot replace effective supervision because Event
filtering and callbacks may remove targets after Prepare.

## Abstractions and efficiency alternatives

No public abstraction, registry, compatibility layer or native dependency is
added. Existing SnapshotHistory, GraphBlock cache, CommScheduler and run_epoch
provide the boundaries. The new optimizer helper is reused by both branches.

- Torch handles masks, arithmetic, gather/scatter and scalar reduction; DGL
  operators continue graph math.
- DGL kernel-inspired work cannot remove redundant object/layout construction
  or a collective whose output has no consumer; reuse and deletion can.
- Custom C++/CUDA is unnecessary for static route metadata, object reuse or the
  semantic optimizer guard and is deferred unless measured work requires it.
- Existing buffer/layout reuse is selected. Python only visits task/window/
  snapshot boundaries; no per-node, edge, message or update loop is added.
  Prepared route bounds are computed once on CPU, avoiding repeated GPU index
  reductions before exchange.

## Validation and evidence limits

V3 matches its 106-module manifest; longest production module is 478 lines.
Pre-integration focused CPU: 50 passed, 10 distributed tests skipped. Archived
V3 NCCL: seven tests per rank, covering empty/mixed supervision, exact GC/DC
output/gradient parity and public sliding fit. Main validation follows promotion.

V1/V2 timings and accuracy remain historical measurements. V3 changes Flickr
from 28 Adam updates to 27, while keeping 28 forward/backward/state windows;
old/new seed statistics cannot be pooled. The additional scalar reduction has
an unmeasured cost. Static reuse has a controlled GConvGRU V2 measurement only;
DCRNN speedup, total overlap, checkpoint/resume and final multi-seed convergence
remain separate validation tasks.
