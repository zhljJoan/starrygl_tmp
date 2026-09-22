# Event row identity for MemShare parity (2026-09-13)

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
