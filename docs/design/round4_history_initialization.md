# Round 4: bounded device memory while binding snapshot history

The canonical spine remains `window row -> prepared task slice -> optional
negatives -> graph accessor -> Batch -> dependency access -> model -> task ->
state update`. Event/snapshot use the shared state builder. Snapshot neighbor
history is its existing one-time specialization, selected in state/build.py
for neighbor_recurrent and prepared snapshot CSC. No loader, runtime, state
interface, artifact or owner policy changes.

Current bind_snapshot_history gathers the union of owner, hot and every
snapshot's source IDs with a single GPU cat/unique. All T inputs coexist on
the device with the concatenated result: at least 16*sum_t(N_src,t) bytes of
GPU input+cat when CPU source rows move to CUDA, plus unique's workspace.
Flickr rank1 header shapes imply 1,211,992,576 bytes before unique; this is a
capacity calculation, not a GPU measurement. Final history still correctly
has W+1 slots; the issue is transient initialization growth in T.

Audit found no existing read_dist_index. FeatureManager.node_ids is an exact
union only for appropriate static-one-hop artifact construction, a condition
not certified by current metadata. Avoid that assumption. Use a CPU bool
membership vector of global graph.num_nodes, index_fill_ owner IDs and each
snapshot's source IDs, then nonzero yields the identical sorted unique union
and moves it to the manager device once. The CPU scratch is N bytes plus
one live row and the 8U-byte result, independent of T. Existing Prepare
contracts guarantee nonnegative node IDs below graph.num_nodes. Per-snapshot
iteration occurs once during construction; all per-node work remains Torch.
No persistent cache, custom operator, new field or generic helper is needed.

Torch's CPU indexing/nonzero directly implements set union. DGL adjacency
operators do not apply, and DGL-inspired/custom C++/CUDA work adds unnecessary
scope. Empty hot history row_map remains unchanged deliberately. Hot replicas
are still rejected before union construction. Owner count, sorted node order,
row maps, packet allocation, versions, boundary subscriptions and ordered
CommScheduler collectives remain the existing implementation. Consumers are
history hydrate/read, commit/install, boundary push and reset; none see a
changed node set or row order.

Focused CPU tests compare against the literal old cat/unique expression,
cover duplicate rows, a remote node appearing only in the final snapshot,
empty owner/empty sources, hot rejection, and history update/read round trips.
Existing snapshot-history and coupled-ablation tests remain the semantic
regression. A thin synthetic GPU diagnostic uses the real bind helper with
one CPU source tensor repeated T=8/64/128, fixed N=1e6, U=6e5, W=8, h=8;
root alone runs it. It records peak/current allocation and reserved memory,
time, and exact sorted IDs. This is an initialization stress test, not a
large-graph training benchmark. CPU startup may cost more; report separately
from epoch timing. Source is frozen after CPU tests before GPU checks.
