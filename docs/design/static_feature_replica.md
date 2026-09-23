# Rejected static node-feature replica parity step

Shared path: prepared Event row -> native sampling -> Batch -> dependency
access -> model/task -> state update. Static node features specialize only the
dependency provider; model, task, loader, state and communication interfaces
remain unchanged.

WIKI node features are immutable and small enough to materialize as existing
`node_replicas`. Preparing each rank with those rows removes the per-window
node-feature owner request/count/response phase. The option is off by default,
so larger datasets keep partitioned features and their existing `shared_nodes`
replicas. This reuses Torch indexing and `FeatureManager`; DGL and a custom
kernel do not apply, and no Python entity loop is added.

The 50-epoch result preserved convergence but did not improve train time, so
the Trainer/config wiring was removed. The existing low-level shard-builder
capability remains unchanged; no runtime cache or second execution path was
added. Sampled edge features, not node features, are the next feature-volume
candidate.

## Sampled edge-feature upper bound

The same optional Prepare lowering can replicate immutable edge features and
let `FeatureManager` satisfy sampled-edge reads locally. This removes the
edge-feature owner request/count/response phase without changing Batch or model
math. It defaults off; WIKI enables it only to test the communication upper
bound before considering a capacity-bounded prepared replica set.

## Reject identity-replica pre-compaction bypass

The common dependency provider receives node IDs already compacted by the
native sampled-feature graph. A Stage-B breakdown still attributed about
0.15 s/epoch to feature launch because generic fetch deduplicated IDs before
checking the existing replicated identity layout. The minimum candidate moves
that layout check ahead of generic compaction. Direct Torch indexing preserves
requested order and duplicates, so no model, Batch, route, cache API, DGL path,
or kernel changes. Partitioned/non-identity reads retained their current compact
request path. Focused tests passed, but four-A40 epochs 2--10 measured median
0.6352 s versus the retained short screen's 0.5968 s. The apparent feature
launch cost was deferred GPU work settling at compaction, and removing that
boundary worsened scheduling. The candidate was removed.
