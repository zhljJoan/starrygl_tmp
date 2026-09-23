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
