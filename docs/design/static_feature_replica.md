# Static feature replica parity step

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

Acceptance requires unchanged TGN convergence and a four-GPU train-only gain.
If full replication is useful but exceeds a later dataset's memory budget, the
same feature-provider boundary can admit a fixed prepared subset; no second
loader or model path is needed.
