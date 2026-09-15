# V10 opt-in snapshot device materialization

The complete common spine remains `window row -> prepared task slice -> optional
negatives -> graph accessor -> Batch -> dependency access -> model -> task ->
state update`. This experiment changes graph-access placement inside that spine.
No loader, queue, task, model, state manager or collective schedule is added.
The existing `snapshot_materialize_on_device` option must be explicitly true;
otherwise prepared snapshot graph access remains CPU. The first supported set
is exact built-in TGCN/MPNN-LSTM/EvolveGCN, node prediction, full snapshot or
chunk-decay windows with full sampling. Coupled/custom models, Event, neighbor
sampling, edge tasks and reverse-direction materialization are rejected.

Source access uses the loader's existing prefetch stream for both threaded and
synchronous iteration. The stream waits once for the creating thread's current
stream, so already-resident input tensors are ready. Source access still binds
`comm=None` and `defer_feature_launch=True`; feature/state/embedding collectives
remain in their existing stages and ranks retain the same ordering. Queue
handoff orders each item's source enqueue before stage enqueue. Existing finish
events, consumer wait_event and record_batch_stream provide the lifetime and
execution handoff. No free p2p or global synchronization is introduced.

For explicitly enabled device access, full rows move before graph construction;
chunk blobs move before reordering/prefix slicing. The existing epoch caches
retain graph topology with their current limits/eviction. Nested route indices
and node features move with the row; route send/receive sizes stay host
metadata. Chunk reordering remaps send_index but preserves the unchanged remote
suffix; strict partial rows keep the existing local induced graph and drop
routes. Existing per-entry embedding_send_rows uses the final row layout.

TaskTarget is moved using the existing dataclass-aware pipeline mover before
any lookup or keep-index operation; IDs, node IDs, labels and timestamps
therefore share the graph device. Leaving the query on CPU would move the GPU
graph column back to CPU and defeat the optimization. Target ownership,
supervision choices, model math and state commit authority do not change.

Chunk bucket concatenation becomes a vectorized Torch pointer gather. Stable
sorting is only over chunks/local bucket IDs as before; counts stay on the
input device, and known-size repeat_interleave avoids a host size round trip.
This reuses the library operators and CSC-column concatenation already used in
V8. A DGL/custom native kernel is unnecessary for the prototype. Python
orchestrates windows/fields only, never individual nodes/edges.

Known limits: GPU nonzero, max/min/any and prefix boundary .item operations can
still synchronize once per row/slot. A snapshot first used as full and later as
partial may upload a full row again into its blob; this is not a claim of
exactly one upload per snapshot. Concurrent source/stage enqueue may cause an
event to include some next-window source work. GPU topology retention raises
memory use. Benchmarking must report these limits and preserve the same steps,
MSE and collective sequence for the existing Flickr configuration.

This version also fixes a separate shared-slice correctness issue: an empty
strict partial selection, or one containing every local owner but excluding
global chunks, used to retain full remote routes. Both now produce a local
induced graph with `chunk_limited=True`. An explicit chunk-order length is the
known global J, retained in packed rows as `_chunk_global_count`. Thus `limit=J`
preserves full topology and routes for the s=1 ablation, including on empty
ranks or ranks missing trailing buckets; `limit<J` stays local. Without a
known global J, nonnegative limits are partial and only -1 means full. A local
observed maximum cannot safely determine global fullness. These corner fixes
also apply to CPU access and must be reported separately from placement.

Validation: CPU snapshot math/cache/route tests, default CPU and explicit option
rejection tests, vectorized chunk ordering including empty buckets, plus tiny
opt-in CUDA comparisons of CPU-placement and GPU-placement outputs, gradients,
targets, routes and stream handoff. CUDA tests are authored here but executed
only by root after the frozen V9 campaign.
