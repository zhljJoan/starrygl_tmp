# Independent snapshot neighbor sampling prerequisite

2026-09-13, before implementation. This is a sampling prerequisite, not an
implementation or reproduction of R-GraphSAGE. The verified primary source is
Yao et al., Pattern Recognition 154 (2024), 110577,
https://www.sciencedirect.com/science/article/pii/S0031320324003285.
Its public abstract and conclusion describe integrated recurrence without
additional LSTM/GRU blocks. No method equations or verified author code were
available; do not substitute GraphSAGE followed by GRU under that model name.
The missing input is the Method section, including recurrence/aggregation and
sampling definitions, or the author's implementation.

## Shared path and specialization

Prepare's fixed owner partition emits snapshot_csc and temporal_csr for sampled
Snapshot plans. Prepared owner task slices feed the common DataLoader, optional
negatives, graph accessor, native MFG sampling, FeatureManager, Batch, dependency
hydrate, encode, task loss/backward, and runtime StateDelta commit. No new model,
loader, target constructor, queue, state manager or training loop is introduced.

Existing Snapshot access passes max(edge timestamp)+1 to temporal sampling,
which samples historical edges. Native `dtdg_uniform` already supports equality
to a time key at every sampled hop (sampler.h:643 onward). The new explicit
`sampler_options.policy="snapshot_uniform"` uses that native operation; its name
describes graph semantics without exposing a backend name. Existing policies
and Event sampling retain their behavior. Event use of this policy is rejected
at sampler construction.

Prepared physical edge rows (`edge_feature_ids`) and `time_ptr_2` determine each
edge's snapshot ID once while the task-static native topology is bound. Thus
different physical timestamps within one snapshot and gaps/empty snapshots are
handled without treating wall-clock timestamp equality as snapshot membership.
Forward and reverse edge copies share the same physical row. The source Prepare
view is never mutated, and the existing topology/native caches hold the bound
tables. Cache identity must include the window pointer table. Accessor roots
use row.snapshot_id only for the explicit policy, preserving target timestamps.
Malformed/uncovered/overlapping prepared ranges fail at construction.

## Efficiency and checks

Use one vectorized Torch bucketize/indexing pass at sampler construction and
the existing C++ sampler for per-node/per-edge work. No Python loops over graph
entities or new kernel/cache framework. No post-sampling lower-bound filtering:
that would change the fanout distribution and could keep cross-snapshot MFG
dependencies. Actual native CPU tests will check two hops, old/future exclusion,
empty snapshots, feature IDs, default temporal behavior and cache reuse. A small
sampled Batch forward/backward check uses an existing direct model; it makes no
claim about the unresolved R-GraphSAGE equations or distributed performance.


## Implemented and checked

Only runtime/sample/__init__.py, runtime/snapshot/materialize.py and
runtime/dataloader/loader.py changed. The cached per-device loader CUDA streams
remain intact. Use `runtime.sampling.neighbor.policy="snapshot_uniform"` with
`backbone.num_layers=1` (or `fit(sampler_options={"policy":"snapshot_uniform"})`
with the same one-layer model). The default historical policies are unchanged.
This selects the exact intra-snapshot **neighbor universe**, but finite fanout
is still a sampled-neighbor approximation; it does not add boundary decay or
make the model's sampling exact. No Prepare format/signature change is needed.

The initial real-native two-layer check exposed an existing MFG limitation:
compact and parallel outputs use the original roots in block 0 but only sampled
neighbors in block 1. Neither ordering produces the required destination-to-next-
source chain; isolated original roots disappear from the final block. The old
source-row target fallback can consequently exceed destination output rows.
The new policy therefore explicitly rejects `num_layers != 1` at construction.
It does not pretend this is repaired, alter generic MFG normalization, or add a
new sampler framework. Native equality sampling preserves snapshot keys across
hops, but that alone is insufficient for multi-layer model execution.

Final focused CPU command (OMP/MKL/OPENBLAS=1, CUDA_VISIBLE_DEVICES='', PYTHONPATH=src):
`/home/zlj/.miniconda3/envs/tgnn_3.10/bin/python -m pytest tests/test_snapshot_uniform_sampling.py tests/test_runtime_graph_blocks.py tests/test_runtime_targets.py tests/test_dataloader_features.py tests/test_access_double_buffer.py -q --tb=short`.
Result: **56 passed, 3 skipped**, 1.91s. Actual native tests executed. Coverage
includes nonidentity edge IDs/physical-row permutations/reverse duplicates,
empty snapshots, old/future exclusion, topology/native cache reuse, Event and
multi-layer rejection, independent dense-matrix mean-operator values and input/
parameter gradients, duplicate owner targets, differing task time vs native
snapshot key, and four real TGCN/Adam steps through the common loader/run_epoch.
No GPU or distributed performance run was executed. XML and per-file hashes:
`paper_method/rebuttal_20260913/snapshot_uniform_cpu_tests.xml` and
`snapshot_uniform_implementation.json`. All production modules remain <=500 lines.

R-GraphSAGE is still blocked on unavailable Method equations/verified author
code. This prerequisite must not be counted as named-model implementation,
reproduction, or evaluation coverage.
