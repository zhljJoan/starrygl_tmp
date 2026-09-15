# Revision component experiments, 2026-09-15

Before implementation: reuse Prepare -> owner task slice -> optional negatives
-> graph accessor -> Batch -> dependency access -> model -> task -> state update.
Event and Snapshot retain the same DataLoader and run_epoch. The unavoidable
specialization is the bound sampler/view and the model's recurrent dependency;
the existing experiment CLI will accept regular sg.from_config configurations.
No new loader, model wrapper, public policy or training runtime is introduced.

Use gpu06+gpu07 (eight A40 ranks), one job at a time, frozen source and native
libraries. Seeds 42/43/44 are paired. TGN and three-view experiments are excluded.
DCRNN/GConvGRU compare exact, stale cache, and stale cache plus fixed mean
increment, including matched alpha scans. TGCN uses its existing layerwise DAG
environment switch and chunk_decay configuration. TGAT uses the existing native
boundary sampler and prepared hot-node set; owner assignments must remain fixed
when the hot fraction changes. DConv is provisionally the GConvGRU in plan C2.

First validate small CPU runs and two-node smoke runs. Epoch timing synchronizes
all ranks before work and reduces elapsed time with MAX after CUDA completion.
Report prepared owner target counts and window throughput; diagnostic counters
are separate from clean timing. Fixed-budget validation selects checkpoints;
never label an unfinished or fixed-budget run converged. Validation RNG is
isolated from training, and evaluation negative generation is fixed across arms.

Efficiency alternatives: use existing Torch reductions, tensor task tables,
DGL graph operators and the existing native sampler. A DGL kernel rewrite or
new C++/CUDA operator adds no value to epoch-level experiment orchestration.
No per-node/edge Python work is added. Cross-rank wall-time reduction and metric
aggregation execute only at epoch boundaries. Profiling host wait counters do
not establish GPU communication overlap; total wire bytes require separate
instrumentation and must not be inferred from boundary-state bytes alone.

Remaining planned work: least-edge partitioning and adaptive theta/s control
require separate implementations and validation; they are not silently replaced
with round-robin or relabeled fixed configurations. Three-view work is deferred
explicitly by the author.


## TGAT two-layer gate failure (before repair)

The actual two-layer native-sampler CPU experiment failed in TGAT's base feature
gather. The accessor fetches the unique union of sampled node IDs but the model
labels these rows with the first block's (possibly duplicated) src IDs. The
common feature handoff drops those IDs. Additionally, non-chain compact native
frontiers place task roots in block0, while native_target_block selects the last
block; TGAT's private node-only endpoint lookup hides this and loses query time.
The repair keeps feature node IDs in Batch.features, selects the actual root
block in the shared sampler helper, and reuses the existing target-route edge
scorer. Hidden frontier lookup must preserve (node, cutoff) using existing Torch
temporal_lookup_rows. No new model/runtime family or native kernel is needed.
Python orchestration remains per layer/window only. Test both actual native
two-layer training and different cutoffs for the same node before releasing it.
The queue supervisor is temporarily stopped at the GConvGRU preparation boundary;
its already-running CPU Prepare continues, and three completed DCRNN smokes are
preserved. A new frozen source revision is required before formal training.


## 2026-09-15 TGAT repair validated; frozen revision v2 and queue resumed

Implemented in model/tgat.py, runtime/dataloader/features.py,
runtime/sample/blocks.py and view/base.py. Preserve the fetched feature-table
node IDs; shared MFG-chain detection compares query times as well as node IDs.
Compact-frontier hidden gathers use existing temporal_lookup_rows; endpoint
scores use the common target route and correct root block. Removed duplicate
TGAT endpoint/chain lookup code. Torch tensor operations handle all rows; no
new per-node/edge loop or native operator. Existing layer-level orchestration
remains; temporal sorting adds GPU work and is included in measured training.

Tests: 84 passed, 7 skipped across test_experiment_cli.py,
test_starrygl_model.py, test_runtime_graph_blocks.py,
test_dataloader_features.py and test_runtime_targets.py. This includes native
2-layer train/eval/checkpoint repeatability, W3 TGCN layerwise/sequential
checkpoint equality, differing query cutoffs for one node, and backward gradient
routing. All 107 production modules compile; maximum 489 physical lines.

Frozen v1 and its manifest are archived under the experiment root as code_v1
and source_manifest_v1.json. Three DCRNN smokes remain explicitly v1 evidence.
No formal run existed before revision. Frozen v2 is code/, with four changed
files, all 131 file hashes verified; source_revision.json records provenance.
The GConvGRU CPU Prepare completed successfully while the supervisor was stopped.
Supervisor 3316044 resumed, releasing GConvGRU eight-rank smoke jobs. Further
model smokes and all formal results are pending. Existing experimental limits
in the launch plan remain in force.


## 2026-09-15 Sequential distributed GCN gate: before repair

Eight-rank TGCN layerwise passed; sequential failed before epoch1 because
cell.materialize() calls the full GCN stack without exchanging intermediate
owner embeddings. Layer2 then indexes a src layout containing remote rows with
an owner-only hidden tensor. The DGL error is a symptom; padding would silently
lose remote messages. The shared Prepare/task/accessor/Batch/dependency/model/
task/state spine is unchanged. In the existing sequential scan, cells exposing
per-layer methods will use the same compute_gcn_layer, finalize_gcn and existing
materialize_embedding_src_async as layerwise, with an immediate stream-aware
wait before the next layer. Collective order is global (window, layer), without
rank-local skipping. This fixes both TGCN and MPNN-LSTM; custom local cells retain
their current materialize contract. No new runtime/API/adapter is introduced.

Reuse Torch/DGL math and scheduled autograd all_to_all; custom/native kernels
cannot fix a missing dependency edge. Loops remain per window/layer, never per
node/edge. Validate two-rank Gloo forward and backward parity, then rerun the
failed eight-rank smoke under a new name. Source v2 and failed logs stay archived.
Also correct inherited coupled-only CLI manifest labels for config-driven models
and use setsid --wait in the temporary launcher so SSH returns torchrun's real
exit status. Formal jobs have not started; freeze a new revision before restart.
