# Flickr DCRNN W=3 evaluation

2026-09-12: continue the previously requested DCRNN evaluation after GConvGRU finished. Use the same hash-frozen corrected package as the completed GConvGRU W=3 experiment, without production source edits. GConvGRU's early first-epoch gain did not persist: final smooth test MSE 0.06049860 vs exact 0.04670891. DCRNN results must be measured independently.

DCRNN, Flickr node regression for next-snapshot log(1+in-degree), two A40 GPUs (0/1), W=3 sliding full snapshots, unit stride, graph/feature access_pipeline enabled, seed 42, hidden dimension 8, Adam lr 0.001. Reuse existing two-rank DCRNN owner-only and hot-compute artifacts only after signature verification; no new graph partition or conversion.

Run one-epoch smooth pilot, validate finite losses, nonzero gamma gradients and causal versions, then three sequential 100-epoch arms: exact; bounded cache plus cold cumulative-mean extrapolation; bounded cache plus cold extrapolation and learnable hot fusion. Validation at epoch 1/every 5/100, checkpoint selected by actual-policy validation MSE, test replays actual arm. Hot ratio 0.1, cosine threshold 0.3, max_staleness=1 denotes publisher skip bound, no injected delay. Shared candidate merge selects newest version with deterministic tie resolution; sum/count refers to increments, not multi-rank state averaging. DCRNN retains current reset-gate communication inside each snapshot in every arm.

Timing includes read audits, computation, gradients, state updates and final flush; epoch reset and final audit reduction are outside train_seconds. Evaluation includes preceding-split warm replay. CPU layerwise profiling is not a GPU overlap measure. Fixed cold subscriber-union push can increase payload. Report stable epoch median, validation vs epoch/training time/wall time, operational test MSE, per-slot hot/cold ages and gamma gradients. Single seed, 100-epoch budget; no convergence claim from the pilot.

Outputs stay under the workspace .experiment_artifacts/flickr_dcrnn_smoothing_w3_20260912. The launcher archives metrics and updates convergence.png/pdf, report.md and time_to_target.csv automatically; it stops on a failed process or source-hash change.

## User revision: 10 epochs

On 2026-09-12 the user reduced the budget to 10 epochs and requested DCRNN/GConvGRU MSE curves. Stop the old 100-epoch supervisor, preserve the one-epoch pilot training/validation diagnostics, and stop its remaining test replay if still running. Run DCRNN exact/cache on GPU 0/1 sequentially and smooth on GPU 2/3, each two ranks and 10 epochs. Reuse the completed identical-config GConvGRU first 10 epochs: no epoch-based LR scheduler is used, evaluation remains 1/5/10, so no rerun is needed for learning curves. Do not use GConvGRU's epoch-100 selected checkpoint/test as a 10-epoch metric. Concurrent DCRNN wall times are not a controlled cross-model speed comparison. Combined comparison plots and raw CSV are generated separately.
