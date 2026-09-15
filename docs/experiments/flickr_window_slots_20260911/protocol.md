# Flickr sliding-window cache rerun

- Code: archived current src/starrygl, SHA256 recorded in every run manifest.
- Data: same Flickr temporal conversion as the preceding experiment: 2,302,925 nodes,
  70 snapshots, next-snapshot log-in-degree regression, raw in/out-degree inputs.
- Models: DCRNN and GConvGRU; each runs exact, bounded_stale without compensation,
  bounded_stale with cumulative increment and learnable sigmoid(gamma).
- Fixed settings: 100 epochs, seed 42, hidden=8, Adam lr=0.001, two ranks,
  hot ratio=0.1, max_staleness=1, cosine threshold=0.3, one input snapshot per batch.
  W=1 uses one circular output slot and one predecessor. This preserves the
  prior accuracy protocol; a multi-snapshot unroll is a separate experiment.
- Hot replicas are computed locally; task/loss/authoritative state stay on owners.
  Batch-end filtered hot all_gather and scheduled cold-owner pushes update cache.
  No synthetic delay. DCRNN still exchanges reset-gate state each snapshot.
- Train/validation/test: chronological 28/14/28 snapshots; final unlabeled snapshot
  advances state, yielding 27/13/27 scored windows. Reset/replay before evaluation;
  checkpoint chosen only by operational validation at epoch 1 and every 5 epochs.
- Approximate checkpoints are tested under operational and exact state reads.
  Exact replays use the same graph layout as the trained checkpoint.
- Hardware: four A40s. DCRNN uses GPUs 0/1; GConvGRU uses 2/3. Within each pair
  exact, no-compensation and compensation run sequentially. No GPU sharing; CPU
  and storage remain shared between the model jobs. Single-seed results.
- Timing: synchronized rank-0 train intervals include read audit; train reset
  and audit reduction are outside train timing. Validation includes reset/warm
  replay. New CLI wall intervals include setup/checkpoint/reset and final test.
  Prepare logs are archived separately. Do not compare combined final-test time: cache
  arms replay twice. GPU peak is the largest allocator peak across ranks over
  the entire run; reserved and allocated are both recorded, not nvidia-smi total.
- In the new history path, the legacy shared mask means all nonowner history
  rows, including locally computed remote-hot replicas and cold-owner cache.
  Report hot/cold counts separately; never compare its fraction directly with
  the preceding experiment's hot-only shared-source fraction.
- Outputs: /mnt/data/zlj/starrygl-experiments/flickr_window_slots_20260911.
  Original outputs and report remain unchanged.

Prepared-layout check: node/edge ownership and time splits match the old artifacts.
Across both ranks and all 70 snapshots, hot replication increases destination
rows by 10.0%, incoming edges by 86.55%, and DCRNN outgoing edges by 87.47%.
These are static layout counts, not measured operator times; see layout_work.json.
