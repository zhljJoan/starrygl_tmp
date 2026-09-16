# View ablation benchmark

The benchmark isolates physical graph access while keeping the canonical event
stream, query roots, fanout, snapshot boundaries, and model-facing `Batch`
contract fixed.

Two comparisons are measured:

1. Event scan versus T-CSR lookup for one batch of historical-neighbor roots.
2. Event-to-snapshot, Event-bounded T-CSR-to-CSC construction, and prepared
   Snapshot-CSC access for one complete snapshot. The Event interval supplies
   the physical edge range and active nodes. A prepared edge-row-to-T-CSR-position
   map gathers the neighbor indices and constructs `indptr` directly, without
   sampling, reverse expansion, MFG compaction, or node renumbering.

Timing ends after construction of the `GraphBlock` and `Batch` accepted by a
model. Model forward, feature communication, and optimization are outside this
microbenchmark. Each path must return the same sorted physical edge IDs before
timings are accepted. Repeated-query totals include the one-time T-CSR or
Snapshot-CSC build cost for 1, 5, 10, 50, and 100 accesses.

The Event baseline uses vectorized Torch filtering and compaction. T-CSR uses
its prepared pointer/index buffers. Snapshot-CSC directly wraps its prepared
row. These paths reuse Torch operators and existing StarryGL builders; no new
Python per-node or per-edge hot loop is introduced.
