# Third Flare parity iteration: proved target rows

## Pre-integration scope

The canonical path remains Prepare -> window row -> prepared task slice -> graph
accessor -> Batch -> dependency -> model -> task -> state commit. The snapshot
window_mean accessor already joins target IDs to graph destination rows and
filters all misses. For N destination rows, the retained route therefore satisfies
0 <= target_rows < N. Record that internal fact as TargetRoute.target_row_bound.
The shared task selector skips its repeated GPU all(valid).item only when the
model output has exactly N rows. Unknown bounds, mismatched output sizes,
root_lids, Event and custom routes retain their existing filtering behavior.
Repeated target IDs, label order, losses, gradients, and ownership are unchanged.

This reuses the existing route and task selector, not a new runtime/interface
framework. No new public configuration, GPU state cache, communication protocol,
model math, or optimizer behavior. All node/edge tensor work stays in Torch.
DGL and custom CUDA are unnecessary: the optimization avoids redundant validation
using information already established by materialization.

## Measurements before integration

Four A40 GPUs, TGCN/Flickr node regression, W8/F2/J128/s=.1, hidden8, two GCN
layers, Adam .001, seed42, identical native partition/chunks/initial weights and
priorities. The benchmark explicitly scales synchronized gradients by .25 to
match native DDP; this is not StarryGL's default training objective. No data
preparation or evaluation is included in epoch timing.

Fresh serial ten-epoch runs, mean rank-maximum time over epochs 2–10:
- Flare: 2.010964218 s.
- Previous implementation control: 2.278488272 s.
- Selected target-bound version: 2.175101861 s.

All slow epochs are retained. The previous published run was 2.223034097 s;
use the fresh control for this round's 4.54% reduction. The selected version
is still 8.16% slower than fresh Flare. One seed/run is not a stability claim.
Clean curves/final parameters pass unchanged rtol=1e-4, atol=2e-6 (maximum
parameter absolute difference 6.5565109e-7). Independent four-rank audit of
batches 1/8/27 also passes (overall maximum absolute difference 5.0663948e-7).
Exact IDs/labels and within-system rank equality are verified separately.

Rejected isolated screens: step host-read short circuit, known CSC output size,
gradient buckets, pinning, unused row metadata transfer removal, pageable async
copy, and target+step combination. No stable additional gain was observed in
those short screens. They remain isolated; this is not a statistically powered
ranking. The profile's large scalar-wait totals largely include preceding GPU
work and cannot be added up as saved wall time.

Selected source: .worktrees/rebuttal_flare_round3_target/starrygl-open. Main's
pre-integration source/tests are preserved in
.experiment_artifacts/rebuttal_20260913/main_before_round3_integration.
Only runtime/snapshot/materialize.py, task/target.py, task/prediction.py and
tests/test_target_row_bound.py are to be copied. Main regression results follow
in the migration log. Prior nine test failures, full minimal model-interface
migration, test-set/multi-seed quality, and DCRNN/GConvGRU parity remain open.
