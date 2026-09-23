# Rank-local Event increment statistics

## Reject Store-bound capacity (2026-09-22)

The canonical path remains Prepare -> Store -> Batch -> dependency access ->
model -> task -> state update.  The unavoidable specialization is the detached
TGN/JODIE/APAN shared-hot increment estimator; DCRNN continues to use the same
runtime state/cache managers through its snapshot dependency kind.

`sg.compile()` may construct an Event model before a Store exists, so the
estimator must initially remain growable.  Once the Store is bound, its existing
`hot_node_ids` tensor fixes the row domain.  The candidate bound that capacity
once to remove two `rows.max().item()` device synchronizations per batch without
a new cache policy, state manager, route, or model stack.  A first version also
let empty shared selections flow through native Torch indexing instead of a
third `.item()` guard.  Dynamic direct-model use retained growth.

Alternatives considered are the existing Torch indexing path, rebuilding the
model from configuration, and a custom C++/CUDA operator.  Reuse Torch and the
Store capacity: rebuilding changes model identity, DGL has no role in scratch
row allocation, and a native operator does not address the host-side check.
Focused checks passed, but the combined and capacity-only variants measured
0.33192 and 0.33117 s/epoch on four A40s versus the adjacent 0.32248 control.
The checks are hidden by other work, so all implementation and tests were
removed; the growable estimator remains.

2026-09-13, before implementation. The full WIKI/TGN shared-hot run completed
epoch 1, then failed at the next model synchronization. Distributed debug reported
broadcast sequence 613 with rank 0 shape `[1, 1]` and other ranks `[875, 1]`.

The common spine remains Prepare -> prepared task slice -> optional negatives ->
graph accessor -> Batch -> dependency hydration -> `model.encode` -> task loss ->
optimizer -> `model.state_update` -> authoritative owner commit/shared refresh.
`run_epoch` calls `sync_model_parameters` before starting the common DataLoader.
Event TGN/JODIE/APAN use `StateIncrementEstimator`; its observation count and
increment sum are rank-local, reset each training epoch, and grow to the local
observed shared-node row range. The learnable gamma remains a model parameter.
This is the necessary state specialization, not a second execution path.

The bug is broadcasting all `model.buffers()` as if each buffer were a replicated
model coefficient. The canonical hot owner can observe no remote-hot rows, while
other ranks grow their estimator. Making every rank's temporary buffer equally
large would hide the shape failure without fixing the ownership mistake.

Minimal change: mark estimator count/sum as nonpersistent Torch buffers, retaining
device movement and runtime access; synchronize Tensor entries of `state_dict()`
so parameters and persistent model buffers retain their existing synchronization.
Ignore non-Tensor extra state because `dist.broadcast` accepts tensors. No custom
kernel, registry, manager, or communication plan is needed: native PyTorch buffer
persistence already distinguishes model/checkpoint state from scratch state.
No gamma, cumulative-mean, normalization, owner, or shared-refresh math changes.

Verification will cover rank-local state surviving synchronization, persistent
buffers/parameters still broadcasting, non-Tensor extra state, checkpoint
exclusion, device/dtype movement, and two actual Gloo ranks with different
estimator row counts between two epoch-boundary synchronizations. Root performs
NCCL/full training. Historical checkpoints containing the old scratch-buffer
keys will need those obsolete keys removed for strict loading; no broad fallback
or legacy load hook is added. This change does not establish MemShare numerical
or performance parity.

## Combined historical read response candidate (2026-09-23)

The common path remains prepared event window -> Batch -> dependency access ->
model/task -> state update. TGN's unavoidable specialization is a combined
node-memory/mailbox read, but its four response tensors currently use four
collectives despite sharing one owner route. Pack their raw bytes into one
two-dimensional tensor, run the existing owner-response all-to-all once, and
restore the original dtype and trailing shape before the existing scatter.

This reuses Torch tensor views/concatenation and the existing `CommScheduler`;
DGL has no state-payload primitive. A custom C++/CUDA pack kernel is deferred
unless profiling shows Torch packing dominates. No request, cache, freshness,
negative-sampling, model, task, communication order, or public API changes.
