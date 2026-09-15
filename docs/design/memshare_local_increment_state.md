# Rank-local Event increment statistics

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
