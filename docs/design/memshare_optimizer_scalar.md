# Supervision flag: isolated MemShare optimizer candidate

## Reject fused CUDA Adam (2026-09-22)

The canonical path already has one optimizer construction before every epoch
loop and one shared optimizer step after Event/Snapshot backward. The profile
attributes about 50 ms across 30 TGN optimizer steps to the current foreach
Adam path, although asynchronous GPU timing means only an end-to-end run can
establish removable time.

The minimum candidate keeps the public optimizer names and hyperparameters but
uses PyTorch's public `fused=True` lowering for framework-created CUDA Adam and
AdamW. CPU construction and any optimizer passed by the user are unchanged.
This is a physical operator choice, not a new option or optimizer API. DGL is
unrelated; a custom CUDA optimizer is unjustified while the installed Torch
primitive exists. Retention requires optimizer-state/numerical checks and a
material four-A40 TGN gain without losing the DCRNN target.

The CUDA trajectory check matched foreach Adam for five steps, but the real
WIKI/TGN run averaged 0.33782 s/epoch over epochs 2--10 versus 0.32967 for the
adjacent control, a 2.47% regression. The candidate was removed without running
DCRNN or full accuracy because the first performance gate failed.

## Reject persistent gradient views (2026-09-22)

The shared boundary is unchanged: Event/Snapshot and node/edge execution all
reach backward -> `sync_gradients` -> optimizer before runtime-owned state
commit. The earlier dense-bucket experiment allocated, flattened and copied
gradients every step; native coalescing retained separate gradient buffers.
Neither tested a persistent flat buffer whose slices are the parameter gradient
buffers themselves.

The bounded candidate creates one zeroed Torch buffer per device/dtype at epoch
setup, binds every trainable parameter's `.grad` to a shaped view, preserves
those views with `optimizer.zero_grad(set_to_none=False)`, and reduces each flat
buffer once after backward. Missing/empty-owner gradients are already zero
views, so collective membership is static. It adds no public option, DDP hook,
cache manager, model branch or custom kernel. DGL is unrelated to parameter
reduction; custom C++/CUDA is unjustified while public Torch tensor views and
collectives express the operation. Retention requires focused Gloo/NCCL
optimizer parity, TGN gain, and DCRNN staying at the 2 s/epoch target.

The implementation passed focused local, Gloo and NCCL checks, but was removed
after timing. Two ten-epoch candidate runs averaged 0.33035 and 0.32078 s/epoch
over epochs 2--10. Their pooled 0.32556 s/epoch was only 0.33% faster than
three control runs pooled at 0.32663 s/epoch, with opposite direction across
individual pairs. That is run jitter, not a reusable performance win. The TGN
gate failed, so no accuracy or DCRNN campaign was spent on this candidate.

## Reject direct native DDP event lowering (2026-09-22)

Before implementation, Event and Snapshot share Prepare -> accessor -> Batch ->
dependencies -> model -> task -> backward -> optimizer -> state commit. Native
MemShare wraps TGN in PyTorch DDP, while StarryGL currently waits until backward
finishes and then reduces every parameter separately. The unavoidable boundary
is model execution: direct Event models already implement `forward(Batch)` and
can enter DDP; runtime-owned Snapshot recurrent scans cannot be wrapped without
moving scan semantics into the model.

The minimum candidate bound DDP once in `fit`, called the standard model
forward path, and kept state update on the underlying StarryModel. Focused
single-process coverage passed, but the first four-rank smoke deadlocked before
epoch 1 when DDP and endpoint autograd collectives shared the default group. A
second smoke gave DDP a separate all-rank process group, matching MemShare's
physical separation, but still deadlocked before epoch 1. The measured Event
configuration has local target rows and does not use endpoint-embedding
autograd exchange, so the earlier endpoint-specific attribution was too strong.
The remaining reducer/rank divergence is unresolved. Both attempts were
terminated and production code was removed.

Direct DDP wrapping is therefore rejected. The current explicit
post-backward synchronization remains the safe common path. Reconsider overlap
only after the complete per-rank forward/backward collective sequence is
traced; adding another wrapper or process group is insufficient.

## Native gradient coalescing candidate (2026-09-22)

Before implementation, the shared path remains Prepare -> accessor -> Batch ->
dependencies -> model -> task -> backward -> `sync_gradients` -> optimizer ->
state commit. Event/Snapshot and node/edge execution all meet at this boundary;
only parameter device/dtype is a necessary physical grouping. TGN currently
issues 27 same-dtype gradient all-reduces per batch. The candidate will reuse
PyTorch distributed's native coalescing context plus `torch._foreach_div_`,
without flat-buffer copies, a DDP wrapper, model branch or configuration. The
existing per-parameter path is restored unless focused Gloo/NCCL correctness and
adjacent four-A40 timing both hold; custom CUDA is unjustified before that test.
The candidate was removed: CPU/Gloo and NCCL each passed 6 tests per rank, but
two adjacent timing pairs pooled to only 0.32512 -> 0.32247 s/epoch (-0.81%).
Its full evaluation ended at test AP/AUC 0.91145/0.90607, and the implementation
depended on a private PyTorch API. That combination does not clear the retention
gate; production stays on the public per-parameter collectives.

Status before implementation: frozen baseline copied; only runtime/epoch.py will change.

Shared path: Prepare owner task table -> task slice/optional negatives -> Event T-CSR or Snapshot graph accessor -> unified Batch -> dependency hydration -> encode_model -> endpoint/task loss -> backward -> step_optimizer -> runtime-owned state commit. Both empty and supervised branches in runtime/loop.py call the same step_optimizer; Event/Snapshot and node/edge ownership do not specialize at this boundary. CPU/GPU device is the only physical distinction.

Evidence: 30 calls over three diagnostic epochs on rank0 spent .272989s inside step_optimizer torch.tensor, versus .001316s in active.item. Constructor time may include already-enqueued GPU work; cProfile does not prove the copy alone costs this much. The local-positive graph is known on the host, so MAX(active) must be1 when local active is1.

Selected minimum: create an int64 scalar via torch.full((), int(has_supervision), device=parameter.device), then retain localTrue or read the reduced scalar on localFalse. All ranks enter existing per-parameter gradient reductions, then the unchanged scheduled MAX, in exactly the original order. Nonempty ranks skip only the redundant host read. Empty ranks still wait and follow their peer's supervision. Globally empty batches still skip Adam, preserving moments/step and state commits. No gradients, shapes, owners, subgroup selection, epoch loop or flags change.

Alternatives: torch.tensor -> torch.full alone likely moves waiting to the immediate item; not chosen. PyTorch full directly fills the device scalar, avoiding CPU tensor construction/copy. DGL, DGL-inspired/native kernels and custom CUDA are unnecessary for one scalar. Gradient bucketing is a separate change and is explicitly excluded. Reusing a persistent scalar would add lifetime/state machinery without need.

Validation: reuse test_empty_supervision_step.py unchanged, first CPU then2-rank Gloo. Tests cover active/all-empty/skewed ownership after task callbacks, exact optimizer parameters/moments, unused gradients, autograd reductions/state commits and collective order/dtypes. Root alone runs NCCL correctness and clean four-GPU baseline/candidate timing. Scope is current diagnostic protocol, not MemShare numerical parity; no acceleration assumed.

2026-09-13 complete/frozen: only runtime/epoch.py differs from frozen device-fix base (2 changed statements,388 lines). CPU test_empty_supervision_step.py:3 passed/9.08s. Two-rank Gloo:3 passed per rank/10.19s; exact optimizer state and parameter comparisons, unused gradients, all-active/all-empty/skewed supervision, unchanged collective order verified. Compile passed. No new test framework; original focused fixture reused unchanged. No GPU performed, gain unresolved. The source remains the pre-math prototype; root may apply only candidate.patch to frozen memshare_wiki_math and preserve both histories.
