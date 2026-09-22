# Supervision flag: isolated MemShare optimizer candidate

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
