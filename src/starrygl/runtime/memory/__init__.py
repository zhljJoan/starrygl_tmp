from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from starrygl.model import StateDelta
from ..comm import distributed
from .access import (
    PendingMemoryMailboxRead,
    PendingRuntimeRead,
    finish_pair,
    materialize_local_pair,
    submit_pair,
)
from .ops import (
    commit_mailbox_local,
    commit_state_local,
    filter_aligned_mailbox,
    filter_delta,
    filter_mailbox_to_nodes,
    overlay_mailbox,
    overlay_state,
    present_rows,
)
from .shared import (
    PendingSharedSync,
    finish_shared_sync,
    launch_shared_sync,
    shared_delta,
    shared_mailbox_delta,
    shared_sync_ready,
)
from .historical import (
    PendingStaleRefresh,
    finish_stale_refreshes,
    materialize_bounded,
)


class SharedStateRefreshFilter:
    """Admit shared-hot refreshes; model-side increment estimation is separate."""

    def __init__(
        self,
        *,
        dim: int,
        num_rows: int = 0,
        min_change_norm: float = 0.0,
        min_cosine_distance: float | None = None,
        max_skip: int = 10,
    ) -> None:
        self.dim = int(dim)
        self.min_change_norm = float(min_change_norm)
        self.min_cosine_distance = None if min_cosine_distance is None else float(min_cosine_distance)
        self.max_skip = int(max_skip)
        self.count = torch.zeros(int(num_rows), 1)
        self.historical = torch.zeros(int(num_rows), self.dim)

    def allow(self, rows: Tensor, change: Tensor | None = None, *, values: Tensor | None = None,
              reference: Tensor | None = None) -> Tensor:
        source = values if values is not None else change
        if not int(rows.numel()):
            return torch.empty(0, dtype=torch.bool, device=source.device if source is not None else rows.device)
        if values is not None and self.min_cosine_distance is not None:
            rows, values = self._prepare(rows, values)
            previous = self.historical.index_select(0, rows) if reference is None else reference
            distance = 1.0 - torch.nn.functional.cosine_similarity(values, previous.to(values), dim=1)
            forced = self.count.index_select(0, rows).reshape(-1) >= self.max_skip
            return (distance > self.min_cosine_distance) | forced
        if change is None or self.min_change_norm <= 0.0:
            return torch.ones(int(rows.numel()), dtype=torch.bool, device=source.device if source is not None else rows.device)
        rows, change = self._prepare(rows, change)
        forced = self.count.index_select(0, rows).reshape(-1) >= self.max_skip
        return (torch.linalg.vector_norm(change, dim=1) >= self.min_change_norm) | forced

    @torch.no_grad()
    def update(
        self,
        rows: Tensor,
        change: Tensor | None = None,
        *,
        values: Tensor | None = None,
        keep: Tensor | None = None,
    ) -> None:
        if not int(rows.numel()):
            return
        source = values if values is not None else change
        if source is None:
            return
        rows, source = self._prepare(rows, source)
        keep = torch.ones(int(rows.numel()), dtype=torch.bool, device=source.device) if keep is None else keep.to(source.device).bool()
        accepted = keep.nonzero(as_tuple=True)[0]
        if int(accepted.numel()):
            accepted_rows = rows.index_select(0, accepted)
            if values is not None:
                self.historical.index_copy_(0, accepted_rows, source.index_select(0, accepted).to(self.historical))
            self.count.index_fill_(0, accepted_rows, 0)
        skipped = rows.index_select(0, (~keep).nonzero(as_tuple=True)[0])
        if int(skipped.numel()):
            self.count.index_add_(0, skipped, self.count.new_ones((int(skipped.numel()), 1)))

    @torch.no_grad()
    def clear(self) -> None:
        self.count.zero_()
        self.historical.zero_()

    def _prepare(self, rows: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        rows = rows.long().to(values.device)
        self._ensure(rows, values.device)
        values = values.detach().reshape(int(rows.numel()), -1)
        if int(values.shape[1]) > self.dim:
            values = values[:, : self.dim]
        elif int(values.shape[1]) < self.dim:
            values = torch.nn.functional.pad(values, (0, self.dim - int(values.shape[1])))
        return rows, values

    def _ensure(self, rows: Tensor, device: torch.device) -> None:
        if self.count.device != device:
            self.count, self.historical = (
                value.to(device, non_blocking=True) for value in (self.count, self.historical)
            )
        extra = int(rows.max().item()) + 1 - int(self.count.shape[0])
        if extra > 0:
            self.count = torch.cat((self.count, self.count.new_zeros((extra, 1))))
            self.historical = torch.cat((self.historical, self.historical.new_zeros((extra, self.dim))))


@dataclass
class AsyncMemoryCommitHandle:
    shared_delta: StateDelta | None = None
    owner_future: Future | None = None
    owner_handle: Any | None = None
    memory_applied: bool = False
    shared_launched: bool = False
    shared_applied: bool = False

    def done(self) -> bool:
        return bool(
            (self.owner_future is None or self.owner_future.done())
            and self.owner_handle is None
            and self.memory_applied
            and self.shared_applied
        )


class AsyncMemoryCommitter:
    """Owns authoritative state commits and optional shared-hot refresh."""

    def __init__(
        self,
        memory_manager: Any,
        *,
        shared_manager: Any | None = None,
        mailbox_manager: Any | None = None,
        shared_mailbox_manager: Any | None = None,
        change_filter: SharedStateRefreshFilter | None = None,
        staged_shared: bool = True,
        use_shared_filter: bool = True,
        background_owner: bool = False,
        bounded_stale_reads: bool = False,
        max_staleness: int = 0,
        stale_miss_fallback: str = "owner",
        shared_flush_interval: int = 1,
        skip_remote_non_hot_owner_commit: bool = False,
    ) -> None:
        self.memory_manager = memory_manager
        self.shared_manager = shared_manager
        self.mailbox_manager = mailbox_manager
        self.shared_mailbox_manager = shared_mailbox_manager
        self.change_filter = change_filter
        self.staged_shared = bool(staged_shared)
        self.use_shared_filter = bool(use_shared_filter)
        self.async_owner_collective = bool(background_owner) and distributed()
        self.background_owner = bool(background_owner) and not distributed()
        self.bounded_stale_reads = bool(bounded_stale_reads)
        self.freshness_policy = "bounded_stale" if self.bounded_stale_reads else "exact"
        self.max_staleness = max(0, int(max_staleness))
        self.stale_miss_fallback = str(stale_miss_fallback)
        self.shared_flush_interval = max(1, int(shared_flush_interval))
        self.skip_remote_non_hot_owner_commit = bool(skip_remote_non_hot_owner_commit)
        self.kind = getattr(memory_manager, "kind", "node_memory")
        self.comm = getattr(memory_manager, "comm", None)
        self.pending: list[AsyncMemoryCommitHandle] = []
        self.pending_shared_sync: list[PendingSharedSync] = []
        self.pending_stale_refresh: list[PendingStaleRefresh] = []
        self.commit_count = 0
        self.snapshot_history = None
        self.snapshot_shared_history = None
        self.pending_snapshot_pushes = []
        self._profile_counters: dict[str, int] = {}
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"starrygl-{self.kind}-commit")
            if self.background_owner else None
        )

    def read(self, *args: Any, **kwargs: Any):
        self.handle_owner_async()
        return self._overlay_state(self.memory_manager.read(*args, **kwargs))

    def materialize(self, *args: Any, **kwargs: Any):
        self.handle_owner_async()
        return materialize_bounded(self, args[0]) if self.bounded_stale_reads and args else self._overlay_state(self.memory_manager.materialize(*args, **kwargs))

    def submit_materialize_async(self, node_ids: Tensor, **kwargs: Any) -> PendingRuntimeRead:
        self.handle_owner_async()
        if self.bounded_stale_reads:
            return PendingRuntimeRead(None, materialize_bounded(self, node_ids))
        submit = getattr(self.memory_manager, "submit_materialize_async", None)
        pending = submit(node_ids, **kwargs) if callable(submit) else self.memory_manager.materialize(node_ids, **kwargs)
        return PendingRuntimeRead(self.memory_manager, pending, True)

    def finish_materialize_async(self, pending: PendingRuntimeRead):
        if pending.manager is None:
            return pending.pending
        finish = getattr(pending.manager, "finish_materialize_async", None)
        read = finish(pending.pending) if callable(finish) else pending.pending
        return self._overlay_state(read) if pending.shared else read

    def submit_materialize_with_mailbox_async(self, node_ids: Tensor, **kwargs: Any):
        if self.mailbox_manager is None:
            return None
        self.handle_owner_async()
        return submit_pair(self, node_ids, **kwargs)

    def finish_materialize_with_mailbox_async(self, pending: PendingMemoryMailboxRead):
        return finish_pair(self, pending)

    def materialize_local_with_mailbox_if_all_present(self, node_ids: Tensor):
        if self.mailbox_manager is None:
            return None
        self.handle_owner_async()
        return materialize_local_pair(self, node_ids)

    def materialize_mailbox(self, node_ids: Tensor):
        if self.mailbox_manager is None:
            return None
        self.handle_owner_async()
        read = self.mailbox_manager.materialize(node_ids)
        return self._overlay_mailbox(read)

    def submit_materialize_mailbox_async(self, node_ids: Tensor):
        if self.mailbox_manager is None:
            return None
        self.handle_owner_async()
        submit = getattr(self.mailbox_manager, "submit_materialize_async", None)
        pending = submit(node_ids) if callable(submit) else self.mailbox_manager.materialize(node_ids)
        return PendingRuntimeRead(self.mailbox_manager, pending, True)

    def finish_materialize_mailbox_async(self, pending: PendingRuntimeRead | None):
        if pending is None:
            return None
        finish = getattr(pending.manager, "finish_materialize_async", None)
        read = finish(pending.pending) if callable(finish) else pending.pending
        return self._overlay_mailbox(read) if pending.shared else read

    def read_all_rows(self, *args: Any, **kwargs: Any):
        self.handle_owner_async()
        return self.memory_manager.read_all_rows(*args, **kwargs)

    def reset(self) -> None:
        self.handle_last_async()
        for manager in (self.memory_manager, self.mailbox_manager, self.shared_manager, self.shared_mailbox_manager):
            reset = getattr(manager, "reset", None)
            if callable(reset):
                reset()
        self.pending.clear()
        self.pending_shared_sync.clear()
        self.pending_stale_refresh.clear()
        self.commit_count = 0
        if self.change_filter is not None:
            self.change_filter.clear()
        if self.snapshot_history is not None:
            self.snapshot_history.reset()
            self.snapshot_shared_history.reset()
            self.snapshot_version = 0

    def commit(self, delta: StateDelta) -> None:
        handle = self.submit_commit(delta)
        if self.staged_shared:
            self.pending = [item for item in self.pending if not item.done()]
        else:
            self.wait(handle)

    def submit_commit(self, delta: StateDelta) -> AsyncMemoryCommitHandle:
        if self.snapshot_history is not None:
            from .snapshot import commit_snapshot_history
            commit_snapshot_history(self, delta)
        owner_delta = self._owner_delta(delta)
        no_shared = self.shared_manager is None and self.shared_mailbox_manager is None
        handle = AsyncMemoryCommitHandle(shared_applied=no_shared)
        owner_handle = None if self.skip_remote_non_hot_owner_commit and distributed() else self._submit_owner_async(owner_delta)
        if owner_handle is not None:
            handle.owner_handle = owner_handle
        elif self.skip_remote_non_hot_owner_commit and distributed():
            self._apply_owner_local(owner_delta)
            handle.memory_applied = True
        elif self._executor is None:
            self._apply_owner(owner_delta)
            handle.memory_applied = True
        else:
            handle.owner_future = self._executor.submit(self._apply_owner, owner_delta)
        memory_delta = shared_delta(self, delta)
        mailbox_delta = shared_mailbox_delta(self, delta, None if memory_delta is None else memory_delta.metadata.get("shared_allowed_node_ids"))
        if self.staged_shared:
            handle.shared_delta = _merge_shared_delta(memory_delta, mailbox_delta)
        else:
            self._apply_shared(memory_delta, mailbox_delta)
            handle.shared_applied = True
        self.pending.append(handle)
        self.commit_count += 1
        return handle

    def launch_shared(self) -> None:
        self._launch_shared(force=False)
        if self.snapshot_history is not None:
            from .snapshot import launch_snapshot_push
            launch_snapshot_push(self)

    def _launch_shared(self, *, force: bool) -> None:
        candidates = [
            handle for handle in self.pending
            if not handle.shared_launched and not handle.shared_applied
        ]
        if not candidates or (not force and len(candidates) < self.shared_flush_interval):
            return
        deltas = [handle.shared_delta for handle in candidates]
        for handle in candidates:
            handle.shared_delta = None
            handle.shared_launched = True
        self.pending_shared_sync.append(launch_shared_sync(self, candidates, deltas))

    def finish_shared_async(self) -> None:
        for sync in self.pending_shared_sync:
            finish_shared_sync(self, sync)
        self.pending_shared_sync.clear()
        self.pending = [item for item in self.pending if not item.done()]
        if self.snapshot_history is not None:
            from .snapshot import finish_snapshot_pushes
            finish_snapshot_pushes(self, ready_only=False)

    def finish_shared_ready(self) -> None:
        waiting = []
        for sync in self.pending_shared_sync:
            if shared_sync_ready(sync):
                finish_shared_sync(self, sync)
            else:
                waiting.append(sync)
        self.pending_shared_sync = waiting
        self.pending = [item for item in self.pending if not item.done()]
        if self.snapshot_history is not None:
            from .snapshot import finish_snapshot_pushes
            finish_snapshot_pushes(self, ready_only=True)

    def handle_last_async(self) -> None:
        finish_stale_refreshes(self)
        self._launch_shared(force=True)
        for handle in self.pending:
            self._finish_owner(handle)
        self.finish_shared_async()
        self.pending = [item for item in self.pending if not item.done()]

    def handle_owner_async(self) -> None:
        finish_stale_refreshes(self)
        for handle in self.pending:
            self._finish_owner(handle)
        self.pending = [item for item in self.pending if not item.done()]

    def wait(self, handle: AsyncMemoryCommitHandle) -> None:
        self._finish_owner(handle)
        if not handle.shared_applied:
            sync = launch_shared_sync(self, [handle], [handle.shared_delta])
            handle.shared_delta = None
            finish_shared_sync(self, sync)

    def _submit_owner_async(self, delta: StateDelta):
        submit = getattr(self.memory_manager, "submit_commit_async", None)
        if not self.async_owner_collective or not callable(submit):
            return None
        memory_handle = submit(delta)
        mailbox_handle = None
        if self.mailbox_manager is not None:
            mailbox_submit = getattr(self.mailbox_manager, "submit_commit_async", None)
            if callable(mailbox_submit):
                mailbox_handle = mailbox_submit(delta)
            else:
                self.mailbox_manager.commit(delta)
        return memory_handle, mailbox_handle

    def _finish_owner(self, handle: AsyncMemoryCommitHandle) -> None:
        if handle.memory_applied:
            return
        if handle.owner_future is not None:
            handle.owner_future.result()
            handle.owner_future = None
        if handle.owner_handle is not None:
            for manager, pending in zip((self.memory_manager, self.mailbox_manager), handle.owner_handle):
                finish = getattr(manager, "finish_commit_async", None)
                if pending is not None and callable(finish):
                    finish(pending)
            handle.owner_handle = None
        handle.memory_applied = True

    def _apply_owner(self, delta: StateDelta) -> None:
        with torch.no_grad():
            self.memory_manager.commit(delta)
            if self.mailbox_manager is not None:
                self.mailbox_manager.commit(delta)

    def _apply_owner_local(self, delta: StateDelta) -> None:
        with torch.no_grad():
            commit_state_local(self.memory_manager, delta)
            nodes, values, timestamps = (delta.metadata.get(name) for name in ("mailbox_nodes", "mailbox_messages", "mailbox_timestamps"))
            if self.mailbox_manager is not None and all(isinstance(value, Tensor) for value in (nodes, values, timestamps)):
                commit_mailbox_local(self.mailbox_manager, nodes, values, timestamps)

    def _owner_delta(self, delta: StateDelta) -> StateDelta:
        if not self.skip_remote_non_hot_owner_commit or not int(delta.node_ids.numel()):
            return delta
        rows = present_rows(self.memory_manager, delta.node_ids.detach().long())
        if rows is None:
            return delta
        keep = rows >= 0
        if bool(keep.all().item()):
            return delta
        pos = keep.nonzero(as_tuple=True)[0]
        nodes = delta.node_ids.index_select(0, pos.to(delta.node_ids.device))
        metadata = dict(delta.metadata)
        if not filter_aligned_mailbox(metadata, delta.node_ids.detach().long(), pos):
            filter_mailbox_to_nodes(metadata, nodes)
        return StateDelta(
            node_ids=nodes,
            values=delta.values.index_select(0, pos.to(delta.values.device)),
            kind=delta.kind,
            timestamps=None if delta.timestamps is None else delta.timestamps.index_select(0, pos.to(delta.timestamps.device)),
            metadata=metadata,
        )

    def _apply_shared(self, memory_delta: StateDelta | None, mailbox_delta: StateDelta | None) -> None:
        if self.shared_manager is not None and memory_delta is not None:
            filtered = filter_delta(memory_delta, self.shared_manager)
            if filtered is not None:
                self.shared_manager.commit(filtered)
        if self.shared_mailbox_manager is not None and mailbox_delta is not None:
            self.shared_mailbox_manager.commit(mailbox_delta)

    def _overlay_state(self, read):
        return overlay_state(read, self.shared_manager)

    def _overlay_mailbox(self, read):
        return overlay_mailbox(read, self.shared_mailbox_manager)

    def profile_counters(self) -> dict[str, int]:
        return dict(self._profile_counters)

    def _count(self, key: str, value: int, *, replace: bool = False) -> None:
        key = f"{key}:{self.kind}"
        self._profile_counters[key] = int(value) if replace else self._profile_counters.get(key, 0) + int(value)


def _merge_shared_delta(memory: StateDelta | None, mailbox: StateDelta | None) -> StateDelta | None:
    if memory is None:
        return mailbox
    if mailbox is None:
        return memory
    return StateDelta(memory.node_ids, memory.values, memory.kind, memory.timestamps, {**memory.metadata, **mailbox.metadata})


__all__ = ["AsyncMemoryCommitHandle", "AsyncMemoryCommitter", "SharedStateRefreshFilter"]
