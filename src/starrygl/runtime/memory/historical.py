from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from starrygl.model import StateDelta
from starrygl.runtime.comm import distributed
from .ops import commit_state_local, present_rows, state_read_rows


@dataclass
class PendingStaleRefresh:
    manager: Any
    pending: Any
    target_manager: Any | None


def materialize_bounded(runtime, node_ids: Tensor):
    from starrygl.store.state import StateRead

    manager, nodes = runtime.memory_manager, node_ids.long()
    count = int(nodes.numel())
    values = manager.values.new_zeros((count, *manager.values.shape[1:]))
    timestamps = None if manager.timestamps is None else manager.timestamps.new_zeros((count, *manager.timestamps.shape[1:]))
    filled = torch.zeros(count, dtype=torch.bool, device=nodes.device)
    shared_mask = torch.zeros(count, dtype=torch.bool, device=nodes.device)
    shared_rows = torch.full((count,), -1, dtype=torch.long, device=nodes.device)
    for source in (manager, runtime.shared_manager):
        rows = present_rows(source, nodes)
        if rows is None:
            continue
        pos = ((rows >= 0) & ~filled.to(rows.device)).nonzero(as_tuple=True)[0]
        if not int(pos.numel()):
            continue
        read = state_read_rows(source, nodes.index_select(0, pos.to(nodes.device)), rows.index_select(0, pos))
        _copy_state(values, timestamps, read, pos)
        filled.index_fill_(0, pos.to(filled.device), True)
        if source is runtime.shared_manager:
            shared_mask.index_fill_(0, pos.to(shared_mask.device), True)
            shared_rows.index_copy_(0, pos.to(shared_rows.device), rows.index_select(0, pos).to(shared_rows))
    missing_pos = (~filled).nonzero(as_tuple=True)[0]
    remote_epoch = distributed() and getattr(manager, "node_dist_index", None) is not None
    if remote_epoch or int(missing_pos.numel()):
        missing = nodes.index_select(0, missing_pos) if int(missing_pos.numel()) else nodes.new_empty((0,))
        submit = getattr(manager, "submit_materialize_async", None)
        if callable(submit):
            pending = submit(missing)
            if runtime.stale_miss_fallback == "owner":
                _copy_state(values, timestamps, manager.finish_materialize_async(pending), missing_pos)
            elif runtime.shared_manager is not None:
                runtime.pending_stale_refresh.append(PendingStaleRefresh(manager, pending, runtime.shared_manager))
        elif runtime.stale_miss_fallback == "owner":
            _copy_state(values, timestamps, manager.materialize(missing), missing_pos)
    runtime._count("bounded_stale_read_rows", count)
    runtime._count("bounded_stale_cache_rows", int(filled.sum().item()))
    runtime._count("bounded_stale_miss_rows", int(missing_pos.numel()))
    return StateRead(
        nodes,
        values,
        timestamps,
        {"shared_mask": shared_mask, "shared_rows": shared_rows},
    )


def finish_stale_refreshes(runtime) -> None:
    for refresh in runtime.pending_stale_refresh:
        finish = getattr(refresh.manager, "finish_materialize_async", None)
        if not callable(finish):
            continue
        read = finish(refresh.pending)
        if refresh.target_manager is not None and int(read.node_ids.numel()):
            commit_state_local(refresh.target_manager, StateDelta(
                node_ids=read.node_ids.to(read.values.device), values=read.values,
                kind=getattr(refresh.target_manager, "kind", runtime.kind), timestamps=read.timestamps,
            ))
    runtime.pending_stale_refresh.clear()


def _copy_state(values, timestamps, read, pos) -> None:
    for out, value in ((values, read.values), (timestamps, read.timestamps)):
        if out is not None and value is not None:
            out.index_copy_(0, pos.to(out.device), value.to(out))
