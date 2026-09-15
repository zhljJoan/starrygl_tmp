from __future__ import annotations

from typing import Mapping

import torch
import torch.distributed as dist
from torch import Tensor

from starrygl.model import StateDelta


def launch_state_update(
    state_manager: object | Mapping[str, object] | None,
    delta: StateDelta | None,
) -> None:
    if state_manager is None:
        return
    managers = tuple(_state_managers(state_manager))
    selected = state_manager.get(delta.kind) if delta is not None and isinstance(state_manager, Mapping) else state_manager
    committed = []
    for manager in managers:
        if delta is not None and manager is selected:
            manager.commit(delta)
        elif needs_empty_state_commit(manager):
            manager.commit(_empty_delta(manager))
        else:
            continue
        committed.append(manager)
    for manager in committed:
        launch = getattr(manager, "launch_shared", None)
        if callable(launch):
            launch()


def finish_state_update(
    state_manager: object | Mapping[str, object] | None,
    *,
    final: bool = False,
) -> None:
    _call_managers(state_manager, "handle_last_async" if final else "handle_owner_async")


def poll_state_update(state_manager: object | Mapping[str, object] | None) -> None:
    _call_managers(state_manager, "finish_shared_ready")


def needs_empty_state_commit(manager: object) -> bool:
    active = dist.is_available() and dist.is_initialized() and int(dist.get_world_size()) > 1
    return active and bool(getattr(manager, "async_owner_collective", False))


def _empty_delta(manager: object) -> StateDelta:
    base = getattr(manager, "memory_manager", manager)
    values = getattr(base, "values", None)
    if not isinstance(values, Tensor):
        raise TypeError("state manager must expose tensor values")
    timestamps = getattr(base, "timestamps", None)
    kind = getattr(manager, "kind", getattr(base, "kind", "state"))
    metadata = {}
    mailbox = getattr(manager, "mailbox_manager", None)
    if mailbox is not None and isinstance(getattr(mailbox, "values", None), Tensor):
        metadata = {
            "mailbox_nodes": values.new_empty((0,), dtype=torch.long),
            "mailbox_messages": mailbox.values.new_empty((0, mailbox.values.shape[-1])),
            "mailbox_timestamps": mailbox.timestamps.new_empty((0,)),
        }
    return StateDelta(
        node_ids=values.new_empty((0,), dtype=torch.long),
        values=values.new_empty((0, *values.shape[1:])),
        kind=kind,
        timestamps=(
            timestamps.new_empty((0,))
            if isinstance(timestamps, Tensor) and kind in {"node_memory", "neighbor_recurrent"}
            else None
        ),
        metadata=metadata,
    )


def _state_managers(state_manager: object | Mapping[str, object] | None):
    if state_manager is None:
        return ()
    return state_manager.values() if isinstance(state_manager, Mapping) else (state_manager,)


def _call_managers(state_manager: object | Mapping[str, object] | None, method: str) -> None:
    for manager in _state_managers(state_manager):
        callback = getattr(manager, method, None)
        if callable(callback):
            callback()


__all__ = ["finish_state_update", "launch_state_update", "needs_empty_state_commit", "poll_state_update"]
