from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from starrygl.runtime.comm import CommScheduler
from starrygl.model import StateDelta
from .ops import (
    commit_mailbox_local,
    commit_state_local,
    filter_delta,
    filter_mailbox_to_manager,
    latest_by_node,
    present_nodes,
)


@dataclass
class SharedPayload:
    manager: Any
    mailbox: bool
    nodes: Any
    values: Any
    timestamps: Any


@dataclass
class PendingSharedSync:
    handles: list[Any]
    payloads: list[SharedPayload]


def shared_delta(runtime, delta: StateDelta) -> StateDelta | None:
    manager = runtime.shared_manager
    if manager is None:
        nodes, values = delta.node_ids.detach(), delta.values.detach()
        timestamps = None if delta.timestamps is None else delta.timestamps.detach()
        return StateDelta(nodes, values, delta.kind, timestamps, {
            **delta.metadata, "shared_allowed_node_ids": nodes.long(),
        })
    nodes, pos, rows = present_nodes(delta.node_ids.detach().long(), manager)
    if runtime.snapshot_history is not None:
        runtime._count("shared_hot_candidate_rows", nodes.numel())
    values = delta.values.detach().index_select(0, pos.to(delta.values.device))
    timestamps = None if delta.timestamps is None else delta.timestamps.detach().index_select(0, pos.to(delta.timestamps.device))
    if runtime.change_filter is not None and int(nodes.numel()):
        previous = manager.values.index_select(0, rows.to(manager.values.device)).to(values)
        change = values - previous
        if runtime.change_filter.min_cosine_distance is not None:
            keep = runtime.change_filter.allow(rows, values=values, reference=previous)
            runtime.change_filter.update(rows, values=values, keep=keep)
        else:
            keep = runtime.change_filter.allow(rows, change)
            runtime.change_filter.update(rows, change, keep=keep)
        if keep is not None and runtime.use_shared_filter:
            pos = keep.nonzero(as_tuple=True)[0]
            nodes = nodes.index_select(0, pos.to(nodes.device))
            rows = rows.index_select(0, pos.to(rows.device))
            values = values.index_select(0, pos.to(values.device))
            if timestamps is not None:
                timestamps = timestamps.index_select(0, pos.to(timestamps.device))
    if runtime.snapshot_history is not None:
        runtime._count("shared_hot_published_rows", nodes.numel())
    return StateDelta(
        node_ids=nodes.to(values.device), values=values, kind=delta.kind,
        timestamps=None if timestamps is None else timestamps.to(values.device),
        metadata={"shared_rows": rows, "shared_allowed_node_ids": nodes.detach().long()},
    )


def shared_mailbox_delta(runtime, delta: StateDelta, allowed_nodes: Tensor | None) -> StateDelta | None:
    manager = runtime.shared_mailbox_manager
    nodes, values, timestamps = (delta.metadata.get(name) for name in ("mailbox_nodes", "mailbox_messages", "mailbox_timestamps"))
    if manager is None or not all(isinstance(value, Tensor) for value in (nodes, values, timestamps)):
        return None
    nodes, values, timestamps = nodes.detach().long(), values.detach(), timestamps.detach()
    _, present_pos, _ = present_nodes(nodes, manager)
    pos = present_pos
    if allowed_nodes is not None and int(pos.numel()):
        candidates = nodes.index_select(0, pos)
        allowed = torch.sort(allowed_nodes.to(candidates.device).long()).values
        if not int(allowed.numel()):
            pos = pos.new_empty(0)
        else:
            lookup = torch.searchsorted(allowed, candidates)
            valid = lookup < int(allowed.numel())
            matched = valid & (allowed.index_select(0, lookup.clamp_max(int(allowed.numel()) - 1)) == candidates)
            pos = pos.index_select(0, matched.nonzero(as_tuple=True)[0])
    nodes = nodes.index_select(0, pos.to(nodes.device))
    values = values.index_select(0, pos.to(values.device))
    timestamps = timestamps.index_select(0, pos.to(timestamps.device))
    metadata = {"mailbox_nodes": nodes, "mailbox_messages": values, "mailbox_timestamps": timestamps}
    return StateDelta(nodes, values, "mailbox", timestamps, metadata)


def launch_shared_sync(runtime, handles: list[Any], deltas: list[StateDelta | None]) -> PendingSharedSync:
    scheduler = runtime.comm or CommScheduler()
    payloads: list[SharedPayload] = []
    if runtime.shared_manager is not None:
        delta = _compact_state([item for item in deltas if item is not None and item.kind != "mailbox"], runtime.shared_manager)
        nodes, values, timestamps = _state_payload(delta, runtime.shared_manager)
        if runtime.snapshot_history is not None:
            history = runtime.snapshot_history
            values = history.read(nodes, timestamps.long())
            timestamps = values[:, -1]
        payloads.append(_launch_payload(scheduler, runtime.shared_manager, False, nodes, values, timestamps, runtime.kind))
    if runtime.shared_mailbox_manager is not None:
        delta = _compact_mailbox([item for item in deltas if item is not None], runtime.shared_mailbox_manager)
        nodes, values, timestamps = _mailbox_payload(delta, runtime.shared_mailbox_manager)
        payloads.append(_launch_payload(scheduler, runtime.shared_mailbox_manager, True, nodes, values, timestamps, runtime.kind))
    return PendingSharedSync(handles, payloads)


def _launch_payload(scheduler, manager, mailbox, nodes, values, timestamps, kind) -> SharedPayload:
    prefix = f"shared_hot:{kind}:{'mailbox' if mailbox else 'memory'}"
    node_handle = scheduler.launch_all_gather(nodes.long(), name=f"{prefix}:nodes")
    counts = node_handle.recv_counts
    return SharedPayload(
        manager=manager,
        mailbox=mailbox,
        nodes=node_handle,
        values=scheduler.launch_all_gather_with_counts(values.contiguous(), counts, name=f"{prefix}:values"),
        timestamps=scheduler.launch_all_gather_with_counts(timestamps.contiguous(), counts, name=f"{prefix}:timestamps"),
    )


def finish_shared_sync(runtime, sync: PendingSharedSync) -> None:
    scheduler = runtime.comm or CommScheduler()
    for payload in sync.payloads:
        nodes = scheduler.finish_all_gather(payload.nodes).long()
        values = scheduler.finish_all_gather(payload.values)
        timestamps = scheduler.finish_all_gather(payload.timestamps)
        if not int(nodes.numel()):
            continue
        nodes, values, timestamps = latest_by_node(nodes, values, timestamps)
        if payload.mailbox:
            commit_mailbox_local(payload.manager, nodes.to(values.device), values, timestamps.to(values.device))
        else:
            if runtime.snapshot_history is not None:
                from .snapshot import install_shared_history
                install_shared_history(runtime, nodes, values)
                values = values[:, :runtime.snapshot_history.dim]
            commit_state_local(payload.manager, StateDelta(
                node_ids=nodes.to(values.device), values=values,
                kind=getattr(payload.manager, "kind", runtime.kind), timestamps=timestamps.to(values.device),
            ))
    for handle in sync.handles:
        handle.shared_applied = True


def shared_sync_ready(sync: PendingSharedSync) -> bool:
    return all(callable(getattr(handle, "ready", None)) and handle.ready() for payload in sync.payloads for handle in (payload.nodes, payload.values, payload.timestamps))


def _compact_state(deltas: list[StateDelta], manager) -> StateDelta | None:
    parts = [filtered for delta in deltas if (filtered := filter_delta(delta, manager)) is not None and int(filtered.node_ids.numel())]
    if not parts:
        return None
    nodes = torch.cat([item.node_ids.detach().long() for item in parts])
    values = torch.cat([item.values.detach() for item in parts])
    timestamps = None if any(item.timestamps is None for item in parts) else torch.cat([item.timestamps.detach() for item in parts])
    nodes, values, timestamps = latest_by_node(nodes, values, timestamps)
    return StateDelta(nodes.to(values.device), values, getattr(manager, "kind", "node_memory"), None if timestamps is None else timestamps.to(values.device))


def _compact_mailbox(deltas: list[StateDelta], manager) -> StateDelta | None:
    parts = []
    for delta in deltas:
        metadata = dict(delta.metadata)
        filter_mailbox_to_manager(metadata, manager)
        nodes, values, timestamps = (metadata.get(name) for name in ("mailbox_nodes", "mailbox_messages", "mailbox_timestamps"))
        if all(isinstance(value, Tensor) for value in (nodes, values, timestamps)) and int(nodes.numel()):
            parts.append((nodes.detach().long(), values.detach(), timestamps.detach()))
    if not parts:
        return None
    nodes, values, timestamps = (torch.cat([item[index] for item in parts]) for index in range(3))
    nodes, values, timestamps = latest_by_node(nodes, values, timestamps)
    metadata = {"mailbox_nodes": nodes, "mailbox_messages": values, "mailbox_timestamps": timestamps}
    return StateDelta(nodes.to(values.device), values, "mailbox", timestamps.to(values.device), metadata)


def _state_payload(delta: StateDelta | None, manager):
    if delta is None:
        return manager.values.new_empty((0,), dtype=torch.long), manager.values.new_empty((0, *manager.values.shape[1:])), manager.timestamps.new_empty((0,))
    timestamps = manager.timestamps.new_zeros((int(delta.node_ids.numel()),)) if delta.timestamps is None else delta.timestamps.to(manager.timestamps)
    return delta.node_ids.to(manager.values.device), delta.values.to(manager.values), timestamps


def _mailbox_payload(delta: StateDelta | None, manager):
    if delta is None:
        return manager.values.new_empty((0,), dtype=torch.long), manager.values.new_empty((0, manager.values.shape[-1])), manager.timestamps.new_empty((0,))
    count = int(delta.node_ids.numel())
    return delta.node_ids.to(manager.values.device), delta.values.to(manager.values).reshape(count, -1), delta.timestamps.to(manager.timestamps).reshape(count, -1).amax(dim=1)
