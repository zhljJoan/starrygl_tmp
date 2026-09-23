from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from starrygl.runtime.comm import (
    CommScheduler,
    Route,
    all_to_all_counts,
)
from starrygl.utils.route import dist_part


@dataclass(frozen=True)
class OwnerRequest:
    order: Tensor
    send_counts: Tensor
    recv_counts: Tensor
    recv_nodes: Tensor
    scheduler: CommScheduler


@dataclass(frozen=True)
class OwnerPush:
    local_pos: Tensor
    handles: dict[str, Any]
    scheduler: CommScheduler


def owner_route(node_ids: Tensor, dist_index: Tensor) -> tuple[Tensor, Tensor]:
    owner = (
        dist_part(dist_index.index_select(0, node_ids.to(dist_index.device))).to(node_ids.device)
        if int(node_ids.numel())
        else node_ids.new_empty((0,))
    )
    order = torch.argsort(owner, stable=True) if int(owner.numel()) else node_ids.new_empty((0,))
    return order.long(), torch.bincount(owner.long(), minlength=dist.get_world_size()).long().cpu()


def submit_owner_request(
    node_ids: Tensor,
    dist_index: Tensor,
    *,
    scheduler: CommScheduler | None,
    name: str,
    order: Tensor | None = None,
    send_counts: Tensor | None = None,
    packet_capacity: int | None = None,
) -> OwnerRequest:
    scheduler = scheduler or CommScheduler()
    if order is None or send_counts is None:
        order, send_counts = owner_route(node_ids, dist_index)
    if packet_capacity is not None:
        return _submit_fixed_owner_request(
            node_ids, order, send_counts, scheduler, name, packet_capacity,
        )
    recv_counts = all_to_all_counts(send_counts.long().cpu(), group=scheduler.group)
    route = Route(tuple(send_counts.tolist()), tuple(recv_counts.tolist()), send_index=order.long())
    recv_nodes = scheduler.finish_push(scheduler.launch_push(route, node_ids.long(), name=f"{name}:nodes"))
    return OwnerRequest(order.long(), send_counts.long().cpu(), recv_counts.long().cpu(), recv_nodes.long(), scheduler)


def _submit_fixed_owner_request(
    node_ids: Tensor,
    order: Tensor,
    send_counts: Tensor,
    scheduler: CommScheduler,
    name: str,
    capacity: int,
) -> OwnerRequest:
    send_counts = send_counts.long().to(node_ids.device)
    world_size = scheduler.world_size
    capacity = int(capacity)
    if capacity < int(send_counts.max().item()):
        raise ValueError("owner request packet capacity is too small")
    packet = node_ids.new_full((world_size, capacity + 1), -1)
    packet[:, 0] = send_counts
    if int(node_ids.numel()):
        offsets = send_counts.cumsum(0) - send_counts
        owners = torch.repeat_interleave(
            torch.arange(world_size, device=node_ids.device), send_counts,
        )
        columns = torch.arange(int(node_ids.numel()), device=node_ids.device)
        columns -= torch.repeat_interleave(offsets, send_counts)
        packet[owners, columns + 1] = node_ids.index_select(0, order.to(node_ids.device))
    sizes = (capacity + 1,) * world_size
    received = scheduler.finish_push(scheduler.launch_push(
        Route(sizes, sizes), packet.view(-1), name=f"{name}:packet",
    )).view(world_size, capacity + 1)
    recv_counts = received[:, 0].long().cpu()
    columns = torch.arange(capacity, device=received.device)
    recv_nodes = received[:, 1:][columns.unsqueeze(0) < received[:, :1]]
    return OwnerRequest(
        order.long(), send_counts.cpu(), recv_counts, recv_nodes.long(), scheduler,
    )


def submit_owner_responses(request: OwnerRequest, values: dict[str, Tensor], *, name: str) -> dict[str, Any]:
    route = Route(tuple(request.recv_counts.tolist()), tuple(request.send_counts.tolist()))
    return {
        key: request.scheduler.launch_push(route, value.detach().contiguous(), name=f"{name}:{key}")
        for key, value in values.items()
    }


def finish_owner_responses(
    scheduler: CommScheduler,
    order: Tensor,
    handles: dict[str, Any],
) -> dict[str, Tensor]:
    return {key: _restore_order(scheduler.finish_push(handle), order) for key, handle in handles.items()}


def submit_owner_push(
    node_ids: Tensor,
    dist_index: Tensor,
    values: dict[str, Tensor],
    *,
    scheduler: CommScheduler | None,
    name: str,
) -> OwnerPush:
    scheduler = scheduler or CommScheduler()
    owner = (
        dist_part(dist_index.index_select(0, node_ids.to(dist_index.device))).long().cpu()
        if int(node_ids.numel())
        else torch.empty(0, dtype=torch.long)
    )
    local = owner == dist.get_rank()
    local_pos = local.nonzero(as_tuple=True)[0]
    remote_pos = (~local).nonzero(as_tuple=True)[0]
    remote_owner = owner.index_select(0, remote_pos)
    order = torch.argsort(remote_owner, stable=True) if int(remote_owner.numel()) else remote_owner
    send_pos = remote_pos.index_select(0, order) if int(order.numel()) else remote_pos
    send_counts = torch.bincount(remote_owner.long(), minlength=dist.get_world_size()).long().cpu()
    recv_counts = all_to_all_counts(send_counts)
    route = Route(tuple(send_counts.tolist()), tuple(recv_counts.tolist()), send_index=send_pos)
    target = next(iter(values.values())).device
    payloads = {"node_ids": node_ids.long().to(target, non_blocking=True), **values}
    handles = {
        key: scheduler.launch_push(route, value, name=f"{name}:{key}")
        for key, value in payloads.items()
    }
    return OwnerPush(local_pos=local_pos, handles=handles, scheduler=scheduler)


def finish_owner_push(pending: OwnerPush) -> dict[str, Tensor]:
    return {key: pending.scheduler.finish_push(handle) for key, handle in pending.handles.items()}


def submit_remote_fetch(
    manager: Any,
    node_ids: Tensor,
    *,
    pending_type: type,
    read_type: type,
    prefix: str,
    ready: Any = None,
    out_values: Tensor | None = None,
    out_timestamps: Tensor | None = None,
    missing_pos: Tensor | None = None,
    result_node_ids: Tensor | None = None,
) -> Any:
    device = manager.values.device
    node_ids = node_ids.long().to(device=device, non_blocking=True)
    result_node_ids = node_ids if result_node_ids is None else result_node_ids.long()
    query_nodes, inverse = node_ids, None
    if int(node_ids.numel()) > 1:
        query_nodes, inverse = torch.unique(node_ids, sorted=True, return_inverse=True)
    request = submit_owner_request(
        query_nodes,
        manager._node_dist_index_on(query_nodes.device),
        scheduler=manager.comm,
        name=f"{prefix}_request:{manager.kind}",
    )
    recv_nodes = request.recv_nodes
    response = manager.read(recv_nodes.to(device=device)) if int(recv_nodes.numel()) else read_type(
        node_ids=recv_nodes,
        values=manager.values.new_empty((0, *manager.values.shape[1:])),
        timestamps=None if manager.timestamps is None else manager.timestamps.new_empty((0, *manager.timestamps.shape[1:])),
    )
    values = {"values": response.values}
    if manager.timestamps is not None:
        values["timestamps"] = (
            response.timestamps
            if response.timestamps is not None
            else manager.timestamps.new_empty((0, *manager.timestamps.shape[1:]))
        )
    handles = submit_owner_responses(request, values, name=f"{prefix}_response:{manager.kind}")
    return pending_type(
        node_ids=result_node_ids.to(device=device),
        ready=ready,
        out_values=out_values,
        out_timestamps=out_timestamps,
        missing_pos=missing_pos,
        order=request.order,
        recv_values=handles["values"],
        recv_timestamps=handles.get("timestamps"),
        scheduler=request.scheduler,
        remote_inverse=inverse,
    )


def finish_remote_fetch(manager: Any, pending: Any, read_type: type) -> Any:
    if pending.ready is not None:
        _finish_pending_handles(pending)
        return pending.ready
    if pending.scheduler is None or pending.recv_values is None:
        raise RuntimeError("pending remote fetch is missing response handles")
    handles = {"values": pending.recv_values}
    if pending.recv_timestamps is not None:
        handles["timestamps"] = pending.recv_timestamps
    received = finish_owner_responses(pending.scheduler, pending.order, handles)
    restored_values = received["values"]
    restored_timestamps = received.get("timestamps")
    if pending.remote_inverse is not None:
        inverse = pending.remote_inverse.long()
        restored_values = restored_values.index_select(0, inverse.to(restored_values.device))
        if restored_timestamps is not None:
            restored_timestamps = restored_timestamps.index_select(0, inverse.to(restored_timestamps.device))
    if pending.out_values is None or pending.missing_pos is None:
        return read_type(
            node_ids=pending.node_ids.to(device=manager.values.device),
            values=restored_values.to(device=manager.values.device, dtype=manager.values.dtype),
            timestamps=(
                None
                if restored_timestamps is None
                else restored_timestamps.to(device=manager.timestamps.device, dtype=manager.timestamps.dtype)
            ),
        )
    rows = pending.missing_pos
    pending.out_values.index_copy_(0, rows.to(pending.out_values.device), restored_values.to(pending.out_values))
    if pending.out_timestamps is not None and restored_timestamps is not None:
        pending.out_timestamps.index_copy_(
            0,
            rows.to(pending.out_timestamps.device),
            restored_timestamps.to(pending.out_timestamps),
        )
    return read_type(
        node_ids=pending.node_ids,
        values=pending.out_values,
        timestamps=pending.out_timestamps,
    )


def ready_fetch(read: Any, pending_type: type) -> Any:
    return pending_type(
        node_ids=read.node_ids,
        ready=read,
        out_values=None,
        out_timestamps=None,
        missing_pos=None,
        order=torch.empty(0, dtype=torch.long, device=read.node_ids.device),
        recv_values=None,
        recv_timestamps=None,
        scheduler=None,
    )


def cached_tensor(value: Tensor | None, cache: dict[str, Tensor], device: torch.device | str, name: str) -> Tensor:
    if value is None:
        raise KeyError(f"{name} is not available")
    target = torch.device(device)
    if value.device == target:
        return value
    key = str(target)
    if key not in cache:
        cache[key] = value.to(device=target, non_blocking=True)
    return cache[key]


def row_map_all_present(row_map: Tensor | None) -> bool:
    return row_map is not None and int(row_map.numel()) > 0 and bool(torch.all(row_map >= 0).item())


def row_map_is_identity(row_map: Tensor | None) -> bool:
    return row_map is not None and int(row_map.numel()) > 0 and torch.equal(
        row_map.cpu(), torch.arange(int(row_map.numel()), dtype=row_map.dtype)
    )


def _finish_pending_handles(pending: Any) -> None:
    if pending.scheduler is None or pending.recv_values is None:
        return
    for handle in (pending.recv_values, pending.recv_timestamps):
        if handle is not None:
            pending.scheduler.finish_push(handle)


def _restore_order(value: Tensor, order: Tensor) -> Tensor:
    out = value.new_empty(value.shape)
    if int(order.numel()):
        out.index_copy_(0, order.to(value.device), value)
    return out
