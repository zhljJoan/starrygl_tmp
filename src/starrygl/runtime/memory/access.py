from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from starrygl.runtime.comm import CommScheduler, collective_needed
from starrygl.store.remote_fetch import OwnerRequest, finish_owner_responses, submit_owner_request, submit_owner_responses
from .ops import mailbox_read_rows, overlay_mailbox, overlay_state, present_rows, state_read_rows
from starrygl.runtime.comm import distributed


@dataclass
class PendingRuntimeRead:
    manager: Any
    pending: Any
    shared: bool = False


@dataclass
class PendingMemoryMailboxRead:
    node_ids: Tensor
    ready: tuple[Any, Any] | None = None
    outputs: dict[str, Tensor | None] = field(default_factory=dict)
    missing_pos: Tensor | None = None
    order: Tensor | None = None
    handles: dict[str, Any] = field(default_factory=dict)
    scheduler: Any | None = None
    remote_inverse: Tensor | None = None
    memory_metadata: dict[str, Tensor] = field(default_factory=dict)


def materialize_local_pair(runtime, node_ids: Tensor):
    memory, mailbox = runtime.memory_manager, runtime.mailbox_manager
    if mailbox is None or not all(hasattr(manager, "row_map") and hasattr(manager, "read") for manager in (memory, mailbox)):
        return None
    nodes = node_ids.long().to(memory.values.device, non_blocking=True)
    memory_rows, mailbox_rows = present_rows(memory, nodes), present_rows(mailbox, nodes)
    if memory_rows is None or mailbox_rows is None:
        return None
    missing = bool(((memory_rows < 0) | (mailbox_rows < 0)).any().item())
    if distributed():
        missing = collective_needed(missing)
    if missing:
        return None
    return (
        overlay_state(state_read_rows(memory, nodes, memory_rows), runtime.shared_manager),
        overlay_mailbox(mailbox_read_rows(mailbox, nodes, mailbox_rows), runtime.shared_mailbox_manager),
    )


def submit_pair(runtime, node_ids: Tensor, **kwargs: Any) -> PendingMemoryMailboxRead | None:
    memory, mailbox = runtime.memory_manager, runtime.mailbox_manager
    if mailbox is None or not all(hasattr(manager, "row_map") and hasattr(manager, "read") for manager in (memory, mailbox)):
        return None
    nodes = node_ids.long()
    if not (distributed() and getattr(memory, "node_dist_index", None) is not None):
        ready = materialize_local_pair(runtime, nodes)
        if ready is None:
            return None
        return PendingMemoryMailboxRead(nodes, ready=ready)
    return _submit_remote_pair(runtime, nodes, kwargs.pop("node_access", None))


def _submit_remote_pair(runtime, nodes: Tensor, node_access) -> PendingMemoryMailboxRead:
    memory, mailbox = runtime.memory_manager, runtime.mailbox_manager
    device = memory.values.device
    query = nodes.to(device, non_blocking=True)
    memory_rows, mailbox_rows = present_rows(memory, query), present_rows(mailbox, query)
    if memory_rows is None or mailbox_rows is None:
        raise RuntimeError("combined memory/mailbox hydrate requires row maps")
    present = (memory_rows >= 0) & (mailbox_rows >= 0)
    outputs = None
    metadata: dict[str, Tensor] = {}
    if runtime.bounded_stale_reads:
        outputs = _pair_outputs(memory, mailbox, int(nodes.numel()))
        local_pos = present.nonzero(as_tuple=True)[0]
        if int(local_pos.numel()):
            _copy_read(outputs, "memory", state_read_rows(memory, query.index_select(0, local_pos), memory_rows.index_select(0, local_pos)), local_pos)
            _copy_read(outputs, "mailbox", mailbox_read_rows(mailbox, query.index_select(0, local_pos), mailbox_rows.index_select(0, local_pos)), local_pos)
        shared_mask, shared_rows = _fill_shared_pair(runtime, query, outputs, present)
        present |= shared_mask
        metadata = {"shared_mask": shared_mask, "shared_rows": shared_rows}
    full_local = bool(present.all().item()) if int(present.numel()) else True
    remote_needed = collective_needed(not full_local)
    if full_local and not remote_needed:
        if outputs is not None:
            return PendingMemoryMailboxRead(nodes, ready=_reads_from_outputs(runtime, nodes, outputs, metadata))
        return PendingMemoryMailboxRead(nodes, ready=(
            overlay_state(state_read_rows(memory, query, memory_rows), runtime.shared_manager),
            overlay_mailbox(mailbox_read_rows(mailbox, query, mailbox_rows), runtime.shared_mailbox_manager),
        ))
    if outputs is None:
        outputs = _pair_outputs(memory, mailbox, int(nodes.numel()))
        local_pos = present.nonzero(as_tuple=True)[0]
        if int(local_pos.numel()):
            _copy_read(outputs, "memory", state_read_rows(memory, query.index_select(0, local_pos), memory_rows.index_select(0, local_pos)), local_pos)
            _copy_read(outputs, "mailbox", mailbox_read_rows(mailbox, query.index_select(0, local_pos), mailbox_rows.index_select(0, local_pos)), local_pos)
    missing_pos = (~present).nonzero(as_tuple=True)[0]
    missing = query.index_select(0, missing_pos)
    remote, inverse = missing, None
    if int(missing.numel()) > 1:
        remote, inverse = torch.unique(missing, sorted=True, return_inverse=True)
    request = _request_nodes(runtime, memory, remote, node_access)
    memory_read = memory.read(request.recv_nodes.to(memory.values.device))
    mailbox_read = mailbox.read(request.recv_nodes.to(mailbox.values.device))
    responses = {}
    for prefix, read in (("memory", memory_read), ("mailbox", mailbox_read)):
        responses[f"{prefix}_values"] = read.values
        if read.timestamps is not None:
            responses[f"{prefix}_timestamps"] = read.timestamps
    handles = submit_owner_responses(request, responses, name=f"state_mailbox_response:{runtime.kind}")
    return PendingMemoryMailboxRead(
        nodes.to(device), outputs=outputs, missing_pos=missing_pos, order=request.order,
        handles=handles, scheduler=request.scheduler, remote_inverse=inverse,
        memory_metadata=metadata,
    )


def _request_nodes(runtime, memory, nodes, node_access):
    if isinstance(node_access, OwnerRequest):
        return node_access
    return submit_owner_request(
        nodes,
        memory._node_dist_index_on(nodes.device),
        scheduler=runtime.comm or CommScheduler(),
        name=f"state_mailbox_request:{runtime.kind}",
        packet_capacity=int(memory.node_dist_index.numel()),
    )


def finish_pair(runtime, pending: PendingMemoryMailboxRead):
    if pending.ready is not None:
        return pending.ready
    if pending.scheduler is None or pending.missing_pos is None or pending.order is None:
        raise RuntimeError("pending memory/mailbox read is incomplete")
    received = finish_owner_responses(pending.scheduler, pending.order, pending.handles)
    if pending.remote_inverse is not None:
        received = {name: value.index_select(0, pending.remote_inverse.to(value.device)) for name, value in received.items()}
    for prefix in ("memory", "mailbox"):
        for field_name in ("values", "timestamps"):
            key = f"{prefix}_{field_name}"
            output, value = pending.outputs.get(key), received.get(key)
            if output is not None and value is not None:
                output.index_copy_(0, pending.missing_pos.to(output.device), value.to(output))
    memory_read, mailbox_read = _reads_from_outputs(
        runtime, pending.node_ids, pending.outputs, pending.memory_metadata,
    )
    if runtime.bounded_stale_reads:
        return memory_read, mailbox_read
    return overlay_state(memory_read, runtime.shared_manager), overlay_mailbox(mailbox_read, runtime.shared_mailbox_manager)


def _fill_shared_pair(runtime, query: Tensor, outputs, owner_present: Tensor) -> tuple[Tensor, Tensor]:
    memory, mailbox = runtime.shared_manager, runtime.shared_mailbox_manager
    mask = torch.zeros(int(query.numel()), dtype=torch.bool, device=query.device)
    rows_out = torch.full((int(query.numel()),), -1, dtype=torch.long, device=query.device)
    memory_rows, mailbox_rows = present_rows(memory, query), present_rows(mailbox, query)
    if memory_rows is None or mailbox_rows is None:
        return mask, rows_out
    mask = (~owner_present) & (memory_rows >= 0) & (mailbox_rows >= 0)
    pos = mask.nonzero(as_tuple=True)[0]
    if int(pos.numel()):
        nodes = query.index_select(0, pos)
        _copy_read(outputs, "memory", state_read_rows(memory, nodes, memory_rows.index_select(0, pos)), pos)
        _copy_read(outputs, "mailbox", mailbox_read_rows(mailbox, nodes, mailbox_rows.index_select(0, pos)), pos)
        rows_out.index_copy_(0, pos, memory_rows.index_select(0, pos))
    return mask, rows_out


def _reads_from_outputs(runtime, nodes: Tensor, outputs, metadata):
    from starrygl.store.mailbox import MailboxRead
    from starrygl.store.state import StateRead

    memory, mailbox = runtime.memory_manager, runtime.mailbox_manager
    return (
        StateRead(
            nodes.to(memory.values.device), outputs["memory_values"],
            outputs["memory_timestamps"], metadata,
        ),
        MailboxRead(
            nodes.to(mailbox.values.device), outputs["mailbox_values"],
            outputs["mailbox_timestamps"],
        ),
    )


def _pair_outputs(memory, mailbox, count: int) -> dict[str, Tensor | None]:
    return {
        "memory_values": memory.values.new_empty((count, *memory.values.shape[1:])),
        "memory_timestamps": None if memory.timestamps is None else memory.timestamps.new_empty((count, *memory.timestamps.shape[1:])),
        "mailbox_values": mailbox.values.new_empty((count, *mailbox.values.shape[1:])),
        "mailbox_timestamps": mailbox.timestamps.new_empty((count, *mailbox.timestamps.shape[1:])),
    }


def _copy_read(outputs, prefix: str, read, pos: Tensor) -> None:
    for name in ("values", "timestamps"):
        output, value = outputs[f"{prefix}_{name}"], getattr(read, name)
        if output is not None and value is not None:
            output.index_copy_(0, pos.to(output.device), value.to(output))
