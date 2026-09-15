from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor

from starrygl.runtime.comm import collective_needed, distributed as _distributed
from .remote_fetch import (
    OwnerPush,
    cached_tensor,
    finish_owner_push,
    finish_remote_fetch,
    ready_fetch,
    row_map_all_present as _row_map_all_present,
    row_map_is_identity as _row_map_is_identity,
    submit_owner_push,
    submit_remote_fetch,
)


@dataclass(frozen=True)
class MailboxRead:
    node_ids: Tensor
    values: Tensor
    timestamps: Tensor
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass
class PendingMailboxCommit:
    local_nodes: Tensor
    local_messages: Tensor
    local_timestamps: Tensor
    push: OwnerPush


@dataclass
class PendingMailboxFetch:
    node_ids: Tensor
    ready: MailboxRead | None
    out_values: Tensor | None
    out_timestamps: Tensor | None
    missing_pos: Tensor | None
    order: Tensor
    recv_values: Any | None
    recv_timestamps: Any | None
    scheduler: Any | None
    remote_inverse: Tensor | None = None


class MailboxManager:
    """Runtime-owned TGN/APAN mailbox state.

    `values[n, k]` stores the k-th message slot for node n.  The first
    implementation keeps writes timestamp ordered by replacing the oldest slot;
    for the common TGN case `mailbox_size == 1`, this is exactly latest-message
    semantics.
    """

    def __init__(
        self,
        *,
        values: Tensor,
        timestamps: Tensor | None = None,
        row_map: Tensor | None = None,
        node_dist_index: Tensor | None = None,
        kind: str = "mailbox",
        comm: Any | None = None,
    ) -> None:
        if values.dim() != 3:
            raise ValueError("MailboxManager.values must have shape [num_nodes, mailbox_size, msg_dim]")
        self.kind = str(kind)
        self.values = values
        shape = values.shape[:2]
        self.timestamps = timestamps if timestamps is not None else torch.zeros(shape, dtype=torch.float32, device=values.device)
        self.row_map = row_map.long() if row_map is not None else None
        self.row_map_all_present = _row_map_all_present(self.row_map)
        self.row_map_is_identity = _row_map_is_identity(self.row_map)
        self.node_dist_index = node_dist_index.long() if node_dist_index is not None else None
        self._row_map_cache: dict[str, Tensor] = {}
        self._node_dist_index_cache: dict[str, Tensor] = {}
        self.commit_count = 0
        self.comm = comm
        self.async_owner_collective = self.node_dist_index is not None

    def read(self, node_ids: Tensor, **_: object) -> MailboxRead:
        rows = self._rows(node_ids.long()).to(device=self.values.device)
        return MailboxRead(
            node_ids=node_ids.long(),
            values=self.values.index_select(0, rows),
            timestamps=self.timestamps.index_select(0, rows),
        )

    def materialize(self, node_ids: Tensor) -> MailboxRead:
        return self.finish_materialize_async(self.submit_materialize_async(node_ids))

    def submit_materialize_async(self, node_ids: Tensor) -> PendingMailboxFetch:
        node_ids = node_ids.long()
        remote = _distributed() and self.node_dist_index is not None
        if int(node_ids.numel()) == 0:
            ready = MailboxRead(
                node_ids=node_ids,
                values=self.values.new_empty((0, *self.values.shape[1:])),
                timestamps=self.timestamps.new_empty((0, *self.timestamps.shape[1:])),
            )
            if remote and self._remote_fetch_needed(False):
                return self._submit_fetch_remote(node_ids, ready=ready)
            return _ready_mailbox_fetch(ready)
        if self.row_map is None:
            ready = self.read(node_ids)
            if remote and self._remote_fetch_needed(False):
                return self._submit_fetch_remote(node_ids.new_empty((0,)), ready=ready)
            return _ready_mailbox_fetch(ready)

        rows = node_ids.long() if self.row_map_is_identity else self._row_map_on(node_ids.device).index_select(0, node_ids)
        present = rows >= 0
        missing = ~present
        local_missing = bool(missing.any().item())
        fetch = self._remote_fetch_needed(local_missing) if remote else local_missing
        if not local_missing:
            local = self.read(node_ids)
            if fetch:
                return self._submit_fetch_remote(node_ids.new_empty((0,)), ready=local)
            return _ready_mailbox_fetch(local)
        if not remote:
            raise KeyError("mailbox row is not local")

        out_values = self.values.new_empty((int(node_ids.numel()), *self.values.shape[1:]))
        out_timestamps = self.timestamps.new_empty((int(node_ids.numel()), *self.timestamps.shape[1:]))
        if bool(present.any().item()):
            local_pos = present.nonzero(as_tuple=True)[0]
            local = self.read(node_ids.index_select(0, local_pos))
            out_values.index_copy_(0, local_pos.to(device=out_values.device), local.values)
            out_timestamps.index_copy_(0, local_pos.to(device=out_timestamps.device), local.timestamps)

        missing_pos = missing.nonzero(as_tuple=True)[0]
        return self._submit_fetch_remote(
            node_ids.index_select(0, missing_pos),
            out_values=out_values,
            out_timestamps=out_timestamps,
            missing_pos=missing_pos,
            result_node_ids=node_ids,
        )

    def finish_materialize_async(self, pending: PendingMailboxFetch) -> MailboxRead:
        return finish_remote_fetch(self, pending, MailboxRead)

    def append(self, node_ids: Tensor, messages: Tensor, timestamps: Tensor) -> None:
        if int(node_ids.numel()) == 0:
            return
        rows, messages, timestamps = self._deduplicate_latest(node_ids, messages, timestamps)
        rows = self._rows(rows.to(device=self.values.device)).to(device=self.values.device)
        messages = messages.to(device=self.values.device, dtype=self.values.dtype).reshape(int(rows.numel()), -1)
        timestamps = timestamps.to(device=self.values.device, dtype=self.timestamps.dtype).reshape(-1)
        if int(messages.shape[1]) != int(self.values.shape[2]):
            messages = _fit_last_dim(messages, int(self.values.shape[2]))
        slot = self.timestamps.index_select(0, rows).argmin(dim=1)
        self.values[rows, slot] = messages
        self.timestamps[rows, slot] = timestamps
        self.commit_count += 1

    def commit(self, delta) -> None:
        self.finish_commit_async(self.submit_commit_async(delta))

    def submit_commit_async(self, delta) -> PendingMailboxCommit | None:
        messages = delta.metadata.get("mailbox_messages")
        timestamps = delta.metadata.get("mailbox_timestamps")
        nodes = delta.metadata.get("mailbox_nodes", delta.node_ids)
        if messages is None or timestamps is None:
            if _distributed() and self.node_dist_index is not None:
                return self._submit_push_remote(
                    torch.empty(0, dtype=torch.long),
                    self.values.new_empty((0, int(self.values.shape[2]))),
                    self.timestamps.new_empty((0,)),
                )
            return None
        if not (_distributed() and self.node_dist_index is not None):
            self.append(nodes.long(), messages, timestamps)
            return None
        return self._submit_push_remote(nodes.long(), messages, timestamps)

    def finish_commit_async(self, pending: PendingMailboxCommit | None) -> None:
        if pending is None:
            return
        nodes, messages, timestamps = self._finish_push_remote(pending)
        self.append(nodes.long(), messages, timestamps)

    def reset(self) -> None:
        self.values.zero_()
        self.timestamps.zero_()
        self.commit_count = 0

    def _rows(self, node_ids: Tensor) -> Tensor:
        if self.row_map is None:
            rows = node_ids.long()
        elif self.row_map_is_identity:
            rows = node_ids.long()
        elif self.node_dist_index is not None:
            rows = self._row_map_on(node_ids.device).index_select(0, node_ids.long())
        else:
            rows = self._row_map_on(node_ids.device).index_select(0, node_ids.long())
        if bool(torch.any(rows < 0).item()):
            raise KeyError("mailbox row is not local")
        return rows.long()

    def _submit_push_remote(self, node_ids: Tensor, messages: Tensor, timestamps: Tensor) -> PendingMailboxCommit:
        node_ids = node_ids.long()
        device = self.values.device
        msg = messages.detach().to(device=device, dtype=self.values.dtype)
        ts = timestamps.detach().to(device=self.timestamps.device, dtype=self.timestamps.dtype)
        push = submit_owner_push(
            node_ids,
            self.node_dist_index,
            {"messages": msg, "timestamps": ts},
            scheduler=self.comm,
            name=f"mailbox_push:{self.kind}",
        )
        local_pos = push.local_pos
        local_nodes = node_ids.index_select(0, local_pos.to(node_ids.device)).to(device)
        local_messages = msg.index_select(0, local_pos.to(device=device)) if int(local_pos.numel()) else msg.new_empty((0, *msg.shape[1:]))
        local_ts = ts.index_select(0, local_pos.to(device=ts.device)) if int(local_pos.numel()) else ts.new_empty((0,))
        return PendingMailboxCommit(
            local_nodes=local_nodes,
            local_messages=local_messages,
            local_timestamps=local_ts,
            push=push,
        )

    def _finish_push_remote(self, pending: PendingMailboxCommit) -> tuple[Tensor, Tensor, Tensor]:
        received = finish_owner_push(pending.push)
        recv_nodes = received["node_ids"]
        recv_messages = received["messages"]
        recv_ts = received["timestamps"]
        return (
            torch.cat((pending.local_nodes.to(device=recv_nodes.device), recv_nodes), dim=0).to(device=self.values.device),
            torch.cat((pending.local_messages.to(device=recv_messages.device), recv_messages), dim=0).to(device=self.values.device, dtype=self.values.dtype),
            torch.cat((pending.local_timestamps.to(device=recv_ts.device), recv_ts), dim=0).to(device=self.timestamps.device, dtype=self.timestamps.dtype),
        )

    def _fetch_remote(self, node_ids: Tensor) -> MailboxRead:
        return self.finish_materialize_async(self._submit_fetch_remote(node_ids.long()))

    def _submit_fetch_remote(
        self,
        node_ids: Tensor,
        *,
        ready: MailboxRead | None = None,
        out_values: Tensor | None = None,
        out_timestamps: Tensor | None = None,
        missing_pos: Tensor | None = None,
        result_node_ids: Tensor | None = None,
    ) -> PendingMailboxFetch:
        return submit_remote_fetch(
            self,
            node_ids,
            pending_type=PendingMailboxFetch,
            read_type=MailboxRead,
            prefix="mailbox",
            ready=ready,
            out_values=out_values,
            out_timestamps=out_timestamps,
            missing_pos=missing_pos,
            result_node_ids=result_node_ids,
        )

    def _remote_fetch_needed(self, local_needed: bool) -> bool:
        return collective_needed(local_needed)

    def _deduplicate_latest(self, node_ids: Tensor, messages: Tensor, timestamps: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        node_ids = node_ids.long()
        timestamps = timestamps.reshape(-1)
        if int(node_ids.numel()) <= 1:
            return node_ids, messages, timestamps
        unique_nodes, inverse = torch.unique(node_ids, sorted=True, return_inverse=True)
        if int(unique_nodes.numel()) == int(node_ids.numel()):
            return node_ids, messages, timestamps
        latest_ts = timestamps.new_full((int(unique_nodes.numel()),), -torch.inf)
        latest_ts.scatter_reduce_(0, inverse.to(device=timestamps.device), timestamps, reduce="amax", include_self=True)
        keep = timestamps == latest_ts.index_select(0, inverse.to(device=timestamps.device))
        pos = torch.arange(int(node_ids.numel()), dtype=torch.long, device=node_ids.device)
        latest_pos = torch.full((int(unique_nodes.numel()),), -1, dtype=torch.long, device=node_ids.device)
        latest_pos.scatter_reduce_(0, inverse, pos.masked_fill(~keep.to(device=pos.device), -1), reduce="amax", include_self=True)
        return (
            unique_nodes,
            messages.index_select(0, latest_pos.to(device=messages.device)),
            timestamps.index_select(0, latest_pos.to(device=timestamps.device)),
        )

    def _row_map_on(self, device: torch.device | str) -> Tensor:
        return cached_tensor(self.row_map, self._row_map_cache, device, "mailbox row_map")

    def _node_dist_index_on(self, device: torch.device | str) -> Tensor:
        return cached_tensor(self.node_dist_index, self._node_dist_index_cache, device, "mailbox node_dist_index")



def _fit_last_dim(value: Tensor, width: int) -> Tensor:
    if int(value.shape[-1]) == int(width):
        return value
    if int(value.shape[-1]) > int(width):
        return value[..., : int(width)]
    pad = value.new_zeros((*value.shape[:-1], int(width) - int(value.shape[-1])))
    return torch.cat((value, pad), dim=-1)


def _ready_mailbox_fetch(read: MailboxRead) -> PendingMailboxFetch:
    return ready_fetch(read, PendingMailboxFetch)


__all__ = ["MailboxManager", "MailboxRead", "PendingMailboxCommit"]
