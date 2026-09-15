from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor

from starrygl.model import StateDelta
from starrygl.runtime.comm import collective_needed, distributed as _distributed
from starrygl.utils.route import dist_loc
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
class StateRead:
    node_ids: Tensor
    values: Tensor
    timestamps: Tensor | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass
class PendingStateCommit:
    delta: StateDelta
    local_nodes: Tensor
    local_values: Tensor
    local_timestamps: Tensor | None
    push: OwnerPush


@dataclass
class PendingStateFetch:
    node_ids: Tensor
    ready: StateRead | None
    out_values: Tensor | None
    out_timestamps: Tensor | None
    missing_pos: Tensor | None
    order: Tensor
    recv_values: Any | None
    recv_timestamps: Any | None
    scheduler: Any | None
    remote_inverse: Tensor | None = None


class StateManager:
    def __init__(
        self,
        *,
        values: Tensor,
        timestamps: Tensor | None = None,
        row_map: Tensor | None = None,
        node_dist_index: Tensor | None = None,
        kind: str = "state",
        comm: Any | None = None,
    ) -> None:
        self.kind = str(kind)
        self.values = values
        self.timestamps = timestamps if timestamps is not None else torch.zeros(int(values.shape[0]), dtype=torch.float32, device=values.device)
        self.row_map = row_map.long() if row_map is not None else None
        self.row_map_all_present = _row_map_all_present(self.row_map)
        self.row_map_is_identity = _row_map_is_identity(self.row_map)
        self.node_dist_index = node_dist_index.long() if node_dist_index is not None else None
        self._row_map_cache: dict[str, Tensor] = {}
        self._node_dist_index_cache: dict[str, Tensor] = {}
        self.commit_count = 0
        self.comm = comm
        self.async_owner_collective = self.node_dist_index is not None

    def read(self, node_ids: Tensor) -> StateRead:
        rows = self._rows(node_ids.long()).to(device=self.values.device)
        return StateRead(
            node_ids=node_ids.long(),
            values=self.values.index_select(0, rows),
            timestamps=None if self.timestamps is None else self.timestamps.index_select(0, rows),
        )

    def materialize(self, node_ids: Tensor) -> StateRead:
        return self.finish_materialize_async(self.submit_materialize_async(node_ids))

    def submit_materialize_async(self, node_ids: Tensor) -> PendingStateFetch:
        node_ids = node_ids.long()
        remote = _distributed() and self.node_dist_index is not None
        if int(node_ids.numel()) == 0:
            ready = StateRead(
                node_ids=node_ids,
                values=self.values.new_empty((0, *self.values.shape[1:])),
                timestamps=None if self.timestamps is None else self.timestamps.new_empty((0,)),
            )
            if remote and self._remote_fetch_needed(False):
                return self._submit_fetch_remote(node_ids, ready=ready)
            return _ready_state_fetch(ready)
        if self.row_map is None:
            ready = self.read(node_ids)
            if remote and self._remote_fetch_needed(False):
                return self._submit_fetch_remote(node_ids.new_empty((0,)), ready=ready)
            return _ready_state_fetch(ready)

        rows = node_ids.long() if self.row_map_is_identity else self._row_map_on(node_ids.device).index_select(0, node_ids)
        present = rows >= 0
        missing = ~present
        local_missing = bool(missing.any().item())
        fetch = self._remote_fetch_needed(local_missing) if remote else local_missing
        if not local_missing:
            local = self.read(node_ids)
            if fetch:
                return self._submit_fetch_remote(node_ids.new_empty((0,)), ready=local)
            return _ready_state_fetch(local)
        if not remote:
            raise KeyError("state row is not local")

        out_values = self.values.new_empty((int(node_ids.numel()), *self.values.shape[1:]))
        out_timestamps = None if self.timestamps is None else self.timestamps.new_empty((int(node_ids.numel()),))
        if bool(present.any().item()):
            local_pos = present.nonzero(as_tuple=True)[0]
            local = self.read(node_ids.index_select(0, local_pos))
            out_values.index_copy_(0, local_pos.to(device=out_values.device), local.values)
            if out_timestamps is not None and local.timestamps is not None:
                out_timestamps.index_copy_(0, local_pos.to(device=out_timestamps.device), local.timestamps)

        missing_pos = missing.nonzero(as_tuple=True)[0]
        return self._submit_fetch_remote(
            node_ids.index_select(0, missing_pos),
            out_values=out_values,
            out_timestamps=out_timestamps,
            missing_pos=missing_pos,
            result_node_ids=node_ids,
        )

    def finish_materialize_async(self, pending: PendingStateFetch) -> StateRead:
        return finish_remote_fetch(self, pending, StateRead)

    def read_all_rows(self) -> StateRead:
        node_ids = torch.arange(int(self.values.shape[0]), dtype=torch.long, device=self.values.device)
        return StateRead(
            node_ids=node_ids,
            values=self.values,
            timestamps=self.timestamps,
            metadata={"full_rows": True},
        )

    def commit(self, delta: StateDelta) -> None:
        self.finish_commit_async(self.submit_commit_async(delta))

    def submit_commit_async(self, delta: StateDelta) -> PendingStateCommit | None:
        if not (_distributed() and self.node_dist_index is not None):
            self._commit_local(delta)
            return None
        return self._submit_push_remote(delta)

    def finish_commit_async(self, pending: PendingStateCommit | None) -> None:
        if pending is None:
            return
        self._commit_local(self._finish_push_remote(pending))

    def reset(self) -> None:
        self.values.zero_()
        if self.timestamps is not None:
            self.timestamps.zero_()
        self.commit_count = 0
        if getattr(self, "snapshot_history", None) is not None:
            self.snapshot_history.reset()
            self.snapshot_shared_history.reset()
            self.snapshot_version = 0

    def _submit_push_remote(self, delta: StateDelta) -> PendingStateCommit:
        node_ids = delta.node_ids.long()
        values_device = self.values.device
        values = delta.values.detach().to(device=values_device, dtype=self.values.dtype)
        payloads = {"values": values}
        timestamps = None
        if delta.timestamps is not None:
            ts_device = self.timestamps.device if self.timestamps is not None else values_device
            ts_dtype = self.timestamps.dtype if self.timestamps is not None else delta.timestamps.dtype
            timestamps = delta.timestamps.detach().to(device=ts_device, dtype=ts_dtype)
            payloads["timestamps"] = timestamps
        push = submit_owner_push(
            node_ids,
            self.node_dist_index,
            payloads,
            scheduler=self.comm,
            name=f"state_push:{self.kind}",
        )
        local_pos = push.local_pos
        local_nodes = node_ids.index_select(0, local_pos.to(node_ids.device)).to(values_device)
        local_values = values.index_select(0, local_pos.to(device=values_device)) if int(local_pos.numel()) else values.new_empty((0, *values.shape[1:]))
        local_timestamps = None if timestamps is None else timestamps.index_select(0, local_pos.to(timestamps.device))
        return PendingStateCommit(
            delta=delta,
            local_nodes=local_nodes,
            local_values=local_values,
            local_timestamps=local_timestamps,
            push=push,
        )

    def _finish_push_remote(self, pending: PendingStateCommit) -> StateDelta:
        received = finish_owner_push(pending.push)
        recv_nodes = received["node_ids"]
        recv_values = received["values"]
        recv_ts = received.get("timestamps")
        merged_nodes = torch.cat((pending.local_nodes.to(device=recv_nodes.device), recv_nodes), dim=0)
        merged_values = torch.cat((pending.local_values.to(device=recv_values.device), recv_values), dim=0).to(device=self.values.device, dtype=self.values.dtype)
        merged_ts = None
        if pending.delta.timestamps is not None:
            local_ts = pending.local_timestamps
            if local_ts is None:
                ts_device = recv_ts.device if recv_ts is not None else self.values.device
                local_ts = torch.empty(0, dtype=(self.timestamps.dtype if self.timestamps is not None else pending.delta.timestamps.dtype), device=ts_device)
            merged_ts = torch.cat((local_ts.to(device=recv_ts.device), recv_ts), dim=0).to(device=self.timestamps.device if self.timestamps is not None else self.values.device, dtype=self.timestamps.dtype if self.timestamps is not None else pending.delta.timestamps.dtype)
        return StateDelta(
            node_ids=merged_nodes.to(device=self.values.device),
            values=merged_values,
            kind=pending.delta.kind,
            timestamps=merged_ts,
            metadata=pending.delta.metadata,
        )

    def _fetch_remote(self, node_ids: Tensor) -> StateRead:
        return self.finish_materialize_async(self._submit_fetch_remote(node_ids.long()))

    def _submit_fetch_remote(
        self,
        node_ids: Tensor,
        *,
        ready: StateRead | None = None,
        out_values: Tensor | None = None,
        out_timestamps: Tensor | None = None,
        missing_pos: Tensor | None = None,
        result_node_ids: Tensor | None = None,
    ) -> PendingStateFetch:
        return submit_remote_fetch(
            self,
            node_ids,
            pending_type=PendingStateFetch,
            read_type=StateRead,
            prefix="state",
            ready=ready,
            out_values=out_values,
            out_timestamps=out_timestamps,
            missing_pos=missing_pos,
            result_node_ids=result_node_ids,
        )

    def _remote_fetch_needed(self, local_needed: bool) -> bool:
        return collective_needed(local_needed)

    def _commit_local(self, delta: StateDelta) -> None:
        if getattr(self, "snapshot_history", None) is not None:
            from starrygl.runtime.memory.snapshot import commit_snapshot_history
            commit_snapshot_history(self, delta)
        if int(delta.node_ids.numel()) == 0:
            return
        rows = self._rows(delta.node_ids.long().to(device=self.values.device))
        values = delta.values.to(device=self.values.device, dtype=self.values.dtype)
        keep = torch.ones(int(rows.numel()), dtype=torch.bool, device=self.values.device)
        timestamps = None
        if delta.timestamps is not None and self.timestamps is not None:
            timestamps = delta.timestamps.to(device=self.values.device, dtype=self.timestamps.dtype)
            keep = timestamps >= self.timestamps.index_select(0, rows)
        if not bool(keep.any().item()):
            return
        rows = rows[keep]
        values = values[keep]
        if timestamps is not None:
            timestamps = timestamps[keep]
        self.values.index_copy_(0, rows, values)
        if timestamps is not None and self.timestamps is not None:
            self.timestamps.index_copy_(0, rows, timestamps)
        self.commit_count += 1

    def _rows(self, node_ids: Tensor) -> Tensor:
        if self.row_map_is_identity:
            rows = node_ids.long()
        elif self.row_map is not None:
            rows = self._row_map_on(node_ids.device).index_select(0, node_ids.long())
        elif self.node_dist_index is not None:
            rows = dist_loc(self._node_dist_index_on(node_ids.device).index_select(0, node_ids.long()))
        else:
            rows = node_ids.long()
        if bool(torch.any(rows < 0).item()):
            raise KeyError("state row is not local")
        return rows.long()

    def _row_map_on(self, device: torch.device | str) -> Tensor:
        return cached_tensor(self.row_map, self._row_map_cache, device, "state row_map")

    def _node_dist_index_on(self, device: torch.device | str) -> Tensor:
        return cached_tensor(self.node_dist_index, self._node_dist_index_cache, device, "state node_dist_index")


def _ready_state_fetch(read: StateRead) -> PendingStateFetch:
    return ready_fetch(read, PendingStateFetch)


__all__ = ["PendingStateCommit", "StateManager", "StateRead"]
