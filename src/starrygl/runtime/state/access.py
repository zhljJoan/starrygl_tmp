from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from starrygl.utils.index import compact_node_time
from torch import Tensor

from starrygl.runtime.state import poll_state_update
from starrygl.view import GraphBlock


@dataclass
class PendingHydrateRead:
    kind: str
    nodes: Tensor
    compact_nodes: Tensor
    inverse: Tensor | None
    manager: Any
    read_pending: Any
    mailbox_pending: Any | None = None
    mailbox_manager: Any | None = None
    metadata_manager: Any | None = None
    combined: bool = False


def hydrate_state(batch, state_manager: object | Mapping[str, object] | None):
    return finish_hydrate_state(batch, submit_hydrate_state(batch, state_manager))


def submit_hydrate_state(
    batch,
    state_manager: object | Mapping[str, object] | None,
    *,
    node_access=None,
) -> list[PendingHydrateRead]:
    if state_manager is None:
        return []
    poll_state_update(state_manager)
    managers = state_manager if isinstance(state_manager, Mapping) else {getattr(state_manager, "kind", "state"): state_manager}
    pending: list[PendingHydrateRead] = []
    for kind, manager in managers.items():
        if getattr(manager, "snapshot_history", None) is not None:
            from starrygl.runtime.memory.snapshot import hydrate_snapshot_history
            hydrate_snapshot_history(batch, manager)
            continue
        if not (hasattr(manager, "submit_materialize_async") or hasattr(manager, "materialize") or hasattr(manager, "read")):
            continue
        nodes = _state_nodes_for_batch(batch, str(kind))
        if nodes is None:
            continue
        node_ts = _state_node_timestamps_for_batch(batch, str(kind), int(nodes.numel()))
        if _state_query_reuses_block_compaction(batch, str(kind), int(nodes.numel())):
            compact_nodes = nodes.long()
            inverse = None
        else:
            compact_nodes, inverse = compact_node_time(nodes.long(), node_ts)
        combined_async = getattr(manager, "submit_materialize_with_mailbox_async", None)
        if str(kind) == "node_memory" and callable(combined_async):
            combined_pending = combined_async(compact_nodes.long(), node_access=node_access)
            if combined_pending is not None:
                pending.append(
                    PendingHydrateRead(
                        kind=str(kind),
                        nodes=nodes.long(),
                        compact_nodes=compact_nodes.long(),
                        inverse=inverse,
                        manager=manager,
                        read_pending=combined_pending,
                        metadata_manager=manager,
                        combined=True,
                    )
                )
                continue
        combined_local = getattr(manager, "materialize_local_with_mailbox_if_all_present", None)
        if str(kind) == "node_memory" and callable(combined_local):
            combined = combined_local(compact_nodes.long())
            if combined is not None:
                read, mailbox = combined
                pending.append(
                    PendingHydrateRead(
                        kind=str(kind),
                        nodes=nodes.long(),
                        compact_nodes=compact_nodes.long(),
                        inverse=inverse,
                        manager=None,
                        read_pending=read,
                        mailbox_pending=mailbox,
                        metadata_manager=manager,
                    )
                )
                continue
        submit = getattr(manager, "submit_materialize_async", None)
        if callable(submit):
            read_pending = submit(compact_nodes.long())
        else:
            read_pending = manager.materialize(compact_nodes.long()) if hasattr(manager, "materialize") else manager.read(compact_nodes.long())
        mailbox_pending = None
        mailbox_manager = None
        submit_mailbox = getattr(manager, "submit_materialize_mailbox_async", None)
        materialize_mailbox = getattr(manager, "materialize_mailbox", None)
        raw_mailbox_manager = getattr(manager, "mailbox_manager", None)
        if str(kind) == "node_memory" and callable(submit_mailbox):
            mailbox_pending = submit_mailbox(compact_nodes.long())
            mailbox_manager = manager
        elif str(kind) == "node_memory" and callable(materialize_mailbox):
            mailbox_pending = materialize_mailbox(compact_nodes.long())
        elif str(kind) == "node_memory" and raw_mailbox_manager is not None:
            mailbox_manager = raw_mailbox_manager
            raw_submit = getattr(raw_mailbox_manager, "submit_materialize_async", None)
            if callable(raw_submit):
                mailbox_pending = raw_submit(compact_nodes.long())
            else:
                mailbox_pending = raw_mailbox_manager.materialize(compact_nodes.long()) if hasattr(raw_mailbox_manager, "materialize") else raw_mailbox_manager.read(compact_nodes.long())
        pending.append(
            PendingHydrateRead(
                kind=str(kind),
                nodes=nodes.long(),
                compact_nodes=compact_nodes.long(),
                inverse=inverse,
                manager=manager,
                read_pending=read_pending,
                mailbox_pending=mailbox_pending,
                mailbox_manager=mailbox_manager,
                metadata_manager=manager,
            )
        )
    return pending

def finish_hydrate_state(batch, pending: Sequence[PendingHydrateRead]):
    if not pending:
        return batch
    state = dict(batch.state)
    for item in pending:
        if item.combined:
            finish_combined = getattr(item.manager, "finish_materialize_with_mailbox_async", None) if item.manager is not None else None
            read, mailbox = finish_combined(item.read_pending) if callable(finish_combined) else item.read_pending
            _store_hydrated_read(
                state,
                kind=item.kind,
                read=read,
                nodes=item.nodes.long(),
                compact_nodes=item.compact_nodes.long(),
                inverse=item.inverse,
                manager=item.metadata_manager,
            )
            if mailbox is not None:
                _store_hydrated_read(
                    state,
                    kind="mailbox",
                    read=mailbox,
                    nodes=item.nodes.long(),
                    compact_nodes=item.compact_nodes.long(),
                    inverse=item.inverse,
                )
            continue
        finish = getattr(item.manager, "finish_materialize_async", None) if item.manager is not None else None
        read = finish(item.read_pending) if callable(finish) else item.read_pending
        _store_hydrated_read(
            state,
            kind=item.kind,
            read=read,
            nodes=item.nodes.long(),
            compact_nodes=item.compact_nodes.long(),
            inverse=item.inverse,
            manager=item.metadata_manager,
        )
        mailbox = item.mailbox_pending
        if mailbox is not None:
            mailbox_finish = None
            if item.mailbox_manager is not None:
                mailbox_finish = getattr(item.mailbox_manager, "finish_materialize_mailbox_async", None)
                if not callable(mailbox_finish):
                    mailbox_finish = getattr(item.mailbox_manager, "finish_materialize_async", None)
            mailbox = mailbox_finish(mailbox) if callable(mailbox_finish) else mailbox
            _store_hydrated_read(
                state,
                kind="mailbox",
                read=mailbox,
                nodes=item.nodes.long(),
                compact_nodes=item.compact_nodes.long(),
                inverse=item.inverse,
            )
    batch.state = state
    return batch


def _state_nodes_for_batch(batch, kind: str) -> Tensor | None:
    graph = batch.graph or (batch.blocks[-1][-1] if batch.blocks is not None else None)
    first = batch.blocks[0][0] if batch.blocks is not None else graph
    target = batch.targets.get("task") if isinstance(batch.targets, Mapping) else None
    if kind in {"node_memory", "mailbox"}:
        return _mfg_src_nodes(batch.blocks) if batch.blocks is not None else (None if first is None else first.src_nodes)
    if kind == "neighbor_recurrent":
        if batch.blocks is None:
            return None if first is None else first.src_nodes
        nodes = [window[-1].src_nodes.long() for window in batch.blocks if window and window[-1].src_nodes.numel()]
        return torch.cat(nodes) if nodes else torch.empty(0, dtype=torch.long)
    if kind == "node_recurrent":
        if getattr(target, "target_kind", None) == "edge":
            return None if graph is None else graph.dst_nodes
        target_ids = getattr(target, "target_ids", None)
        return target_ids if target_ids is not None else (None if graph is None else graph.dst_nodes)
    if kind == "model_recurrent":
        return torch.zeros(1, dtype=torch.long)
    return graph.dst_nodes if graph is not None else getattr(target, "target_ids", None)


def _state_node_timestamps_for_batch(batch, kind: str, count: int) -> Tensor | None:
    graph = batch.graph or (batch.blocks[-1][-1] if batch.blocks is not None else None)
    first = batch.blocks[0][0] if batch.blocks is not None else graph
    if kind in {"node_memory", "mailbox", "neighbor_recurrent"} and first is not None:
        if batch.blocks is not None:
            value = _mfg_src_timestamps(batch.blocks, last_block_only=kind == "neighbor_recurrent")
            if value is not None and value.numel() == count:
                return value
        value = first.srcdata.get("ts")
        if value is not None and value.numel() == count:
            return value.reshape(-1)
    value = None if graph is None else graph.dstdata.get("ts")
    return value.reshape(-1) if value is not None and value.numel() == count else None


def _state_query_reuses_block_compaction(batch, kind: str, count: int) -> bool:
    graph = batch.graph or (batch.blocks[-1][-1] if batch.blocks is not None else None)
    first = batch.blocks[0][0] if batch.blocks is not None else graph
    if kind in {"node_memory", "mailbox", "neighbor_recurrent"} and first is not None:
        return bool(first.cache.get("src_node_ts_compacted", False)) and first.src_nodes.numel() == count
    return bool(graph is not None and graph.cache.get("dst_node_ts_compacted", False) and graph.dst_nodes.numel() == count)


def _mfg_src_nodes(mfgs: Sequence[Sequence[GraphBlock]]) -> Tensor:
    pieces = [block.src_nodes.long() for window in mfgs for block in window if block.src_nodes.numel()]
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.long)


def _mfg_src_timestamps(
    mfgs: Sequence[Sequence[GraphBlock]],
    *,
    last_block_only: bool,
) -> Tensor | None:
    pieces: list[Tensor] = []
    for window in mfgs:
        for block in window[-1:] if last_block_only else window:
            if not block.src_nodes.numel():
                continue
            value = block.srcdata.get("ts")
            if not isinstance(value, Tensor) or value.numel() != block.src_nodes.numel():
                return None
            pieces.append(value.reshape(-1))
    return torch.cat(pieces) if pieces else None


def _expand_state_read(read, nodes: Tensor, inverse: Tensor):
    metadata = dict(getattr(read, "metadata", {}))
    for key in ("shared_mask", "shared_rows"):
        value = metadata.get(key)
        if isinstance(value, Tensor) and int(value.numel()) == int(read.node_ids.numel()):
            metadata[key] = value.index_select(0, inverse.to(value.device))
    return read.__class__(
        node_ids=nodes.long(),
        values=read.values[inverse.to(read.values.device)],
        timestamps=None if read.timestamps is None else read.timestamps[inverse.to(read.timestamps.device)],
        metadata=metadata,
    )


def _store_hydrated_read(
    state: dict[str, Any],
    *,
    kind: str,
    read,
    nodes: Tensor,
    compact_nodes: Tensor,
    inverse: Tensor | None,
    manager: object | None = None,
) -> None:
    layout_key = f"{kind}_layout_inverse"
    retain_compact = (
        kind in {"node_memory", "mailbox"}
        and inverse is not None
        and int(compact_nodes.numel()) < int(nodes.numel())
    )
    if retain_compact:
        materialized = read
        materialized_nodes = compact_nodes.long()
        state[layout_key] = inverse.long()
    else:
        materialized = read
        if inverse is not None:
            materialized = _expand_state_read(read, nodes.long(), inverse)
        materialized_nodes = nodes.long()
        state.pop(layout_key, None)

    if kind == "model_recurrent" and manager is not None and int(getattr(manager, "commit_count", 0)) == 0:
        state.pop(kind, None)
        state.pop(f"{kind}_node_ids", None)
        state.pop(f"{kind}_ts", None)
        return
    state[kind] = materialized.values
    state[f"{kind}_node_ids"] = materialized_nodes.to(device=materialized.values.device)
    if materialized.timestamps is not None:
        state[f"{kind}_ts"] = materialized.timestamps
    _apply_shared_state_metadata(state, kind, nodes.long(), manager, materialized, inverse)


def _apply_shared_state_metadata(
    state: dict[str, Any],
    kind: str,
    nodes: Tensor,
    manager: object | None,
    read: object,
    inverse: Tensor | None,
) -> None:
    if manager is None or int(nodes.numel()) == 0:
        return
    shared = getattr(manager, "shared_manager", None)
    row_map = getattr(shared, "row_map", None)
    if shared is None or row_map is None:
        return
    metadata = getattr(read, "metadata", {})
    mask = metadata.get("shared_mask") if isinstance(metadata, Mapping) else None
    rows = metadata.get("shared_rows") if isinstance(metadata, Mapping) else None
    if isinstance(mask, Tensor) and isinstance(rows, Tensor):
        if int(mask.numel()) != int(nodes.numel()) and inverse is not None:
            mask = mask.index_select(0, inverse.to(mask.device))
            rows = rows.index_select(0, inverse.to(rows.device))
        mask = mask.to(device=nodes.device, dtype=torch.bool)
        rows = rows.to(device=nodes.device, dtype=torch.long)
    else:
        row_map_on = getattr(shared, "_row_map_on", None)
        mapped = row_map_on(nodes.device) if callable(row_map_on) else row_map.to(device=nodes.device)
        rows = mapped.index_select(0, nodes.long())
        mask = rows >= 0
    if not bool(mask.any().item()):
        return
    values = read.values
    state[f"{kind}_shared_mask"] = mask.to(device=values.device)
    state[f"{kind}_shared_rows"] = rows.to(device=values.device)
    state[f"{kind}_historical"] = values.detach()
    timestamps = getattr(read, "timestamps", None)
    if timestamps is not None:
        state[f"{kind}_historical_ts"] = timestamps.detach()


__all__ = [
    "PendingHydrateRead",
    "finish_hydrate_state",
    "hydrate_state",
    "submit_hydrate_state",
]
