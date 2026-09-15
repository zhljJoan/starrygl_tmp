from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from starrygl.model import StateDelta
from starrygl.runtime.comm import distributed
from starrygl.utils.index import compact_lookup_rows


def present_rows(manager: Any | None, node_ids: Tensor) -> Tensor | None:
    if manager is None:
        return None
    nodes = node_ids.long()
    row_map = getattr(manager, "row_map", None)
    if row_map is not None:
        row_map_on = getattr(manager, "_row_map_on", None)
        mapped = row_map_on(nodes.device) if callable(row_map_on) else row_map.to(nodes.device)
        return mapped.index_select(0, nodes)
    index = getattr(manager, "node_dist_index", None)
    if index is not None and distributed():
        from starrygl.utils.route import dist_loc, dist_part

        index_on = getattr(manager, "_node_dist_index_on", None)
        index = index_on(nodes.device) if callable(index_on) else index.to(nodes.device)
        selected = index.index_select(0, nodes)
        rows = torch.where(
            dist_part(selected) == int(dist.get_rank()),
            dist_loc(selected),
            torch.full_like(selected, -1),
        )
    else:
        rows = nodes
    values = getattr(manager, "values", None)
    if isinstance(values, Tensor):
        rows = torch.where(rows < int(values.shape[0]), rows, torch.full_like(rows, -1))
    return rows.long()


def present_nodes(node_ids: Tensor, manager: Any) -> tuple[Tensor, Tensor, Tensor]:
    rows = present_rows(manager, node_ids)
    assert rows is not None
    pos = (rows >= 0).nonzero(as_tuple=True)[0]
    return node_ids.index_select(0, pos), pos, rows.index_select(0, pos)


def state_read_rows(manager: Any, node_ids: Tensor, rows: Tensor):
    from starrygl.store.state import StateRead

    rows = rows.to(manager.values.device)
    return StateRead(
        node_ids=node_ids.long(),
        values=manager.values.index_select(0, rows),
        timestamps=None if manager.timestamps is None else manager.timestamps.index_select(0, rows),
    )


def mailbox_read_rows(manager: Any, node_ids: Tensor, rows: Tensor):
    from starrygl.store.mailbox import MailboxRead

    rows = rows.to(manager.values.device)
    return MailboxRead(
        node_ids=node_ids.long(),
        values=manager.values.index_select(0, rows),
        timestamps=manager.timestamps.index_select(0, rows),
    )


def overlay_state(owner, shared_manager):
    if shared_manager is None or int(owner.node_ids.numel()) == 0:
        return owner
    _, pos, rows = present_nodes(owner.node_ids, shared_manager)
    if not int(pos.numel()):
        return owner
    shared = state_read_rows(shared_manager, owner.node_ids.index_select(0, pos), rows)
    if owner.timestamps is None or shared.timestamps is None:
        return owner
    keep = shared.timestamps.to(owner.timestamps) > owner.timestamps.index_select(0, pos.to(owner.timestamps.device))
    return _overlay_read(owner, shared, pos, keep)


def overlay_mailbox(owner, shared_manager):
    if owner is None or shared_manager is None or int(owner.node_ids.numel()) == 0:
        return owner
    _, pos, rows = present_nodes(owner.node_ids, shared_manager)
    if not int(pos.numel()):
        return owner
    shared = mailbox_read_rows(shared_manager, owner.node_ids.index_select(0, pos), rows)
    keep = shared.timestamps.amax(dim=1).to(owner.timestamps) > owner.timestamps.amax(dim=1).index_select(0, pos.to(owner.timestamps.device))
    return _overlay_read(owner, shared, pos, keep)


def _overlay_read(owner, shared, owner_pos: Tensor, keep: Tensor):
    if not bool(keep.any().item()):
        return owner
    from starrygl.store.mailbox import MailboxRead
    from starrygl.store.state import StateRead

    shared_pos = keep.nonzero(as_tuple=True)[0]
    owner_pos = owner_pos.index_select(0, shared_pos.to(owner_pos.device))
    values = owner.values.clone()
    values.index_copy_(0, owner_pos.to(values.device), shared.values.index_select(0, shared_pos.to(shared.values.device)).to(values))
    timestamps = None if owner.timestamps is None else owner.timestamps.clone()
    if timestamps is not None:
        timestamps.index_copy_(0, owner_pos.to(timestamps.device), shared.timestamps.index_select(0, shared_pos.to(shared.timestamps.device)).to(timestamps))
    read_type = MailboxRead if owner.__class__.__name__ == "MailboxRead" else StateRead
    return read_type(owner.node_ids, values, timestamps, owner.metadata)


def commit_state_local(manager: Any, delta: StateDelta) -> None:
    commit = getattr(manager, "_commit_local", None)
    (commit if callable(commit) else manager.commit)(delta)


def commit_mailbox_local(manager: Any, nodes: Tensor, values: Tensor, timestamps: Tensor) -> None:
    append = getattr(manager, "append", None)
    if callable(append):
        append(nodes.long(), values, timestamps)
        return
    manager.commit(StateDelta(
        node_ids=nodes.long(), values=values, kind="mailbox", timestamps=timestamps,
        metadata={"mailbox_nodes": nodes.long(), "mailbox_messages": values, "mailbox_timestamps": timestamps},
    ))


def latest_by_node(node_ids: Tensor, values: Tensor, timestamps: Tensor | None):
    nodes, pos = latest_positions(node_ids, timestamps)
    return (
        nodes,
        values.index_select(0, pos.to(values.device)),
        None if timestamps is None else timestamps.index_select(0, pos.to(timestamps.device)),
    )


def latest_positions(node_ids: Tensor, timestamps: Tensor | None) -> tuple[Tensor, Tensor]:
    node_ids = node_ids.long()
    if int(node_ids.numel()) <= 1:
        return node_ids, torch.arange(int(node_ids.numel()), device=node_ids.device)
    nodes, inverse = torch.unique(node_ids, sorted=True, return_inverse=True)
    if int(nodes.numel()) == int(node_ids.numel()):
        return node_ids, torch.arange(int(node_ids.numel()), device=node_ids.device)
    pos = torch.arange(int(node_ids.numel()), device=node_ids.device)
    if timestamps is None:
        latest = torch.full((int(nodes.numel()),), -1, dtype=torch.long, device=node_ids.device)
        latest.scatter_reduce_(0, inverse, pos, reduce="amax", include_self=True)
        return nodes, latest
    score = timestamps.reshape(int(node_ids.numel()), -1).amax(dim=1)
    floor = -torch.inf if score.is_floating_point() else torch.iinfo(score.dtype).min
    latest_score = score.new_full((int(nodes.numel()),), floor)
    latest_score.scatter_reduce_(0, inverse.to(score.device), score, reduce="amax", include_self=True)
    keep = score == latest_score.index_select(0, inverse.to(score.device))
    latest = torch.full((int(nodes.numel()),), -1, dtype=torch.long, device=node_ids.device)
    latest.scatter_reduce_(0, inverse, pos.masked_fill(~keep.to(pos.device), -1), reduce="amax", include_self=True)
    return nodes, latest


def filter_delta(delta: StateDelta, manager: Any) -> StateDelta | None:
    rows = present_rows(manager, delta.node_ids)
    if rows is None or int(delta.node_ids.numel()) == 0:
        return delta
    pos = (rows >= 0).nonzero(as_tuple=True)[0]
    if not int(pos.numel()):
        return None
    metadata = dict(delta.metadata)
    filter_mailbox_to_manager(metadata, manager)
    return StateDelta(
        node_ids=delta.node_ids.index_select(0, pos.to(delta.node_ids.device)),
        values=delta.values.index_select(0, pos.to(delta.values.device)),
        kind=delta.kind,
        timestamps=None if delta.timestamps is None else delta.timestamps.index_select(0, pos.to(delta.timestamps.device)),
        metadata=metadata,
    )


def filter_mailbox_to_manager(metadata: dict[str, Any], manager: Any) -> None:
    nodes = metadata.get("mailbox_nodes")
    if not _valid_mailbox(metadata) or not isinstance(nodes, Tensor):
        return
    rows = present_rows(manager, nodes)
    if rows is not None:
        _filter_mailbox(metadata, (rows >= 0).nonzero(as_tuple=True)[0])


def filter_mailbox_to_nodes(metadata: dict[str, Any], allowed: Tensor) -> None:
    nodes = metadata.get("mailbox_nodes")
    if not _valid_mailbox(metadata) or not isinstance(nodes, Tensor):
        return
    if not int(allowed.numel()):
        return _filter_mailbox(metadata, nodes.new_empty(0))
    allowed = torch.sort(allowed.to(nodes.device).long()).values
    lookup = torch.searchsorted(allowed, nodes.long())
    valid = lookup < int(allowed.numel())
    safe = lookup.clamp_max(int(allowed.numel()) - 1)
    _filter_mailbox(metadata, (valid & (allowed.index_select(0, safe) == nodes)).nonzero(as_tuple=True)[0])


def filter_aligned_mailbox(metadata: dict[str, Any], delta_nodes: Tensor, keep_pos: Tensor) -> bool:
    nodes = metadata.get("mailbox_nodes")
    if not _valid_mailbox(metadata) or not isinstance(nodes, Tensor) or int(nodes.numel()) != int(delta_nodes.numel()):
        return False
    if not torch.equal(nodes.long(), delta_nodes.to(nodes.device).long()):
        return False
    _filter_mailbox(metadata, keep_pos)
    return True


def _valid_mailbox(metadata: dict[str, Any]) -> bool:
    nodes, values, timestamps = (metadata.get(name) for name in ("mailbox_nodes", "mailbox_messages", "mailbox_timestamps"))
    return all(isinstance(value, Tensor) for value in (nodes, values, timestamps)) and int(values.shape[0]) == int(nodes.numel()) == int(timestamps.shape[0])


def _filter_mailbox(metadata: dict[str, Any], pos: Tensor) -> None:
    for name in ("mailbox_nodes", "mailbox_messages", "mailbox_timestamps"):
        value = metadata[name]
        metadata[name] = value.index_select(0, pos.to(value.device))


def select_nodes(base_ids: Tensor, values: Tensor, node_ids: Tensor) -> Tensor:
    if not int(node_ids.numel()):
        return values.new_empty((0, *values.shape[1:]))
    pos = compact_lookup_rows(base_ids, node_ids)
    if bool((pos < 0).any().item()):
        raise KeyError("shared memory node is not present in StateDelta")
    return values.index_select(0, pos.to(values.device))
