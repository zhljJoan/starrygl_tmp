from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor

from starrygl.batch import Batch
from starrygl.utils.index import compact_lookup_rows
from starrygl.view import GraphBlock


@dataclass(frozen=True)
class CoupledStateMaterialization:
    src_state: Tensor
    dst_state: Tensor
    dst_rows: Tensor
    dst_rows_identity: bool = False


def previous_state(
    *,
    batch_state: Any,
    current: Tensor,
    carried: Tensor | None,
    persist_state: bool,
    previous_block: GraphBlock | None = None,
    current_block: GraphBlock | None = None,
) -> Tensor:
    if carried is not None:
        if previous_block is not None and current_block is not None:
            # Prepared full rows share owner order; prefixes share one epoch permutation.
            previous_layout = previous_block.cache.get("chunk_limited")
            current_layout = current_block.cache.get("chunk_limited")
            previous_prefix = previous_block.cache.get("chunk_prefix_ordered", False)
            current_prefix = current_block.cache.get("chunk_prefix_ordered", False)
            same_layout = previous_prefix and current_prefix or (
                previous_layout is False and current_layout is False
                and not previous_prefix and not current_prefix
            )
            if not same_layout:
                if not int(carried.shape[0]):
                    return current.new_zeros((int(current.shape[0]), int(carried.shape[1])))
                rows = compact_lookup_rows(previous_block.dst_nodes, current_block.dst_nodes)
                values = carried.index_select(0, rows.clamp_min(0))
                return torch.where((rows >= 0).unsqueeze(1), values, torch.zeros_like(values))
        return carried if int(carried.shape[0]) == int(current.shape[0]) else _fit_rows(carried, current)
    if persist_state and isinstance(batch_state, Tensor) and int(batch_state.shape[0]):
        return _fit_rows(batch_state.to(device=current.device, dtype=current.dtype), current)
    return current.new_zeros((int(current.shape[0]), int(current.shape[1])))


def owned_previous_state(
    *,
    batch_state: Any,
    current: Tensor,
    carried: Tensor | None,
    persist_state: bool,
    state_reader: Any | None = None,
) -> Tensor:
    if carried is not None:
        return carried
    if state_reader is not None and hasattr(state_reader, "read_all_rows"):
        return state_reader.read_all_rows().values.to(device=current.device, dtype=current.dtype)
    if persist_state and isinstance(batch_state, Tensor) and int(batch_state.shape[0]):
        return batch_state.to(device=current.device, dtype=current.dtype)
    return current.new_zeros((int(current.shape[0]), int(current.shape[1])))


def materialize_coupled_neighbor_state(
    *,
    batch: Batch | None,
    blocks: Sequence[GraphBlock],
    block: GraphBlock,
    owned_state: Tensor,
    dst_owned_state: Tensor | None = None,
    state_key: str,
    state_reader: Any | None,
    prefer_state_reader: bool = True,
) -> CoupledStateMaterialization:
    del blocks
    dst_owned = owned_state if dst_owned_state is None else dst_owned_state
    table_state = _materialize_from_external_state_table(
        batch,
        block,
        state_key,
        src_table=owned_state,
        dst_table=dst_owned,
    )
    if table_state is not None:
        return table_state
    if prefer_state_reader and state_reader is not None and hasattr(state_reader, "materialize") and _has_node_ids(block):
        src_state = state_reader.materialize(block.src_nodes.long()).values.to(device=owned_state.device, dtype=owned_state.dtype)
        dst_state = state_reader.materialize(block.dst_nodes.long()).values.to(device=dst_owned.device, dtype=dst_owned.dtype)
        dst_rows = _state_rows(dst_owned, block, "dst_state_rows", int(dst_state.shape[0]))
        return CoupledStateMaterialization(src_state, dst_state, dst_rows)
    src_state = _gather_state_rows(owned_state, block, "src_state_rows", _num_src(block))
    dst_read_rows = _state_rows(dst_owned, block, "dst_state_rows", _num_dst(block))
    return CoupledStateMaterialization(
        src_state=src_state,
        dst_state=_safe_index_select_rows(dst_owned, dst_read_rows),
        dst_rows=_write_state_rows(block, "dst_state_rows", dst_read_rows),
        dst_rows_identity=bool(block.cache.get("dst_state_rows_identity", False)),
    )


def has_external_state_table(batch: Batch, state_key: str) -> bool:
    values = batch.state.get(state_key)
    node_ids = batch.state.get(f"{state_key}_node_ids")
    return isinstance(values, Tensor) and isinstance(node_ids, Tensor) and int(values.shape[0]) == int(node_ids.numel())


def materialize_snapshot_history(batch, window_id, block, previous_block, previous_values):
    src = batch.state["neighbor_recurrent_window_state"][window_id]
    if previous_block is not None and int(previous_block.cache["snapshot_id"]) + 1 == int(block.cache["snapshot_id"]):
        rows = compact_lookup_rows(previous_block.dst_nodes, block.src_nodes)
        pos = (rows >= 0).nonzero(as_tuple=True)[0]
        src = src.index_copy(0, pos, previous_values.index_select(0, rows[pos]))
    dst_rows = compact_lookup_rows(block.src_nodes, block.dst_nodes)
    return CoupledStateMaterialization(src, src.index_select(0, dst_rows), dst_rows)


def scatter_dst_state(state: Tensor, rows: Tensor, values: Tensor, *, identity_rows: bool = False) -> Tensor:
    rows = rows.to(device=state.device).long()
    if not int(rows.numel()):
        return state
    valid = rows >= 0
    if not bool(torch.all(valid).item()):
        if not bool(torch.any(valid).item()):
            return state
        values = values.index_select(0, torch.nonzero(valid, as_tuple=True)[0].to(device=values.device))
        rows = rows[valid]
    if identity_rows and int(rows.numel()) == int(state.shape[0]) and tuple(values.shape) == tuple(state.shape):
        return values.to(device=state.device, dtype=state.dtype)
    target_rows = int(rows.max().item()) + 1
    if target_rows > int(state.shape[0]):
        out = state.new_zeros((target_rows, int(state.shape[1])))
        out[: int(state.shape[0])] = state
    else:
        out = state.clone()
    out.index_copy_(0, rows, values.to(device=state.device, dtype=state.dtype))
    return out


def scatter_state_table(state: Tensor, node_ids: Tensor, updated_node_ids: Tensor, values: Tensor) -> Tensor:
    """Update every hydrated table row matching a locally produced node."""

    table_nodes = node_ids.to(device=state.device).long()
    update_rows = compact_lookup_rows(updated_node_ids.to(device=state.device).long(), table_nodes)
    present = update_rows >= 0
    if not bool(present.any().item()):
        return state
    table_rows = present.nonzero(as_tuple=True)[0]
    updates = values.index_select(0, update_rows[present].to(device=values.device))
    return scatter_dst_state(state, table_rows, updates)


def _fit_rows(state: Tensor, like: Tensor) -> Tensor:
    rows = int(like.shape[0])
    if int(state.shape[0]) >= rows:
        return state[:rows].to(device=like.device, dtype=like.dtype)
    out = like.new_zeros((rows, int(state.shape[1])))
    if int(state.shape[0]):
        out[: int(state.shape[0])] = state.to(device=like.device, dtype=like.dtype)
    return out


def _materialize_from_external_state_table(
    batch: Batch | None,
    block: GraphBlock,
    state_key: str,
    *,
    src_table: Tensor,
    dst_table: Tensor,
) -> CoupledStateMaterialization | None:
    if batch is None:
        return None
    table = batch.state.get(state_key)
    node_ids = batch.state.get(f"{state_key}_node_ids")
    if not isinstance(table, Tensor) or not isinstance(node_ids, Tensor) or int(table.shape[0]) != int(node_ids.numel()):
        return None
    if int(src_table.shape[0]) != int(node_ids.numel()) or int(dst_table.shape[0]) != int(node_ids.numel()):
        return None
    node_ids = node_ids.to(device=block.src_nodes.device).long()
    src_state, _ = _lookup_state_table(src_table, node_ids, block.src_nodes.long(), src_table.device)
    dst_state, dst_rows = _lookup_state_table(dst_table, node_ids, block.dst_nodes.long(), dst_table.device)
    return CoupledStateMaterialization(src_state, dst_state, dst_rows)


def _lookup_state_table(table: Tensor, node_ids: Tensor, query: Tensor, device: torch.device) -> tuple[Tensor, Tensor]:
    query = query.to(device=node_ids.device).long()
    if not int(query.numel()):
        return table.new_empty((0, int(table.shape[1]))), torch.empty(0, dtype=torch.long, device=device)
    order = torch.argsort(node_ids)
    sorted_ids = node_ids.index_select(0, order)
    pos = torch.searchsorted(sorted_ids, query)
    in_range = pos < int(sorted_ids.numel())
    found = torch.zeros_like(in_range, dtype=torch.bool)
    found[in_range] = sorted_ids.index_select(0, pos[in_range]) == query[in_range]
    rows = torch.full_like(query, -1)
    rows[found] = order.index_select(0, pos[found])
    out = table.new_zeros((int(query.numel()), int(table.shape[1])))
    out[found.to(device=table.device)] = table.index_select(0, rows[found].to(device=table.device))
    return out.to(device=device), rows.to(device=device)


def _gather_state_rows(state: Tensor, block: GraphBlock, key: str, count: int) -> Tensor:
    if bool(block.cache.get(f"{key}_identity", False)) and count == int(state.shape[0]):
        return state
    return _safe_index_select_rows(state, _state_rows(state, block, key, count))


def _state_rows(state: Tensor, block: GraphBlock, key: str, count: int) -> Tensor:
    rows = block.cache.get(key)
    if isinstance(rows, Tensor):
        rows = rows.to(device=state.device).long()
        if not int(rows.numel()) or int(rows.max().item()) < int(state.shape[0]):
            return rows
        if int(state.shape[0]) in {int(count), _num_src(block)}:
            return torch.arange(int(count), dtype=torch.long, device=state.device)
        return rows
    return torch.arange(min(int(count), int(state.shape[0])), dtype=torch.long, device=state.device)


def _write_state_rows(block: GraphBlock, key: str, fallback: Tensor) -> Tensor:
    rows = block.cache.get(key)
    return rows.to(device=fallback.device).long() if isinstance(rows, Tensor) and int(rows.numel()) == int(fallback.numel()) else fallback


def _safe_index_select_rows(state: Tensor, rows: Tensor) -> Tensor:
    if not int(rows.numel()):
        return state.new_empty((0, int(state.shape[1])))
    present = rows < int(state.shape[0])
    if bool(torch.all(present).item()):
        return state.index_select(0, rows)
    out = state.new_zeros((int(rows.numel()), int(state.shape[1])))
    out[present] = state.index_select(0, rows[present])
    return out


def _num_src(block: GraphBlock) -> int:
    return int(block.num_src) if block.num_src is not None else int(block.src_nodes.numel())


def _num_dst(block: GraphBlock) -> int:
    return int(block.num_dst) if block.num_dst is not None else int(block.dst_nodes.numel())


def _has_node_ids(block: GraphBlock) -> bool:
    return bool(int(block.src_nodes.numel()) and int(block.dst_nodes.numel()))


__all__ = ["CoupledStateMaterialization", "materialize_coupled_neighbor_state", "scatter_state_table"]
