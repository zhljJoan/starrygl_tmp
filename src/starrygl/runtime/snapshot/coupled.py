from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import torch
from torch import Tensor

from starrygl.batch import Batch
from starrygl.view import GraphBlock
from ..state.recurrent import (
    CoupledStateMaterialization,
    has_external_state_table as _has_external_state_table,
    materialize_coupled_neighbor_state,
    owned_previous_state as _owned_previous_state,
    scatter_dst_state as _scatter_dst_state,
    scatter_state_table as _scatter_state_table,
)

if TYPE_CHECKING:
    from .scan import WindowScanResult


def run_coupled_window_scan(
    batch: Batch,
    *,
    input_project: Callable[[Tensor], Tensor],
    cell: Any,
    feature_key: str = "x",
    fallback_feature_key: str = "feat",
    persist_state: bool = False,
    state_reader: Any | None = None,
    state_materializer: Callable[..., CoupledStateMaterialization | tuple[Tensor, Tensor, Tensor]] | None = None,
    neighbor_state_delay: int = 0,
    state_transform: Callable[[Batch, int, GraphBlock, Tensor], Tensor] | None = None,
) -> WindowScanResult:
    """Run a coupled recurrent GNN that reads neighbor previous state.

    The runtime owns state layout conversion: each window receives
    ``src["h_prev"]`` in src layout and ``dst["h_prev"]`` in dst layout before
    local cell computation. Distributed state materialization plugs in through
    ``state_materializer`` or a comm-backed ``state_reader.materialize(...)``.
    """

    from .scan import WindowScanResult, _feature_for_window

    carried_owned: Tensor | None = None
    carried_dst: Tensor | None = None
    state_history: list[Tensor] = []
    final_block: GraphBlock | None = None
    window_embeddings: list[Tensor] = []
    delay = max(0, int(neighbor_state_delay))
    external_state_table = _has_external_state_table(batch, cell.state_key)
    snapshots = batch.state.get("neighbor_recurrent_snapshots")
    computed_history = []
    cache_history: dict[str, list[tuple[Tensor, int, Tensor]]] = {}
    previous_block = None
    for window_id, blocks in enumerate(batch.iter_blocks()):
        x = input_project(_feature_for_window(batch, feature_key, window_id, fallback_feature_key).float())
        block = blocks[-1]
        owned_prev = _owned_previous_state(
            batch_state=batch.state.get(cell.state_key),
            current=x,
            carried=carried_owned,
            persist_state=bool(persist_state),
            state_reader=state_reader,
        )
        if not state_history:
            state_history.append(owned_prev)
        neighbor_owned_prev = state_history[max(0, len(state_history) - 1 - delay)]
        if snapshots is not None:
            from ..state.recurrent import materialize_snapshot_history
            materialized = materialize_snapshot_history(
                batch, window_id, block, previous_block, carried_dst,
            )
        else:
            materializer = state_materializer or materialize_coupled_neighbor_state
            materialized = materializer(
                batch=batch,
                blocks=blocks,
                block=block,
                owned_state=neighbor_owned_prev,
                dst_owned_state=owned_prev,
                state_key=cell.state_key,
                state_reader=state_reader,
                prefer_state_reader=bool(external_state_table or state_reader is not None),
            )
        if isinstance(materialized, CoupledStateMaterialization):
            state = materialized
        else:
            src_state, dst_state, dst_rows = materialized
            state = CoupledStateMaterialization(src_state=src_state, dst_state=dst_state, dst_rows=dst_rows)
        from .layerwise import materialize_coupled_cell, materialize_embedding_src_async
        if batch.state.get("neighbor_recurrent_exact", False):
            src_state = materialize_embedding_src_async(
                block, state.dst_state, comm=block.cache.get("comm"), name="snapshot_exact_previous_state",
            ).wait()
            state = CoupledStateMaterialization(src_state, state.dst_state, state.dst_rows)
        channel_packets = batch.state.get("snapshot_cache_channels", {})
        if channel_packets and hasattr(cell, "materialize_cached"):
            cached = {}
            for name, policy in getattr(cell, "cache_channels", {}).items():
                if policy != "increment":
                    raise ValueError(f"unsupported snapshot cache prediction: {policy!r}")
                packet = channel_packets[name][window_id]
                dim = (int(packet.shape[1]) - 2) // 2
                age = packet.new_full(
                    (packet.shape[0], 1), int(block.cache["snapshot_id"]) + 1) - packet[:, -1:]
                cached[name] = packet[:, :dim] + age * packet[:, dim:2 * dim]
            src, final_block = cell.materialize_cached(
                block, x, state.src_state, state.dst_state, cached)
        else:
            src, final_block = materialize_coupled_cell(cell, blocks, x, state.src_state)
        if "h_prev" not in src:
            src = dict(src)
            src["h_prev"] = state.src_state
        updated_dst = cell.local_forward(final_block, src, {"h_prev": state.dst_state})
        if state_transform is not None:
            updated_dst = state_transform(batch, window_id, final_block, updated_dst)
        carried_dst = updated_dst
        table_node_ids = batch.state.get(f"{cell.state_key}_node_ids")
        if snapshots is not None:
            carried_owned = _scatter_state_table(state.src_state, block.src_nodes, block.dst_nodes, updated_dst)
            computed_history.append((block.dst_nodes, int(block.cache["snapshot_id"]) + 1, updated_dst.detach()))
            bound_channels = batch.state.get("snapshot_cache_channels", {})
            for name in getattr(cell, "cache_channels", {}):
                if name not in bound_channels:
                    continue
                value = src.get(name)
                if value is not None:
                    cache_history.setdefault(name, []).append(
                        (block.dst_nodes, int(block.cache["snapshot_id"]) + 1, value.detach()))
            previous_block = block
        else:
            carried_owned = (
                _scatter_state_table(owned_prev, table_node_ids, block.dst_nodes, updated_dst)
                if external_state_table and isinstance(table_node_ids, Tensor)
                else _scatter_dst_state(owned_prev, state.dst_rows, updated_dst, identity_rows=state.dst_rows_identity)
            )
        state_history.append(carried_owned)
        window_embeddings.append(
            cell.embedding_from_state(updated_dst) if hasattr(cell, "embedding_from_state") else updated_dst
        )
    if carried_owned is None or carried_dst is None or final_block is None:
        raise ValueError("batch has no recurrent windows")
    return WindowScanResult(
        embeddings=window_embeddings[-1],
        state_embeddings=carried_owned,
        final_block=final_block,
        window_embeddings=tuple(window_embeddings),
        state_history=tuple(computed_history),
        cache_history={name: tuple(values) for name, values in cache_history.items()},
    )
