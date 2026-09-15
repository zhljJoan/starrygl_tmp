from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from starrygl.store import StoreBundle
from starrygl.view import GraphBlock

from starrygl.runtime.comm import CommScheduler
from starrygl.runtime.exchange import PendingNodeFeatureFetch
from starrygl.runtime.dataloader.blocks import (
    move_graph_block as _move_snapshot_graph_block,
    snapshot_graph_block as _snapshot_graph_block,
)
from starrygl.runtime.dataloader.materialize import attach_comm_to_blocks
from .features import (
    _attach_node_dist_index,
    _read_snapshot_features,
)
from .rows import (
    _apply_chunk_limit_to_snapshot_row,
    _move_snapshot_features,
    _move_snapshot_row,
    _reorder_snapshot_row_for_chunk_prefix,
)


@dataclass(frozen=True)
class _SnapshotEntry:
    row: Mapping[str, Any]
    graph: GraphBlock
    features: Mapping[str, Tensor]
    chunk_limit: int
    requires_node_exchange: bool = False
    pending_node_features: PendingNodeFeatureFetch | None = None
    feature_launch_deferred: bool = False


@dataclass
class _SnapshotGraphBlob:
    """Per-epoch packed snapshot row used to serve chunk prefixes.

    The blob is an internal materialization cache. It keeps the model-facing
    contract unchanged: callers still receive normal GraphBlock/features in
    Batch.blocks and Batch.features.
    """

    sid: int
    row: Mapping[str, Any]
    full_entry: _SnapshotEntry | None = None
    prefix_entries: dict[int, _SnapshotEntry] = field(default_factory=dict)


def _materialize_snapshot_entry(
    store: StoreBundle,
    row: Mapping[str, Any],
    *,
    chunk_limit: int,
    chunk_order: Tensor | None = None,
    comm: CommScheduler | None = None,
    defer_finish: bool = False,
    defer_launch: bool = False,
    device: str | torch.device | None = None,
    attach_reverse: bool = True,
    precompute_edge_rows: bool = False,
    use_sparse_gcn: bool = False,
    use_dgl_gcn: bool = False,
) -> _SnapshotEntry:
    if device is not None:
        row = _move_snapshot_row(row, torch.device(device))
    limited = _apply_chunk_limit_to_snapshot_row(row, int(chunk_limit), chunk_order=chunk_order)
    graph = _snapshot_graph_block(
        limited,
        attach_reverse=attach_reverse,
        precompute_edge_rows=precompute_edge_rows,
        use_sparse_gcn=use_sparse_gcn,
        use_dgl_gcn=use_dgl_gcn,
    )
    _attach_node_dist_index(graph, store)
    attach_comm_to_blocks((graph,), comm)
    if bool(defer_launch):
        features: dict[str, Tensor] = {}
        requires_node_exchange = False
        pending = None
    else:
        features, requires_node_exchange, pending = _read_snapshot_features(
            store,
            limited,
            comm=comm,
            defer_finish=bool(defer_finish),
        )
    if device is not None:
        target = torch.device(device)
        graph = _move_snapshot_graph_block(graph, target)
        features = _move_snapshot_features(features, target)
    return _SnapshotEntry(
        row=limited,
        graph=graph,
        features=features,
        chunk_limit=int(chunk_limit),
        requires_node_exchange=bool(requires_node_exchange),
        pending_node_features=pending,
        feature_launch_deferred=bool(defer_launch),
    )


def _materialize_snapshot_entries(
    store: StoreBundle,
    rows: Sequence[tuple[Mapping[str, Any], int]],
    *,
    chunk_order: Tensor | None,
    comm: CommScheduler | None,
    options: Mapping[str, Any] | None,
    entry_cache: dict[tuple[int, int], _SnapshotEntry] | None = None,
    blob_cache: dict[int, _SnapshotGraphBlob] | None = None,
) -> list[_SnapshotEntry]:
    options = options or {}
    device = options.get("_materialize_device")
    defer_finish = bool(options.get("defer_node_feature_finish", False))
    defer_launch = bool(options.get("defer_feature_launch", False))
    attach_reverse = bool(options.get("snapshot_reverse_direction", True))
    precompute_edge_rows = bool(options.get("snapshot_precompute_edge_rows", False))
    use_sparse_gcn = bool(options.get("snapshot_sparse_gcn", False))
    use_dgl_gcn = bool(options.get("snapshot_dgl_gcn", False))
    ordered_full = (
        chunk_order is not None and options.get("snapshot_materialize_on_device") is True
        and options.get("_snapshot_train_loss_mode") == "window_mean"
    )

    def materialize(row: Mapping[str, Any], limit: int) -> _SnapshotEntry:
        return _materialize_snapshot_entry(
            store,
            row,
            chunk_limit=int(limit),
            chunk_order=chunk_order,
            comm=comm,
            defer_finish=defer_finish,
            defer_launch=defer_launch,
            device=device,
            attach_reverse=attach_reverse,
            precompute_edge_rows=precompute_edge_rows,
            use_sparse_gcn=use_sparse_gcn,
            use_dgl_gcn=use_dgl_gcn,
        )

    if entry_cache is None or blob_cache is None:
        if not ordered_full:
            return [materialize(row, limit) for row, limit in rows]
        entry_cache, blob_cache = {}, {}

    entries = []
    for row, limit in rows:
        sid = int(row["snapshot_id"])
        limit = int(limit)
        entry = entry_cache.get((sid, limit))
        if entry is not None:
            attach_comm_to_blocks((entry.graph,), comm)
        elif limit >= 0 or ordered_full:
            blob = _snapshot_graph_blob(
                blob_cache,
                sid=sid,
                row=row,
                chunk_order=chunk_order,
                device=device,
                cache_row_on_device=bool(options.get("snapshot_row_cache_on_device", True)),
            )
            entry = _snapshot_blob_entry(
                blob,
                store=store,
                chunk_limit=limit,
                comm=comm,
                defer_finish=defer_finish,
                defer_launch=defer_launch,
                device=device,
                attach_reverse=attach_reverse,
                precompute_edge_rows=precompute_edge_rows,
                use_sparse_gcn=use_sparse_gcn,
                use_dgl_gcn=use_dgl_gcn,
            )
        else:
            entry = _get_persistent_snapshot_entry(store, sid=sid, device=device)
            if entry is None:
                entry = materialize(row, limit)
            else:
                attach_comm_to_blocks((entry.graph,), comm)
            if entry.pending_node_features is None and (
                not entry.feature_launch_deferred
                or bool(options.get("_reuse_static_snapshot_graph", False))
            ):
                entry_cache[(sid, limit)] = entry
                if not entry.feature_launch_deferred:
                    _put_persistent_snapshot_entry(store, sid=sid, device=device, entry=entry)
        if bool(options.get("_reuse_static_snapshot_graph", False)) and entry.pending_node_features is None:
            entry_cache[(sid, limit)] = entry
        entries.append(entry)
    return entries


def _snapshot_graph_blob(
    cache: dict[int, _SnapshotGraphBlob],
    *,
    sid: int,
    row: Mapping[str, Any],
    chunk_order: Tensor | None,
    device: str | torch.device | None,
    cache_row_on_device: bool,
) -> _SnapshotGraphBlob:
    blob = cache.get(int(sid))
    if blob is not None:
        return blob
    if device is not None and bool(cache_row_on_device):
        row = _move_snapshot_row(row, torch.device(device))
    packed = _reorder_snapshot_row_for_chunk_prefix(row, chunk_order=chunk_order)
    blob = _SnapshotGraphBlob(sid=int(sid), row=packed)
    cache[int(sid)] = blob
    return blob


def _snapshot_blob_entry(
    blob: _SnapshotGraphBlob,
    *,
    store: StoreBundle,
    chunk_limit: int,
    comm: CommScheduler | None,
    defer_finish: bool,
    defer_launch: bool,
    device: str | torch.device | None,
    attach_reverse: bool,
    precompute_edge_rows: bool,
    use_sparse_gcn: bool,
    use_dgl_gcn: bool,
) -> _SnapshotEntry:
    limit = int(chunk_limit)
    cached = blob.prefix_entries.get(limit)
    if cached is not None:
        attach_comm_to_blocks((cached.graph,), comm)
        return cached
    if limit >= 0:
        limited = _apply_chunk_limit_to_snapshot_row(blob.row, limit, chunk_order=None)
        entry = _materialize_snapshot_entry(
            store,
            limited,
            chunk_limit=-1,
            chunk_order=None,
            comm=comm,
            defer_finish=bool(defer_finish),
            defer_launch=bool(defer_launch),
            device=device,
            attach_reverse=attach_reverse,
            precompute_edge_rows=precompute_edge_rows,
            use_sparse_gcn=use_sparse_gcn,
            use_dgl_gcn=use_dgl_gcn,
        )
        entry = _SnapshotEntry(
            row=entry.row,
            graph=entry.graph,
            features=entry.features,
            chunk_limit=limit,
            requires_node_exchange=entry.requires_node_exchange,
            pending_node_features=entry.pending_node_features,
            feature_launch_deferred=entry.feature_launch_deferred,
        )
        entry.graph.cache["chunk_prefix_ordered"] = True
        if entry.pending_node_features is None and not entry.feature_launch_deferred:
            blob.prefix_entries[limit] = entry
        return entry
    if blob.full_entry is None:
        full_entry = _materialize_snapshot_entry(
            store,
            blob.row,
            chunk_limit=-1,
            chunk_order=None,
            comm=comm,
            defer_finish=bool(defer_finish),
            defer_launch=bool(defer_launch),
            device=device,
            attach_reverse=attach_reverse,
            precompute_edge_rows=precompute_edge_rows,
            use_sparse_gcn=use_sparse_gcn,
            use_dgl_gcn=use_dgl_gcn,
        )
        full_entry.graph.cache["chunk_prefix_ordered"] = True
        if full_entry.feature_launch_deferred or full_entry.pending_node_features is not None:
            return full_entry
        blob.full_entry = full_entry
    attach_comm_to_blocks((blob.full_entry.graph,), comm)
    return blob.full_entry


def _truncate_snapshot_entry(
    entry: _SnapshotEntry,
    *,
    chunk_limit: int,
    chunk_order: Tensor | None = None,
    comm: CommScheduler | None = None,
    device: str | torch.device | None = None,
) -> _SnapshotEntry:
    limited = _apply_chunk_limit_to_snapshot_row(entry.row, int(chunk_limit), chunk_order=chunk_order)
    graph = _snapshot_graph_block(
        limited,
        attach_reverse="reverse_row" in entry.graph.cache,
        precompute_edge_rows=entry.graph.row is not None and entry.graph.col is not None,
        use_sparse_gcn=bool(entry.graph.cache.get("use_sparse_tensor_gcn", False)),
        use_dgl_gcn=bool(entry.graph.cache.get("use_dgl_gcn", False)),
    )
    if "node_dist_index" in entry.graph.cache:
        graph.cache["node_dist_index"] = entry.graph.cache["node_dist_index"]
    attach_comm_to_blocks((graph,), comm)
    features = _slice_snapshot_entry_features(entry.features, entry.row, limited)
    if device is not None:
        target = torch.device(device)
        graph = _move_snapshot_graph_block(graph, target)
        features = _move_snapshot_features(features, target)
    return _SnapshotEntry(
        row=limited,
        graph=graph,
        features=features,
        chunk_limit=int(chunk_limit),
        requires_node_exchange=bool(entry.requires_node_exchange),
        pending_node_features=None,
        feature_launch_deferred=bool(entry.feature_launch_deferred),
    )


def _persistent_snapshot_cache(store: StoreBundle) -> dict[tuple[int, str], _SnapshotEntry]:
    cache = store.graph.runtime_cache.setdefault("snapshot_full_entries", {})
    return cache


def _snapshot_cache_device_key(device: str | torch.device | None) -> str:
    if device is None:
        return "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None and torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return str(resolved)


def _get_persistent_snapshot_entry(
    store: StoreBundle,
    *,
    sid: int,
    device: str | torch.device | None,
) -> _SnapshotEntry | None:
    return _persistent_snapshot_cache(store).get((int(sid), _snapshot_cache_device_key(device)))


def _put_persistent_snapshot_entry(
    store: StoreBundle,
    *,
    sid: int,
    device: str | torch.device | None,
    entry: _SnapshotEntry,
) -> None:
    _persistent_snapshot_cache(store)[(int(sid), _snapshot_cache_device_key(device))] = entry


def _slice_snapshot_entry_features(
    features: Mapping[str, Tensor],
    old_row: Mapping[str, Any],
    new_row: Mapping[str, Any],
) -> dict[str, Tensor]:
    src_parent = new_row.get("_src_parent_row")
    edge_parent = new_row.get("_edge_parent_row")
    old_src_count = int(old_row["src_nodes"].numel())
    old_edge_count = int(old_row["edge_ids"].numel())
    out: dict[str, Tensor] = {}
    for key, value in features.items():
        if not isinstance(value, Tensor) or value.dim() == 0:
            out[key] = value
            continue
        if key.startswith("edge") and isinstance(edge_parent, Tensor) and int(value.shape[0]) == old_edge_count:
            out[key] = value.index_select(0, edge_parent.long().to(device=value.device))
        elif isinstance(src_parent, Tensor) and int(value.shape[0]) == old_src_count:
            out[key] = value.index_select(0, src_parent.long().to(device=value.device))
        elif isinstance(edge_parent, Tensor) and int(value.shape[0]) == old_edge_count:
            out[key] = value.index_select(0, edge_parent.long().to(device=value.device))
        else:
            out[key] = value
    return out


def _evict_snapshot_entry_cache(entry_cache: dict[tuple[int, int], _SnapshotEntry], min_active: int) -> None:
    if not entry_cache:
        return
    for key in [key for key in entry_cache if int(key[0]) < min_active]:
        del entry_cache[key]


def _evict_snapshot_blob_cache(blob_cache: dict[int, _SnapshotGraphBlob], min_active: int) -> None:
    if not blob_cache:
        return
    for sid in [sid for sid in blob_cache if int(sid) < min_active]:
        del blob_cache[sid]
