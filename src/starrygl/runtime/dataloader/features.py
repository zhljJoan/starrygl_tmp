from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from starrygl.batch import Batch, EventRows
from starrygl.store import StoreBundle
from starrygl.view import GraphBlock

from starrygl.runtime.comm import CommScheduler
from starrygl.runtime.exchange import (
    launch_edge_feature_fetch,
    materialize_local_edge_features,
)
from .materialize import (
    attach_comm_to_blocks,
    available_feature_names,
)

def launch_batch_features(
    batch: Batch,
    store: StoreBundle,
    *,
    comm: CommScheduler | None,
    sampler_options: Mapping[str, Any] | None = None,
    feature_node_ids: Sequence[Tensor] | None = None,
    edge_ids: Sequence[Tensor] | None = None,
):
    """Launch feature dependencies for either model-facing Batch mode."""

    options = sampler_options or {}
    for blocks in batch.iter_blocks():
        attach_comm_to_blocks(blocks, comm)

    if batch.mode == "event":
        nodes = feature_node_ids[0] if feature_node_ids else None
        events = batch.targets.get("events") if isinstance(batch.targets, Mapping) else None
        features_by_window, pending_nodes = _launch_event_nodes(
            batch,
            store,
            node_ids=nodes,
            target_edge_ids=events.edge_ids if isinstance(events, EventRows) else None,
            comm=comm,
            options=options,
        )
    else:
        features_by_window, pending_nodes = _launch_snapshot_nodes(
            batch,
            store,
            comm=comm,
            options=options,
        )

    pending_edges = None
    unique_edge_ids = edge_ids[0] if batch.mode == "event" and edge_ids else None
    pending_edges = launch_block_edge_features(
        store,
        tuple(batch.iter_blocks()),
        comm=comm,
        names=options.get("_snapshot_edge_feature_names") if batch.mode == "snapshot" else None,
        unique_edge_ids=unique_edge_ids,
        include_feature_windows=batch.mode == "snapshot",
        defer_local_read=bool(options.get("defer_node_feature_finish", False)),
    )

    from starrygl.runtime.snapshot.features import _window_feature_mapping

    batch.features = _window_feature_mapping(features_by_window)
    return batch, tuple(pending_nodes), pending_edges


def launch_block_edge_features(
    store: StoreBundle,
    blocks: Sequence[Sequence[GraphBlock]],
    *,
    comm: CommScheduler | None,
    names: Sequence[str] | None = None,
    unique_edge_ids: Tensor | None = None,
    include_feature_windows: bool = False,
    defer_local_read: bool = False,
) -> tuple[Mapping[str, Any], ...] | None:
    available = available_feature_names(store.features.edge_features)
    names = available if names is None else tuple(name for name in available if name in names)
    if not names:
        return None
    block_items: list[dict[str, Any]] = []
    pieces: list[Tensor] = []
    for window_id, window in enumerate(blocks):
        for layer_id, block in enumerate(window):
            edge_ids = block.cache.get("edge_feature_ids", block.edge_ids)
            if isinstance(edge_ids, Tensor) and int(edge_ids.numel()):
                edge_ids = edge_ids.long()
                block_items.append({"window": window_id, "layer": layer_id, "edge_ids": edge_ids})
                pieces.append(edge_ids)
    if unique_edge_ids is None:
        unique_edge_ids = (
            torch.unique(torch.cat(pieces, dim=0), sorted=True)
            if pieces
            else next(iter(store.features.edge_features.values())).new_empty(0, dtype=torch.long)
        )
    pending = launch_edge_feature_fetch(
        store,
        unique_edge_ids.long(),
        names=names,
        comm=comm,
        assume_unique=True,
        defer_local_read=bool(defer_local_read),
    )
    item: dict[str, Any] = {
        "pending": pending,
        "unique_edge_ids": unique_edge_ids.long(),
        "blocks": tuple(block_items),
    }
    if include_feature_windows:
        item["feature_windows"] = tuple(value for value in block_items if int(value["layer"]) == 0)
    return (item,)


def _launch_event_nodes(batch, store, *, node_ids, target_edge_ids, comm, options):
    from starrygl.runtime.event.features import _read_event_features, _set_positive_edge_features

    if not isinstance(node_ids, Tensor):
        node_ids = _first_feature_block(batch).src_nodes
    features, _, pending = _read_event_features(
        store,
        node_ids.long(),
        comm=comm,
        defer_finish=bool(options.get("defer_node_feature_finish", False)),
        read_edge_features=False,
        assume_unique_node_features=False,
    )
    features.setdefault("x", torch.empty((int(node_ids.numel()), 0), dtype=torch.float32, device=node_ids.device))
    features["node_ids"] = node_ids
    edge_names = available_feature_names(store.features.edge_features)
    if isinstance(target_edge_ids, Tensor) and int(target_edge_ids.numel()) and edge_names:
        _set_positive_edge_features(
            features,
            materialize_local_edge_features(store, target_edge_ids.long(), names=edge_names),
        )
    pending_items = [] if pending is None else [{"window": 0, "pending": pending}]
    return [features], pending_items


def _launch_snapshot_nodes(batch, store, *, comm, options):
    from starrygl.runtime.snapshot.features import _read_snapshot_block_features

    features_by_window = []
    pending_items = []
    for window_id, blocks in enumerate(batch.iter_blocks()):
        features, _, pending = _read_snapshot_block_features(
            store,
            blocks[0],
            comm=comm,
            defer_finish=bool(options.get("defer_node_feature_finish", False)),
            read_edge_features=False,
        )
        features_by_window.append(features)
        if pending is not None:
            pending_items.append({"window": window_id, "pending": pending})
    return features_by_window, pending_items


def _first_feature_block(batch: Batch) -> GraphBlock:
    return next(iter(batch.iter_blocks()))[0]


__all__ = ["launch_batch_features", "launch_block_edge_features"]
