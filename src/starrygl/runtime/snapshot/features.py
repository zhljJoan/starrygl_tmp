from __future__ import annotations

from typing import Any, Mapping, Sequence

from torch import Tensor

from starrygl.store import StoreBundle
from starrygl.view import GraphBlock

from starrygl.runtime.comm import CommScheduler
from starrygl.runtime.exchange import (
    PendingNodeFeatureFetch,
    launch_node_feature_fetch,
    materialize_node_features,
)


def _attach_node_dist_index(block: GraphBlock, store: StoreBundle) -> None:
    index = store.graph.partition.get("node_dist_index") if store.graph.prepare is not None else None
    if isinstance(index, Tensor):
        block.cache["node_dist_index"] = index.long()


def _read_snapshot_features(
    store: StoreBundle,
    row: Mapping[str, Any],
    *,
    comm: CommScheduler | None = None,
    defer_finish: bool = False,
) -> tuple[dict[str, Tensor], bool, PendingNodeFeatureFetch | None]:
    src_nodes = row.get("src_nodes")
    count = int(src_nodes.numel()) if isinstance(src_nodes, Tensor) else 0
    return _read_features(
        store,
        src_nodes=src_nodes,
        edge_ids=row.get("edge_feature_ids", row.get("edge_ids")),
        snapshot_id=int(row.get("snapshot_id", 0)),
        embedded=_embedded_snapshot_node_features(row, count),
        src_feature_row=row.get("src_feature_row"),
        comm=comm,
        defer_finish=defer_finish,
    )


def _embedded_snapshot_node_features(values: Mapping[str, Any], count: int) -> Tensor | None:
    node_data = values.get("node_data")
    x = node_data.get("x") if isinstance(node_data, Mapping) else None
    if not isinstance(x, Tensor):
        return None
    rows = values.get("src_data_row")
    if isinstance(rows, Tensor) and int(rows.numel()) == int(count):
        return x.index_select(0, rows.long().to(device=x.device))
    if int(x.shape[0]) != int(count):
        raise ValueError("embedded Snapshot-CSC node features do not match src_nodes")
    return x


def _empty_node_features(store: StoreBundle, *, snapshot_id: int) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, value in store.features.node_features.items():
        tensor = value
        if tensor.dim() >= 3:
            sid = max(0, min(int(snapshot_id), int(tensor.shape[0]) - 1))
            tensor = tensor[int(sid)]
        out[str(name)] = tensor.new_empty((0, *tensor.shape[1:]))
    return out


def _read_snapshot_block_features(
    store: StoreBundle,
    block: GraphBlock,
    *,
    comm: CommScheduler | None = None,
    defer_finish: bool = False,
    read_edge_features: bool = True,
) -> tuple[dict[str, Tensor], bool, PendingNodeFeatureFetch | None]:
    snapshot_id = int(block.cache.get("snapshot_id", 0))
    return _read_features(
        store,
        src_nodes=block.src_nodes,
        edge_ids=block.cache.get("edge_feature_ids", block.edge_ids),
        snapshot_id=snapshot_id,
        embedded=_embedded_snapshot_node_features(block.cache, int(block.src_nodes.numel())),
        src_feature_row=block.cache.get("src_feature_row"),
        comm=comm,
        defer_finish=defer_finish,
        read_edge_features=read_edge_features,
    )


def _read_features(
    store: StoreBundle,
    *,
    src_nodes: Tensor | None,
    edge_ids: Tensor | None,
    snapshot_id: int,
    embedded: Tensor | None,
    src_feature_row: Any = None,
    comm: CommScheduler | None,
    defer_finish: bool,
    read_edge_features: bool = True,
) -> tuple[dict[str, Tensor], bool, PendingNodeFeatureFetch | None]:
    features: dict[str, Tensor] = {}
    requires_node_exchange = False
    pending: PendingNodeFeatureFetch | None = None
    if embedded is not None:
        features["x"] = embedded
    elif isinstance(src_nodes, Tensor) and store.features.node_features:
        if _has_temporal_node_features(store):
            node_features = _empty_node_features(store, snapshot_id=snapshot_id)
            if int(src_nodes.numel()):
                node_features = store.features.read_nodes_at(src_nodes.long(), snapshot_id)
            features.update(node_features)
        elif bool(defer_finish):
            pending = launch_node_feature_fetch(store, src_nodes.long(), comm=comm, defer_local_read=True)
            requires_node_exchange = bool(pending.remote)
            if not pending.remote and not pending.response_handles and pending.local_rows is None:
                from starrygl.runtime.exchange import finish_node_feature_fetch

                features.update(finish_node_feature_fetch(pending))
                pending = None
        else:
            node_features, requires_node_exchange = materialize_node_features(
                store,
                src_nodes.long(),
                comm=comm,
                assume_unique=True,
            )
            features.update(node_features)
    if (
        bool(read_edge_features)
        and isinstance(edge_ids, Tensor)
        and int(edge_ids.numel()) > 0
        and store.features.edge_features
    ):
        features.update(store.features.read_edges(edge_ids.long()))
    return features, requires_node_exchange, pending


def _window_feature_mapping(features_by_window: Sequence[Mapping[str, Tensor]]) -> dict[str, tuple[Tensor | None, ...]]:
    keys = sorted({key for features in features_by_window for key in features.keys()})
    return {
        key: tuple(features.get(key) for features in features_by_window)
        for key in keys
    }


def _has_temporal_node_features(store: StoreBundle) -> bool:
    return any(value.dim() >= 3 for value in store.features.node_features.values())
