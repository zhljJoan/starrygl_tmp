from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Mapping, Sequence

import torch
from torch import Tensor

from starrygl.runtime.comm import CommScheduler, distributed
from starrygl.store import FeatureManager, StoreBundle
from starrygl.store.remote_fetch import (
    OwnerRequest,
    finish_owner_responses,
    submit_owner_request,
    submit_owner_responses,
)


@dataclass
class PendingNodeFeatureFetch:
    keys: tuple[str, ...]
    out: dict[str, Tensor]
    kind: str = "node"
    remote: bool = False
    compact: PendingNodeFeatureFetch | None = None
    inverse: Tensor | None = None
    missing_pos: Tensor | None = None
    order: Tensor | None = None
    response_handles: dict[str, object] = field(default_factory=dict)
    scheduler: CommScheduler | None = None
    feature_manager: FeatureManager | None = None
    local_rows: Tensor | None = None
    local_pos: Tensor | None = None
    local_read_kind: str | None = None
    node_access: OwnerRequest | None = None


def materialize_node_features(store: StoreBundle, node_ids: Tensor, **kwargs) -> tuple[dict[str, Tensor], bool]:
    pending = launch_node_feature_fetch(store, node_ids, **kwargs)
    return finish_node_feature_fetch(pending), pending.remote


def materialize_local_edge_features(
    store: StoreBundle,
    edge_ids: Tensor,
    *,
    names: Sequence[str] | None = None,
    **_: object,
) -> dict[str, Tensor]:
    keys = _keys(store.features, "edge", names)
    if not keys:
        return {}
    rows = _mapped_rows(
        edge_ids,
        store.features.edge_row_map,
        _row_count(store.features, "edge", keys),
        "local edge",
    )
    return _read_rows(store.features, "edge", rows, keys)


def launch_node_feature_fetch(
    store: StoreBundle,
    node_ids: Tensor,
    *,
    names: Sequence[str] | None = None,
    comm: CommScheduler | None = None,
    assume_unique: bool = False,
    defer_local_read: bool = False,
    **_: object,
) -> PendingNodeFeatureFetch:
    return _launch(store, "node", node_ids, names, comm, assume_unique, defer_local_read)


def launch_edge_feature_fetch(
    store: StoreBundle,
    edge_ids: Tensor,
    *,
    names: Sequence[str] | None = None,
    comm: CommScheduler | None = None,
    assume_unique: bool = False,
    defer_local_read: bool = False,
    **_: object,
) -> PendingNodeFeatureFetch:
    return _launch(store, "edge", edge_ids, names, comm, assume_unique, defer_local_read)


def finish_node_feature_fetch(pending: PendingNodeFeatureFetch) -> dict[str, Tensor]:
    if pending.compact is not None:
        compact = finish_node_feature_fetch(pending.compact)
        return compact if pending.inverse is None else {
            key: value.index_select(0, pending.inverse.to(value.device)) for key, value in compact.items()
        }
    fetched = {}
    if pending.scheduler is not None and pending.order is not None:
        fetched = finish_owner_responses(pending.scheduler, pending.order, pending.response_handles)
        if pending.missing_pos is not None:
            _scatter(pending.out, fetched, pending.missing_pos)
    if pending.local_rows is not None:
        local = _read_pending_local(pending)
        if pending.local_pos is None:
            pending.out.update(local)
        else:
            _scatter(pending.out, local, pending.local_pos)
    pending.response_handles.clear()
    return pending.out


def finish_edge_feature_fetch(pending: PendingNodeFeatureFetch) -> dict[str, Tensor]:
    return finish_node_feature_fetch(pending)


def node_access_request_context(pending: PendingNodeFeatureFetch | None) -> OwnerRequest | None:
    if pending is None:
        return None
    return pending.node_access or node_access_request_context(pending.compact)


def _launch(store, kind, ids, names, comm, assume_unique, defer_local_read):
    manager = store.features
    keys = _keys(manager, kind, names)
    if not keys:
        return PendingNodeFeatureFetch(keys=keys, out={}, kind=kind)
    ids = ids.long()
    if not assume_unique and int(ids.numel()) > 1:
        unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
        if int(unique.numel()) < int(ids.numel()):
            compact = _launch(store, kind, unique, keys, comm, True, defer_local_read)
            return _restore_duplicates(compact, inverse)
    replicated = manager.node_features_replicated if kind == "node" else manager.edge_features_replicated
    uses_remote = distributed() and not replicated
    if not int(ids.numel()):
        return _launch_remote(store, kind, ids, keys, comm, None, ids, False) if uses_remote else PendingNodeFeatureFetch(
            keys=keys, out=_empty(manager, kind, keys, 0), kind=kind
        )
    row_map = _row_map(manager, kind)
    if replicated and _identity(manager, kind):
        if not defer_local_read:
            return PendingNodeFeatureFetch(keys=keys, out=_read_ids(manager, kind, ids, keys), kind=kind)
        pending = PendingNodeFeatureFetch(keys=keys, out={}, kind=kind)
        return _defer_local(pending, manager, ids, None, f"{kind}_ids")
    if row_map is None:
        if not defer_local_read:
            return PendingNodeFeatureFetch(keys=keys, out=_read_ids(manager, kind, ids, keys), kind=kind)
        pending = PendingNodeFeatureFetch(keys=keys, out=_empty(manager, kind, keys, int(ids.numel())), kind=kind)
        return _defer_local(pending, manager, ids, None, f"{kind}_ids")
    rows = ids if _identity(manager, kind) else row_map.index_select(0, ids.to(row_map.device))
    present = rows >= 0
    out = _empty(manager, kind, keys, int(ids.numel()))
    local_pos = present.nonzero(as_tuple=True)[0]
    local_rows = rows.index_select(0, local_pos.to(rows.device)).long()
    if int(local_pos.numel()) and not defer_local_read:
        _scatter(out, _read_rows(manager, kind, local_rows, keys), local_pos)
        local_pos = local_rows = None
    missing_pos = (~present).nonzero(as_tuple=True)[0]
    missing_ids = ids.index_select(0, missing_pos.to(ids.device))
    if int(missing_ids.numel()) and not distributed():
        return PendingNodeFeatureFetch(keys=keys, out={}, kind=kind, remote=True)
    pending = (
        _launch_remote(store, kind, missing_ids, keys, comm, out, missing_pos, bool(int(missing_ids.numel())))
        if uses_remote
        else PendingNodeFeatureFetch(keys=keys, out=out, kind=kind)
    )
    if local_rows is not None:
        _defer_local(pending, manager, local_rows, local_pos, f"{kind}_rows")
    return pending


def _launch_remote(store, kind, ids, keys, comm, out, missing_pos, remote, order=None, send_counts=None):
    scheduler = comm or CommScheduler()
    request = submit_owner_request(
        ids,
        _partition_index(store, f"{kind}_dist_index", ids.device),
        scheduler=scheduler,
        name=f"{kind}_feature_request",
        order=order,
        send_counts=send_counts,
    )
    response = _owner_read(store.features, kind, request.recv_nodes, keys)
    handles = submit_owner_responses(request, response, name=f"{kind}_feature_response")
    if missing_pos is None:
        missing_pos = torch.arange(int(ids.numel()), dtype=torch.long, device=ids.device)
    return PendingNodeFeatureFetch(
        keys=tuple(keys),
        out=_empty(store.features, kind, keys, int(ids.numel())) if out is None else out,
        kind=kind,
        remote=remote,
        missing_pos=missing_pos.long(),
        order=request.order,
        response_handles=handles,
        scheduler=request.scheduler,
        feature_manager=store.features,
        node_access=request if kind == "node" else None,
    )


def _restore_duplicates(pending, inverse):
    return pending if inverse is None else PendingNodeFeatureFetch(
        keys=pending.keys, out={}, kind=pending.kind, remote=pending.remote, compact=pending, inverse=inverse.long()
    )


def _defer_local(pending, manager, rows, pos, read_kind):
    pending.feature_manager, pending.local_rows, pending.local_pos = manager, rows, pos
    pending.local_read_kind = read_kind
    return pending


def _read_pending_local(pending):
    kind = "edge" if str(pending.local_read_kind).startswith("edge") else "node"
    return (
        _read_ids(pending.feature_manager, kind, pending.local_rows, pending.keys)
        if str(pending.local_read_kind).endswith("ids")
        else _read_rows(pending.feature_manager, kind, pending.local_rows, pending.keys)
    )


def _keys(manager, kind, names):
    values = manager.node_features if kind == "node" else manager.edge_features
    return tuple(values) if names is None else tuple(names)


def _row_map(manager, kind):
    return manager.node_row_map if kind == "node" else manager.edge_row_map


def _identity(manager, kind):
    return bool(getattr(manager, f"{kind}_row_map_is_identity", False))


def _features(manager, kind):
    return manager.node_features if kind == "node" else manager.edge_features


def _read_ids(manager, kind, ids, keys):
    values = manager.read_nodes(ids, keys) if kind == "node" else manager.read_edges(ids, keys)
    return _normalize(values) if kind == "edge" else values


def _read_rows(manager, kind, rows, keys):
    values = manager.read_node_rows(rows, keys) if kind == "node" else manager.read_edge_rows(rows, keys)
    return _normalize(values) if kind == "edge" else values


def _owner_read(manager, kind, ids, keys):
    rows = _mapped_rows(ids, _row_map(manager, kind), _row_count(manager, kind, keys), kind)
    return _read_rows(manager, kind, rows, keys)


def _mapped_rows(ids, row_map, row_count, label):
    query = ids.long() if row_map is None else ids.to(row_map.device).long()
    if row_map is not None and (bool((query < 0).any()) or bool((query >= int(row_map.numel())).any())):
        raise KeyError(f"owner missing requested {label} features outside row map")
    rows = query if row_map is None else row_map.index_select(0, query)
    if bool((rows < 0).any()) or bool((rows >= int(row_count)).any()):
        raise KeyError(f"owner missing requested {label} features")
    return rows


def _row_count(manager, kind, keys):
    values = _features(manager, kind)
    return int(values[keys[0]].shape[0]) if keys else 0


def _empty(manager, kind, keys, count):
    return {
        key: value.new_empty((int(count), *value.shape[1:]), dtype=torch.uint8 if kind == "edge" and value.dtype == torch.bool else value.dtype)
        for key, value in _features(manager, kind).items() if key in keys
    }


def _normalize(values: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.to(torch.uint8) if value.dtype == torch.bool else value for key, value in values.items()}


def _scatter(out, values, rows):
    for key, value in values.items():
        out[key].index_copy_(0, rows.to(out[key].device), value.to(out[key]))


def _partition_index(store: StoreBundle, name: str, device: torch.device | str) -> Tensor:
    index = store.graph.partition[name].long()
    target = torch.device(device)
    limit = int(os.environ.get("STARRYGL_GPU_PARTITION_INDEX_MAX_ELEMENTS", "8000000") or 0)
    if target.type != "cuda" or int(index.numel()) > limit:
        return index
    key = ("partition_index", name, str(target))
    cached = store.graph.runtime_cache.get(key)
    if not isinstance(cached, Tensor):
        cached = index.to(target, non_blocking=True)
        store.graph.runtime_cache[key] = cached
    return cached


__all__ = "PendingNodeFeatureFetch finish_edge_feature_fetch finish_node_feature_fetch launch_edge_feature_fetch launch_node_feature_fetch materialize_local_edge_features materialize_node_features node_access_request_context".split()
