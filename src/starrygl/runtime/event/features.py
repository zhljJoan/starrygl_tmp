from __future__ import annotations

import time
from typing import Mapping

import torch
from torch import Tensor

from starrygl.store import StoreBundle

from starrygl.runtime.exchange import (
    PendingNodeFeatureFetch,
    launch_node_feature_fetch,
    materialize_node_features,
)
from starrygl.runtime.comm import CommScheduler
from starrygl.runtime.dataloader.materialize import available_feature_names


def _set_positive_edge_features(features: dict[str, Tensor], values: Mapping[str, Tensor]) -> None:
    if "edge" in values:
        edge = values["edge"]
    elif values:
        edge = next(iter(values.values()))
    else:
        return
    features["pos_edge_feat"] = edge
    features.setdefault("edge", edge)


def _read_event_features(
    store: StoreBundle,
    node_ids: Tensor,
    *,
    edge_ids: Tensor | None = None,
    comm: CommScheduler | None = None,
    defer_finish: bool = False,
    read_edge_features: bool = True,
    profile: dict[str, float] | None = None,
    profile_device: torch.device | None = None,
    assume_unique_node_features: bool = True,
) -> tuple[dict[str, Tensor], bool, PendingNodeFeatureFetch | None]:
    features: dict[str, Tensor] = {}
    requires_node_exchange = False
    pending: PendingNodeFeatureFetch | None = None
    enabled = profile is not None
    node_feature_names = available_feature_names(store.features.node_features)
    if node_feature_names:
        if bool(defer_finish):
            start = _profile_start(enabled, profile_device)
            pending = launch_node_feature_fetch(
                store,
                node_ids,
                names=node_feature_names,
                comm=comm,
                assume_unique=bool(assume_unique_node_features),
                defer_local_read=True,
                profile=profile,
                profile_prefix="launch_feature_node_fetch",
                profile_device=profile_device,
            )
            _profile_add(profile, "launch_feature_node_fetch_launch", start, enabled, profile_device)
            requires_node_exchange = bool(pending.remote)
            if (
                pending.compact is None
                and not pending.remote
                and not pending.response_handles
                and pending.local_rows is None
            ):
                from starrygl.runtime.exchange import finish_node_feature_fetch

                start = _profile_start(enabled, profile_device)
                features.update(finish_node_feature_fetch(pending))
                _profile_add(profile, "launch_feature_node_fetch_finish_local", start, enabled, profile_device)
                pending = None
        else:
            start = _profile_start(enabled, profile_device)
            node_features, requires_node_exchange = materialize_node_features(
                store,
                node_ids,
                names=node_feature_names,
                comm=comm,
                assume_unique=bool(assume_unique_node_features),
                profile=profile,
                profile_prefix="launch_feature_node_fetch",
                profile_device=profile_device,
            )
            _profile_add(profile, "launch_feature_node_materialize", start, enabled, profile_device)
            features.update(node_features)
    edge_feature_names = available_feature_names(store.features.edge_features)
    if (
        bool(read_edge_features)
        and isinstance(edge_ids, Tensor)
        and int(edge_ids.numel()) > 0
        and edge_feature_names
    ):
        start = _profile_start(enabled, profile_device)
        features.update(store.features.read_edges(edge_ids, names=edge_feature_names))
        _profile_add(profile, "launch_feature_inline_edge_read", start, enabled, profile_device)
    return features, requires_node_exchange, pending


def _profile_start(enabled: bool, device: torch.device | None = None) -> float:
    if enabled:
        _profile_sync(device)
    return time.perf_counter() if enabled else 0.0


def _profile_add(
    stats: dict[str, float] | None,
    key: str,
    start: float,
    enabled: bool,
    device: torch.device | None = None,
) -> None:
    if enabled and stats is not None:
        _profile_sync(device)
        stats[key] = stats.get(key, 0.0) + (time.perf_counter() - float(start))


def _profile_sync(device: torch.device | None) -> None:
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
