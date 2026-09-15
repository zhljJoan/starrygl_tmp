from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any, Callable, Mapping, Sequence, TypeVar

import torch
from torch import Tensor

from starrygl.batch import Batch, EventRows
from starrygl.store import StoreBundle
from starrygl.task import NegativeSamplePool, TargetRoute, TaskTarget
from starrygl.view import GraphBlock

from starrygl.runtime.comm import CommScheduler
from starrygl.runtime.exchange import (
    finish_edge_feature_fetch,
    finish_node_feature_fetch,
    node_access_request_context,
)
from starrygl.runtime.state.access import (
    PendingHydrateRead,
    finish_hydrate_state,
)
from .blocks import move_graph_block
from .features import launch_batch_features


_T = TypeVar("_T")
ReadObserver = Callable[[str, Callable[[], Any]], Any]


def move_batch(
    batch: Batch,
    device: str | torch.device | None,
    *,
    non_blocking: bool = False,
    pin_memory: bool = False,
    graph_cache: dict[int, tuple[GraphBlock, GraphBlock]] | None = None,
) -> Batch:
    if device is None and graph_cache is None:
        return batch
    target = torch.device(device or "cpu")
    memo: dict[int, GraphBlock] = {}
    if graph_cache is not None:
        sources = {id(block): block for window in batch.iter_blocks() for block in window}
        if batch.graph is not None:
            sources[id(batch.graph)] = batch.graph
        for key, source in sources.items():
            cached = graph_cache.get(key)
            moved = cached[1] if cached is not None else move_graph_block(
                source, target, non_blocking=non_blocking, pin_memory=pin_memory,
            )
            # Keep the source topology cache independent of model layout additions.
            memo[key] = moved if cached is not None else replace(moved, cache=dict(moved.cache))
        # Strong source references prevent id reuse; retain only this window.
        graph_cache.clear()
        graph_cache.update({key: (source, memo[key]) for key, source in sources.items()})
        # Feature completion writes edata; each batch owns these dictionaries.
        memo = {key: replace(block, srcdata=dict(block.srcdata),
                             dstdata=dict(block.dstdata), edata=dict(block.edata))
                for key, block in memo.items()}
    batch.features = _move_value(batch.features, target, non_blocking, pin_memory)
    batch.state = _move_value(batch.state, target, non_blocking, pin_memory)
    batch.targets = _move_value(batch.targets, target, non_blocking, pin_memory)
    if batch.graph is not None:
        batch.graph = move_graph_block(
            batch.graph,
            target,
            memo,
            non_blocking=non_blocking,
            pin_memory=pin_memory,
        )
    if batch.blocks is not None:
        batch.blocks = tuple(
            tuple(
                move_graph_block(
                    block,
                    target,
                    memo,
                    non_blocking=non_blocking,
                    pin_memory=pin_memory,
                )
                for block in window
            )
            for window in batch.blocks
        )
    return batch


def record_batch_stream(batch: Batch, stream: torch.cuda.Stream) -> None:
    """Keep Batch CUDA storage alive on the stream that consumes it."""

    seen: set[int] = set()

    def record(value: Any) -> None:
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(value, Tensor):
            if value.device.type == "cuda":
                value.record_stream(stream)
        elif is_dataclass(value) and not isinstance(value, type):
            for item in fields(value):
                record(getattr(value, item.name))
        elif isinstance(value, Mapping):
            for item in value.values():
                record(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                record(item)

    record(batch)


def launch_batch(
    batch: Batch,
    *,
    store: StoreBundle,
    comm: CommScheduler,
    options: Mapping[str, Any],
    device: str | torch.device | None,
    prefetch_state: Callable[..., Sequence[PendingHydrateRead]] | None = None,
    feature_node_ids: Sequence[Tensor] | None = None,
    edge_ids: Sequence[Tensor] | None = None,
    non_blocking: bool = False,
    pin_memory: bool = False,
    graph_cache: dict[int, tuple[GraphBlock, GraphBlock]] | None = None,
):
    """Launch Stage-B reads without exposing their handles to Stage A."""

    batch, pending_nodes, pending_edges = launch_batch_features(
        batch,
        store=store,
        comm=comm,
        sampler_options=options,
        feature_node_ids=feature_node_ids,
        edge_ids=edge_ids,
    )
    batch = move_batch(
        batch,
        device,
        non_blocking=non_blocking,
        pin_memory=pin_memory,
        graph_cache=graph_cache,
    )
    pending_state = (
        prefetch_state(batch, node_access=_node_access_context(pending_nodes))
        if prefetch_state is not None
        else ()
    )
    return batch, pending_nodes, pending_edges, tuple(pending_state or ())


def finish_batch(
    batch: Batch,
    pending_nodes: Sequence[Any],
    pending_edges: Sequence[Any] | None,
    pending_state: Sequence[PendingHydrateRead],
    *,
    device: str | torch.device | None,
    non_blocking: bool = False,
    pin_memory: bool = False,
) -> Batch:
    """Resolve Stage-B reads and return the Batch-only handoff to Stage A."""

    batch = finish_pending_feature_fetches(
        batch,
        pending_nodes,
        pending_edges,
        device=device,
        non_blocking=non_blocking,
        pin_memory=pin_memory,
    )
    return finish_hydrate_state(batch, pending_state)


def state_prefetch_enabled(
    state_manager: object | Mapping[str, object] | None,
    *,
    pipeline_enabled: bool,
) -> bool:
    return bool(pipeline_enabled and _bounded_stale_state(state_manager))


def _bounded_stale_state(state_manager: object | Mapping[str, object] | None) -> bool:
    if state_manager is None:
        return False
    managers = state_manager.values() if isinstance(state_manager, Mapping) else (state_manager,)
    managers = tuple(managers)
    return bool(managers) and all(
        str(getattr(manager, "freshness_policy", "exact")) == "bounded_stale"
        and getattr(manager, "snapshot_history", None) is None
        for manager in managers
    )


def finish_pending_feature_fetches(
    batch: Any,
    pending_items: Sequence[Any] = (),
    pending_edge_items: Sequence[Any] | None = None,
    *,
    device: str | torch.device | None = None,
    non_blocking: bool = False,
    pin_memory: bool = False,
    observe: ReadObserver | None = None,
) -> Any:
    target = torch.device(device) if device is not None else None
    features = dict(batch.features)
    if pending_items:
        for item in pending_items:
            if not isinstance(item, Mapping):
                continue
            pending = item.get("pending")
            if pending is None:
                continue
            fetched = _observe(
                observe,
                "finish_node_feature_fetch_total",
                lambda: finish_node_feature_fetch(pending),
            )
            window = item.get("window")
            for key, value in fetched.items():
                if target is not None:
                    value = _observe(
                        observe,
                        "finish_node_feature_to_device",
                        lambda value=value: _move_value(
                            value,
                            target,
                            non_blocking,
                            pin_memory,
                        ),
                    )
                if window is None:
                    features[key] = value
                    continue
                width = len(batch.blocks) if batch.blocks is not None else int(window) + 1
                current = features.get(key)
                slots = list(current) if isinstance(current, tuple) else [None for _ in range(int(width))]
                while len(slots) <= int(window):
                    slots.append(None)
                slots[int(window)] = value
                features[key] = tuple(slots)
    if pending_edge_items:
        _finish_pending_edge_feature_fetches(
            batch,
            pending_edge_items,
            features=features,
            target=target,
            non_blocking=non_blocking,
            pin_memory=pin_memory,
            observe=observe,
        )
    batch.features = features
    return batch


def _node_access_context(pending_items: Sequence[Any]):
    for item in pending_items:
        pending = item.get("pending") if isinstance(item, Mapping) else item
        context = node_access_request_context(pending)
        if context is not None:
            return context
    return None


def _finish_pending_edge_feature_fetches(
    batch: Any,
    pending_items: Sequence[Any],
    *,
    features: dict[str, Any] | None = None,
    target: torch.device | None,
    non_blocking: bool,
    pin_memory: bool,
    observe: ReadObserver | None = None,
) -> None:
    if batch.blocks is None:
        return
    for item in pending_items:
        if not isinstance(item, Mapping):
            continue
        pending = item.get("pending")
        unique_edge_ids = item.get("unique_edge_ids")
        block_items = item.get("blocks")
        if pending is None or not isinstance(unique_edge_ids, Tensor):
            continue
        fetched = _observe(
            observe,
            "finish_edge_feature_fetch_total",
            lambda: finish_edge_feature_fetch(pending),
        )
        if target is not None:
            fetched = _observe(
                observe,
                "finish_edge_feature_to_device",
                lambda: {
                    key: _move_value(value, target, non_blocking, pin_memory)
                    for key, value in fetched.items()
                },
            )
            unique_edge_ids = unique_edge_ids.to(device=target)
        if not block_items:
            continue
        for block_item in block_items:
            if not isinstance(block_item, Mapping):
                continue
            window = int(block_item.get("window", 0))
            layer = int(block_item.get("layer", 0))
            if window < 0 or layer < 0 or window >= len(batch.blocks) or layer >= len(batch.blocks[window]):
                continue
            block = batch.blocks[window][layer]
            item_edge_ids = block_item.get("edge_ids")
            edge_ids = item_edge_ids.long() if isinstance(item_edge_ids, Tensor) else block.edge_ids.long()
            if target is not None:
                edge_ids = edge_ids.to(device=target)
            rows = _observe(
                observe,
                "finish_edge_feature_searchsorted",
                lambda: torch.searchsorted(unique_edge_ids.to(device=edge_ids.device), edge_ids),
            )
            for name, value in fetched.items():
                key = "edge_feat" if str(name) == "edge" else str(name)
                block.edata[key] = _observe(
                    observe,
                    "finish_edge_feature_index_select",
                    lambda value=value: value.index_select(0, rows.to(device=value.device)),
                )
        feature_windows = item.get("feature_windows")
        if features is None or not feature_windows:
            continue
        for feature_item in feature_windows:
            if not isinstance(feature_item, Mapping):
                continue
            window = int(feature_item.get("window", 0))
            edge_ids = feature_item.get("edge_ids")
            if not isinstance(edge_ids, Tensor):
                continue
            if target is not None:
                edge_ids = edge_ids.to(device=target)
            rows = torch.searchsorted(unique_edge_ids.to(device=edge_ids.device), edge_ids.long())
            width = len(batch.blocks) if batch.blocks is not None else int(window) + 1
            for name, value in fetched.items():
                selected = value.index_select(0, rows.to(device=value.device))
                current = features.get(str(name))
                slots = list(current) if isinstance(current, tuple) else [None for _ in range(int(width))]
                while len(slots) <= int(window):
                    slots.append(None)
                slots[int(window)] = selected
                features[str(name)] = tuple(slots)


def _observe(
    observer: ReadObserver | None,
    name: str,
    operation: Callable[[], _T],
) -> _T:
    if observer is None:
        return operation()
    return observer(name, operation)


def _move_value(
    value: Any,
    device: torch.device,
    non_blocking: bool = False,
    pin_memory: bool = False,
):
    if isinstance(value, Tensor):
        source = (
            value.pin_memory()
            if pin_memory and value.device.type == "cpu" and not value.is_pinned()
            else value
        )
        return source.to(device, non_blocking=non_blocking)
    if isinstance(value, (EventRows, TaskTarget, NegativeSamplePool, TargetRoute)):
        return replace(
            value,
            **{
                name: _move_value(
                    getattr(value, name),
                    device,
                    non_blocking,
                    pin_memory,
                )
                for name in value.__dataclass_fields__
            },
        )
    if isinstance(value, Mapping):
        return {
            key: _move_value(item, device, non_blocking, pin_memory)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_move_value(item, device, non_blocking, pin_memory) for item in value)
    if isinstance(value, list):
        return [_move_value(item, device, non_blocking, pin_memory) for item in value]
    return value


__all__ = [
    "finish_batch",
    "finish_pending_feature_fetches",
    "launch_batch",
    "ReadObserver",
    "record_batch_stream",
    "state_prefetch_enabled",
]
