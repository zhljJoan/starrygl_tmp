from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch

from starrygl.batch import SamplingPolicy
from starrygl.runtime.dataloader.materialize import AccessedWindow
from starrygl.runtime.dataloader.pipeline import _move_value
from starrygl.runtime.sample import access_native_graphs, native_target_block
from starrygl.store import StoreBundle
from starrygl.task import (
    SamplingRoot,
    TargetRoute,
    attach_target_route,
    build_window_task_target,
    sampling_roots_from_target,
)
from starrygl.task.negative import snapshot_negative_dst_pool
from starrygl.task.target import lookup_target_rows

from .cache import (
    _SnapshotEntry,
    _SnapshotGraphBlob,
    _evict_snapshot_blob_cache,
    _evict_snapshot_entry_cache,
    _materialize_snapshot_entries,
)
from .features import _attach_node_dist_index
from .rows import _apply_chunk_limit_to_snapshot_row, _chunk_order_tensor


def access_snapshot_window(
    store: StoreBundle,
    slices: Any,
    *,
    split: str,
    window_id: int,
    input_window: range | Sequence[int],
    chunk_limits: Sequence[int],
    sampling_policy: SamplingPolicy,
    native_sampler: Any | None = None,
    sampler_options: Mapping[str, Any] | None = None,
    num_negatives: int = 0,
    generator: object | None = None,
    entry_cache: dict[tuple[int, int], _SnapshotEntry] | None = None,
    blob_cache: dict[int, _SnapshotGraphBlob] | None = None,
) -> AccessedWindow:
    """Resolve native-sampled or prepared-CSC topology to one short tuple."""

    options = sampler_options or {}
    window_mean = options.get("_snapshot_train_loss_mode") == "window_mean"
    if window_mean and (store.labels.task_kind != "node" or sampling_policy == "neighbor"):
        raise ValueError("window_mean requires full/chunk snapshot node prediction")
    chunk_order = _chunk_order_tensor(options)
    rows = []
    for snapshot_id, chunk_limit in zip(input_window, chunk_limits):
        if 0 <= int(snapshot_id) < len(slices):
            key = int(snapshot_id), int(chunk_limit)
            cached = entry_cache.get(key) if entry_cache is not None else None
            rows.append((cached.row if cached is not None else slices[key[0]], key[1]))
    rows = tuple(rows)
    if not rows:
        raise ValueError(f"snapshot window {int(window_id)} has no prepared graph row")
    target_row = _apply_chunk_limit_to_snapshot_row(
        rows[-1][0],
        rows[-1][1],
        chunk_order=chunk_order,
    )
    negative_pool = None
    if store.labels.task_kind == "edge":
        negative_pool = snapshot_negative_dst_pool(
            local_dst_ids=target_row["dst_nodes"].long(),
            global_dst_ids=_snapshot_global_dst_pool(store),
            split=split,
            options=options,
        )
    target = build_window_task_target(
        store.labels,
        int(window_id),
        target_ts=_snapshot_target_ts(target_row, 1),
        negative_pool=negative_pool,
        num_negatives=int(num_negatives),
        generator=generator if isinstance(generator, torch.Generator) else None,
    )

    if sampling_policy == "neighbor":
        if native_sampler is None:
            raise ValueError("neighbor sampling requires an initialized native sampler")
        root = sampling_roots_from_target(target)
        roots = tuple(
            SamplingRoot(
                node_ids=root.node_ids,
                ts=(torch.full_like(root.node_ids, int(row["snapshot_id"]))
                    if options.get("policy") == "snapshot_uniform"
                    else _snapshot_target_ts(row, int(root.node_ids.numel()))),
                groups=root.groups,
            )
            for row, _ in rows
        )
        blocks, feature_node_ids, edge_ids = access_native_graphs(native_sampler, roots)
        for (row, _), window in zip(rows, blocks):
            for block in window:
                block.cache.setdefault("snapshot_id", int(row.get("snapshot_id", 0)))
                _attach_node_dist_index(block, store)
    else:
        options = dict(options)
        options["defer_feature_launch"] = True
        if options.get("snapshot_materialize_on_device") is not True:
            options.pop("_materialize_device", None)
        entries = _materialize_snapshot_entries(
            store,
            rows,
            chunk_order=chunk_order,
            comm=None,
            options=options,
            entry_cache=entry_cache,
            blob_cache=blob_cache,
        )
        blocks = tuple((entry.graph,) for entry in entries)
        feature_node_ids = tuple(entry.graph.src_nodes.long() for entry in entries)
        # Snapshot feature launch reads physical edge IDs directly from blocks.
        edge_ids = ()
        minimum = int(rows[0][0].get("snapshot_id", window_id))
        if entry_cache is not None:
            _evict_snapshot_entry_cache(entry_cache, minimum)
        if blob_cache is not None:
            _evict_snapshot_blob_cache(blob_cache, minimum)

    graph = native_target_block(blocks[-1])
    if target.target_ids.device != graph.dst_nodes.device:
        target = _move_value(target, graph.dst_nodes.device)
    targets = {"task": target}
    if window_mean:
        window_tasks = []
        for slot, ((row, _), window) in enumerate(zip(rows, blocks)):
            graph = native_target_block(window)
            item = target if slot == len(rows) - 1 else build_window_task_target(
                store.labels, int(row["snapshot_id"]), target_ts=_snapshot_target_ts(row, 1))
            if item.target_ids.device != graph.dst_nodes.device:
                item = _move_value(item, graph.dst_nodes.device)
            target_rows = lookup_target_rows(graph.dst_nodes, item.target_ids)
            keep = (target_rows >= 0).nonzero(as_tuple=True)[0]
            item = replace(item, target_ids=item.target_ids[keep], node_ids=item.node_ids[keep],
                label=None if item.label is None else item.label[keep],
                target_ts=None if item.target_ts is None else item.target_ts[keep],
                target_route=TargetRoute(target_rows=target_rows[keep],
                                         target_row_bound=int(graph.dst_nodes.numel())))
            window_tasks.append(item)
        targets["window_tasks"] = tuple(window_tasks)
        targets["task"] = window_tasks[-1]
    else:
        targets["task"] = attach_target_route(graph, target, collect_remote_endpoints=True)
    return int(window_id), targets, blocks, feature_node_ids, edge_ids


def _snapshot_target_ts(row: Mapping[str, Any], count: int) -> torch.Tensor | None:
    ts = row.get("ts")
    if ts is None or int(count) == 0:
        return None
    if int(ts.numel()) == 0:
        return ts.new_empty((0,))
    return (torch.floor(ts.max()).long() + 1).reshape(1).expand(int(count)).clone()


def _snapshot_global_dst_pool(store: StoreBundle) -> torch.Tensor:
    cached = store.graph.runtime_cache.get("snapshot_global_dst_pool")
    if isinstance(cached, torch.Tensor):
        return cached
    if store.graph.num_nodes > 0:
        pool = torch.arange(int(store.graph.num_nodes), dtype=torch.long)
    elif store.features.node_ids is not None and int(store.features.node_ids.numel()) > 0:
        pool = store.features.node_ids.long().cpu()
    else:
        values = [
            row["dst_nodes"].long().cpu()
            for row in store.graph.snapshot_csc_view.get("slices", [])
            if "dst_nodes" in row and int(row["dst_nodes"].numel()) > 0
        ]
        pool = (
            torch.unique(torch.cat(values), sorted=True)
            if values
            else torch.empty(0, dtype=torch.long)
        )
    store.graph.runtime_cache["snapshot_global_dst_pool"] = pool
    return pool


__all__ = ["access_snapshot_window"]
