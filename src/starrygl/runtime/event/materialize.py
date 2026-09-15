from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import torch

from starrygl.batch import EventRows, SamplingPolicy
from starrygl.runtime.dataloader.blocks import event_rows_to_graph_block
from starrygl.runtime.dataloader.materialize import AccessedWindow
from starrygl.runtime.sample import access_native_graphs, native_target_block
from starrygl.store import StoreBundle
from starrygl.store.graph import split_window_range
from starrygl.task import SamplingRoot, attach_target_route, build_window_task_target
from starrygl.task.target import with_negative_samples

from .target import (
    compatible_generator as _compatible_generator,
    deduplicate_roots as _deduplicate_roots,
    negative_dst_pool as _negative_dst_pool,
    target_extra_nodes as _target_extra_nodes,
    train_positive_rows as _train_positive_rows,
)


def event_window_ids(
    store: StoreBundle,
    split: str,
    *,
    drop_last: bool = False,
) -> range:
    """Return prepared Event rows eligible for this epoch."""

    window_ids = split_window_range(store.graph.split_time_ptr_2, split)
    if not window_ids or not drop_last:
        return window_ids
    prepare = store.graph.prepare
    meta = prepare.get("meta", {}) if isinstance(prepare, Mapping) else {}
    batch_size = int(meta.get("target_batch_size") or 0)
    if meta.get("time_split") not in {"batch", "adaptive_batch"} or batch_size < 1:
        return window_ids
    begin, end = store.graph.time_ptr_2[window_ids.stop - 1].long().tolist()
    return window_ids[:-1] if int(end) - int(begin) < batch_size else window_ids


def access_event_window(
    store: StoreBundle,
    view: Mapping[str, Any],
    *,
    split: str,
    window_id: int,
    input_window: range,
    chunk_limits: tuple[int, ...],
    sampling_policy: SamplingPolicy,
    native_sampler: Any | None = None,
    sampler_options: Mapping[str, Any] | None = None,
    num_negatives: int = 0,
    generator: object | None = None,
) -> AccessedWindow:
    """Build one Event task slice and return the common graph-access tuple."""

    del input_window, chunk_limits
    begin, end = view["time_ptr_2"][int(window_id)].long().tolist()
    edge_rows = torch.arange(int(begin), int(end), dtype=torch.long, device=view["src"].device)
    src = view["src"].index_select(0, edge_rows).long()
    dst = view["dst"].index_select(0, edge_rows).long()
    event_edge_ids = view["edge_ids"].index_select(0, edge_rows).long()
    ts = view["ts"].index_select(0, edge_rows) if "ts" in view else None
    target = build_window_task_target(
        store.labels,
        int(window_id),
    )
    if target.target_kind == "edge":
        keep = _train_positive_rows(
            store,
            target.pos_src,
            target.pos_dst,
            split=split,
            sampler_options=sampler_options,
            generator=generator,
        )
        edge_rows = edge_rows.index_select(0, keep.to(edge_rows.device))
        src = src.index_select(0, keep.to(src.device))
        dst = dst.index_select(0, keep.to(dst.device))
        event_edge_ids = event_edge_ids.index_select(0, keep.to(event_edge_ids.device))
        ts = None if ts is None else ts.index_select(0, keep.to(ts.device))
        target = replace(
            target,
            target_ids=target.target_ids.index_select(0, keep.to(target.target_ids.device)),
            target_ts=(
                None
                if target.target_ts is None
                else target.target_ts.index_select(0, keep.to(target.target_ts.device))
            ),
            label=(
                None
                if target.label is None
                else target.label.index_select(0, keep.to(target.label.device))
            ),
            pos_src=target.pos_src.index_select(0, keep.to(target.pos_src.device)),
            pos_dst=target.pos_dst.index_select(0, keep.to(target.pos_dst.device)),
            edge_ids=target.edge_ids.index_select(0, keep.to(target.edge_ids.device)),
            negative_pool=_negative_dst_pool(view, sampler_options),
        )
        if int(target.target_ids.numel()) and int(num_negatives) > 0:
            target = replace(
                with_negative_samples(
                    target,
                    num_negatives=int(num_negatives),
                    generator=_compatible_generator(target.pos_src, generator),
                ),
                negative_pool=None,
            )

    events = EventRows(src=src, dst=dst, edge_ids=event_edge_ids, ts=ts)
    if sampling_policy == "neighbor" and (int(edge_rows.numel()) or int(target.target_ids.numel())):
        if native_sampler is None:
            raise ValueError("neighbor sampling requires a native sampler")
        roots = _event_sampling_roots(target, events)
        root_nodes, root_ts = _deduplicate_roots(
            roots.node_ids,
            roots.ts,
            enabled=bool((sampler_options or {}).get("deduplicate_roots", False)),
        )
        blocks, feature_nodes, sampled_edge_ids = access_native_graphs(
            native_sampler,
            (SamplingRoot(node_ids=root_nodes, ts=root_ts, groups=roots.groups),),
        )
    else:
        graph = event_rows_to_graph_block(view, edge_rows, extra_nodes=_target_extra_nodes(target))
        blocks = ((graph,),)
        feature_nodes = (graph.src_nodes.long(),)
        sampled_edge_ids = (
            torch.unique(graph.edge_ids.long(), sorted=True)
            if int(graph.edge_ids.numel())
            else graph.edge_ids.long()
        ,)

    graph = native_target_block(blocks[-1])
    mask = view.get("state_write_mask")
    events = replace(
        events,
        state_write_mask=None if mask is None else mask.index_select(0, edge_rows),
    )
    target = attach_target_route(
        graph,
        target,
        lazy=bool((sampler_options or {}).get("lazy_target_rows", False)),
    )
    return int(window_id), {"task": target, "events": events}, blocks, feature_nodes, sampled_edge_ids


def _event_sampling_roots(target, events: EventRows) -> SamplingRoot:
    from starrygl.task import sampling_roots_from_target

    roots = sampling_roots_from_target(target)
    if target.target_kind == "edge" or not int(events.src.numel()):
        return roots
    count = int(roots.node_ids.numel())
    edge_count = int(events.src.numel())
    groups = dict(roots.groups or {})
    groups["event_src"] = (count, count + edge_count)
    groups["event_dst"] = (count + edge_count, count + 2 * edge_count)
    ts = None
    if roots.ts is not None and events.ts is not None:
        ts = torch.cat((roots.ts, events.ts, events.ts), dim=0)
    return SamplingRoot(
        node_ids=torch.cat((roots.node_ids, events.src, events.dst), dim=0),
        ts=ts,
        groups=groups,
    )


__all__ = ["access_event_window", "event_window_ids"]
