from __future__ import annotations

from typing import Any, Mapping, Sequence
import time

import torch
from torch import Tensor

from starrygl.native import NativeTemporalSampler
from starrygl.store import StoreBundle
from starrygl.utils.route import dist_part
from starrygl.view import GraphBlock
from .blocks import (
    empty_sample_block,
    native_feature_block,
    native_target_block,
    normalize_mfg_blocks,
    sampled_edge_ids,
    sampled_feature_graph,
)


def build_native_sampler(
    store: StoreBundle,
    view: Mapping[str, Any],
    *,
    fanouts: Sequence[int] | Tensor | None,
    num_layers: int,
    options: Mapping[str, Any] | None = None,
    snapshot: bool = False,
) -> NativeTemporalSampler:
    opts = dict(options or {})
    snapshot_uniform = opts.get("policy") == "snapshot_uniform"
    if snapshot_uniform and not snapshot:
        raise ValueError("snapshot_uniform requires Snapshot neighbor sampling")
    if snapshot_uniform and int(num_layers) != 1:
        raise ValueError("snapshot_uniform currently requires num_layers=1; native multilayer MFGs do not preserve the root chain")
    profile = bool(opts.get("profile_runtime", False))
    start = time.perf_counter()
    sample_fanouts = tuple(int(v) for v in (fanouts if fanouts is not None else opts.get("fanouts", (20,))) if int(v) > 0)
    if not sample_fanouts:
        raise ValueError("neighbor sampling requires at least one positive fanout")
    topology = _sampler_topology(store, view, opts)
    cache_key = _native_sampler_cache_key(store, topology, sample_fanouts, int(num_layers), opts)
    if bool(opts.get("cache_native_sampler", True)):
        cached = store.graph.runtime_cache.get(cache_key)
        if isinstance(cached, NativeTemporalSampler):
            if profile:
                print(
                    f"[starrygl.profile] build_native_sampler_cache_hit "
                    f"rank={int(store.graph.rank)} edges={int(topology['edge_ids'].numel())} "
                    f"fanouts={sample_fanouts} seconds={time.perf_counter() - start:.6f}",
                    flush=True,
                )
            return cached
    node_part = dist_part(store.graph.partition["node_dist_index"]).to(torch.int32).cpu().contiguous()
    node_is_hot = store.graph.partition.get("node_is_hot")
    if isinstance(topology.get("node_is_hot"), Tensor):
        node_is_hot = topology["node_is_hot"]
    edge_dist_index = topology.get("edge_dist_index")
    edge_part = (
        dist_part(edge_dist_index.long()).to(torch.int32).cpu().contiguous()
        if isinstance(edge_dist_index, Tensor) and int(edge_dist_index.numel()) == int(topology["edge_ids"].numel())
        else torch.full((int(topology["edge_ids"].numel()),), int(store.graph.rank), dtype=torch.int32)
    )
    sampler = NativeTemporalSampler.from_graph(
        {
            "src": topology["src"].long(),
            "dst": topology["dst"].long(),
            "edge_ids": topology["edge_ids"].long(),
            "edge_feature_ids": topology.get("edge_feature_ids"),
            "timestamps": topology.get("ts"),
            "num_nodes": store.graph.num_nodes,
        },
        fanouts=sample_fanouts,
        num_layers=int(num_layers),
        workers=int(opts.get("workers", opts.get("num_workers", 1))),
        policy="dtdg_uniform" if snapshot_uniform else str(opts.get("policy", "boundary_recent_decay")),
        local_part=int(store.graph.rank),
        edge_part=edge_part,
        node_part=node_part,
        node_is_hot=node_is_hot if isinstance(node_is_hot, Tensor) else None,
        probability=float(opts.get("probability", opts.get("sample_probability", 1.0))),
        graph_name=str(
            opts.get(
                "graph_name",
                f"starrygl_rank{store.graph.rank}_{id(store.graph)}_{_tensor_identity(topology.get('src'))[1]}",
            )
        ),
        output=str(opts.get("output", "compact")),
        add_reverse_edges=bool(opts.get("add_reverse_edges", opts.get("reverse", True)))
        and not bool(topology.get("_reverse_edges_materialized", False)),
        materialize_col=bool(opts.get("materialize_col", True)),
        deduplicate_edges=bool(opts.get("deduplicate_edges", True)),
        deduplicate_nodes=bool(opts.get("deduplicate_nodes", False)),
        approximate_node_compaction=bool(opts.get("approximate_node_compaction", False)),
        seed=None if opts.get("seed") is None else int(opts["seed"]),
    )
    if profile:
        print(
            f"[starrygl.profile] build_native_sampler "
            f"rank={int(store.graph.rank)} edges={int(topology['edge_ids'].numel())} "
            f"topology={topology.get('format', 'event_view')} "
            f"fanouts={sample_fanouts} seconds={time.perf_counter() - start:.6f}",
            flush=True,
        )
    if bool(opts.get("cache_native_sampler", True)):
        store.graph.runtime_cache[cache_key] = sampler
    return sampler


def _sampler_topology(store: StoreBundle, view: Mapping[str, Any], opts: Mapping[str, Any]) -> Mapping[str, Any]:
    topology = store.graph.temporal_csr_view if bool(opts.get("distributed_topology", True)) else {}
    if (
        isinstance(topology, Mapping)
        and isinstance(topology.get("src"), Tensor)
        and isinstance(topology.get("dst"), Tensor)
        and isinstance(topology.get("edge_ids"), Tensor)
        and int(topology["edge_ids"].numel()) > 0
    ):
        source = topology
    else:
        source = view
    snapshot_uniform = opts.get("policy") == "snapshot_uniform"
    if not snapshot_uniform and (not isinstance(source.get("ts"), Tensor) or int(source["ts"].numel()) == 0):
        return source
    reverse = bool(opts.get("add_reverse_edges", opts.get("reverse", True)))
    cache_key = _temporal_event_topology_cache_key(store, source, reverse)
    if snapshot_uniform:
        cache_key += ("snapshot_uniform", _tensor_identity(store.graph.time_ptr_2),
                      _tensor_identity(source.get("edge_feature_ids")))
    cached = store.graph.runtime_cache.get(cache_key)
    if isinstance(cached, Mapping):
        return cached
    if snapshot_uniform:
        source = {**source, "ts": _snapshot_sample_times(source, store.graph.time_ptr_2)}
    out = _temporal_event_topology(source, reverse=reverse)
    store.graph.runtime_cache[cache_key] = out
    return out


def _snapshot_sample_times(source: Mapping[str, Any], windows: Tensor) -> Tensor:
    rows = source.get("edge_feature_ids")
    if not isinstance(rows, Tensor):
        raise ValueError("snapshot_uniform requires prepared physical edge_feature_ids")
    windows = windows.long().to(rows.device).reshape(-1, 2)
    if (bool((windows[:, 1] < windows[:, 0]).any())
            or bool((windows[1:, 0] < windows[:-1, 1]).any())):
        raise ValueError("snapshot_uniform requires ordered non-overlapping snapshot ranges")
    if not int(rows.numel()):
        return rows.new_empty(0, dtype=torch.long)
    if not int(windows.shape[0]):
        raise ValueError("snapshot_uniform requires prepared snapshot ranges")
    ids = torch.bucketize(rows.long(), windows[:, 1].contiguous(), right=True)
    if bool((ids >= windows.shape[0]).any()) or bool((rows < windows[ids, 0]).any()):
        raise ValueError("snapshot_uniform edge rows are not covered by snapshot ranges")
    return ids


def _temporal_event_topology(source: Mapping[str, Any], *, reverse: bool) -> Mapping[str, Any]:
    src = source["src"].long()
    dst = source["dst"].long()
    ts = source["ts"].long()
    edge_ids = source["edge_ids"].long()
    edge_feature_ids = source.get("edge_feature_ids", edge_ids).long()
    if int(src.numel()) <= 1:
        order = torch.arange(int(src.numel()), dtype=torch.long, device=src.device)
    else:
        order = torch.argsort(ts, stable=True)
    src = src.index_select(0, order)
    dst = dst.index_select(0, order)
    ts = ts.index_select(0, order)
    edge_ids = edge_ids.index_select(0, order)
    edge_feature_ids = edge_feature_ids.index_select(0, order)
    edge_dist_index = source.get("edge_dist_index")
    edge_dist_index = edge_dist_index.index_select(0, order) if isinstance(edge_dist_index, Tensor) else None
    node_is_hot = source.get("node_is_hot")

    if reverse and not bool(source.get("bidirectional", False)):
        src, dst = _interleave_pair(src, dst)
        ts = _interleave_same(ts)
        edge_ids = _interleave_same(edge_ids)
        edge_feature_ids = _interleave_same(edge_feature_ids)
        if isinstance(edge_dist_index, Tensor):
            edge_dist_index = _interleave_same(edge_dist_index)

    out: dict[str, Any] = {
        "format": "temporal_event_topology",
        "src": src.contiguous(),
        "dst": dst.contiguous(),
        "edge_ids": edge_ids.contiguous(),
        "edge_feature_ids": edge_feature_ids.contiguous(),
        "ts": ts.contiguous(),
        "_reverse_edges_materialized": bool(reverse or source.get("bidirectional", False)),
    }
    if isinstance(edge_dist_index, Tensor):
        out["edge_dist_index"] = edge_dist_index.contiguous()
    if isinstance(node_is_hot, Tensor):
        out["node_is_hot"] = node_is_hot
    return out


def _interleave_pair(src: Tensor, dst: Tensor) -> tuple[Tensor, Tensor]:
    return torch.stack((src, dst), dim=1).reshape(-1), torch.stack((dst, src), dim=1).reshape(-1)


def _interleave_same(value: Tensor) -> Tensor:
    return torch.stack((value, value), dim=1).reshape(-1)


def _temporal_event_topology_cache_key(store: StoreBundle, source: Mapping[str, Any], reverse: bool) -> tuple[Any, ...]:
    return (
        "temporal_event_topology",
        int(store.graph.rank),
        _tensor_identity(source.get("src")),
        _tensor_identity(source.get("dst")),
        _tensor_identity(source.get("edge_ids")),
        _tensor_identity(source.get("ts")),
        bool(reverse),
    )


def _native_sampler_cache_key(
    store: StoreBundle,
    topology: Mapping[str, Any],
    fanouts: tuple[int, ...],
    num_layers: int,
    opts: Mapping[str, Any],
) -> tuple[Any, ...]:
    edge_ids = topology.get("edge_ids")
    src = topology.get("src")
    dst = topology.get("dst")
    node_is_hot = topology.get("node_is_hot")
    return (
        "native_temporal_sampler",
        int(store.graph.rank),
        int(store.graph.num_nodes),
        _tensor_identity(src),
        _tensor_identity(dst),
        _tensor_identity(edge_ids),
        _tensor_identity(topology.get("edge_feature_ids")),
        _tensor_identity(node_is_hot),
        tuple(int(v) for v in fanouts),
        int(num_layers),
        max(1, int(opts.get("workers", opts.get("num_workers", 1)))),
        str(opts.get("policy", "boundary_recent_decay")),
        float(opts.get("probability", opts.get("sample_probability", 1.0))),
        str(opts.get("output", "compact")),
        bool(opts.get("add_reverse_edges", opts.get("reverse", True))),
        bool(opts.get("materialize_col", True)),
        bool(opts.get("deduplicate_edges", True)),
        bool(opts.get("deduplicate_nodes", False)),
        bool(opts.get("approximate_node_compaction", False)),
        None if opts.get("seed") is None else int(opts["seed"]),
    )


def _tensor_identity(value: Any) -> tuple[int, int, str]:
    if not isinstance(value, Tensor):
        return (0, 0, "")
    storage_ptr = int(value.untyped_storage().data_ptr()) if int(value.numel()) else 0
    return (int(value.numel()), storage_ptr, str(value.device))


def sample_native_blocks(
    sampler: NativeTemporalSampler,
    *,
    root_nodes: Tensor,
    root_ts: Tensor | None,
) -> tuple[GraphBlock, ...]:
    blocks = sampler.sample_blocks(root_nodes.long(), root_ts)
    if not blocks:
        return (empty_sample_block(root_nodes),)
    return normalize_mfg_blocks(tuple(blocks))


def access_native_graphs(
    sampler: NativeTemporalSampler,
    roots: Sequence[Any],
) -> tuple[tuple[tuple[GraphBlock, ...], ...], tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Sample one or more temporal units into the common graph tuple."""

    windows = []
    feature_nodes = []
    edge_ids = []
    for root in roots:
        nodes = root.node_ids.long()
        blocks = (
            (empty_sample_block(nodes),)
            if not int(nodes.numel())
            else sample_native_blocks(sampler, root_nodes=nodes, root_ts=root.ts)
        )
        windows.append(tuple(blocks))
        feature_nodes.append(sampled_feature_graph(blocks).src_nodes.long())
        edge_ids.append(sampled_edge_ids(blocks))
    return tuple(windows), tuple(feature_nodes), tuple(edge_ids)


__all__ = [
    "access_native_graphs",
    "build_native_sampler",
    "native_feature_block",
    "native_target_block",
    "sample_native_blocks",
]
