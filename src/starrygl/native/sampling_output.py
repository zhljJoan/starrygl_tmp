from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from starrygl.batch import Batch
from starrygl.view import GraphBlock

_NATIVE_UTILS_MODULE: Any | None = None
_NATIVE_UTILS_LOAD_FAILED = False

def blocks_from_native_sampling_output(
    output: Any,
    *,
    materialize_col: bool = True,
    edge_id_map: Tensor | None = None,
    edge_feature_id_map: Tensor | None = None,
    deduplicate_edges: bool = True,
    deduplicate_nodes: bool = False,
) -> tuple[GraphBlock, ...]:
    node_gids = _optional_tensor_field(output, "node_gids")
    node_ts = _optional_tensor_field(output, "node_ts")
    edge_gids = _optional_tensor_field(output, "edge_gids")
    return tuple(
        graph_block_from_native_mfg(
            mfg,
            node_gids=node_gids,
            node_ts=node_ts,
            edge_gids=edge_gids,
            edge_id_map=edge_id_map,
            edge_feature_id_map=edge_feature_id_map,
            materialize_col=materialize_col,
            deduplicate_edges=deduplicate_edges,
            deduplicate_nodes=deduplicate_nodes,
        )
        for mfg in output.mfgs
    )


def batch_from_native_sampling_output(
    output: Any,
    *,
    features: Mapping[str, Any] | None = None,
    targets: Mapping[str, Any] | None = None,
    materialize_col: bool = False,
    edge_id_map: Tensor | None = None,
    edge_feature_id_map: Tensor | None = None,
    deduplicate_edges: bool = True,
    deduplicate_nodes: bool = False,
) -> Batch:
    blocks = blocks_from_native_sampling_output(
        output,
        materialize_col=materialize_col,
        edge_id_map=edge_id_map,
        edge_feature_id_map=edge_feature_id_map,
        deduplicate_edges=deduplicate_edges,
        deduplicate_nodes=deduplicate_nodes,
    )
    return Batch(
        mode="sampled",  # type: ignore[arg-type]
        features={} if features is None else features,
        targets={} if targets is None else targets,
        blocks=(tuple(blocks),),
        num_layers=len(blocks),
    )


def graph_block_from_native_mfg(
    mfg: Any,
    *,
    node_gids: Tensor | None = None,
    node_ts: Tensor | None = None,
    edge_gids: Tensor | None = None,
    edge_id_map: Tensor | None = None,
    edge_feature_id_map: Tensor | None = None,
    materialize_col: bool = True,
    deduplicate_edges: bool = True,
    deduplicate_nodes: bool = False,
) -> GraphBlock:
    dst_lids = _tensor_field(mfg, "dst_lids")
    src_lids = _tensor_field(mfg, "src_lids")
    indptr = _tensor_field(mfg, "csc_indptr")
    indices = _tensor_field(mfg, "csc_indices")
    edge_lids = _tensor_field(mfg, "edge_lids")
    delta_t = _optional_tensor_field(mfg, "delta_t")
    raw_src_nodes = int(src_lids.numel())
    raw_dst_nodes = int(dst_lids.numel())
    num_dst = int(dst_lids.numel())
    col = None
    if materialize_col:
        counts = indptr[1:] - indptr[:-1]
        col = torch.repeat_interleave(torch.arange(num_dst, dtype=torch.long, device=indptr.device), counts)
    raw_edges = int(edge_lids.numel())
    if bool(deduplicate_edges) and raw_edges > 1:
        native_dedup = _native_deduplicate_csc_edges(indptr, edge_lids)
        if native_dedup is not None:
            keep_index, new_indptr = native_dedup
            keep_index = keep_index.to(device=edge_lids.device)
            if int(keep_index.numel()) != raw_edges:
                indices = indices.index_select(0, keep_index)
                edge_lids = edge_lids.index_select(0, keep_index)
                if col is not None:
                    col = col.index_select(0, keep_index)
                if delta_t is not None:
                    delta_t = delta_t.index_select(0, keep_index)
                indptr = new_indptr.to(device=indptr.device)
                if col is None and bool(materialize_col):
                    counts = indptr[1:] - indptr[:-1]
                    col = torch.repeat_interleave(torch.arange(num_dst, dtype=torch.long, device=indptr.device), counts)
        else:
            if col is None:
                counts = indptr[1:] - indptr[:-1]
                col = torch.repeat_interleave(torch.arange(num_dst, dtype=torch.long, device=indptr.device), counts)
            keep = _first_occurrence_mask(_dedup_key(col.long(), edge_lids.long()))
            if int(keep.sum().item()) != raw_edges:
                keep_index = keep.nonzero(as_tuple=True)[0]
                indices = indices.index_select(0, keep_index)
                edge_lids = edge_lids.index_select(0, keep_index)
                col = col.index_select(0, keep_index)
                if delta_t is not None:
                    delta_t = delta_t.index_select(0, keep_index)
                counts = torch.bincount(col.long(), minlength=num_dst)
                indptr = torch.empty(num_dst + 1, dtype=torch.long, device=counts.device)
                indptr[0] = 0
                indptr[1:] = torch.cumsum(counts, dim=0)
    src_nodes = _gather_if_index(node_gids, src_lids).long()
    dst_nodes = _gather_if_index(node_gids, dst_lids).long()
    src_ts = _gather_if_index(node_ts, src_lids) if node_ts is not None else None
    dst_ts = _gather_if_index(node_ts, dst_lids) if node_ts is not None else None
    if bool(deduplicate_nodes) and int(src_nodes.numel()) > 1:
        if col is None:
            counts = indptr[1:] - indptr[:-1]
            col = torch.repeat_interleave(torch.arange(num_dst, dtype=torch.long, device=indptr.device), counts)
        compact = _compact_csc_nodes(
            src_lids=src_lids,
            dst_lids=dst_lids,
            src_nodes=src_nodes,
            dst_nodes=dst_nodes,
            src_ts=src_ts,
            dst_ts=dst_ts,
            indptr=indptr,
            indices=indices,
            col=col,
            edge_lids=edge_lids,
            delta_t=delta_t,
            materialize_col=materialize_col,
        )
        src_nodes = compact["src_nodes"]
        dst_nodes = compact["dst_nodes"]
        src_ts = compact["src_ts"]
        dst_ts = compact["dst_ts"]
        indptr = compact["indptr"]
        indices = compact["indices"]
        col = compact["col"]
        edge_lids = compact["edge_lids"]
        delta_t = compact["delta_t"]
        num_dst = int(dst_nodes.numel())
        if bool(deduplicate_edges):
            indptr, indices, col, edge_lids, delta_t = _deduplicate_csc_edges(
                indptr=indptr,
                indices=indices,
                col=col,
                edge_lids=edge_lids,
                delta_t=delta_t,
                num_dst=num_dst,
                materialize_col=materialize_col,
            )
    native_edge_ids = _gather_if_index(edge_gids, edge_lids)
    edge_feature_ids = native_edge_ids
    if edge_feature_id_map is not None and int(native_edge_ids.numel()) > 0:
        edge_feature_ids = edge_feature_id_map.index_select(0, native_edge_ids.cpu().long()).to(
            device=native_edge_ids.device
        )
    edge_ids = native_edge_ids
    if edge_id_map is not None and int(native_edge_ids.numel()) > 0:
        edge_ids = edge_id_map.index_select(0, native_edge_ids.cpu().long()).to(device=native_edge_ids.device)
    srcdata: dict[str, Tensor] = {}
    dstdata: dict[str, Tensor] = {}
    if node_ts is not None:
        srcdata["ts"] = src_ts
        dstdata["ts"] = dst_ts
    edata: dict[str, Tensor] = {}
    if delta_t is not None:
        edata["delta_t"] = delta_t
    deduped_edges = int(edge_lids.numel())
    return GraphBlock(
        src_nodes=src_nodes.long(),
        dst_nodes=dst_nodes.long(),
        edge_ids=edge_ids.long(),
        format="csc",
        indptr=indptr.long(),
        indices=indices.long(),
        row=indices.long(),
        col=col,
        num_src=int(src_nodes.numel()),
        num_dst=num_dst,
        srcdata=srcdata,
        dstdata=dstdata,
        edata=edata,
        cache={
            "raw_edge_count": raw_edges,
            "deduped_edge_count": deduped_edges,
            "raw_src_node_count": raw_src_nodes,
            "deduped_src_node_count": int(src_nodes.numel()),
            "raw_dst_node_count": raw_dst_nodes,
            "deduped_dst_node_count": int(dst_nodes.numel()),
            "edge_feature_ids": edge_feature_ids.long(),
            "src_node_ts_compacted": node_ts is not None and not bool(deduplicate_nodes),
            "dst_node_ts_compacted": node_ts is not None and not bool(deduplicate_nodes),
        },
        exec_mode="LOCAL_SAMPLE",
    )


def _first_occurrence_mask(values: Tensor) -> Tensor:
    pos = torch.arange(int(values.numel()), dtype=torch.long, device=values.device)
    unique, inverse = torch.unique(values, sorted=False, return_inverse=True)
    del unique
    first = torch.full((int(inverse.max().item()) + 1,), int(values.numel()), dtype=torch.long, device=values.device)
    first.scatter_reduce_(0, inverse, pos, reduce="amin", include_self=True)
    return first.index_select(0, inverse) == pos


def _dedup_key(col: Tensor, edge_lids: Tensor) -> Tensor:
    if int(edge_lids.numel()) == 0:
        return edge_lids.long()
    width = int(edge_lids.max().item()) + 1
    return col.long() * max(1, width) + edge_lids.long()


def _compact_csc_nodes(
    *,
    src_lids: Tensor,
    dst_lids: Tensor,
    src_nodes: Tensor,
    dst_nodes: Tensor,
    src_ts: Tensor | None,
    dst_ts: Tensor | None,
    indptr: Tensor,
    indices: Tensor,
    col: Tensor,
    edge_lids: Tensor,
    delta_t: Tensor | None,
    materialize_col: bool,
) -> dict[str, Tensor | None]:
    del src_lids, dst_lids
    dst_keep = _first_occurrence_mask(dst_nodes.long()) if int(dst_nodes.numel()) else torch.empty(0, dtype=torch.bool, device=dst_nodes.device)
    compact_dst_nodes = dst_nodes.index_select(0, dst_keep.nonzero(as_tuple=True)[0]) if int(dst_nodes.numel()) else dst_nodes
    compact_dst_ts = None if dst_ts is None else dst_ts.index_select(0, dst_keep.nonzero(as_tuple=True)[0])

    src_first = _first_occurrence_mask(src_nodes.long()) if int(src_nodes.numel()) else torch.empty(0, dtype=torch.bool, device=src_nodes.device)
    src_not_dst = ~_isin_int(src_nodes.long(), compact_dst_nodes.long()) if int(src_nodes.numel()) else src_first
    src_keep = src_first & src_not_dst
    src_keep_pos = src_keep.nonzero(as_tuple=True)[0]
    compact_src_nodes = torch.cat((compact_dst_nodes, src_nodes.index_select(0, src_keep_pos)), dim=0)
    if src_ts is None:
        compact_src_ts = None
    else:
        compact_src_ts = torch.cat((compact_dst_ts, src_ts.index_select(0, src_keep_pos)), dim=0) if compact_dst_ts is not None else src_ts.index_select(0, src_keep_pos)

    old_src_keys = src_nodes.index_select(0, indices.long()) if int(indices.numel()) else src_nodes.new_empty((0,))
    new_indices = _map_keys_to_rows(compact_src_nodes.long(), old_src_keys.long()) if int(old_src_keys.numel()) else indices.new_empty((0,))
    old_dst_keys = dst_nodes.index_select(0, col.long()) if int(col.numel()) else dst_nodes.new_empty((0,))
    new_col = _map_keys_to_rows(compact_dst_nodes.long(), old_dst_keys.long()) if int(old_dst_keys.numel()) else col.new_empty((0,))
    if int(new_col.numel()) > 1:
        order = torch.argsort(new_col, stable=True)
        new_col = new_col.index_select(0, order)
        new_indices = new_indices.index_select(0, order)
        edge_lids = edge_lids.index_select(0, order)
        if delta_t is not None:
            delta_t = delta_t.index_select(0, order)
    counts = torch.bincount(new_col.long(), minlength=int(compact_dst_nodes.numel())) if int(new_col.numel()) else torch.zeros(int(compact_dst_nodes.numel()), dtype=torch.long, device=indptr.device)
    new_indptr = torch.empty(int(compact_dst_nodes.numel()) + 1, dtype=torch.long, device=indptr.device)
    new_indptr[0] = 0
    new_indptr[1:] = torch.cumsum(counts, dim=0)
    return {
        "src_nodes": compact_src_nodes,
        "dst_nodes": compact_dst_nodes,
        "src_ts": compact_src_ts,
        "dst_ts": compact_dst_ts,
        "indptr": new_indptr,
        "indices": new_indices,
        "col": new_col if bool(materialize_col) else None,
        "edge_lids": edge_lids,
        "delta_t": delta_t,
    }


def _deduplicate_csc_edges(
    *,
    indptr: Tensor,
    indices: Tensor,
    col: Tensor | None,
    edge_lids: Tensor,
    delta_t: Tensor | None,
    num_dst: int,
    materialize_col: bool,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor, Tensor | None]:
    if int(edge_lids.numel()) <= 1:
        return indptr, indices, col, edge_lids, delta_t
    native_dedup = _native_deduplicate_csc_edges(indptr, edge_lids)
    if native_dedup is not None:
        keep_index, new_indptr = native_dedup
        if int(keep_index.numel()) == int(edge_lids.numel()):
            return indptr, indices, col if bool(materialize_col) else None, edge_lids, delta_t
        keep_index = keep_index.to(device=edge_lids.device)
        indices = indices.index_select(0, keep_index)
        edge_lids = edge_lids.index_select(0, keep_index)
        if col is not None:
            col = col.index_select(0, keep_index)
        elif bool(materialize_col):
            counts = new_indptr.to(device=indptr.device)[1:] - new_indptr.to(device=indptr.device)[:-1]
            col = torch.repeat_interleave(torch.arange(int(num_dst), dtype=torch.long, device=indptr.device), counts)
        if delta_t is not None:
            delta_t = delta_t.index_select(0, keep_index)
        return new_indptr.to(device=indptr.device), indices, col if bool(materialize_col) else None, edge_lids, delta_t
    if col is None:
        counts = indptr[1:] - indptr[:-1]
        col = torch.repeat_interleave(torch.arange(int(num_dst), dtype=torch.long, device=indptr.device), counts)
    keep = _first_occurrence_mask(_dedup_key(col.long(), edge_lids.long()))
    if int(keep.sum().item()) == int(edge_lids.numel()):
        return indptr, indices, col if bool(materialize_col) else None, edge_lids, delta_t
    keep_index = keep.nonzero(as_tuple=True)[0]
    indices = indices.index_select(0, keep_index)
    edge_lids = edge_lids.index_select(0, keep_index)
    col = col.index_select(0, keep_index)
    if delta_t is not None:
        delta_t = delta_t.index_select(0, keep_index)
    counts = torch.bincount(col.long(), minlength=int(num_dst))
    indptr = torch.empty(int(num_dst) + 1, dtype=torch.long, device=counts.device)
    indptr[0] = 0
    indptr[1:] = torch.cumsum(counts, dim=0)
    return indptr, indices, col if bool(materialize_col) else None, edge_lids, delta_t


def _native_deduplicate_csc_edges(indptr: Tensor, edge_lids: Tensor) -> tuple[Tensor, Tensor] | None:
    if indptr.device.type != "cpu" or edge_lids.device.type != "cpu":
        return None
    mod = _load_native_utils_module()
    if mod is None or not hasattr(mod, "deduplicate_csc_edges"):
        return None
    try:
        keep_index, new_indptr = mod.deduplicate_csc_edges(indptr.long().contiguous(), edge_lids.long().contiguous())
    except Exception:
        return None
    return keep_index.long(), new_indptr.long()


def _isin_int(values: Tensor, candidates: Tensor) -> Tensor:
    if int(values.numel()) == 0 or int(candidates.numel()) == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    if _can_dense_map(values, candidates):
        size = int(torch.maximum(values.max(), candidates.max()).item()) + 1
        table = torch.zeros(size, dtype=torch.bool, device=values.device)
        table[candidates.long()] = True
        return table.index_select(0, values.long())
    candidates_sorted = torch.sort(candidates.long()).values
    pos = torch.searchsorted(candidates_sorted, values.long())
    valid = pos < int(candidates_sorted.numel())
    pos = pos.clamp_max(max(0, int(candidates_sorted.numel()) - 1))
    return valid & (candidates_sorted.index_select(0, pos) == values.long())


def _map_keys_to_rows(keys: Tensor, query: Tensor) -> Tensor:
    if int(query.numel()) == 0:
        return query.new_empty((0,))
    if _can_dense_map(keys, query):
        size = int(torch.maximum(keys.max(), query.max()).item()) + 1
        row = torch.full((size,), -1, dtype=torch.long, device=keys.device)
        row[keys.long()] = torch.arange(int(keys.numel()), dtype=torch.long, device=keys.device)
        out = row.index_select(0, query.long())
        if bool(torch.any(out < 0).item()):
            raise KeyError("sampled node key is not present in compacted node table")
        return out
    order = torch.argsort(keys.long(), stable=True)
    sorted_keys = keys.long().index_select(0, order)
    pos = torch.searchsorted(sorted_keys, query.long())
    if bool(torch.any(pos >= int(sorted_keys.numel())).item()):
        raise KeyError("sampled node key is not present in compacted node table")
    rows = order.index_select(0, pos)
    if bool(torch.any(sorted_keys.index_select(0, pos) != query.long()).item()):
        raise KeyError("sampled node key is not present in compacted node table")
    return rows.long()


def _can_dense_map(left: Tensor, right: Tensor, *, max_size: int = 20_000_000) -> bool:
    if left.device != right.device or left.device.type != "cpu":
        return False
    if int(left.numel()) == 0 or int(right.numel()) == 0:
        return False
    min_key = int(torch.minimum(left.min(), right.min()).item())
    if min_key < 0:
        return False
    max_key = int(torch.maximum(left.max(), right.max()).item())
    return max_key < int(max_size)


def _tensor_field(obj: Any, name: str) -> Tensor:
    value = getattr(obj, name)
    if callable(value):
        value = value()
    tensor = torch.as_tensor(value)
    return tensor if name == "delta_t" else tensor.long()


def _optional_tensor_field(obj: Any, name: str) -> Tensor | None:
    if not hasattr(obj, name):
        return None
    value = getattr(obj, name)
    if callable(value):
        value = value()
    tensor = torch.as_tensor(value)
    if int(tensor.numel()) == 0:
        return None
    return tensor


def _gather_if_index(values: Tensor | None, index: Tensor) -> Tensor:
    if values is None or int(values.numel()) == 0 or int(index.numel()) == 0:
        return index.long()
    return values.to(device=index.device).index_select(0, index.long())


def _int_part(value: Tensor | None, size: int) -> Tensor:
    if value is None:
        return torch.zeros(int(size), dtype=torch.int32)
    return value.to(dtype=torch.int32).cpu().contiguous()


def _uint8_mask(value: Tensor | None, size: int) -> Tensor:
    if value is None:
        return torch.empty(0, dtype=torch.uint8)
    mask = value.to(dtype=torch.uint8).cpu().contiguous()
    if int(mask.numel()) == int(size):
        return mask
    return torch.empty(0, dtype=torch.uint8)


def _infer_num_nodes(src: Tensor, dst: Tensor) -> int:
    if int(src.numel()) == 0 and int(dst.numel()) == 0:
        return 0
    return int(torch.cat((src, dst), dim=0).max().item()) + 1

def _load_native_utils_module():
    global _NATIVE_UTILS_MODULE, _NATIVE_UTILS_LOAD_FAILED
    if _NATIVE_UTILS_MODULE is not None:
        return _NATIVE_UTILS_MODULE
    if _NATIVE_UTILS_LOAD_FAILED:
        return None
    try:
        import starrygl.native.lib.native_utils as mod

        _NATIVE_UTILS_MODULE = mod
        return mod
    except Exception:
        _NATIVE_UTILS_LOAD_FAILED = True
        return None

__all__ = ["batch_from_native_sampling_output", "blocks_from_native_sampling_output", "graph_block_from_native_mfg"]
