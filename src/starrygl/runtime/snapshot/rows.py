from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from starrygl.runtime.dataloader.blocks import _move_graph_value


def _move_snapshot_features(features: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device=device)
        if isinstance(value, Tensor)
        else value
        for key, value in features.items()
    }


def _move_snapshot_row(row: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    out = {key: _move_graph_value(value, device) for key, value in row.items() if key != "route"}
    route = row.get("route")
    if isinstance(route, Mapping):
        out["route"] = {
            key: value if key in {"send_sizes", "recv_sizes"} else _move_graph_value(value, device)
            for key, value in route.items()
        }
    elif route is not None:
        out["route"] = _move_graph_value(route, device)
    return out


def _chunk_order_tensor(sampler_options: Mapping[str, Any] | None) -> Tensor | None:
    value = (sampler_options or {}).get("chunk_order")
    if isinstance(value, Tensor):
        return value.long().flatten()
    return None


def _apply_chunk_limit_to_snapshot_row(
    row: Mapping[str, Any],
    chunk_limit: int,
    *,
    chunk_order: Tensor | None = None,
) -> Mapping[str, Any]:
    limit = int(chunk_limit)
    if limit < 0:
        return row
    global_count = int(chunk_order.numel()) if chunk_order is not None and chunk_order.numel() else row.get("_chunk_global_count")
    if global_count is not None and limit >= int(global_count):
        return row
    chunk_ptr = row.get("_chunk_rank_indptr")
    if isinstance(chunk_ptr, Tensor):
        end = int(chunk_ptr.long()[min(max(0, limit), int(chunk_ptr.numel()) - 1)].item())
        return _slice_csc_prefix(row, end)
    dst_nodes = row["dst_nodes"].long()
    node_chunk = row.get("node_chunk")
    if node_chunk is None:
        return row
    dst_chunk = node_chunk.long()[: int(dst_nodes.numel())]
    chunk_rank = _chunk_rank(row, dst_chunk, chunk_order=chunk_order)
    keep_dst = torch.nonzero(chunk_rank < limit, as_tuple=True)[0].long()
    return _slice_csc_row(row, keep_dst)


def _reorder_snapshot_row_for_chunk_prefix(
    row: Mapping[str, Any],
    *,
    chunk_order: Tensor | None,
) -> Mapping[str, Any]:
    dst_nodes = row["dst_nodes"].long()
    src_nodes = row["src_nodes"].long()
    node_chunk = row.get("node_chunk")
    dst_count = int(row["indptr"].long().numel()) - 1
    if node_chunk is None:
        out = dict(row)
        out["_chunk_rank_indptr"] = torch.tensor([0, dst_count], dtype=torch.long, device=dst_nodes.device)
        return out
    device = dst_nodes.device
    dst_chunk = node_chunk.long()[:dst_count]
    dst_perm, chunk_counts = _chunk_ordered_dst_perm(row, dst_chunk, chunk_order=chunk_order)
    chunk_ptr = torch.empty(int(chunk_counts.numel()) + 1, dtype=torch.long, device=device)
    chunk_ptr[0] = 0
    chunk_ptr[1:] = torch.cumsum(chunk_counts, dim=0)

    old_to_new_dst = torch.empty(dst_count, dtype=torch.long, device=device)
    old_to_new_dst[dst_perm] = torch.arange(dst_count, dtype=torch.long, device=device)
    indptr = row["indptr"].long()
    indices = row["indices"].long()
    counts = indptr[1:] - indptr[:-1]
    ordered_counts = counts.index_select(0, dst_perm)
    new_indptr = torch.empty(dst_count + 1, dtype=torch.long, device=device)
    new_indptr[0] = 0
    new_indptr[1:] = torch.cumsum(ordered_counts, dim=0)
    edge_order = torch.arange(indices.numel(), dtype=torch.long, device=device) + torch.repeat_interleave(
        indptr[:-1].index_select(0, dst_perm) - new_indptr[:-1], ordered_counts,
        output_size=indices.numel(),
    )
    old_src = indices.index_select(0, edge_order)
    new_src = old_src.clone()
    in_dst = old_src < dst_count
    if bool(torch.any(in_dst).item()):
        new_src[in_dst] = old_to_new_dst.index_select(0, old_src[in_dst])
    out = dict(row)
    route = row.get("route")
    if isinstance(route, Mapping):
        route = dict(route)
        send_index = route.get("send_index")
        if isinstance(send_index, Tensor):
            route["send_index"] = old_to_new_dst.index_select(0, send_index.long().to(device=device))
        out["route"] = route
    out["src_nodes"] = torch.cat((dst_nodes.index_select(0, dst_perm), src_nodes[dst_count:]), dim=0)
    out["dst_nodes"] = dst_nodes.index_select(0, dst_perm)
    out["edge_ids"] = row["edge_ids"].long().index_select(0, edge_order)
    out["indptr"] = new_indptr
    out["indices"] = new_src
    if "ts" in row:
        out["ts"] = row["ts"].index_select(0, edge_order)
    if "node_chunk" in row:
        chunks = row["node_chunk"].long()
        out["node_chunk"] = torch.cat((chunks[:dst_count].index_select(0, dst_perm), chunks[dst_count:]), dim=0)
    if "src_feature_row" in row:
        feature_row = row["src_feature_row"].long()
        out["src_feature_row"] = torch.cat((feature_row[:dst_count].index_select(0, dst_perm), feature_row[dst_count:]), dim=0)
    data_row = _src_data_rows(row, src_nodes)
    if data_row is not None:
        out["src_data_row"] = torch.cat((data_row[:dst_count].index_select(0, dst_perm), data_row[dst_count:]), dim=0)
    if "edge_feature_row" in row:
        out["edge_feature_row"] = row["edge_feature_row"].long().index_select(0, edge_order)
    if "edge_feature_ids" in row:
        out["edge_feature_ids"] = row["edge_feature_ids"].long().index_select(0, edge_order)
    if "edge_gcn_norm" in row:
        out["edge_gcn_norm"] = row["edge_gcn_norm"].float().index_select(0, edge_order)
    if "self_gcn_norm" in row:
        out["self_gcn_norm"] = row["self_gcn_norm"].float().index_select(0, dst_perm)
    out["_src_parent_row"] = torch.arange(int(src_nodes.numel()), dtype=torch.long, device=device)
    out["_dst_parent_row"] = dst_perm.long()
    out["_edge_parent_row"] = edge_order.long()
    out["_chunk_rank_indptr"] = chunk_ptr
    if chunk_order is not None and chunk_order.numel():
        out["_chunk_global_count"] = int(chunk_order.numel())
    out["_chunk_base"] = 0
    return out


def _chunk_ordered_dst_perm(
    row: Mapping[str, Any],
    dst_chunk: Tensor,
    *,
    chunk_order: Tensor | None,
) -> tuple[Tensor, Tensor]:
    cached = row.get("_chunk_bucket_order")
    cached_ptr = row.get("_chunk_bucket_indptr")
    if not isinstance(cached, Tensor) or not isinstance(cached_ptr, Tensor):
        local_chunk = _chunk_rank(row, dst_chunk, chunk_order=None)
        order = torch.argsort(local_chunk, stable=True)
        sorted_chunk = local_chunk.index_select(0, order)
        chunk_count = int(sorted_chunk.max().item()) + 1 if int(sorted_chunk.numel()) else 1
        counts = torch.bincount(sorted_chunk, minlength=max(1, chunk_count))
        ptr = torch.empty(int(counts.numel()) + 1, dtype=torch.long, device=dst_chunk.device)
        ptr[0] = 0
        ptr[1:] = torch.cumsum(counts, dim=0)
        if isinstance(row, dict):
            row["_chunk_bucket_order"] = order
            row["_chunk_bucket_indptr"] = ptr
        cached = order
        cached_ptr = ptr
    order = cached.long().to(device=dst_chunk.device)
    ptr = cached_ptr.long().to(device=dst_chunk.device)
    chunk_count = int(ptr.numel()) - 1
    if chunk_order is None or int(chunk_order.numel()) == 0:
        counts = ptr[1:] - ptr[:-1]
        return order, counts
    priority = chunk_order.long().to(device=dst_chunk.device)
    if int(priority.numel()) < chunk_count:
        raise ValueError("chunk_order must provide one priority per local chunk")
    if int(priority.numel()) > chunk_count:
        ptr = torch.cat((ptr, ptr[-1:].expand(int(priority.numel()) - chunk_count)))
        chunk_count = int(priority.numel())
    requested = torch.argsort(priority[:chunk_count], stable=True)
    counts = (ptr[1:] - ptr[:-1]).index_select(0, requested)
    start = torch.cumsum(counts, dim=0) - counts
    take = torch.arange(order.numel(), dtype=torch.long, device=order.device) + torch.repeat_interleave(
        ptr[:-1].index_select(0, requested) - start, counts, output_size=order.numel(),
    )
    return order.index_select(0, take), counts


def _slice_csc_prefix(row: Mapping[str, Any], dst_end: int) -> Mapping[str, Any]:
    dst_nodes = row["dst_nodes"].long()
    src_nodes = row["src_nodes"].long()
    indptr = row["indptr"].long()
    indices = row["indices"].long()
    edge_ids = row["edge_ids"].long()
    device = dst_nodes.device
    end = max(0, min(int(dst_end), int(dst_nodes.numel())))
    if end == 0:
        return _slice_csc_row(row, dst_nodes.new_empty((0,)))
    edge_end = int(indptr[end].item())
    old_dst = torch.repeat_interleave(
        torch.arange(end, dtype=torch.long, device=device),
        indptr[1 : end + 1] - indptr[:end],
        output_size=edge_end,
    )
    keep_edge = indices[:edge_end].long() < end
    selected_edges = torch.nonzero(keep_edge, as_tuple=True)[0].long()
    new_src = indices[:edge_end].long()[keep_edge]
    new_counts = torch.bincount(old_dst[keep_edge], minlength=end)
    new_indptr = torch.empty(end + 1, dtype=torch.long, device=device)
    new_indptr[0] = 0
    new_indptr[1:] = torch.cumsum(new_counts, dim=0)
    keep_src = torch.arange(end, dtype=torch.long, device=device)
    out = dict(row)
    out.pop("route", None)
    _drop_chunk_cache(out)
    out["src_nodes"] = src_nodes.index_select(0, keep_src)
    out["dst_nodes"] = dst_nodes[:end]
    out["edge_ids"] = edge_ids[:edge_end].index_select(0, selected_edges)
    out["indptr"] = new_indptr
    out["indices"] = new_src
    if "ts" in row:
        out["ts"] = row["ts"][:edge_end].index_select(0, selected_edges)
    if "node_chunk" in row:
        out["node_chunk"] = row["node_chunk"].long().index_select(0, keep_src)
    if "src_feature_row" in row:
        out["src_feature_row"] = row["src_feature_row"].long().index_select(0, keep_src)
    data_row = _src_data_rows(row, src_nodes)
    if data_row is not None:
        out["src_data_row"] = data_row.index_select(0, keep_src)
    if "edge_feature_row" in row:
        out["edge_feature_row"] = row["edge_feature_row"].long()[:edge_end].index_select(0, selected_edges)
    if "edge_feature_ids" in row:
        out["edge_feature_ids"] = row["edge_feature_ids"].long()[:edge_end].index_select(0, selected_edges)
    if "edge_gcn_norm" in row:
        out["edge_gcn_norm"] = row["edge_gcn_norm"].float()[:edge_end].index_select(0, selected_edges)
    if "self_gcn_norm" in row:
        out["self_gcn_norm"] = row["self_gcn_norm"].float()[:end]
    out["_src_parent_row"] = keep_src.long()
    out["_dst_parent_row"] = torch.arange(end, dtype=torch.long, device=device)
    out["_edge_parent_row"] = selected_edges
    out["_chunk_base"] = row.get("_chunk_base", 0)
    out["chunk_limited"] = torch.tensor(True)
    return out


def _chunk_rank(row: Mapping[str, Any], dst_chunk: Tensor, *, chunk_order: Tensor | None) -> Tensor:
    if int(dst_chunk.numel()) == 0:
        return dst_chunk.long()
    base_value = row.get("_chunk_base")
    if isinstance(base_value, Tensor):
        base = int(base_value.item())
    elif base_value is not None:
        base = int(base_value)
    else:
        base = int(dst_chunk.min().item())
    local_chunk = dst_chunk.long() - base
    if chunk_order is None or int(chunk_order.numel()) == 0:
        return local_chunk
    order = chunk_order.long().to(device=local_chunk.device)
    return order.index_select(0, local_chunk.clamp_(0, int(order.numel()) - 1))


def _slice_csc_row(row: Mapping[str, Any], keep_dst: Tensor) -> Mapping[str, Any]:
    dst_nodes = row["dst_nodes"].long()
    src_nodes = row["src_nodes"].long()
    indptr = row["indptr"].long()
    indices = row["indices"].long()
    edge_ids = row["edge_ids"].long()
    device = dst_nodes.device
    data_row = _src_data_rows(row, src_nodes)
    keep_dst = keep_dst.long().to(device=device)
    if int(keep_dst.numel()) == 0:
        out = dict(row)
        out.pop("route", None)
        _drop_chunk_cache(out)
        out.update(
            {
                "src_nodes": src_nodes.new_empty((0,)),
                "dst_nodes": dst_nodes.new_empty((0,)),
                "edge_ids": edge_ids.new_empty((0,)),
                "ts": row.get("ts", torch.empty(0, dtype=torch.float32)).new_empty((0,)),
                "indptr": torch.zeros(1, dtype=torch.long, device=device),
                "indices": indices.new_empty((0,)),
                "node_chunk": row.get("node_chunk", src_nodes.new_empty((0,))).new_empty((0,)),
                "src_feature_row": row.get("src_feature_row", src_nodes.new_empty((0,))).new_empty((0,)),
                "src_data_row": src_nodes.new_empty((0,)) if data_row is None else data_row.new_empty((0,)),
                "edge_feature_row": row.get("edge_feature_row", edge_ids.new_empty((0,))).new_empty((0,)),
                "edge_feature_ids": row.get("edge_feature_ids", edge_ids.new_empty((0,))).new_empty((0,)),
                "edge_gcn_norm": row.get("edge_gcn_norm", edge_ids.new_empty((0,), dtype=torch.float32)).new_empty((0,)),
                "self_gcn_norm": row.get("self_gcn_norm", dst_nodes.new_empty((0,), dtype=torch.float32)).new_empty((0,)),
                "_src_parent_row": torch.empty(0, dtype=torch.long, device=device),
                "_dst_parent_row": torch.empty(0, dtype=torch.long, device=device),
                "_edge_parent_row": torch.empty(0, dtype=torch.long, device=device),
                "_chunk_base": row.get("_chunk_base", row["node_chunk"].long().min() if "node_chunk" in row and int(row["node_chunk"].numel()) else 0),
                "chunk_limited": torch.tensor(True),
            }
        )
        return out

    counts = indptr[1:] - indptr[:-1]
    old_dst_rows = torch.repeat_interleave(torch.arange(int(counts.numel()), dtype=torch.long, device=device), counts)
    keep_dst_mask = torch.zeros(int(dst_nodes.numel()), dtype=torch.bool, device=device)
    keep_dst_mask[keep_dst] = True
    keep_edge = (
        keep_dst_mask.index_select(0, old_dst_rows)
        if int(old_dst_rows.numel())
        else torch.empty(0, dtype=torch.bool, device=device)
    )
    if int(indices.numel()):
        local_src = indices < int(dst_nodes.numel())
        source_kept = torch.zeros_like(local_src)
        source_kept[local_src] = keep_dst_mask.index_select(0, indices[local_src])
        keep_edge &= source_kept
    selected_old_dst = old_dst_rows[keep_edge]
    selected_old_src = indices[keep_edge]
    selected_edges = torch.nonzero(keep_edge, as_tuple=True)[0].long()

    keep_src = keep_dst

    src_row = torch.full((int(src_nodes.numel()),), -1, dtype=torch.long, device=device)
    dst_row = torch.full((int(dst_nodes.numel()),), -1, dtype=torch.long, device=device)
    src_row[keep_src] = torch.arange(int(keep_src.numel()), dtype=torch.long, device=device)
    dst_row[keep_dst] = torch.arange(int(keep_dst.numel()), dtype=torch.long, device=device)
    new_dst = dst_row.index_select(0, selected_old_dst)
    new_src = src_row.index_select(0, selected_old_src)
    order = torch.argsort(new_dst, stable=True)
    new_dst = new_dst.index_select(0, order)
    new_src = new_src.index_select(0, order)
    sorted_edges = selected_edges.index_select(0, order)
    new_counts = torch.bincount(new_dst, minlength=int(keep_dst.numel()))
    new_indptr = torch.empty(int(keep_dst.numel()) + 1, dtype=torch.long, device=device)
    new_indptr[0] = 0
    new_indptr[1:] = torch.cumsum(new_counts, dim=0)

    out = dict(row)
    out.pop("route", None)
    _drop_chunk_cache(out)
    out["src_nodes"] = src_nodes.index_select(0, keep_src)
    out["dst_nodes"] = dst_nodes.index_select(0, keep_dst)
    out["edge_ids"] = edge_ids.index_select(0, sorted_edges)
    out["indptr"] = new_indptr
    out["indices"] = new_src
    if "ts" in row:
        out["ts"] = row["ts"].index_select(0, sorted_edges)
    if "node_chunk" in row:
        out["node_chunk"] = row["node_chunk"].long().index_select(0, keep_src)
    if "src_feature_row" in row:
        out["src_feature_row"] = row["src_feature_row"].long().index_select(0, keep_src)
    if data_row is not None:
        out["src_data_row"] = data_row.index_select(0, keep_src)
    if "edge_feature_row" in row:
        out["edge_feature_row"] = row["edge_feature_row"].long().index_select(0, sorted_edges)
    if "edge_gcn_norm" in row:
        out["edge_gcn_norm"] = row["edge_gcn_norm"].float().index_select(0, sorted_edges)
    if "self_gcn_norm" in row:
        out["self_gcn_norm"] = row["self_gcn_norm"].float().index_select(0, keep_dst)
    out["_src_parent_row"] = keep_src.long()
    out["_dst_parent_row"] = keep_dst.long()
    out["_edge_parent_row"] = sorted_edges.long()
    out["_chunk_base"] = row.get("_chunk_base", row["node_chunk"].long().min() if "node_chunk" in row and int(row["node_chunk"].numel()) else 0)
    out["chunk_limited"] = torch.tensor(True)
    return out


def _drop_chunk_cache(row: dict[str, Any]) -> None:
    row.pop("_chunk_bucket_order", None)
    row.pop("_chunk_bucket_indptr", None)
    row.pop("_chunk_rank_indptr", None)


def _src_data_rows(row: Mapping[str, Any], src_nodes: Tensor) -> Tensor | None:
    rows = row.get("src_data_row")
    if isinstance(rows, Tensor):
        return rows.long()
    node_data = row.get("node_data")
    if isinstance(node_data, Mapping) and isinstance(node_data.get("x"), Tensor):
        return torch.arange(int(src_nodes.numel()), dtype=torch.long, device=src_nodes.device)
    return None
