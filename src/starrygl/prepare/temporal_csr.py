from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def build_temporal_csr_view(
    *,
    src: Tensor,
    dst: Tensor,
    ts: Tensor | None,
    edge_ids: Tensor,
    node_dist_index: Tensor,
    edge_dist_index: Tensor,
    node_to_chunk: Tensor,
    edge_chunk: Tensor,
    time_ptr_2: Tensor,
    num_nodes: int,
    bidirectional: bool,
    shared: bool,
    node_is_hot: Tensor,
) -> dict[str, Any]:
    edge_rows = torch.arange(int(src.numel()), dtype=torch.long)
    if bidirectional:
        view_src = torch.stack((src, dst), dim=1).reshape(-1)
        view_dst = torch.stack((dst, src), dim=1).reshape(-1)
        view_ts = torch.stack((ts, ts), dim=1).reshape(-1) if ts is not None else None
        view_edge_rows = torch.stack((edge_rows, edge_rows), dim=1).reshape(-1)
        view_edge_ids = torch.stack((edge_ids, edge_ids), dim=1).reshape(-1)
        reverse_edge_chunk = node_to_chunk.index_select(0, src)
        view_edge_chunk = torch.stack((edge_chunk, reverse_edge_chunk), dim=1).reshape(-1)
    else:
        view_src, view_dst, view_ts = src, dst, ts
        view_edge_rows, view_edge_ids, view_edge_chunk = edge_rows, edge_ids, edge_chunk
    indptr, indices, order = compressed(primary=view_src, indices=view_dst, num_nodes=num_nodes)
    return {
        "format": "temporal_csr",
        "bidirectional": bool(bidirectional),
        "shared": bool(shared),
        "indptr": indptr,
        "indices": indices,
        "edge_order": order,
        "edge_ids": view_edge_ids,
        "edge_feature_ids": view_edge_rows,
        "ts": view_ts if view_ts is not None else torch.empty(0, dtype=torch.float32),
        "src": view_src,
        "dst": view_dst,
        "time_ptr_2": time_ptr_2,
        "node_dist_index": node_dist_index,
        "edge_dist_index": edge_dist_index.index_select(0, view_edge_rows),
        "node_chunk": node_to_chunk,
        "edge_chunk": view_edge_chunk,
        "src_chunk": node_to_chunk.index_select(0, view_src),
        "dst_chunk": node_to_chunk.index_select(0, view_dst),
        "node_is_hot": node_is_hot,
        "src_is_hot": node_is_hot.index_select(0, view_src),
        "dst_is_hot": node_is_hot.index_select(0, view_dst),
    }


def compressed(
    *,
    primary: Tensor,
    indices: Tensor,
    num_nodes: int,
    sort_keys: tuple[Tensor, ...] = (),
) -> tuple[Tensor, Tensor, Tensor]:
    if int(primary.numel()) == 0:
        empty = torch.empty(0, dtype=torch.long)
        return torch.zeros(num_nodes + 1, dtype=torch.long), empty, empty
    order = torch.arange(int(primary.numel()), dtype=torch.long)
    for key in reversed(sort_keys):
        order = order.index_select(0, torch.argsort(key.index_select(0, order), stable=True))
    order = order.index_select(0, torch.argsort(primary.long().index_select(0, order), stable=True))
    sorted_primary = primary.long().index_select(0, order)
    counts = torch.bincount(sorted_primary.clamp_min(0), minlength=num_nodes)
    indptr = torch.cat((torch.zeros(1, dtype=torch.long), counts[:num_nodes].cumsum(0)))
    return indptr, indices.long().index_select(0, order), order


__all__ = ["build_temporal_csr_view", "compressed"]
