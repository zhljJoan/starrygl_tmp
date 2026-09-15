from __future__ import annotations

import torch
from torch import Tensor


def compact_lookup_rows(source_ids: Tensor, query_ids: Tensor) -> Tensor:
    """Map query ids to their last source row with O(source + query) scratch."""

    query = query_ids.long()
    if int(query.numel()) == 0:
        return query.new_empty((0,))
    source = source_ids.to(device=query.device).long()
    if int(source.numel()) == 0:
        return torch.full_like(query, -1)

    if source_ids.device.type == "cpu" and query.device.type == "cpu" and source.numel() >= 256:
        low, high = torch.aminmax(source)
        low, high = int(low), int(high)
        span = high - low + 1
        if span <= 4 * max(source.numel(), query.numel()):
            inverse = torch.full((span,), -1, dtype=torch.long, device=source.device)
            inverse.scatter_reduce_(
                0, source - low, torch.arange(source.numel(), device=source.device),
                reduce="amax", include_self=True,
            )
            present = (query >= low) & (query <= high)
            # Clamp first so extreme out-of-range IDs cannot overflow the offset.
            positions = query.clamp(min=low, max=high) - low
            return inverse.index_select(0, positions).masked_fill_(~present, -1)

    order = torch.argsort(source, stable=True)
    sorted_ids = source.index_select(0, order)
    positions = torch.searchsorted(sorted_ids, query, right=True) - 1
    safe = positions.clamp(min=0, max=int(sorted_ids.numel()) - 1)
    found = (positions >= 0) & (sorted_ids.index_select(0, safe) == query)
    matched_rows = order.index_select(0, safe)
    return torch.where(found, matched_rows, torch.full_like(query, -1))


def compact_node_time(nodes: Tensor, timestamps: Tensor | None) -> tuple[Tensor, Tensor]:
    if nodes.numel() <= 1:
        return nodes.long(), torch.arange(nodes.numel(), device=nodes.device)
    if timestamps is None:
        return torch.unique(nodes.long(), sorted=True, return_inverse=True)
    node_ids = nodes.long()
    timestamps = timestamps.to(nodes.device).reshape(-1)
    time_order = torch.argsort(timestamps, stable=True)
    order = time_order[torch.argsort(node_ids[time_order], stable=True)]
    sorted_nodes, sorted_ts = node_ids[order], timestamps[order]
    first = torch.ones(sorted_nodes.numel(), dtype=torch.bool, device=nodes.device)
    first[1:] = (sorted_nodes[1:] != sorted_nodes[:-1]) | (sorted_ts[1:] != sorted_ts[:-1])
    sorted_inverse = torch.cumsum(first.long(), 0) - 1
    inverse = torch.empty_like(sorted_inverse)
    inverse[order] = sorted_inverse
    return sorted_nodes[first].long(), inverse.long()



def temporal_lookup_rows(source_ids: Tensor, source_ts: Tensor, query_ids: Tensor, query_ts: Tensor) -> Tensor:
    """Map exact (node, cutoff) queries to the last matching source row."""

    nodes = torch.cat((source_ids.to(query_ids.device).long(), query_ids.long()))
    timestamps = torch.cat((source_ts.to(query_ids.device), query_ts.to(query_ids.device)))
    _, inverse = compact_node_time(nodes, timestamps)
    count = source_ids.numel()
    latest = torch.full((nodes.numel(),), -1, dtype=torch.long, device=nodes.device)
    latest.scatter_reduce_(0, inverse[:count], torch.arange(count, device=nodes.device),
                           reduce="amax", include_self=True)
    return latest[inverse[count:]]


__all__ = ["compact_lookup_rows", "compact_node_time", "temporal_lookup_rows"]
