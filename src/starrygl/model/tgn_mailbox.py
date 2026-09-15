from __future__ import annotations

import torch
from torch import Tensor

from starrygl.batch import Batch, EventRows
from starrygl.utils.index import compact_lookup_rows
from starrygl.view import GraphBlock


def mailbox_update_values(
    batch: Batch,
    block: GraphBlock,
    embeddings: Tensor,
    edge_dim: int,
) -> tuple[Tensor, Tensor, Tensor] | None:
    events = batch.targets.get("events") if isinstance(batch.targets, dict) else None
    if not isinstance(events, EventRows):
        return None
    target = batch.targets.get("task")
    pos_src = events.src.to(embeddings.device).long()
    pos_dst = events.dst.to(embeddings.device).long()
    if not pos_src.numel():
        return None
    write_mask = events.state_write_mask
    timestamps = events.ts
    if timestamps is None:
        edge_ts = block.edata.get("ts")
        timestamps = edge_ts if edge_ts is not None and edge_ts.numel() == pos_src.numel() else None
    timestamps = (
        embeddings.new_zeros(pos_src.numel())
        if timestamps is None
        else timestamps.to(embeddings).reshape(-1)
    )
    edge_features = _positive_edge_features(batch, block, embeddings, edge_dim, pos_src.numel(), events.edge_ids)
    route = getattr(target, "target_route", None)
    if write_mask is None:
        src_rows = node_rows(block, pos_src, embeddings.device, None if route is None else route.pos_src_rows)
        dst_rows = node_rows(block, pos_dst, embeddings.device, None if route is None else route.pos_dst_rows)
        src_memory, dst_memory = embeddings[src_rows], embeddings[dst_rows]
        src_mail = torch.cat((src_memory, dst_memory, edge_features), 1)
        dst_mail = torch.cat((dst_memory, src_memory, edge_features), 1)
        return torch.cat((pos_src, pos_dst)), torch.cat((src_mail, dst_mail)), torch.cat((timestamps, timestamps))

    mask = write_mask.to(embeddings.device).reshape(-1).long()
    parts: list[tuple[Tensor, Tensor, Tensor]] = []
    for bit, nodes, left, right, route_rows in (
        (1, pos_src, pos_src, pos_dst, None if route is None else (route.pos_src_rows, route.pos_dst_rows)),
        (2, pos_dst, pos_dst, pos_src, None if route is None else (route.pos_dst_rows, route.pos_src_rows)),
    ):
        positions = ((mask & bit) != 0).nonzero(as_tuple=True)[0]
        if not positions.numel():
            continue
        selected_left, selected_right = left[positions], right[positions]
        left_route = None if route_rows is None or route_rows[0] is None else route_rows[0][positions.to(route_rows[0].device)]
        right_route = None if route_rows is None or route_rows[1] is None else route_rows[1][positions.to(route_rows[1].device)]
        left_rows = node_rows(block, selected_left, embeddings.device, left_route)
        right_rows = node_rows(block, selected_right, embeddings.device, right_route)
        messages = torch.cat((embeddings[left_rows], embeddings[right_rows], edge_features[positions]), 1)
        parts.append((nodes[positions], messages, timestamps[positions]))
    if not parts:
        return None
    return tuple(torch.cat([part[index] for part in parts]) for index in range(3))  # type: ignore[return-value]


def fit_last_dim(value: Tensor, width: int) -> Tensor:
    if value.shape[1] == width:
        return value
    if value.shape[1] > width:
        return value[:, :width]
    return torch.cat((value, value.new_zeros((value.shape[0], width - value.shape[1]))), 1)


def _positive_edge_features(
    batch: Batch,
    block: GraphBlock,
    embeddings: Tensor,
    edge_dim: int,
    count: int,
    edge_ids: Tensor,
) -> Tensor:
    if edge_dim == 0:
        return embeddings.new_empty((count, 0))
    candidates = (
        batch.features.get("pos_edge_feat"),
        batch.features.get("edge_feat"),
        batch.features.get("edge"),
        batch.features.get("e"),
        block.edata.get("f"),
        block.edata.get("edge_feat"),
    )
    for value in candidates:
        value = value[0] if isinstance(value, (tuple, list)) else value
        if not isinstance(value, Tensor):
            continue
        if value.shape[0] == count:
            return fit_last_dim(value.to(embeddings).reshape(count, -1), edge_dim)
        if not edge_ids.numel() or value.shape[0] > int(edge_ids.max()):
            return fit_last_dim(value[edge_ids.to(value.device).long()].to(embeddings).reshape(count, -1), edge_dim)
        if value.shape[0] == block.num_edges:
            rows = _target_edge_positions(block, edge_ids)
            if rows is not None:
                return fit_last_dim(value[rows.to(value.device)].to(embeddings).reshape(count, -1), edge_dim)
    if count == 0:
        return embeddings.new_empty((0, edge_dim))
    raise RuntimeError("edge_dim > 0 but positive edge features are missing")


def _target_edge_positions(block: GraphBlock, edge_ids: Tensor) -> Tensor | None:
    if not edge_ids.numel():
        return None
    rows = compact_lookup_rows(block.edge_ids.long(), edge_ids.to(block.edge_ids.device))
    return None if torch.any(rows < 0) else rows


def node_rows(
    block: GraphBlock,
    node_ids: Tensor,
    device: torch.device,
    route_rows: Tensor | None = None,
) -> Tensor:
    if route_rows is not None:
        return route_rows.to(device).long()
    nodes = block.dst_nodes if block.dst_nodes.numel() else block.src_nodes
    rows = compact_lookup_rows(nodes.to(device).long(), node_ids.to(device).long())
    if torch.any(rows < 0) and not torch.equal(block.src_nodes.to(device), nodes.to(device)):
        rows = compact_lookup_rows(block.src_nodes.to(device).long(), node_ids.to(device).long())
    if torch.any(rows < 0):
        raise KeyError("target node is not materialized in graph block")
    return rows


__all__ = ["fit_last_dim", "mailbox_update_values", "node_rows"]
