from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from starrygl.prepare.task import build_task_shards

from .feature import (
    _dist_loc,
    _dist_part,
    _owned_ids_by_part,
    _scatter_by_loc,
    _scatter_temporal_by_loc,
)


def build_label_shards(
    *,
    prepared: Any,
    node_label: Tensor | None = None,
    node_label_nodes: Tensor | None = None,
    node_label_ts: Tensor | None = None,
    node_label_split: Tensor | None = None,
    node_label_temporal: bool = False,
    node_label_horizon: int = 0,
    edge_label: Tensor | None = None,
    task: Mapping[str, Any] | str | None = None,
    temporal: str = "event",
    src: Tensor | None = None,
    dst: Tensor | None = None,
    ts: Tensor | None = None,
    edge_ids: Tensor | None = None,
) -> list[dict[str, Any]]:
    partition = prepared.partition
    world_size = int(prepared.meta.get("world_size", 1))
    node_index = partition["node_dist_index"].long()
    edge_index = partition["edge_dist_index"].long()
    node_part, node_loc = _dist_part(node_index), _dist_loc(node_index)
    edge_part, edge_loc = _dist_part(edge_index), _dist_loc(edge_index)

    out = []
    for rank in range(world_size):
        row = {"rank": torch.tensor(rank, dtype=torch.long)}
        node_ids = _owned_ids_by_part(part=node_part, loc=node_loc, rank=rank)
        owned_edges = _owned_ids_by_part(part=edge_part, loc=edge_loc, rank=rank)
        if node_label is not None and node_label_nodes is not None:
            label_nodes = node_label_nodes.long()
            pos = (node_part.index_select(0, label_nodes) == int(rank)).nonzero(as_tuple=True)[0]
            row["node_label_ids"] = label_nodes.index_select(0, pos)
            row["node_label"] = node_label.index_select(0, pos.to(node_label.device))
        elif node_label is not None:
            row["node_label_ids"] = node_ids
            row["node_label_temporal"] = bool(node_label_temporal)
            values = node_label.index_select(1 if node_label_temporal else 0, node_ids)
            row["node_label"] = (
                _scatter_temporal_by_loc(values=values, loc=node_loc.index_select(0, node_ids))
                if node_label_temporal
                else _scatter_by_loc(values=values, loc=node_loc.index_select(0, node_ids))
            )
        if edge_label is not None:
            row["edge_label"] = _scatter_by_loc(
                values=edge_label.index_select(0, owned_edges),
                loc=edge_loc.index_select(0, owned_edges),
            )
        out.append(row)

    if task is None:
        return out
    if any(value is None for value in (src, dst, ts, edge_ids)):
        raise ValueError("prepared task tables require src, dst, ts, and edge_ids")
    tasks = build_task_shards(
        prepared=prepared,
        task=task,
        temporal=str(temporal),
        src=src,
        dst=dst,
        ts=ts,
        edge_ids=edge_ids,
        node_label=node_label,
        node_label_nodes=node_label_nodes,
        node_label_ts=node_label_ts,
        node_label_split=node_label_split,
        node_label_temporal=bool(node_label_temporal),
        node_label_horizon=int(node_label_horizon),
        edge_label=edge_label,
    )
    for row, task_row in zip(out, tasks):
        row.update(task_row)
    return out


__all__ = ["build_label_shards"]
