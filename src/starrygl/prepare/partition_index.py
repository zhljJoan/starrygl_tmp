from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from starrygl.partition import PartitionPlan
from starrygl.utils.route import DIST_INDEX_LOC_BITS


@dataclass(frozen=True)
class PartitionIndices:
    node_dist_index: Tensor
    edge_dist_index: Tensor
    node_is_hot: Tensor
    node_to_chunk: Tensor
    edge_chunk: Tensor


def build_partition_indices(*, plan: PartitionPlan, num_nodes: int) -> PartitionIndices:
    node_master = plan.node_master
    edge_master = plan.edge_master
    hot_node_ids = plan.shared_nodes
    node_to_chunk = plan.chunk_table.chunk_ids
    edge_chunk = plan.chunk_table.values.get("edge_chunk")
    if node_master is None or edge_master is None or hot_node_ids is None:
        raise ValueError("PartitionPlan must define node_master, edge_master, and shared_nodes")
    if node_to_chunk is None or edge_chunk is None:
        raise ValueError("PartitionPlan must define node and edge chunk assignments")
    node_is_hot = torch.zeros(int(num_nodes), dtype=torch.bool)
    if int(hot_node_ids.numel()) > 0:
        node_is_hot[hot_node_ids] = True
    return PartitionIndices(
        node_dist_index=_build_node_dist_index(
            node_master=node_master,
            hot_node_ids=hot_node_ids,
            node_is_hot=node_is_hot,
        ),
        edge_dist_index=_build_dist_index(edge_master),
        node_is_hot=node_is_hot,
        node_to_chunk=node_to_chunk,
        edge_chunk=edge_chunk,
    )


def _build_node_dist_index(*, node_master: Tensor, hot_node_ids: Tensor, node_is_hot: Tensor) -> Tensor:
    node_master = node_master.long().cpu().contiguous()
    out = torch.empty_like(node_master, dtype=torch.long)
    if int(hot_node_ids.numel()) > 0:
        hot_loc = torch.arange(int(hot_node_ids.numel()), dtype=torch.long)
        out[hot_node_ids] = _pack_dist_index(part=node_master.index_select(0, hot_node_ids), loc=hot_loc)
    nonhot = (~node_is_hot).nonzero(as_tuple=True)[0].long()
    if int(nonhot.numel()) > 0:
        nonhot_part = node_master.index_select(0, nonhot)
        nonhot_loc = _local_rows_by_part(nonhot_part) + int(hot_node_ids.numel())
        out[nonhot] = _pack_dist_index(part=nonhot_part, loc=nonhot_loc)
    return out


def _build_dist_index(part: Tensor) -> Tensor:
    part = part.long().cpu().contiguous()
    if int(part.numel()) == 0:
        return torch.empty(0, dtype=torch.long)
    return _pack_dist_index(part=part, loc=_local_rows_by_part(part))


def _local_rows_by_part(part: Tensor) -> Tensor:
    order = torch.argsort(part, stable=True)
    sorted_part = part.index_select(0, order)
    change = torch.ones(int(sorted_part.numel()), dtype=torch.bool)
    change[1:] = sorted_part[1:] != sorted_part[:-1]
    positions = torch.arange(int(sorted_part.numel()), dtype=torch.long)
    group_start = torch.where(change, positions, torch.zeros_like(positions))
    group_start = torch.cummax(group_start, dim=0).values
    sorted_loc = positions - group_start
    loc = torch.empty_like(part)
    loc[order] = sorted_loc
    return loc


def _pack_dist_index(*, part: Tensor, loc: Tensor) -> Tensor:
    return (part.long() << DIST_INDEX_LOC_BITS) | loc.long()


__all__ = ["PartitionIndices", "build_partition_indices"]
