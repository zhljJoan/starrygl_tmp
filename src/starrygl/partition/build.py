from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor

from .plan import ChunkTable, PartitionPlan

@dataclass(frozen=True)
class PartitionConfig:
    num_parts: int
    chunks_per_rank: int = 1
    backend: str = "speed_partition"
    speed_beta: float = 0.1
    hot_node_ratio: float = 0.01
    hot_node_type: str = "degree"

    def __post_init__(self) -> None:
        if self.num_parts < 1:
            raise ValueError("num_parts must be >= 1")
        if self.chunks_per_rank < 1:
            raise ValueError("chunks_per_rank must be >= 1")
        if self.backend not in {"speed_partition", "round_robin"}:
            raise ValueError("backend must be speed_partition or round_robin")
        if not 0.0 <= self.hot_node_ratio <= 1.0:
            raise ValueError("hot_node_ratio must be in [0, 1]")


def partition_graph(
    *,
    src: Tensor,
    dst: Tensor,
    ts: Tensor | None,
    num_nodes: int,
    config: PartitionConfig,
    node_master: Tensor | None = None,
    edge_master: Tensor | None = None,
    hot_node_ids: Tensor | None = None,
    node_to_chunk: Tensor | None = None,
) -> PartitionPlan:
    """Assign authoritative node/edge owners and chunks within node owners."""

    node_master, edge_master, hot_node_ids = _resolve_owners(
        src=src,
        dst=dst,
        ts=ts,
        num_nodes=int(num_nodes),
        config=config,
        node_master=node_master,
        edge_master=edge_master,
        hot_node_ids=hot_node_ids,
    )
    if node_to_chunk is None:
        node_to_chunk = _assign_chunks_within_master(
            src=src, dst=dst, node_master=node_master, chunks_per_rank=config.chunks_per_rank,
        )
    else:
        if not torch.is_tensor(node_to_chunk) or node_to_chunk.ndim != 1 or node_to_chunk.numel() != num_nodes:
            raise ValueError("node_to_chunk must contain one integer per node in a one-dimensional tensor")
        if node_to_chunk.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("node_to_chunk must have integer dtype")
        node_to_chunk = node_to_chunk.long().cpu().contiguous()
        if torch.any((node_to_chunk < 0) | (node_to_chunk >= config.num_parts * config.chunks_per_rank)):
            raise ValueError("node_to_chunk is outside the configured chunk ID range")
        if not torch.equal(node_to_chunk // config.chunks_per_rank, node_master):
            raise ValueError("node_to_chunk must use owner * chunks_per_rank + local_chunk IDs")
    edge_chunk = node_to_chunk.index_select(0, dst.long())
    return PartitionPlan(
        node_master=node_master,
        edge_master=edge_master,
        shared_nodes=hot_node_ids,
        chunk_table=ChunkTable(
            node_ids=torch.arange(int(num_nodes), dtype=torch.long),
            chunk_ids=node_to_chunk,
            values={"edge_chunk": edge_chunk},
        ),
        metadata={
            "num_parts": config.num_parts,
            "chunks_per_rank": config.chunks_per_rank,
            "partition_backend": config.backend,
            "hot_node_ratio": config.hot_node_ratio,
            "hot_node_type": config.hot_node_type,
        },
    )


def _resolve_owners(
    *,
    src: Tensor,
    dst: Tensor,
    ts: Tensor | None,
    num_nodes: int,
    config: PartitionConfig,
    node_master: Tensor | None,
    edge_master: Tensor | None,
    hot_node_ids: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    if (
        config.num_parts > 1
        and node_master is None
        and edge_master is None
        and hot_node_ids is None
        and config.backend == "speed_partition"
    ):
        speed = _run_speed_partition(
            src=src,
            dst=dst,
            ts=ts,
            num_nodes=num_nodes,
            config=config,
        )
        if speed is not None:
            node_master = speed["node_master"].long().cpu().contiguous()
            edge_master = speed["edge_owner"].long().cpu().contiguous()
            hot_node_ids = speed.get("hot_node_ids", torch.empty(0, dtype=torch.long)).long().cpu().contiguous()
    if node_master is None:
        node_master = torch.arange(num_nodes, dtype=torch.long) % config.num_parts
    else:
        node_master = node_master.long().cpu().contiguous()
    if int(node_master.numel()) != num_nodes:
        raise ValueError("node_master must have one value per node")
    if edge_master is None:
        edge_master = node_master.index_select(0, dst.long())
    else:
        edge_master = edge_master.long().cpu().contiguous()
    if int(edge_master.numel()) != int(src.numel()):
        raise ValueError("edge_master must have one value per edge")
    if hot_node_ids is None:
        hot_node_ids = torch.empty(0, dtype=torch.long)
    else:
        hot_node_ids = hot_node_ids.long().cpu().contiguous()
    return node_master, edge_master, hot_node_ids


def _run_speed_partition(
    *,
    src: Tensor,
    dst: Tensor,
    ts: Tensor | None,
    num_nodes: int,
    config: PartitionConfig,
) -> Mapping[str, Tensor] | None:
    try:
        mod = importlib.import_module("starrygl.native.lib.libstarrygl_sampler")
    except ModuleNotFoundError:
        return None
    ts_arg = torch.arange(int(src.numel()), dtype=torch.float64) if ts is None else ts.to(torch.float64).cpu().contiguous()
    return mod.speed_partition(
        src.long().cpu().contiguous(),
        dst.long().cpu().contiguous(),
        ts_arg,
        int(num_nodes),
        config.num_parts,
        config.speed_beta,
        config.hot_node_ratio,
        config.hot_node_type,
    )


def _assign_chunks_within_master(
    *,
    src: Tensor,
    dst: Tensor,
    node_master: Tensor,
    chunks_per_rank: int,
) -> Tensor:
    num_parts = int(node_master.max().item()) + 1 if int(node_master.numel()) else 1
    out = torch.empty_like(node_master, dtype=torch.long)
    for rank in range(num_parts):
        nodes = (node_master == rank).nonzero(as_tuple=True)[0]
        if int(nodes.numel()) == 0:
            continue
        local_chunk = _metis_partition_subset(
            src=src,
            dst=dst,
            nodes=nodes,
            num_nodes=int(node_master.numel()),
            num_parts=int(chunks_per_rank),
        )
        out[nodes] = rank * int(chunks_per_rank) + local_chunk
    return out.long().contiguous()


def _metis_partition_subset(
    *,
    src: Tensor,
    dst: Tensor,
    nodes: Tensor,
    num_nodes: int,
    num_parts: int,
) -> Tensor:
    nodes = nodes.long().cpu().contiguous()
    if num_parts <= 1:
        return torch.zeros(int(nodes.numel()), dtype=torch.long)
    if int(nodes.numel()) < num_parts:
        return torch.arange(int(nodes.numel()), dtype=torch.long) % num_parts
    local = torch.full((num_nodes,), -1, dtype=torch.long)
    local[nodes] = torch.arange(int(nodes.numel()), dtype=torch.long)
    local_src = local.index_select(0, src.long())
    local_dst = local.index_select(0, dst.long())
    keep = (local_src >= 0) & (local_dst >= 0)
    if not bool(keep.any()):
        return torch.arange(int(nodes.numel()), dtype=torch.long) % num_parts
    edge_ids = keep.nonzero(as_tuple=True)[0]
    return _run_metis_partition(
        src=local_src.index_select(0, edge_ids),
        dst=local_dst.index_select(0, edge_ids),
        num_nodes=int(nodes.numel()),
        num_parts=num_parts,
    )


def _run_metis_partition(*, src: Tensor, dst: Tensor, num_nodes: int, num_parts: int) -> Tensor:
    try:
        import dgl

        graph = dgl.graph((src.cpu(), dst.cpu()), num_nodes=num_nodes)
        graph = dgl.to_bidirected(graph, copy_ndata=False)
        return dgl.metis_partition_assignment(graph, num_parts).long().cpu().contiguous()
    except Exception:
        return torch.arange(num_nodes, dtype=torch.long) % num_parts
