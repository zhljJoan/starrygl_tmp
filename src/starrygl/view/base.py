from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import Tensor


GraphFormat = Literal["csc", "csr", "coo", "native"]


def mfg_chain_matches(blocks) -> bool:
    """Adjacent MFGs must agree on both node IDs and temporal query cutoffs."""
    return all(torch.equal(left.dst_nodes, right.src_nodes) and (
        "ts" not in left.dstdata or "ts" not in right.srcdata
        or torch.equal(left.dstdata["ts"], right.srcdata["ts"]))
        for left, right in zip(blocks, blocks[1:]))


@dataclass
class GraphBlock:
    """Direct graph tensor buffers consumed by model layers."""

    src_nodes: Tensor
    dst_nodes: Tensor
    edge_ids: Tensor
    format: GraphFormat
    indptr: Tensor | None = None
    indices: Tensor | None = None
    row: Tensor | None = None
    col: Tensor | None = None
    edge_index: Tensor | None = None
    num_src: int | None = None
    num_dst: int | None = None
    srcdata: dict[str, Tensor] = field(default_factory=dict)
    dstdata: dict[str, Tensor] = field(default_factory=dict)
    edata: dict[str, Tensor] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict, repr=False)
    route: Any | None = None
    exec_mode: str | None = None

    @property
    def is_csc(self) -> bool:
        return self.format == "csc"

    @property
    def is_csr(self) -> bool:
        return self.format == "csr"

    @property
    def num_edges(self) -> int:
        return int(self.edge_ids.numel())

    def sparse_buffers(self) -> tuple[Tensor | None, Tensor | None]:
        return self.indptr, self.indices


def graph_block_from_coo(
    *,
    src: Tensor,
    dst: Tensor,
    edge_ids: Tensor,
    num_nodes: int,
    format: GraphFormat,
    materialize_coo: bool = True,
    edge_weight: Tensor | None = None,
) -> GraphBlock:
    if format == "csr":
        indptr, indices, order = _compressed(src, dst, num_nodes)
    elif format == "csc":
        indptr, indices, order = _compressed(dst, src, num_nodes)
    else:
        return GraphBlock(
            src_nodes=torch.arange(num_nodes, dtype=torch.long, device=src.device),
            dst_nodes=torch.arange(num_nodes, dtype=torch.long, device=src.device),
            edge_ids=edge_ids,
            format=format,
            row=src,
            col=dst,
            edge_index=torch.stack((src, dst), dim=0),
            num_src=num_nodes,
            num_dst=num_nodes,
            edata=_ordered_edge_data(edge_weight=edge_weight, order=None, num_edges=int(edge_ids.numel())),
        )

    ordered_src = src.index_select(0, order) if materialize_coo else None
    ordered_dst = dst.index_select(0, order) if materialize_coo else None
    return GraphBlock(
        src_nodes=torch.arange(num_nodes, dtype=torch.long, device=src.device),
        dst_nodes=torch.arange(num_nodes, dtype=torch.long, device=src.device),
        edge_ids=edge_ids.index_select(0, order),
        format=format,
        indptr=indptr,
        indices=indices,
        row=ordered_src,
        col=ordered_dst,
        edge_index=None if not materialize_coo else torch.stack((ordered_src, ordered_dst), dim=0),
        num_src=num_nodes,
        num_dst=num_nodes,
        edata=_ordered_edge_data(edge_weight=edge_weight, order=order, num_edges=int(edge_ids.numel())),
    )


def _compressed(primary: Tensor, secondary: Tensor, num_nodes: int) -> tuple[Tensor, Tensor, Tensor]:
    order = torch.argsort(primary, stable=True)
    sorted_primary = primary.index_select(0, order)
    sorted_secondary = secondary.index_select(0, order)
    counts = torch.bincount(sorted_primary, minlength=int(num_nodes))
    indptr = torch.empty(int(num_nodes) + 1, dtype=torch.long, device=primary.device)
    indptr[0] = 0
    indptr[1:] = torch.cumsum(counts, dim=0)
    return indptr, sorted_secondary, order


def _ordered_edge_data(*, edge_weight: Tensor | None, order: Tensor | None, num_edges: int) -> dict[str, Tensor]:
    if edge_weight is None:
        return {}
    weight = torch.as_tensor(edge_weight)
    if order is not None:
        weight = weight.to(device=order.device)
    if weight.dim() == 0 or int(weight.shape[0]) != int(num_edges):
        raise ValueError("edge_weight must have one row per edge")
    if order is not None:
        weight = weight.index_select(0, order)
    return {"w": weight.contiguous()}


__all__ = ["GraphBlock", "GraphFormat", "graph_block_from_coo"]
