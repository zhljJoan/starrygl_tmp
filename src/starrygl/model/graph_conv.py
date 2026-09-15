from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from starrygl.view import GraphBlock


class MeanGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, include_self: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.self_linear = nn.Linear(in_dim, out_dim, bias=False) if include_self else None

    def forward(self, block: GraphBlock, x_src: Tensor) -> Tensor:
        src, dst = edge_rows(block, x_src.device)
        num_dst = int(block.num_dst or block.dst_nodes.numel())
        out = x_src.new_zeros((num_dst, self.linear.out_features))
        if src.numel():
            out.index_add_(0, dst, self.linear(x_src[src]))
            degree = torch.bincount(dst, minlength=num_dst).to(x_src).clamp_min_(1).unsqueeze(1)
            out.div_(degree)
        if self.self_linear is not None and x_src.shape[0] >= num_dst:
            out.add_(self.self_linear(x_src[:num_dst]))
        return out


class GCNConv(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool = True,
        add_self_loops: bool = True,
        improved: bool = False,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.add_self_loops = bool(add_self_loops)
        self.improved = bool(improved)
        self.normalize = bool(normalize)
        self.weight = nn.Parameter(torch.empty(self.in_dim, self.out_dim))
        self.bias = nn.Parameter(torch.empty(self.out_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, block: GraphBlock, x_src: Tensor) -> Tensor:
        src, dst = edge_rows(block, x_src.device)
        num_dst = int(block.num_dst or block.dst_nodes.numel())
        transform_first = self.out_dim <= self.in_dim
        support = x_src @ self.weight if transform_first else x_src
        norm = (
            edge_gcn_norm(block, src, dst, x_src, num_dst, self.add_self_loops)
            if self.normalize and src.numel()
            else None
        )
        out = None
        if self.add_self_loops and support.shape[0] >= num_dst and block.cache.get("use_dgl_gcn", False):
            self_weight = (
                self_gcn_norm(block, x_src, num_dst) * (2.0 if self.improved else 1.0)
                if self.normalize else x_src.new_ones(num_dst)
            )
            out = dgl_gcn_aggregate(
                block, support, src, dst, norm, num_dst, self_weight=self_weight,
            )
        if out is None:
            out = _aggregate(block, support, src, dst, norm, num_dst)
            if self.add_self_loops and support.shape[0] >= num_dst:
                self_values = support[:num_dst]
                if self.normalize:
                    weight = 2.0 if self.improved else 1.0
                    self_values = self_values * (self_gcn_norm(block, x_src, num_dst).unsqueeze(1) * weight)
                out.add_(self_values)
        if not transform_first:
            out = out @ self.weight
        if self.bias is not None:
            out.add_(self.bias)
        return out


class GCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        num_layers: int = 2,
        bias: bool = True,
        add_self_loops: bool = True,
        improved: bool = False,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = max(1, int(num_layers))
        self.convs = nn.ModuleList(
            GCNConv(
                in_dim if layer == 0 else out_dim,
                out_dim,
                bias=bias,
                add_self_loops=add_self_loops,
                improved=improved,
                normalize=normalize,
            )
            for layer in range(self.num_layers)
        )

    def forward_layer(self, layer: int, block: GraphBlock, x_src: Tensor) -> Tensor:
        value = self.convs[layer](block, x_src)
        return F.relu(value) if layer + 1 < len(self.convs) else value

    def forward(
        self,
        blocks: Sequence[GraphBlock],
        x_src: Tensor,
        *,
        materialize_between_layers: Callable[[GraphBlock, Tensor, int], Tensor] | None = None,
    ) -> Tensor:
        if not blocks:
            raise ValueError("GCN requires at least one graph block")
        value = x_src
        for layer in range(len(self.convs)):
            value = self.forward_layer(layer, blocks[min(layer, len(blocks) - 1)], value)
            if materialize_between_layers is not None and layer + 1 < len(self.convs):
                value = materialize_between_layers(blocks[min(layer, len(blocks) - 1)], value, layer)
        return value


def edge_rows(block: GraphBlock, device: torch.device) -> tuple[Tensor, Tensor]:
    if block.row is not None and block.col is not None:
        return block.row.to(device).long(), block.col.to(device).long()
    key = f"edge_rows:{device}"
    cached = block.cache.get(key)
    if isinstance(cached, tuple) and len(cached) == 2:
        return cached
    if block.indptr is None or block.indices is None:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    indptr, indices = block.indptr.to(device).long(), block.indices.to(device).long()
    primary = torch.repeat_interleave(torch.arange(indptr.numel() - 1, device=device), indptr[1:] - indptr[:-1])
    rows = (indices, primary) if block.is_csc else (primary, indices)
    block.cache[key] = rows
    return rows


def edge_gcn_norm(
    block: GraphBlock,
    src: Tensor,
    dst: Tensor,
    x_src: Tensor,
    num_dst: int,
    add_self_loops: bool,
) -> Tensor:
    cached = block.edata.get("gcn_norm")
    if isinstance(cached, Tensor) and cached.numel() == src.numel():
        return cached.to(x_src)
    weight = block.edata.get("w")
    weight = weight.to(x_src).reshape(-1) if isinstance(weight, Tensor) and weight.numel() == src.numel() else x_src.new_ones(src.numel())
    src_degree = x_src.new_zeros(x_src.shape[0]).index_add_(0, src, weight)
    dst_degree = x_src.new_zeros(num_dst).index_add_(0, dst, weight)
    if add_self_loops and x_src.shape[0] >= num_dst:
        src_degree[:num_dst].add_(1)
        dst_degree.add_(1)
    return (weight * torch.rsqrt(src_degree[src] * dst_degree[dst])).nan_to_num_(0)


def self_gcn_norm(block: GraphBlock, x_src: Tensor, num_dst: int) -> Tensor:
    cached = block.edata.get("self_gcn_norm")
    if isinstance(cached, Tensor) and cached.numel() == num_dst:
        return cached.to(x_src)
    src, dst = edge_rows(block, x_src.device)
    weight = block.edata.get("w")
    weight = weight.to(x_src).reshape(-1) if isinstance(weight, Tensor) and weight.numel() == src.numel() else x_src.new_ones(src.numel())
    src_degree, dst_degree = x_src.new_ones(x_src.shape[0]), x_src.new_ones(num_dst)
    if src.numel():
        src_degree.index_add_(0, src, weight)
        dst_degree.index_add_(0, dst, weight)
    return torch.rsqrt(src_degree[:num_dst] * dst_degree).nan_to_num_(0)


def _aggregate(
    block: GraphBlock,
    values: Tensor,
    src: Tensor,
    dst: Tensor,
    norm: Tensor | None,
    num_dst: int,
) -> Tensor:
    if src.numel():
        out = dgl_gcn_aggregate(block, values, src, dst, norm, num_dst)
        out = out if out is not None else sparse_gcn_aggregate(block, values, src, dst, norm, num_dst)
        if out is not None:
            return out
    out = values.new_zeros((num_dst, values.shape[1]))
    if src.numel():
        messages = values[src]
        out.index_add_(0, dst, messages if norm is None else messages * norm.unsqueeze(1))
    return out


def sparse_gcn_aggregate(
    block: GraphBlock,
    values: Tensor,
    src: Tensor,
    dst: Tensor,
    norm: Tensor | None,
    num_dst: int,
) -> Tensor | None:
    if not block.cache.get("use_sparse_tensor_gcn", False):
        return None
    try:
        from torch_sparse import SparseTensor, matmul
    except ImportError:
        return None
    key = f"sparse_gcn:{values.device}:{values.dtype}:{src.numel()}:{num_dst}:{values.shape[0]}"
    adjacency = block.cache.get(key)
    if adjacency is None:
        edge_weight = norm.to(values) if norm is not None else values.new_ones(src.numel())
        adjacency = SparseTensor(row=dst, col=src, value=edge_weight, sparse_sizes=(num_dst, values.shape[0]))
        block.cache[key] = adjacency
    return matmul(adjacency, values)


def dgl_gcn_aggregate(
    block: GraphBlock,
    values: Tensor,
    src: Tensor,
    dst: Tensor,
    norm: Tensor | None,
    num_dst: int,
    *,
    self_weight: Tensor | None = None,
) -> Tensor | None:
    if not block.cache.get("use_dgl_gcn", False):
        return None
    try:
        import dgl
    except ImportError:
        return None
    key = f"dgl_gcn:{values.device}:{src.numel()}:{num_dst}:{values.shape[0]}"
    if self_weight is not None:
        key += ":self"
    graph = block.cache.get(key)
    if graph is None:
        if self_weight is not None:
            diagonal = torch.arange(num_dst, device=src.device)
            graph_src, graph_dst = torch.cat((src, diagonal)), torch.cat((dst, diagonal))
        else:
            graph_src, graph_dst = src, dst
        graph = dgl.graph((graph_src, graph_dst), num_nodes=max(values.shape[0], num_dst), device=values.device)
        block.cache[key] = graph
    if self_weight is not None:
        edge_weight = norm.to(values) if norm is not None else values.new_ones(src.numel())
        norm = torch.cat((edge_weight, self_weight.to(values)))
    out = dgl.ops.copy_u_sum(graph, values) if norm is None else dgl.ops.u_mul_e_sum(graph, values, norm.to(values))
    return out[:num_dst]


NormalizedGraphConv = GCNConv
NormalizedGCN = GCN


__all__ = ["GCN", "GCNConv", "MeanGraphConv", "NormalizedGCN", "NormalizedGraphConv", "edge_rows"]
