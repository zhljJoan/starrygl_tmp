from __future__ import annotations

import torch
from torch import nn

from .gconv_gru import GConvGRUModel
from .graph_conv import edge_rows


class DiffusionGraphConv(nn.Module):
    """Bidirectional random-walk diffusion, including the zero-hop term."""

    def __init__(self, in_dim: int, out_dim: int, diffusion_steps: int = 2):
        super().__init__()
        if diffusion_steps not in (1, 2):
            raise ValueError("DCRNN currently supports diffusion_steps=1 or 2")
        self.weight = nn.Parameter(torch.empty(2, diffusion_steps, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, block, values):
        count = int(block.num_dst or block.dst_nodes.numel())
        result = values[:count] @ (self.weight[0, 0] + self.weight[1, 0])
        if self.weight.shape[1] == 2:
            import dgl

            key = f"dcrnn_diffusion:{values.device}:{values.dtype}"
            cached = block.cache.get(key)
            prepared = block.cache.get("diffusion")
            if cached is None and prepared is not None:
                src, dst = edge_rows(block, values.device)
                graph = dgl.graph((src, dst), num_nodes=values.shape[0], device=values.device)
                reverse = dgl.graph((prepared["reverse_row"], prepared["reverse_col"]),
                                    num_nodes=values.shape[0], device=values.device)
                cached = graph, reverse, prepared["forward_norm"], prepared["reverse_norm"]
                block.cache[key] = cached
            if cached is None:
                if block.num_src != block.num_dst or not torch.equal(block.src_nodes, block.dst_nodes):
                    raise NotImplementedError("DCRNN requires a complete snapshot; partial reverse routes are not implemented")
                src, dst = edge_rows(block, values.device)
                keep = src != dst
                nodes = torch.arange(values.shape[0], device=values.device)
                src, dst = torch.cat((src[keep], nodes)), torch.cat((dst[keep], nodes))
                weight = block.edata.get("w")
                weight = values.new_ones(keep.numel()) if weight is None else weight.to(values).reshape(-1)
                weight = torch.cat((weight[keep], values.new_ones(nodes.numel())))
                graph = dgl.graph((src, dst), num_nodes=values.shape[0], device=values.device)
                reverse = dgl.reverse(graph)
                out_degree = values.new_zeros(values.shape[0]).index_add_(0, src, weight)
                in_degree = values.new_zeros(values.shape[0]).index_add_(0, dst, weight)
                cached = graph, reverse, weight / out_degree[src], weight / in_degree[dst]
                block.cache[key] = cached
            graph, reverse, norm_out, norm_in = cached
            # Project first: DGL avoids allocating an edge-by-feature message tensor.
            forward_values, reverse_values = values @ self.weight[0, 1], values @ self.weight[1, 1]
            result = result + dgl.ops.u_mul_e_sum(graph, forward_values, norm_out)[:count]
            result = result + dgl.ops.u_mul_e_sum(reverse, reverse_values, norm_in)[:count]
            if prepared is not None:
                result = result + forward_values[:count] * prepared["forward_self"].unsqueeze(1)
                result = result + reverse_values[:count] * prepared["reverse_self"].unsqueeze(1)
        return result + self.bias


class DCRNNCell(nn.Module):
    reads_neighbor_state = True
    state_kind = "neighbor_recurrent"
    state_key = "neighbor_recurrent"

    def __init__(self, hidden_dim: int, diffusion_steps: int = 2):
        super().__init__()
        self.gates = DiffusionGraphConv(2 * hidden_dim, 2 * hidden_dim, diffusion_steps)
        self.candidate = DiffusionGraphConv(2 * hidden_dim, hidden_dim, diffusion_steps)

    def materialize(self, blocks, src):
        block = blocks[-1]
        x, previous = src["x"], src["h_prev"]
        update, reset = self.materialize_gates(block, x, previous)
        candidate = self.materialize_candidate(block, x, previous, reset)
        return {"update": update, "candidate": candidate, "state_like": candidate}, block

    def materialize_gates(self, block, x, previous):
        return self.gates(block, torch.cat((x, previous), -1)).sigmoid().chunk(2, -1)

    def materialize_candidate(self, block, x, previous, reset):
        return self.candidate(block, torch.cat((x, reset * previous), -1)).tanh()

    def local_forward(self, block, src, dst):
        return src["update"] * dst["h_prev"] + (1 - src["update"]) * src["candidate"]


class DCRNNModel(GConvGRUModel):
    """Diffusion cell with the shared StarryGL recurrent output/state contract."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *,
                 diffusion_steps: int = 2, node_output_dim: int | None = None,
                 gamma_boundary_init: float | None = None, state_extrapolation: bool = True):
        super().__init__(in_dim, hidden_dim, out_dim, node_output_dim=node_output_dim,
                         gamma_boundary_init=gamma_boundary_init,
                         state_extrapolation=state_extrapolation)
        self.cell = DCRNNCell(hidden_dim, diffusion_steps)
