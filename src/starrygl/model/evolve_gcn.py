from __future__ import annotations

import torch
from torch import Tensor, nn

from starrygl.batch import Batch
from starrygl.view import GraphBlock

from ._graph_ops import (
    EdgeScore,
    edge_scores,
    is_edge_task,
    is_node_task,
)
from .base import ModelOutput, StarryModel, StateDelta
from .graph_conv import (
    dgl_gcn_aggregate as _dgl_gcn_aggregate,
    edge_gcn_norm as _edge_gcn_norm,
    edge_rows,
    self_gcn_norm as _self_gcn_norm,
    sparse_gcn_aggregate as _sparse_gcn_aggregate,
)


class MatGRUCell(nn.Module):
    """Matrix GRU used by the FlareDTDG EvolveGCN implementation."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        gate_dim = int(in_dim) + int(hidden_dim)
        self.update = nn.Linear(gate_dim, int(hidden_dim))
        self.reset = nn.Linear(gate_dim, int(hidden_dim))
        self.candidate = nn.Linear(gate_dim, int(hidden_dim))

    def forward(self, previous: Tensor, context: Tensor) -> Tensor:
        expanded = context.reshape(1, -1).expand(int(previous.shape[0]), -1)
        joined = torch.cat((expanded, previous), dim=1)
        update = torch.sigmoid(self.update(joined))
        reset = torch.sigmoid(self.reset(joined))
        candidate = torch.tanh(self.candidate(torch.cat((expanded, reset * previous), dim=1)))
        return update * previous + (1.0 - update) * candidate


class EvolveGCNOConv(nn.Module):
    """Normalized graph operator with an externally evolved weight matrix."""

    def __init__(self, *, add_self_loops: bool = False, normalize: bool = True) -> None:
        super().__init__()
        self.add_self_loops = bool(add_self_loops)
        self.normalize = bool(normalize)

    def forward(self, block: GraphBlock, x_src: Tensor, weight: Tensor) -> Tensor:
        if x_src.dim() != 2 or weight.dim() != 2 or int(x_src.shape[1]) != int(weight.shape[0]):
            raise ValueError("EvolveGCN graph features and weight matrix dimensions do not match")
        src, dst = edge_rows(block, device=x_src.device)
        num_dst = int(block.num_dst or block.dst_nodes.numel())
        norm = None
        if self.normalize and int(src.numel()) > 0:
            norm = _edge_gcn_norm(
                block,
                src=src,
                dst=dst,
                x_src=x_src,
                num_dst=num_dst,
                add_self_loops=self.add_self_loops,
            )
        aggregated = None
        if int(src.numel()) > 0:
            aggregated = _dgl_gcn_aggregate(
                block,
                x_src,
                src=src,
                dst=dst,
                norm=norm,
                num_dst=num_dst,
            )
            if aggregated is None:
                aggregated = _sparse_gcn_aggregate(
                    block,
                    x_src,
                    src=src,
                    dst=dst,
                    norm=norm,
                    num_dst=num_dst,
                )
            if aggregated is None:
                messages = x_src.index_select(0, src)
                if norm is not None:
                    messages = messages * norm.reshape(-1, 1)
                aggregated = x_src.new_zeros((num_dst, int(x_src.shape[1])))
                aggregated.index_add_(0, dst, messages)
        if aggregated is None:
            aggregated = x_src.new_zeros((num_dst, int(x_src.shape[1])))
        if self.add_self_loops and int(x_src.shape[0]) >= num_dst:
            self_messages = x_src[:num_dst]
            if self.normalize:
                self_messages = self_messages * _self_gcn_norm(
                    block,
                    x_src=x_src,
                    num_dst=num_dst,
                ).reshape(-1, 1)
            aggregated = aggregated + self_messages
        return aggregated @ weight


class EvolveGCNModel(StarryModel):
    """EvolveGCN with globally consistent snapshot contexts."""

    runtime_batch_local_state = True
    state_kind = "model_recurrent"
    state_key = "model_recurrent"

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        num_layers: int = 1,
        node_output_dim: int | None = None,
        persist_state: bool = True,
        pool_mode: str = "mean",
        input_transform: str = "none",
        add_self_loops: bool = False,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if int(num_layers) != 1:
            raise ValueError("EvolveGCN currently supports one spatial layer, matching FlareDTDG")
        mode = str(pool_mode).strip().lower()
        if mode not in {"mean", "max"}:
            raise ValueError("EvolveGCN pool_mode must be 'mean' or 'max'")
        transform = str(input_transform).strip().lower()
        if transform not in {"none", "log1p"}:
            raise ValueError("EvolveGCN input_transform must be 'none' or 'log1p'")
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = 1
        self.num_spatial_layers = 1
        self.persist_state = bool(persist_state)
        self.pool_mode = mode
        self.input_transform = transform
        self.initial_weight = nn.Parameter(torch.empty(self.in_dim, self.hidden_dim))
        self.mat_gru = MatGRUCell(self.in_dim, self.hidden_dim)
        self.pool_proj = nn.Linear(self.in_dim, self.in_dim)
        self.conv_layer = EvolveGCNOConv(add_self_loops=add_self_loops, normalize=normalize)
        self.output = nn.Linear(self.hidden_dim, int(out_dim))
        self.node_head = nn.Linear(self.hidden_dim, int(node_output_dim)) if node_output_dim is not None else None
        self.edge_score = EdgeScore(self.hidden_dim)
        self.reset_parameters()

    @property
    def state_dim(self) -> int:
        return self.in_dim * self.hidden_dim

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.initial_weight)

    @property
    def runtime_cell(self):
        return self

    @property
    def runtime_input_project(self):
        return self._transform_input

    @property
    def runtime_persist_state(self) -> bool:
        return self.persist_state

    @property
    def context_reduction(self) -> str:
        return "sum_count" if self.pool_mode == "mean" else "max"

    def context(self, block: GraphBlock, x: Tensor) -> tuple[Tensor, Tensor | None]:
        count = min(int(block.num_dst or block.dst_nodes.numel()), int(x.shape[0]))
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        owned = x[:count].detach().to(dtype=dtype)
        if self.pool_mode == "max":
            return (
                owned.amax(dim=0)
                if count
                else x.new_full((int(x.shape[1]),), float("-inf"), dtype=dtype),
                None,
            )
        summary = owned.sum(dim=0) if count else x.new_zeros((int(x.shape[1]),), dtype=dtype)
        return summary, summary.new_tensor(count)

    def initial_state(self, batch: Batch) -> Tensor:
        return _model_weight(batch, self.initial_weight, self.state_dim, persist_state=self.persist_state)

    def advance_state(self, previous_state: Tensor, update_input: Tensor) -> Tensor:
        context = self.pool_proj(
            update_input.to(device=self.initial_weight.device, dtype=self.initial_weight.dtype)
        )
        return self.mat_gru(previous_state, context)

    def spatial(
        self,
        layer: int,
        block: GraphBlock,
        x: Tensor,
        dependency: Tensor | None,
    ) -> Tensor:
        if int(layer) != 0 or dependency is None:
            raise ValueError("EvolveGCN requires its one evolved-weight spatial dependency")
        return self.conv_layer(
            block,
            x.to(device=dependency.device, dtype=dependency.dtype),
            dependency,
        )

    def encode(self, batch: Batch) -> ModelOutput:
        from starrygl.runtime.snapshot.scan import encode_model

        return encode_model(self, batch)

    def runtime_output_from_scan(self, batch: Batch, scan) -> ModelOutput:
        embeddings = scan.embeddings
        logits = None
        aux = {}
        if is_node_task(batch):
            head = self.node_head if self.node_head is not None else self.output
            window_logits = tuple(head(value) for value in scan.window_embeddings)
            logits = window_logits[-1]
            aux["window_logits"] = window_logits
        elif is_edge_task(batch):
            aux = edge_scores(batch, scan.final_block, embeddings, self.edge_score)
        return ModelOutput(
            embeddings=embeddings,
            logits=logits,
            state_embeddings=scan.state_embeddings.reshape(1, self.state_dim),
            aux=aux,
        )

    def _transform_input(self, value: Tensor) -> Tensor:
        if self.input_transform == "log1p":
            if bool(torch.any(value < 0).item()):
                raise ValueError("EvolveGCN log1p input_transform requires non-negative features")
            return torch.log1p(value)
        return value

    def state_update(self, batch: Batch, output: ModelOutput) -> StateDelta | None:
        del batch
        if not self.persist_state or output.state_embeddings is None:
            return None
        return StateDelta(
            kind="model_recurrent",
            node_ids=torch.zeros(1, dtype=torch.long, device=output.state_embeddings.device),
            values=output.state_embeddings.detach().reshape(1, self.state_dim),
            metadata={
                "local_owner_only": True,
                "state_layout": "model_weight",
                "weight_shape": (self.in_dim, self.hidden_dim),
                "globally_replicated": True,
            },
        )


def _model_weight(
    batch: Batch,
    initial_weight: Tensor,
    state_dim: int,
    *,
    persist_state: bool,
) -> Tensor:
    if not persist_state:
        return initial_weight
    value = batch.state.get("model_recurrent")
    if isinstance(value, Tensor):
        if int(value.numel()) != int(state_dim):
            raise ValueError(
                "model_recurrent state has the wrong size: "
                f"expected {int(state_dim)}, got {int(value.numel())}"
            )
        return value.to(device=initial_weight.device, dtype=initial_weight.dtype).reshape_as(initial_weight)
    return initial_weight


__all__ = ["EvolveGCNOConv", "EvolveGCNModel", "MatGRUCell"]
