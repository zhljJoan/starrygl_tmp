from __future__ import annotations

import torch
from torch import nn

from starrygl.batch import Batch
from starrygl.view import GraphBlock

from ._graph_ops import EdgeScore, GCN, edge_scores, is_edge_task, is_node_task, state_delta_values
from .base import ModelOutput, StarryModel, StateDelta


class TGCNLocalCell(nn.Module):
    reads_neighbor_state = False
    state_kind = "node_recurrent"
    state_key = "node_recurrent"

    def __init__(self, hidden_dim: int, *, num_layers: int, in_dim: int | None = None) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.in_dim = int(hidden_dim if in_dim is None else in_dim)
        self.gcn = GCN(
            self.in_dim,
            int(hidden_dim) * 3,
            num_layers=max(1, int(num_layers)),
            bias=False,
            add_self_loops=True,
        )
        self.update_gate = nn.Linear(int(hidden_dim) * 2, int(hidden_dim))
        self.reset_gate = nn.Linear(int(hidden_dim) * 2, int(hidden_dim))
        self.candidate = nn.Linear(int(hidden_dim) * 2, int(hidden_dim))

    def materialize(self, blocks, src):
        gates = self.gcn(blocks, src["x"])
        return {"gates": gates, "state_like": gates[..., : self.hidden_dim]}, blocks[-1]

    @property
    def num_gcn_layers(self) -> int:
        return len(self.gcn.convs)

    def compute_gcn_layer(self, layer: int, blocks, x_src: torch.Tensor) -> torch.Tensor:
        block = blocks[min(int(layer), len(blocks) - 1)]
        return self.gcn.forward_layer(int(layer), block, x_src)

    def finalize_gcn(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"gates": value, "state_like": value[..., : self.hidden_dim]}

    def local_forward(self, block: GraphBlock, src, dst):
        del block
        gates = src["gates"]
        h_prev = dst["h_prev"]
        update_in, reset_in, cand_in = gates.chunk(3, dim=-1)
        update = _linear_cat(self.update_gate, update_in, h_prev).sigmoid()
        reset = _linear_cat(self.reset_gate, reset_in, h_prev).sigmoid()
        candidate = _linear_cat(self.candidate, cand_in, reset * h_prev).tanh()
        return update * h_prev + (1.0 - update) * candidate


class TGCNModel(StarryModel):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        num_layers: int = 2,
        node_output_dim: int | None = None,
        persist_state: bool = False,
        replay_snapshot_window: bool = False,
        node_regression_residual: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = max(1, int(num_layers))
        self.persist_state = bool(persist_state)
        self.replay_snapshot_window = bool(replay_snapshot_window)
        self.node_regression_residual = node_regression_residual
        self.input = nn.Identity()
        self.cell = TGCNLocalCell(int(hidden_dim), num_layers=self.num_layers, in_dim=int(in_dim))
        self.gcn = self.cell.gcn
        self.layers = self.cell.gcn.convs
        self.update_gate = self.cell.update_gate
        self.reset_gate = self.cell.reset_gate
        self.candidate = self.cell.candidate
        self.output = nn.Linear(int(hidden_dim), int(out_dim))
        self.node_head = nn.Linear(int(hidden_dim), int(node_output_dim)) if node_output_dim is not None else None
        self.edge_score = EdgeScore(int(hidden_dim))

    @property
    def runtime_cell(self) -> TGCNLocalCell:
        return self.cell

    @property
    def runtime_input_project(self):
        return self.input

    @property
    def runtime_persist_state(self) -> bool:
        return self.persist_state

    def encode(self, batch: Batch) -> ModelOutput:
        from starrygl.runtime.snapshot.scan import encode_model

        return encode_model(self, batch)

    def runtime_output_from_scan(self, batch: Batch, scan) -> ModelOutput:
        h = scan.embeddings
        logits = None
        window_logits = ()
        if is_node_task(batch):
            head = self.node_head if self.node_head is not None else self.output
            window_logits = tuple(head(value) for value in scan.window_embeddings)
            logits = window_logits[-1]
        aux = edge_scores(batch, scan.final_block, h, self.edge_score) if is_edge_task(batch) else {}
        if window_logits:
            aux["window_logits"] = window_logits
        return ModelOutput(
            embeddings=h,
            logits=logits,
            state_embeddings=scan.state_embeddings,
            aux=aux,
        )

    def state_update(self, batch: Batch, output: ModelOutput) -> StateDelta | None:
        if not self.persist_state:
            return None
        if output.state_embeddings is None:
            return None
        values = state_delta_values(batch, output.state_embeddings)
        if values is None:
            return None
        node_ids, state = values
        return StateDelta(kind="node_recurrent", node_ids=node_ids, values=state)


def _linear_cat(linear: nn.Linear, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return linear(torch.cat((left, right), dim=-1))


__all__ = ["TGCNLocalCell", "TGCNModel"]
