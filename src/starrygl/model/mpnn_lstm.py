from __future__ import annotations

import torch
from torch import Tensor, nn

from starrygl.batch import Batch
from starrygl.view import GraphBlock

from ._graph_ops import EdgeScore, GCN, edge_scores, is_edge_task, is_node_task, target_rows
from .base import ModelOutput, StarryModel, StateDelta


class MPNNLSTMLocalCell(nn.Module):
    reads_neighbor_state = False
    state_kind = "node_recurrent"
    state_key = "node_recurrent"

    def __init__(self, hidden_dim: int, *, num_layers: int, in_dim: int | None = None) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.in_dim = int(hidden_dim if in_dim is None else in_dim)
        self.num_layers = max(1, int(num_layers))
        self.gcn = GCN(
            self.in_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            bias=True,
            add_self_loops=True,
        )
        self.rnn1 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.rnn2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

    @property
    def state_dim(self) -> int:
        return 4 * self.hidden_dim

    def materialize(self, blocks, src):
        z = self.gcn(blocks, src["x"])
        final_block = blocks[-1]
        state_like = z.new_zeros((int(z.shape[0]), self.state_dim))
        return {"z": z, "state_like": state_like}, final_block

    @property
    def num_gcn_layers(self) -> int:
        return len(self.gcn.convs)

    def compute_gcn_layer(self, layer: int, blocks, x_src: Tensor) -> Tensor:
        block = blocks[min(int(layer), len(blocks) - 1)]
        return self.gcn.forward_layer(int(layer), block, x_src)

    def finalize_gcn(self, value: Tensor) -> dict[str, Tensor]:
        state_like = value.new_zeros((int(value.shape[0]), self.state_dim))
        return {"z": value, "state_like": state_like}

    def local_forward(self, block: GraphBlock, src, dst):
        del block
        h1, c1, h2, c2 = _unpack_state(dst["h_prev"], self.hidden_dim)
        h1, c1 = self.rnn1(src["z"], (h1, c1))
        h2, c2 = self.rnn2(h1, (h2, c2))
        return torch.cat((h1, c1, h2, c2), dim=-1)

    def embedding_from_state(self, state: Tensor) -> Tensor:
        return state[..., 2 * self.hidden_dim : 3 * self.hidden_dim].contiguous()


class MPNNLSTMModel(StarryModel):
    """Two-layer MPNN-LSTM snapshot model.

    This follows the compact FlareDTDG shape: per window graph convolution
    produces an input embedding, then two LSTM cells scan the temporal window.
    The recurrent state is stored as one node_recurrent tensor with
    ``[h1, c1, h2, c2]`` concatenated on the last dimension.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        num_layers: int = 2,
        node_output_dim: int | None = None,
        persist_state: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = max(1, int(num_layers))
        self.persist_state = bool(persist_state)
        self.input = nn.Identity()
        self.cell = MPNNLSTMLocalCell(self.hidden_dim, num_layers=self.num_layers, in_dim=int(in_dim))
        self.gcn = self.cell.gcn
        self.layers = self.cell.gcn.convs
        self.rnn1 = self.cell.rnn1
        self.rnn2 = self.cell.rnn2
        self.output = nn.Linear(self.hidden_dim, int(out_dim))
        self.node_head = nn.Linear(self.hidden_dim, int(node_output_dim)) if node_output_dim is not None else None
        self.edge_score = EdgeScore(self.hidden_dim)

    @property
    def state_dim(self) -> int:
        return self.cell.state_dim

    @property
    def runtime_cell(self) -> MPNNLSTMLocalCell:
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
        logits = None
        window_logits = ()
        if is_node_task(batch):
            head = self.node_head if self.node_head is not None else self.output
            window_logits = tuple(head(value) for value in scan.window_embeddings)
            logits = window_logits[-1]
        aux = edge_scores(batch, scan.final_block, scan.embeddings, self.edge_score) if is_edge_task(batch) else {}
        if window_logits:
            aux["window_logits"] = window_logits
        return ModelOutput(
            embeddings=scan.embeddings,
            logits=logits,
            state_embeddings=scan.state_embeddings,
            aux=aux,
        )

    def state_update(self, batch: Batch, output: ModelOutput) -> StateDelta | None:
        if not self.persist_state:
            return None
        if output.state_embeddings is None:
            return None
        rows, node_ids = target_rows(batch, int(output.state_embeddings.shape[0]))
        if int(rows.numel()) == 0:
            return None
        rows = rows.to(device=output.state_embeddings.device)
        return StateDelta(
            kind="node_recurrent",
            node_ids=node_ids.to(device=output.state_embeddings.device),
            values=output.state_embeddings.detach().index_select(0, rows),
        )


def _unpack_state(state: Tensor, hidden_dim: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if int(state.shape[-1]) >= 4 * int(hidden_dim):
        return (
            state[..., :hidden_dim].contiguous(),
            state[..., hidden_dim : 2 * hidden_dim].contiguous(),
            state[..., 2 * hidden_dim : 3 * hidden_dim].contiguous(),
            state[..., 3 * hidden_dim : 4 * hidden_dim].contiguous(),
        )
    if int(state.shape[-1]) >= int(hidden_dim):
        h = state[..., :hidden_dim].contiguous()
        zeros = torch.zeros_like(h)
        return h, zeros, h, zeros
    rows = int(state.shape[0])
    zeros = state.new_zeros((rows, int(hidden_dim)))
    return zeros, zeros.clone(), zeros.clone(), zeros.clone()


__all__ = ["MPNNLSTMLocalCell", "MPNNLSTMModel"]
