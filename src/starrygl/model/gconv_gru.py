from __future__ import annotations

from dataclasses import replace
import math

import torch
from torch import Tensor, nn

from starrygl.batch import Batch
from starrygl.utils.index import compact_lookup_rows
from starrygl.view import GraphBlock

from ._graph_ops import EdgeScore, edge_scores, is_edge_task, is_node_task, state_delta_values
from .base import ModelOutput, StarryModel, StateDelta
from .graph_conv import GCNConv


class GConvGRUCell(nn.Module):
    """Graph convolution over neighbor state followed by a local GRU update."""

    reads_neighbor_state = True
    state_kind = "neighbor_recurrent"
    state_key = "neighbor_recurrent"

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.graph_conv = GCNConv(2 * int(hidden_dim), int(hidden_dim))
        self.gru = nn.GRUCell(int(hidden_dim), int(hidden_dim))

    def materialize(self, blocks, src):
        block = blocks[-1]
        message = self.graph_conv(block, torch.cat((src["x"], src["h_prev"]), dim=-1))
        return {"message": message, "state_like": message}, block

    def local_forward(self, block: GraphBlock, src, dst):
        del block
        return self.gru(src["message"], dst["h_prev"])


class GConvGRUModel(StarryModel):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        node_output_dim: int | None = None,
        gamma_boundary_init: float | None = None,
        state_extrapolation: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.state_extrapolation = bool(state_extrapolation)
        self.state_shapes = {"neighbor_recurrent": (self.hidden_dim,)}
        self.input = nn.Linear(int(in_dim), self.hidden_dim)
        self.cell = GConvGRUCell(self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, int(out_dim))
        self.node_head = nn.Linear(self.hidden_dim, int(node_output_dim)) if node_output_dim is not None else None
        self.edge_score = EdgeScore(self.hidden_dim)
        if gamma_boundary_init is not None and not math.isfinite(float(gamma_boundary_init)):
            raise ValueError("gamma_boundary_init must be finite")
        self.gamma_boundary = (
            nn.Parameter(torch.tensor(float(gamma_boundary_init)))
            if self.state_extrapolation and gamma_boundary_init is not None else None
        )

    @property
    def runtime_cell(self) -> GConvGRUCell:
        return self.cell

    @property
    def runtime_input_project(self):
        return self.input

    @property
    def runtime_persist_state(self) -> bool:
        return True

    def runtime_prepare_scan(self, batch: Batch):
        packets = batch.state.get("neighbor_recurrent_snapshots")
        if packets is not None:
            states = []
            for window, packet, rows in zip(batch.iter_blocks(), packets, batch.state["neighbor_recurrent_cold_src_rows"]):
                cold = packet[rows]
                age = int(window[-1].cache["snapshot_id"]) - cold[:, -1:]
                predicted = cold[:, :self.hidden_dim]
                if self.state_extrapolation:
                    scale = self.gamma_boundary.sigmoid() if self.gamma_boundary is not None else 1.0
                    predicted = predicted + scale * age * cold[:, self.hidden_dim:2 * self.hidden_dim]
                states.append(packet[:, :self.hidden_dim].index_copy(0, rows, predicted))
            return replace(batch, state={**batch.state, "neighbor_recurrent_window_state": tuple(states)}), {}
        return batch, {}

    def encode(self, batch: Batch) -> ModelOutput:
        from starrygl.runtime.snapshot.scan import encode_model

        return encode_model(self, batch)

    def runtime_output_from_scan(self, batch: Batch, scan) -> ModelOutput:
        embeddings = scan.embeddings
        logits = None
        window_logits = ()
        if is_node_task(batch):
            head = self.node_head if self.node_head is not None else self.output
            window_logits = tuple(head(value) for value in scan.window_embeddings)
            logits = window_logits[-1]
        aux = edge_scores(batch, scan.final_block, embeddings, self.edge_score) if is_edge_task(batch) else {}
        if window_logits:
            aux["window_logits"] = window_logits
        if scan.state_history:
            aux["snapshot_states"] = scan.state_history
        return ModelOutput(
            embeddings=embeddings,
            logits=logits,
            state_embeddings=scan.state_embeddings,
            aux=aux,
        )

    def state_update(self, batch: Batch, output: ModelOutput) -> StateDelta | None:
        if output.state_embeddings is None:
            return None
        block = tuple(batch.iter_blocks())[-1][-1]
        node_ids = block.dst_nodes.long()
        table_ids = batch.state.get("neighbor_recurrent_node_ids")
        if isinstance(table_ids, Tensor):
            rows = compact_lookup_rows(table_ids, node_ids.to(table_ids.device))
            state = output.state_embeddings.index_select(0, rows.to(output.state_embeddings.device)).detach()
        else:
            node_ids, state = state_delta_values(batch, output.state_embeddings)
        return StateDelta(
            kind="neighbor_recurrent",
            node_ids=node_ids,
            values=state,
            timestamps=_state_timestamps(batch, node_ids, state),
            metadata={"snapshot_states": output.aux["snapshot_states"]} if "snapshot_states" in output.aux else {},
        )


def _state_timestamps(batch: Batch, node_ids: Tensor, values: Tensor) -> Tensor | None:
    block = tuple(batch.iter_blocks())[-1][-1]
    if batch.mode == "snapshot" and "snapshot_id" in block.cache:
        return values.new_full((int(node_ids.numel()),), float(block.cache["snapshot_id"] + 1))
    target = batch.targets.get("task") if isinstance(batch.targets, dict) else None
    target_ids = getattr(target, "target_ids", None)
    target_ts = getattr(target, "target_ts", None)
    if isinstance(target_ids, Tensor) and isinstance(target_ts, Tensor):
        rows = compact_lookup_rows(target_ids.detach(), node_ids.detach())
        if not bool(torch.any(rows < 0).item()):
            return target_ts.detach().to(device=values.device, dtype=torch.float32).index_select(
                0, rows.to(device=values.device)
            )
    snapshot_id = getattr(target, "snapshot_id", None)
    return None if snapshot_id is None else values.new_full((int(node_ids.numel()),), float(snapshot_id))


__all__ = ["GConvGRUCell", "GConvGRUModel"]
