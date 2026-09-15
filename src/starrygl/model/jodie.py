from __future__ import annotations

from torch import Tensor, nn

from starrygl.batch import Batch

from ._graph_ops import feature_for_window, first_block, is_edge_task, is_node_task
from .base import ModelOutput, StarryModel, StateDelta
from .layers import EdgePredictor, IdentityNormLayer, JODIETimeEmbedding, TGNMemoryUpdater
from .tgn import (
    _edge_scores,
    _event_memory_values,
    _input_project,
    _memory_input,
    _memory_update_timestamps,
    _shared_filter_index,
    _state_tensor,
    _state_timestamp,
    _state_value_for_layout,
)
from .tgn_mailbox import mailbox_update_values as _mailbox_update_values


class JODIEModel(StarryModel):
    """JODIE-style CTDG cell on materialized StarryGL event blocks."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        node_output_dim: int | None = None,
        edge_dim: int = 0,
        time_dim: int = 100,
        dropout: float = 0.0,
        state_compensation: bool = False,
        compensation_num_rows: int | None = None,
        gamma_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.input = nn.Linear(int(in_dim), self.hidden_dim)
        self.memory = TGNMemoryUpdater(
            memory_dim=self.hidden_dim,
            message_dim=2 * self.hidden_dim + self.edge_dim,
            time_dim=int(time_dim),
            node_dim=self.hidden_dim,
            combine_node_feature=True,
            memory_update="rnn",
            state_compensation=bool(state_compensation),
            compensation_num_rows=int(compensation_num_rows or 1),
            gamma_init=float(gamma_init),
        )
        self.norm = IdentityNormLayer(self.hidden_dim)
        self.time_embedding = JODIETimeEmbedding(self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        del out_dim
        self.node_head = nn.Linear(self.hidden_dim, int(node_output_dim)) if node_output_dim is not None else None
        self.edge_predictor = EdgePredictor(self.hidden_dim)

    def encode(self, batch: Batch) -> ModelOutput:
        z = None
        updated_memory = None
        state_aux: dict[str, Tensor] = {}
        block = first_block(batch)
        model_layer_edge_count = 0
        model_layer_block_count = 0
        for window_id, layer_blocks in enumerate(batch.iter_blocks()):
            block = layer_blocks[0]
            model_layer_edge_count += int(block.edge_ids.numel())
            model_layer_block_count += 1
            x = _input_project(self.input, feature_for_window(batch, "x", window_id).float())
            memory = _state_tensor(batch, "node_memory", x, self.hidden_dim)
            memory_ts = _state_timestamp(batch, "node_memory_ts", x, int(memory.shape[0]))
            node_ts = _memory_update_timestamps(batch, block, x, int(memory.shape[0]))
            mem_input = _memory_input(batch, x, memory, block, self.edge_dim)
            shared_mask, shared_rows = _shared_filter_index(batch, block, int(memory.shape[0]), x.device)
            updated_memory, h, state_aux = self.memory(
                node_feat=x,
                memory=memory,
                memory_ts=memory_ts,
                node_ts=node_ts,
                mem_input=mem_input,
                historical_memory=_state_value_for_layout(
                    batch,
                    "node_memory_historical",
                    int(memory.shape[0]),
                ),
                shared_mask=shared_mask,
                shared_rows=shared_rows,
                mailbox_ts=_state_value_for_layout(batch, "mailbox_ts", int(memory.shape[0])),
            )
            h = self.time_embedding(self.norm(h), memory_ts, node_ts)
            z = self.dropout(h)
        assert z is not None and updated_memory is not None
        logits = self.node_head(z) if self.node_head is not None and is_node_task(batch) else None
        aux = _edge_scores(batch, block, z, self.edge_predictor) if is_edge_task(batch) else {}
        aux.update(state_aux)
        aux["model_layer_edge_count"] = model_layer_edge_count
        aux["model_layer_block_count"] = model_layer_block_count
        return ModelOutput(embeddings=z, logits=logits, state_embeddings=updated_memory, aux=aux)

    def state_update(self, batch: Batch, output: ModelOutput) -> StateDelta | None:
        block = first_block(batch)
        state = output.state_embeddings
        if state is None:
            return None
        values = _event_memory_values(batch, block, state)
        if values is None:
            return None
        node_ids, rows, event_ts = values
        timestamps = output.aux.get("memory_node_ts")
        node_ts = event_ts if event_ts is not None else None if timestamps is None else timestamps.index_select(0, rows.to(device=timestamps.device))
        metadata: dict[str, Tensor] = {}
        mailbox_update = _mailbox_update_values(batch, block, state, self.edge_dim)
        if mailbox_update is not None:
            mailbox_nodes, mailbox_messages, mailbox_ts = mailbox_update
            metadata["mailbox_nodes"] = mailbox_nodes.detach()
            metadata["mailbox_messages"] = mailbox_messages.detach()
            metadata["mailbox_timestamps"] = mailbox_ts.detach()
        return StateDelta(kind="node_memory", node_ids=node_ids, values=state.detach().index_select(0, rows), timestamps=node_ts, metadata=metadata)

    def clear_state_compensation(self) -> None:
        self.memory.clear_compensation()


__all__ = ["JODIEModel"]
