from __future__ import annotations

import torch
from torch import Tensor, nn

from starrygl.batch import Batch
from starrygl.utils.index import compact_lookup_rows

from ._graph_ops import feature_for_window, first_block, is_edge_task, is_node_task
from .base import ModelOutput, StarryModel, StateDelta
from .graph_conv import edge_rows
from .layers import EdgePredictor, IdentityNormLayer, TGNMemoryUpdater
from .tgn_mailbox import mailbox_update_values
from .tgn import (
    _edge_scores,
    _event_memory_values,
    _fit_last_dim,
    _input_project,
    _memory_update_timestamps,
    _shared_filter_index,
    _state_tensor,
    _state_timestamp,
    _state_value_for_layout,
)


class APANModel(StarryModel):
    """APAN-style CTDG memory model with neighbor-delivered mailbox messages."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        node_output_dim: int | None = None,
        edge_dim: int = 0,
        time_dim: int = 100,
        mailbox_size: int = 10,
        transformer_heads: int = 2,
        dropout: float = 0.1,
        att_dropout: float = 0.1,
        state_compensation: bool = False,
        compensation_num_rows: int | None = None,
        gamma_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.mailbox_size = max(1, int(mailbox_size))
        self.input = nn.Linear(int(in_dim), self.hidden_dim)
        self.message_dim = 2 * self.hidden_dim + self.edge_dim
        self.memory = TGNMemoryUpdater(
            memory_dim=self.hidden_dim,
            message_dim=self.mailbox_size * self.message_dim,
            time_dim=int(time_dim),
            node_dim=self.hidden_dim,
            combine_node_feature=False,
            memory_update="transformer",
            state_compensation=bool(state_compensation),
            compensation_num_rows=int(compensation_num_rows or 1),
            gamma_init=float(gamma_init),
            mailbox_size=self.mailbox_size,
            transformer_heads=int(transformer_heads),
            dropout=float(dropout),
            att_dropout=float(att_dropout),
        )
        self.norm = IdentityNormLayer(self.hidden_dim)
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
            mem_input = _apan_memory_input(batch, x, memory, self.message_dim, self.mailbox_size)
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
            z = self.norm(h)
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
        mailbox_update = _apan_mailbox_update_values(batch, block, state, self.edge_dim)
        if mailbox_update is not None:
            mailbox_nodes, mailbox_messages, mailbox_ts = mailbox_update
            metadata["mailbox_nodes"] = mailbox_nodes.detach()
            metadata["mailbox_messages"] = mailbox_messages.detach()
            metadata["mailbox_timestamps"] = mailbox_ts.detach()
        return StateDelta(kind="node_memory", node_ids=node_ids, values=state.detach().index_select(0, rows), timestamps=node_ts, metadata=metadata)

    def clear_state_compensation(self) -> None:
        self.memory.clear_compensation()


def _apan_memory_input(batch: Batch, like: Tensor, memory: Tensor, message_dim: int, mailbox_size: int) -> Tensor:
    mailbox = _state_value_for_layout(batch, "mailbox", int(memory.shape[0]))
    expected = int(message_dim) * int(mailbox_size)
    if mailbox is not None:
        mailbox = mailbox.to(device=like.device, dtype=like.dtype).reshape(int(memory.shape[0]), -1)
        return _fit_last_dim(mailbox, expected)
    return like.new_zeros((int(memory.shape[0]), expected))


def _apan_mailbox_update_values(batch: Batch, block, embeddings: Tensor, edge_dim: int) -> tuple[Tensor, Tensor, Tensor] | None:
    base = mailbox_update_values(batch, block, embeddings, edge_dim)
    if base is None:
        return None
    base_nodes, base_messages, base_ts = _latest_mail(*base)
    src, dst = edge_rows(block, device=embeddings.device)
    if int(src.numel()) == 0:
        return base_nodes, base_messages, base_ts
    source_nodes = block.src_nodes.to(device=embeddings.device).long().index_select(0, src)
    source_rows = compact_lookup_rows(base_nodes, source_nodes)
    keep = source_rows >= 0
    if not bool(keep.any().item()):
        return base_nodes, base_messages, base_ts
    source_rows = source_rows[keep]
    neighbor_nodes = block.dst_nodes.to(device=embeddings.device).long().index_select(0, dst[keep])
    return _latest_mail(
        torch.cat((base_nodes, neighbor_nodes)),
        torch.cat((base_messages, base_messages.index_select(0, source_rows.to(base_messages.device)))),
        torch.cat((base_ts, base_ts.index_select(0, source_rows.to(base_ts.device)))),
    )


def _latest_mail(nodes: Tensor, messages: Tensor, timestamps: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    nodes, timestamps = nodes.long(), timestamps.reshape(-1)
    unique, inverse = torch.unique(nodes, sorted=True, return_inverse=True)
    if int(unique.numel()) == int(nodes.numel()):
        return nodes, messages, timestamps
    latest_ts = timestamps.new_full((int(unique.numel()),), -torch.inf)
    latest_ts.scatter_reduce_(0, inverse.to(timestamps.device), timestamps, reduce="amax", include_self=True)
    pos = torch.arange(int(nodes.numel()), dtype=torch.long, device=nodes.device)
    candidates = pos.masked_fill(~(timestamps == latest_ts.index_select(0, inverse.to(timestamps.device))).to(pos.device), -1)
    latest = torch.full((int(unique.numel()),), -1, dtype=torch.long, device=nodes.device)
    latest.scatter_reduce_(0, inverse, candidates, reduce="amax", include_self=True)
    return unique, messages.index_select(0, latest.to(messages.device)), timestamps.index_select(0, latest.to(timestamps.device))


__all__ = ["APANModel"]
