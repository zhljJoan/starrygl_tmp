from __future__ import annotations

import torch
from torch import Tensor, nn

from starrygl.batch import Batch, EventRows
from starrygl.view import GraphBlock

from ._graph_ops import (
    edge_endpoint_embeddings,
    feature_for_window,
    first_block,
    is_edge_task,
    is_node_task,
    score_endpoint_embeddings,
)
from .base import ModelOutput, StarryModel, StateDelta
from .graph_conv import edge_rows
from .layers import EdgePredictor, TGNMemoryUpdater, TemporalTransformerAttentionLayer
from .tgn_mailbox import fit_last_dim as _fit_last_dim, mailbox_update_values, node_rows as _node_rows


class TGNModel(StarryModel):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        node_output_dim: int | None = None,
        edge_dim: int = 0,
        time_dim: int = 0,
        num_layers: int = 1,
        num_heads: int = 1,
        dropout: float = 0.0,
        att_dropout: float = 0.0,
        memory_update: str = "gru",
        combine_node_feature: bool = True,
        state_compensation: bool = False,
        compensation_num_rows: int | None = None,
        gamma_init: float = 0.5,
        mailbox_size: int = 1,
        transformer_heads: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.num_layers = max(1, int(num_layers))
        self.input = nn.Linear(int(in_dim), self.hidden_dim)
        message_dim = 2 * self.hidden_dim + self.edge_dim
        self.memory = TGNMemoryUpdater(
            memory_dim=self.hidden_dim,
            message_dim=message_dim,
            time_dim=int(time_dim),
            node_dim=self.hidden_dim,
            combine_node_feature=bool(combine_node_feature),
            memory_update=memory_update,
            state_compensation=bool(state_compensation),
            compensation_num_rows=int(compensation_num_rows or 1),
            gamma_init=float(gamma_init),
            mailbox_size=int(mailbox_size),
            transformer_heads=int(transformer_heads),
            dropout=float(dropout),
            att_dropout=float(att_dropout),
        )
        self.layers = nn.ModuleList(
            TemporalTransformerAttentionLayer(
                node_dim=self.hidden_dim,
                edge_dim=self.edge_dim,
                time_dim=int(time_dim),
                num_heads=int(num_heads),
                out_dim=self.hidden_dim,
                dropout=float(dropout),
                att_dropout=float(att_dropout),
                score_scale=1.0,
            )
            for _ in range(self.num_layers)
        )
        del out_dim
        self.node_head = nn.Linear(self.hidden_dim, int(node_output_dim)) if node_output_dim is not None else None
        self.edge_predictor = EdgePredictor(self.hidden_dim)

    def encode(self, batch: Batch) -> ModelOutput:
        z = None
        updated_memory = None
        state_aux: dict[str, Tensor] = {}
        final_block = first_block(batch)
        model_layer_edge_count = 0
        model_layer_block_count = 0
        for window_id, layer_blocks in enumerate(batch.iter_blocks()):
            memory_block = layer_blocks[0]
            x = _input_project(self.input, feature_for_window(batch, "x", window_id).float())
            memory = _state_tensor(batch, "node_memory", x, self.hidden_dim)
            memory_ts = _state_timestamp(batch, "node_memory_ts", x, int(memory.shape[0]))
            node_ts = _node_timestamps(batch, memory_block, x, int(memory.shape[0]))
            mem_input = _memory_input(batch, x, memory, memory_block, self.edge_dim)
            shared_mask, shared_rows = _shared_filter_index(batch, memory_block, int(memory.shape[0]), x.device)
            historical = _state_value_for_layout(
                batch,
                "node_memory_historical",
                int(memory.shape[0]),
            )
            updated_memory, h, state_aux = self.memory(
                node_feat=x,
                memory=memory,
                memory_ts=memory_ts,
                node_ts=node_ts,
                mem_input=mem_input,
                historical_memory=historical,
                shared_mask=shared_mask,
                shared_rows=shared_rows,
                mailbox_ts=_state_value_for_layout(batch, "mailbox_ts", int(memory.shape[0])),
            )
            for layer_id, block in enumerate(layer_blocks):
                layer = self.layers[min(layer_id, self.num_layers - 1)]
                edge_feat = _edge_features(batch, block, h.device, h.dtype, self.edge_dim)
                edge_dt = _edge_delta_t(batch, block, h.device, h.dtype)
                model_layer_edge_count += int(block.edge_ids.numel())
                model_layer_block_count += 1
                h = layer(block, h, edge_feat, edge_dt)
                final_block = block
            z = h
        assert z is not None and updated_memory is not None
        logits = self.node_head(z) if self.node_head is not None and is_node_task(batch) else None
        aux = _edge_scores(batch, final_block, z, self.edge_predictor) if is_edge_task(batch) else {}
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
        if event_ts is not None:
            node_ts = event_ts.to(device=state.device, dtype=timestamps.dtype if timestamps is not None else state.dtype)
        else:
            node_ts = None if timestamps is None else timestamps.index_select(0, rows.to(device=timestamps.device))
        metadata: dict[str, Tensor] = {}
        mailbox_update = mailbox_update_values(batch, block, state, self.edge_dim)
        if mailbox_update is not None:
            mailbox_nodes, mailbox_messages, mailbox_ts = mailbox_update
            metadata["mailbox_nodes"] = mailbox_nodes.detach()
            metadata["mailbox_messages"] = mailbox_messages.detach()
            metadata["mailbox_timestamps"] = mailbox_ts.detach()
        return StateDelta(kind="node_memory", node_ids=node_ids, values=state.detach().index_select(0, rows), timestamps=node_ts, metadata=metadata)

    def clear_state_compensation(self) -> None:
        self.memory.clear_compensation()


def _state_tensor(batch: Batch, key: str, like: Tensor, dim: int) -> Tensor:
    value = _state_value_for_layout(batch, key, int(like.shape[0]))
    if value is None:
        return like.new_zeros((int(like.shape[0]), int(dim)))
    return value.to(device=like.device, dtype=like.dtype)


def _input_project(layer: nn.Linear, x: Tensor) -> Tensor:
    if int(layer.in_features) == 0 and int(x.shape[-1]) == 0:
        if layer.bias is None:
            return x.new_zeros((int(x.shape[0]), int(layer.out_features)))
        return layer.bias.to(device=x.device, dtype=x.dtype).reshape(1, -1).expand(int(x.shape[0]), -1)
    return layer(x)


def _state_timestamp(batch: Batch, key: str, like: Tensor, rows: int) -> Tensor:
    value = _state_value_for_layout(batch, key, int(rows))
    if value is None:
        return like.new_zeros((int(rows),))
    return value.to(device=like.device, dtype=like.dtype)


def _state_value_for_layout(batch: Batch, key: str, rows: int) -> Tensor | None:
    value = batch.state.get(key)
    if not isinstance(value, Tensor):
        return None
    layout_kind = "mailbox" if key.startswith("mailbox") else "node_memory" if key.startswith("node_memory") else key
    inverse = batch.state.get(f"{layout_kind}_layout_inverse")
    if isinstance(inverse, Tensor):
        if int(inverse.numel()) < int(rows):
            raise RuntimeError(f"{layout_kind} compact layout has fewer than {rows} requested rows")
        return value.index_select(0, inverse[: int(rows)].to(device=value.device).long())
    return value[: int(rows)]


def _memory_input(batch: Batch, x: Tensor, memory: Tensor, block: GraphBlock, edge_dim: int) -> Tensor:
    value = batch.state.get("mem_input")
    expected = int(memory.shape[1]) * 2 + int(edge_dim)
    if value is not None:
        value = value[: int(memory.shape[0])].to(device=x.device, dtype=x.dtype).reshape(int(memory.shape[0]), -1)
        return _fit_last_dim(value, expected)
    mailbox = _state_value_for_layout(batch, "mailbox", int(memory.shape[0]))
    if mailbox is not None:
        mailbox = mailbox.to(device=x.device, dtype=x.dtype).reshape(int(memory.shape[0]), -1)
        return _fit_last_dim(mailbox, expected)
    return x.new_zeros((int(memory.shape[0]), expected))


def _edge_features(batch: Batch, block: GraphBlock, device: torch.device, dtype: torch.dtype, edge_dim: int) -> Tensor:
    if int(edge_dim) == 0:
        return torch.empty((int(block.num_edges), 0), dtype=dtype, device=device)
    value = block.edata.get("f")
    if value is None:
        value = block.edata.get("edge_feat")
        if value is None:
            value = batch.features.get("edge_feat")
            if value is None:
                value = batch.features.get("edge")
            if value is None:
                value = batch.features.get("e")
        if isinstance(value, (tuple, list)):
            value = value[0]
        if isinstance(value, Tensor) and int(value.shape[0]) != int(block.num_edges):
            edge_rows = block.edge_ids.to(device=value.device).long()
            if int(edge_rows.numel()) and int(edge_rows.max().item()) < int(value.shape[0]):
                value = value.index_select(0, edge_rows)
            else:
                value = None
    if value is None:
        if int(block.num_edges) == 0:
            return torch.empty((0, int(edge_dim)), dtype=dtype, device=device)
        if batch.features.get("pos_edge_feat") is not None:
            return torch.zeros((int(block.num_edges), int(edge_dim)), dtype=dtype, device=device)
        raise RuntimeError(
            "edge_dim > 0 but sampled block edge features are missing. "
            "Expected block.edata['edge_feat'] or a compatible batch edge feature tensor."
        )
    value = value.to(device=device, dtype=dtype).reshape(int(block.num_edges), -1)
    return _fit_last_dim(value, int(edge_dim))


def _edge_delta_t(batch: Batch, block: GraphBlock, device: torch.device, dtype: torch.dtype) -> Tensor:
    value = block.edata.get("dt")
    if value is None:
        value = block.edata.get("delta_t")
    if value is None:
        value = batch.features.get("edge_dt")
        if isinstance(value, (tuple, list)):
            value = value[0]
    if value is None:
        return torch.zeros((int(block.num_edges),), dtype=dtype, device=device)
    if int(value.shape[0]) != int(block.num_edges):
        value = value.index_select(0, block.edge_ids.to(device=value.device).long())
    return value.to(device=device, dtype=dtype).reshape(-1)


def _memory_update_timestamps(batch: Batch, block: GraphBlock, like: Tensor, rows: int) -> Tensor:
    mailbox_ts = _state_value_for_layout(batch, "mailbox_ts", int(rows))
    if mailbox_ts is not None:
        mailbox_ts = mailbox_ts.to(device=like.device, dtype=like.dtype).reshape(int(rows), -1)
        return mailbox_ts.amax(dim=1)
    return _node_timestamps(batch, block, like, rows)


def _node_timestamps(batch: Batch, block: GraphBlock, like: Tensor, rows: int) -> Tensor:
    src_ts = block.srcdata.get("ts")
    if src_ts is not None:
        src_ts = src_ts[: int(rows)].to(device=like.device, dtype=like.dtype).reshape(-1)
        if int(src_ts.numel()) == int(rows):
            return src_ts
    out = like.new_zeros((int(rows),))
    edge_ts = block.edata.get("ts")
    if edge_ts is None:
        edge_ts = batch.features.get("edge_ts")
        if edge_ts is None:
            edge_ts = batch.features.get("ts")
        if isinstance(edge_ts, (tuple, list)):
            edge_ts = edge_ts[0]
        if isinstance(edge_ts, Tensor) and int(edge_ts.shape[0]) != int(block.num_edges):
            edge_ts = edge_ts.index_select(0, block.edge_ids.to(device=edge_ts.device).long())
    if edge_ts is not None and int(block.num_edges):
        edge_ts = edge_ts.to(device=like.device, dtype=like.dtype).reshape(-1)
        src, dst = edge_rows(block, device=like.device)
        out.scatter_reduce_(0, src.clamp_max(int(rows) - 1), edge_ts, reduce="amax", include_self=True)
        out.scatter_reduce_(0, dst.clamp_max(int(rows) - 1), edge_ts, reduce="amax", include_self=True)
        return out
    target = batch.targets.get("task") if isinstance(batch.targets, dict) else None
    ts = getattr(target, "target_ts", None)
    if ts is not None and int(ts.numel()):
        out.fill_(float(ts.reshape(-1)[-1].detach().cpu().item()))
    return out


def _shared_filter_index(batch: Batch, block: GraphBlock, rows: int, device: torch.device) -> tuple[Tensor | None, Tensor | None]:
    mask = batch.state.get("node_memory_shared_mask")
    if mask is not None:
        mask = mask[: int(rows)].to(device=device, dtype=torch.bool)
    elif "shared_mask" in block.srcdata:
        mask = block.srcdata["shared_mask"][: int(rows)].to(device=device, dtype=torch.bool)
    else:
        return None, None
    shared_rows = batch.state.get("node_memory_shared_rows")
    if shared_rows is None and "shared_rows" in block.srcdata:
        shared_rows = block.srcdata["shared_rows"]
    if shared_rows is None:
        shared_rows = torch.arange(int(mask.sum().item()), dtype=torch.long, device=device)
    else:
        shared_rows = shared_rows.to(device=device).long()
        if int(shared_rows.numel()) == int(rows):
            shared_rows = shared_rows[mask]
    return mask, shared_rows


def _event_memory_values(batch: Batch, block: GraphBlock, embeddings: Tensor) -> tuple[Tensor, Tensor, Tensor | None] | None:
    events = batch.targets.get("events") if isinstance(batch.targets, dict) else None
    if not isinstance(events, EventRows):
        return None
    state_write_mask = events.state_write_mask
    if state_write_mask is not None:
        mask = state_write_mask.to(device=embeddings.device).reshape(-1).long()
        pos_src = events.src.to(device=embeddings.device).long()
        pos_dst = events.dst.to(device=embeddings.device).long()
        src_keep = (mask & 1) != 0
        dst_keep = (mask & 2) != 0
        nodes = torch.cat((pos_src[src_keep], pos_dst[dst_keep]), dim=0)
        if int(nodes.numel()) == 0:
            return None
        rows = _node_rows(block, nodes, embeddings.device)
        ts = events.ts
        event_ts = None
        if ts is not None:
            ts = ts.to(device=embeddings.device, dtype=embeddings.dtype).reshape(-1)
            event_ts = torch.cat((ts[src_keep], ts[dst_keep]), dim=0)
        return _deduplicate_state_rows(nodes, rows, event_ts)
    nodes = torch.unique(torch.cat((events.src.long(), events.dst.long()), dim=0), sorted=True)
    if int(nodes.numel()) == 0:
        return None
    rows = _node_rows(block, nodes.to(device=embeddings.device), embeddings.device)
    return nodes.to(device=embeddings.device), rows, None


def _deduplicate_state_rows(nodes: Tensor, rows: Tensor, ts: Tensor | None) -> tuple[Tensor, Tensor, Tensor | None]:
    unique_nodes, inverse = torch.unique(nodes, sorted=True, return_inverse=True)
    if int(unique_nodes.numel()) == int(nodes.numel()):
        return nodes, rows, ts
    pos = torch.arange(int(nodes.numel()), dtype=torch.long, device=nodes.device)
    latest = torch.full((int(unique_nodes.numel()),), -1, dtype=torch.long, device=nodes.device)
    latest.scatter_reduce_(0, inverse, pos, reduce="amax", include_self=True)
    out_rows = rows.index_select(0, latest.to(device=rows.device))
    out_ts = None if ts is None else ts.index_select(0, latest.to(device=ts.device))
    return unique_nodes, out_rows, out_ts


def _edge_scores(batch: Batch, block: GraphBlock, embeddings: Tensor, predictor: EdgePredictor) -> dict[str, Tensor]:
    target = batch.targets.get("task") if isinstance(batch.targets, dict) else None
    if target is None or target.pos_src is None or target.pos_dst is None:
        return {}
    endpoints = edge_endpoint_embeddings(batch, block, embeddings)
    if endpoints is None:
        return {}
    return score_endpoint_embeddings(target, endpoints, predictor=predictor)


__all__ = ["TGNModel"]
