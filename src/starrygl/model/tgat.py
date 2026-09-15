from __future__ import annotations

import torch
from torch import Tensor, nn
from starrygl.batch import Batch
from starrygl.utils.index import compact_lookup_rows, temporal_lookup_rows
from starrygl.view.base import mfg_chain_matches

from ._graph_ops import EdgeScore, edge_scores, feature_for_window, first_block, is_edge_task, is_node_task
from .base import ModelOutput, StarryModel
from .layers import TemporalTransformerAttentionLayer
from .tgn import _edge_delta_t, _edge_features


class TGATModel(StarryModel):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        node_output_dim: int | None = None,
        edge_dim: int = 0,
        time_dim: int = 100,
        num_layers: int = 1,
        num_heads: int = 1,
        dropout: float = 0.1,
        att_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.num_layers = max(1, int(num_layers))
        self.layers = nn.ModuleList(
            TemporalTransformerAttentionLayer(
                node_dim=self.in_dim if layer == 0 else self.hidden_dim,
                edge_dim=self.edge_dim,
                time_dim=int(time_dim),
                num_heads=int(num_heads),
                out_dim=self.hidden_dim,
                dropout=float(dropout),
                att_dropout=float(att_dropout),
            )
            for layer in range(self.num_layers)
        )
        node_dim = int(out_dim) if node_output_dim is None else int(node_output_dim)
        self.node_head = nn.Linear(self.hidden_dim, node_dim)
        self.edge_score = EdgeScore(self.hidden_dim)

    def encode(self, batch: Batch) -> ModelOutput:
        h: Tensor | None = None
        model_layer_edge_count = 0
        model_layer_block_count = 0
        for window_id, layer_blocks in enumerate(batch.iter_blocks()):
            base = feature_for_window(batch, "x", window_id).float()
            base_nodes = (feature_for_window(batch, "node_ids", window_id)
                          if "node_ids" in batch.features else layer_blocks[0].src_nodes)
            blocks = tuple(layer_blocks)
            if len(blocks) > 1 and not mfg_chain_matches(blocks) and not mfg_chain_matches(tuple(reversed(blocks))):
                h, h_nodes, final_block, edge_count, block_count = _encode_compact_frontiers(
                    blocks=blocks,
                    base=base,
                    base_nodes=base_nodes,
                    layers=self.layers,
                    edge_dim=self.edge_dim,
                )
                model_layer_edge_count += int(edge_count)
                model_layer_block_count += int(block_count)
            else:
                h = base
                h_nodes = base_nodes
                h_ts = None
                ordered_blocks = _attention_blocks(blocks)
                for layer_id, block in enumerate(ordered_blocks):
                    h = _align_to_src_nodes(h, h_nodes, block.src_nodes, base=base, base_nodes=base_nodes,
                                           h_ts=h_ts, src_ts=block.srcdata.get("ts"))
                    layer = self.layers[min(layer_id, self.num_layers - 1)]
                    edge_feat = _edge_features(batch, block, h.device, h.dtype, self.edge_dim)
                    edge_dt = _edge_delta_t(batch, block, h.device, h.dtype)
                    model_layer_edge_count += int(block.edge_ids.numel())
                    model_layer_block_count += 1
                    h = layer(block, h, edge_feat, edge_dt)
                    h_nodes = block.dst_nodes
                    h_ts = block.dstdata.get("ts")
                    final_block = block
        if h is None:
            block = first_block(batch)
            h = feature_for_window(batch, "x", 0).float()
            h_nodes = block.src_nodes
            for layer_id, layer in enumerate(self.layers):
                h = _align_to_src_nodes(h, h_nodes, block.src_nodes)
                edge_feat = _edge_features(batch, block, h.device, h.dtype, self.edge_dim)
                edge_dt = _edge_delta_t(batch, block, h.device, h.dtype)
                model_layer_edge_count += int(block.edge_ids.numel())
                model_layer_block_count += 1
                h = layer(block, h, edge_feat, edge_dt)
                h_nodes = block.dst_nodes
            final_block = block
        logits = None
        if is_node_task(batch):
            logits = self.node_head(h) if self.node_head is not None else None
        aux = edge_scores(batch, final_block, h, self.edge_score) if is_edge_task(batch) else {}
        aux["model_layer_edge_count"] = model_layer_edge_count
        aux["model_layer_block_count"] = model_layer_block_count
        return ModelOutput(
            embeddings=h,
            logits=logits,
            aux=aux,
        )


def _attention_blocks(blocks: tuple) -> tuple:
    if len(blocks) <= 1:
        return blocks
    if mfg_chain_matches(blocks):
        return blocks
    reversed_blocks = tuple(reversed(blocks))
    if mfg_chain_matches(reversed_blocks):
        return reversed_blocks
    return reversed_blocks


def _encode_compact_frontiers(
    *,
    blocks: tuple,
    base: Tensor,
    base_nodes: Tensor,
    layers: nn.ModuleList,
    edge_dim: int,
) -> tuple[Tensor, Tensor, object, int, int]:
    depth = min(len(blocks), len(layers))
    prev = []
    edge_count = 0
    block_count = 0
    for block in blocks[:depth]:
        src_h = _align_base_to_src(base, base_nodes, block.src_nodes, like=base)
        edge_feat = _edge_features_for_block(block, src_h, int(edge_dim))
        edge_dt = _edge_delta_for_block(block, src_h)
        edge_count += int(block.edge_ids.numel())
        block_count += 1
        out = layers[0](block, src_h, edge_feat, edge_dt)
        prev.append((block, out))
    for layer_id in range(1, depth):
        current = []
        for block_id, block in enumerate(blocks[: depth - layer_id]):
            src_h = _merge_frontier_embeddings(block, prev[block_id], prev[block_id + 1])
            edge_feat = _edge_features_for_block(block, src_h, int(edge_dim))
            edge_dt = _edge_delta_for_block(block, src_h)
            edge_count += int(block.edge_ids.numel())
            block_count += 1
            out = layers[min(layer_id, len(layers) - 1)](block, src_h, edge_feat, edge_dt)
            current.append((block, out))
        prev = current
    return prev[0][1], prev[0][0].dst_nodes, blocks[0], edge_count, block_count


def _merge_frontier_embeddings(block, *frontiers) -> Tensor:
    src_nodes = block.src_nodes
    values = [value for _, value in frontiers if int(value.numel()) > 0]
    if not values:
        raise RuntimeError("TGAT compact frontier has no embeddings to merge")
    like = values[0]
    out = like.new_zeros((int(src_nodes.numel()), int(like.shape[-1])))
    target = src_nodes.to(device=like.device).long()
    if int(target.numel()) == 0:
        return out
    for source_block, value in frontiers:
        nodes = source_block.dst_nodes
        if int(nodes.numel()) == 0:
            continue
        source = nodes.to(device=like.device).long()
        rows = (temporal_lookup_rows(source, source_block.dstdata["ts"], target, block.srcdata["ts"])
                if "ts" in source_block.dstdata and "ts" in block.srcdata
                else compact_lookup_rows(source, target))
        valid = rows >= 0
        if bool(torch.any(valid).item()):
            keep = valid.nonzero(as_tuple=True)[0]
            out.index_copy_(0, keep, value.to(device=like.device).index_select(0, rows.index_select(0, keep)))
    return out


def _edge_features_for_block(block, h: Tensor, edge_dim: int) -> Tensor:
    value = block.edata.get("edge_feat")
    if value is None:
        value = block.edata.get("edge")
    if value is None:
        if int(edge_dim) == 0 or int(block.num_edges) == 0:
            return h.new_empty((int(block.num_edges), int(edge_dim)))
        raise RuntimeError(
            "edge_dim > 0 but TGAT block edge features are missing. "
            "Expected block.edata['edge_feat'] from sampled edge feature materialization."
        )
    value = value.to(device=h.device, dtype=h.dtype).reshape(int(block.num_edges), -1)
    if int(value.shape[1]) == int(edge_dim):
        return value
    if int(value.shape[1]) > int(edge_dim):
        return value[:, : int(edge_dim)]
    return torch.cat((value, h.new_zeros((int(value.shape[0]), int(edge_dim) - int(value.shape[1])))), dim=1)


def _edge_delta_for_block(block, h: Tensor) -> Tensor:
    value = block.edata.get("dt")
    if value is None:
        value = block.edata.get("delta_t")
    if value is None:
        return h.new_zeros((int(block.num_edges),))
    return value.to(device=h.device, dtype=h.dtype).reshape(-1)


def _align_to_src_nodes(
    h: Tensor,
    h_nodes: Tensor,
    src_nodes: Tensor,
    *,
    base: Tensor | None = None,
    base_nodes: Tensor | None = None,
    h_ts: Tensor | None = None,
    src_ts: Tensor | None = None,
) -> Tensor:
    temporal = h_ts is not None and src_ts is not None
    if torch.equal(h_nodes.to(device=src_nodes.device).long(), src_nodes.long()) and (
            not temporal or torch.equal(h_ts, src_ts)):
        return h
    from_nodes = h_nodes.to(device=h.device).long()
    to_nodes = src_nodes.to(device=h.device).long()
    if int(from_nodes.numel()) == 0 or int(to_nodes.numel()) == 0:
        return h.new_zeros((int(to_nodes.numel()), int(h.shape[-1])))
    rows = (temporal_lookup_rows(from_nodes, h_ts, to_nodes, src_ts) if temporal
            else compact_lookup_rows(from_nodes, to_nodes))
    out = _align_base_to_src(base, base_nodes, src_nodes, like=h) if base is not None and base_nodes is not None else h.new_zeros((int(to_nodes.numel()), int(h.shape[-1])))
    valid = rows >= 0
    if bool(torch.any(valid).item()):
        keep = valid.nonzero(as_tuple=True)[0]
        out.index_copy_(0, keep, h.index_select(0, rows.index_select(0, keep)))
    return out


def _align_base_to_src(base: Tensor, base_nodes: Tensor, src_nodes: Tensor, *, like: Tensor) -> Tensor:
    if int(base.shape[-1]) != int(like.shape[-1]):
        return like.new_zeros((int(src_nodes.numel()), int(like.shape[-1])))
    nodes = base_nodes.to(device=like.device).long()
    target = src_nodes.to(device=like.device).long()
    if int(nodes.numel()) == 0 or int(target.numel()) == 0:
        return like.new_zeros((int(target.numel()), int(like.shape[-1])))
    rows = compact_lookup_rows(nodes, target)
    out = like.new_zeros((int(target.numel()), int(like.shape[-1])))
    valid = rows >= 0
    if bool(torch.any(valid).item()):
        keep = valid.nonzero(as_tuple=True)[0]
        out.index_copy_(0, keep, base.to(device=like.device, dtype=like.dtype).index_select(0, rows.index_select(0, keep)))
    return out


__all__ = ["TGATModel"]
