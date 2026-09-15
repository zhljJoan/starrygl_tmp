from __future__ import annotations

import os
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from starrygl.batch import Batch
from starrygl.task import TaskTarget
from starrygl.utils.index import compact_lookup_rows
from starrygl.view import GraphBlock

from .graph_conv import GCN, GCNConv, MeanGraphConv, NormalizedGCN, NormalizedGraphConv




class EdgeScore(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(int(dim) * 2, 1)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        return self.score(torch.cat((left, right), dim=-1)).squeeze(-1)


def first_block(batch: Batch) -> GraphBlock:
    if batch.blocks is not None:
        return batch.blocks[-1][-1]
    if batch.graph is not None:
        return batch.graph
    raise ValueError("batch has no graph block")


def window_blocks(batch: Batch) -> tuple[GraphBlock, ...]:
    if batch.blocks is not None:
        return tuple(window[-1] for window in batch.blocks)
    return (first_block(batch),)


def feature_for_window(batch: Batch, key: str, window_id: int, fallback_key: str = "feat") -> Tensor:
    values = batch.features.get(key)
    if values is None:
        values = batch.features.get(fallback_key)
    if values is None:
        raise KeyError(key)
    if isinstance(values, (tuple, list)):
        return values[int(window_id)]
    if isinstance(values, Tensor) and values.dim() >= 3:
        return values[int(window_id)]
    return values


def target(batch: Batch) -> Any | None:
    value = batch.targets.get("task") if isinstance(batch.targets, Mapping) else None
    return value if hasattr(value, "target_kind") and hasattr(value, "target_ids") else None


def is_edge_task(batch: Batch) -> bool:
    tgt = target(batch)
    return bool(tgt is not None and getattr(tgt, "target_kind", None) == "edge")


def is_node_task(batch: Batch) -> bool:
    tgt = target(batch)
    return bool(tgt is not None and getattr(tgt, "target_kind", None) == "node")


def target_rows(batch: Batch, output_rows: int) -> tuple[Tensor, Tensor]:
    tgt = target(batch)
    if tgt is None:
        rows = torch.arange(int(output_rows), dtype=torch.long)
        return rows, rows
    if getattr(tgt, "target_kind", None) == "edge":
        block = first_block(batch)
        count = int(min(output_rows, block.dst_nodes.numel()))
        rows = torch.arange(count, dtype=torch.long, device=block.dst_nodes.device)
        return rows, block.dst_nodes.long()[:count]
    route = tgt.target_route
    if route is not None and route.target_rows is not None:
        rows = route.target_rows.long()
        return rows, tgt.target_ids.long()
    rows = torch.arange(int(min(output_rows, tgt.target_ids.numel())), dtype=torch.long, device=tgt.target_ids.device)
    return rows, tgt.target_ids.long()[: int(rows.numel())]


def edge_scores(batch: Batch, block: GraphBlock, embeddings: Tensor, scorer: EdgeScore) -> dict[str, Tensor]:
    tgt = target(batch)
    if tgt is None or tgt.pos_src is None or tgt.pos_dst is None:
        return {}
    if tgt.target_route is not None and tgt.target_route.endpoint_collect is not None:
        return {}
    endpoints = edge_endpoint_embeddings(batch, block, embeddings)
    return {} if endpoints is None else score_endpoint_embeddings(tgt, endpoints, scorer=scorer)


def score_endpoint_embeddings(
    target: TaskTarget,
    endpoints: Mapping[str, Tensor],
    *,
    scorer: Any | None = None,
    predictor: Any | None = None,
) -> dict[str, Tensor]:
    neg_dst = endpoints.get("neg_dst")
    if callable(predictor):
        mode = "triplet"
        if target.neg_src is not None and neg_dst is not None:
            mode = "src_dst" if int(target.neg_src.numel()) == int(neg_dst.shape[0]) else "triplet"
        samples = 1 if neg_dst is None else max(1, int(neg_dst.shape[0]) // max(1, int(endpoints["pos_src"].shape[0])))
        pos, neg = predictor(
            endpoints["pos_src"],
            endpoints["pos_dst"],
            endpoints.get("neg_src"),
            neg_dst,
            neg_samples=samples,
            mode=mode,
        )
        return {"pos_score": pos, **({} if neg is None else {"neg_score": neg})}
    if not callable(scorer):
        return {}
    out = {"pos_score": scorer(endpoints["pos_src"], endpoints["pos_dst"])}
    if neg_dst is not None:
        neg_src = endpoints.get("neg_src")
        if neg_src is None:
            repeats = max(1, int(neg_dst.shape[0]) // max(1, int(endpoints["pos_src"].shape[0])))
            neg_src = endpoints["pos_src"].repeat_interleave(repeats, dim=0)
        out["neg_score"] = scorer(neg_src, neg_dst)
    return out


def edge_endpoint_embeddings(batch: Batch, block: GraphBlock, embeddings: Tensor) -> dict[str, Tensor] | None:
    tgt = target(batch)
    if tgt is None or tgt.pos_src is None or tgt.pos_dst is None:
        return None
    route = tgt.target_route
    collect = None if route is None else route.endpoint_collect
    if collect is not None:
        return None
    pos_src_rows, pos_dst_rows = _positive_rows(tgt, block, embeddings.device)
    _debug_check_endpoint_rows("pos_src", pos_src_rows, embeddings)
    _debug_check_endpoint_rows("pos_dst", pos_dst_rows, embeddings)
    out = {
        "pos_src": embeddings.index_select(0, pos_src_rows),
        "pos_dst": embeddings.index_select(0, pos_dst_rows),
    }
    if tgt.neg_dst is not None:
        neg_src = tgt.neg_src if tgt.neg_src is not None else tgt.pos_src.repeat_interleave(max(1, int(tgt.neg_dst.numel()) // max(1, int(tgt.pos_src.numel()))))
        neg_src_rows = (
            route.neg_src_rows.to(device=embeddings.device).long()
            if route is not None and route.neg_src_rows is not None and tgt.neg_src is not None
            else _repeat_positive_rows_for_negatives(route, tgt, embeddings.device)
            if tgt.neg_src is None
            else _node_rows(block, neg_src.to(device=embeddings.device), embeddings.device)
        )
        neg_dst_rows = (
            route.neg_dst_rows.to(device=embeddings.device).long()
            if route is not None and route.neg_dst_rows is not None
            else _node_rows(block, tgt.neg_dst.to(device=embeddings.device), embeddings.device)
        )
        _debug_check_endpoint_rows("neg_src", neg_src_rows, embeddings)
        _debug_check_endpoint_rows("neg_dst", neg_dst_rows, embeddings)
        if tgt.neg_src is not None:
            out["neg_src"] = embeddings.index_select(0, neg_src_rows)
        out["neg_dst"] = embeddings.index_select(0, neg_dst_rows)
    return out


def _repeat_positive_rows_for_negatives(route: Any, target: TaskTarget, device: torch.device) -> Tensor:
    if route is not None and route.pos_src_rows is not None and target.neg_dst is not None and target.pos_src is not None:
        rows = route.pos_src_rows.to(device=device).long()
        repeats = max(1, int(target.neg_dst.numel()) // max(1, int(target.pos_src.numel())))
        return rows.repeat_interleave(repeats)[: int(target.neg_dst.numel())]
    if target.pos_src is None or target.neg_dst is None:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.empty(int(target.neg_dst.numel()), dtype=torch.long, device=device).fill_(-1)


def _debug_check_endpoint_rows(name: str, rows: Tensor | None, embeddings: Tensor) -> None:
    if os.environ.get("STARRYGL_DEBUG_ENDPOINT_ROWS") != "1" or rows is None or int(rows.numel()) == 0:
        return
    rows = rows.long()
    low = int(rows.min().item())
    high = int(rows.max().item())
    limit = int(embeddings.shape[0])
    if low < 0 or high >= limit:
        raise RuntimeError(
            f"invalid endpoint rows for {name}: min={low}, max={high}, "
            f"embedding_rows={limit}, count={int(rows.numel())}"
        )


def state_delta_values(batch: Batch, embeddings: Tensor) -> tuple[Tensor, Tensor] | None:
    rows, node_ids = target_rows(batch, int(embeddings.shape[0]))
    if int(rows.numel()) == 0:
        return None
    rows = rows.to(device=embeddings.device)
    return node_ids.to(device=embeddings.device), embeddings.detach().index_select(0, rows)


def event_state_values(batch: Batch, block: GraphBlock, embeddings: Tensor) -> tuple[Tensor, Tensor] | None:
    tgt = target(batch)
    if tgt is None or tgt.pos_src is None or tgt.pos_dst is None:
        return None
    nodes = torch.unique(torch.cat((tgt.pos_src.long(), tgt.pos_dst.long()), dim=0), sorted=True)
    if int(nodes.numel()) == 0:
        return None
    rows = _node_rows(block, nodes.to(device=embeddings.device), embeddings.device)
    return nodes.to(device=embeddings.device), embeddings.detach().index_select(0, rows)




def _positive_rows(tgt: TaskTarget, block: GraphBlock, device: torch.device) -> tuple[Tensor, Tensor]:
    route = tgt.target_route
    if route is not None and route.pos_src_rows is not None and route.pos_dst_rows is not None:
        return route.pos_src_rows.to(device=device).long(), route.pos_dst_rows.to(device=device).long()
    return _node_rows(block, tgt.pos_src.to(device=device), device), _node_rows(block, tgt.pos_dst.to(device=device), device)


def _node_rows(block: GraphBlock, node_ids: Tensor, device: torch.device) -> Tensor:
    ids = node_ids.long()
    nodes = block.dst_nodes if int(block.dst_nodes.numel()) else block.src_nodes
    nodes = nodes.to(device=device).long()
    if int(nodes.numel()) == 0:
        return ids.new_empty((0,))
    out = compact_lookup_rows(nodes, ids)
    if bool(torch.any(out < 0).item()) and not torch.equal(block.src_nodes.to(device=device), nodes):
        nodes = block.src_nodes.to(device=device).long()
        out = compact_lookup_rows(nodes, ids)
    if bool(torch.any(out < 0).item()):
        raise KeyError("target node is not materialized in graph block")
    return out


__all__ = [
    "EdgeScore",
    "GCN",
    "GCNConv",
    "MeanGraphConv",
    "NormalizedGCN",
    "NormalizedGraphConv",
    "edge_scores",
    "event_state_values",
    "feature_for_window",
    "first_block",
    "is_edge_task",
    "is_node_task",
    "score_endpoint_embeddings",
    "state_delta_values",
    "target_rows",
    "window_blocks",
]
