from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, Mapping

import torch

from torch import Tensor
from starrygl.utils.index import compact_lookup_rows, temporal_lookup_rows
from starrygl.view import GraphBlock


TargetKind = Literal["node", "edge"]
NegativeMode = Literal["dst", "src_dst"]


@dataclass(frozen=True)
class NegativeSamplePool:
    mode: NegativeMode
    local_node_ids: Tensor | None = None
    global_node_ids: Tensor | None = None
    local_src_ids: Tensor | None = None
    global_src_ids: Tensor | None = None
    local_dst_ids: Tensor | None = None
    global_dst_ids: Tensor | None = None
    local_prob: float = 1.0
    global_prob: float = 0.0
    local_loss_weight: float | Tensor = 1.0
    global_loss_weight: float | Tensor = 1.0
    loss_weight_fn: Callable[[Tensor, "NegativeSamplePool"], Tensor] | None = None


@dataclass(frozen=True)
class EndpointCollectRoute:
    """Route metadata for collecting endpoint embeddings on edge owners."""

    num_endpoints: int
    groups: Mapping[str, tuple[int, int]]
    local_endpoint_rows: Tensor | None = None
    local_embedding_rows: Tensor | None = None
    remote_endpoint_nodes: Tensor | None = None
    remote_endpoint_rows: Tensor | None = None
    node_dist_index: Tensor | None = None
    route: Any | None = None
    recv_endpoint_rows: Tensor | None = None


@dataclass(frozen=True)
class TargetRoute:
    target_rows: Tensor | None = None
    pos_src_rows: Tensor | None = None
    pos_dst_rows: Tensor | None = None
    neg_src_rows: Tensor | None = None
    neg_dst_rows: Tensor | None = None
    endpoint_collect: EndpointCollectRoute | None = None
    # Internal proof for target_rows; clear if replacing or mutating those rows.
    target_row_bound: int | None = None


@dataclass(frozen=True)
class TaskTarget:
    target_kind: TargetKind
    target_ids: Tensor
    target_ts: Tensor | None = None
    label: Tensor | None = None
    pos_src: Tensor | None = None
    pos_dst: Tensor | None = None
    neg_src: Tensor | None = None
    neg_dst: Tensor | None = None
    neg_loss_weight: Tensor | None = None
    negative_pool: NegativeSamplePool | None = None
    target_route: TargetRoute | None = None
    node_ids: Tensor | None = None
    edge_ids: Tensor | None = None


@dataclass(frozen=True)
class SamplingRoot:
    node_ids: Tensor
    ts: Tensor | None = None
    groups: Mapping[str, tuple[int, int]] | None = None


def build_task_target(
    *,
    target_kind: TargetKind,
    target_ids: Tensor,
    target_ts: Tensor | None = None,
    label: Tensor | None = None,
    pos_src: Tensor | None = None,
    pos_dst: Tensor | None = None,
    neg_src: Tensor | None = None,
    neg_dst: Tensor | None = None,
    neg_loss_weight: Tensor | None = None,
    negative_pool: NegativeSamplePool | None = None,
    target_route: TargetRoute | None = None,
    node_ids: Tensor | None = None,
    edge_ids: Tensor | None = None,
    num_negatives: int = 0,
    generator: torch.Generator | None = None,
) -> TaskTarget:
    """Build the semantic target once; physical row routes are attached later."""

    target = TaskTarget(
        target_kind=target_kind,
        target_ids=target_ids,
        target_ts=target_ts,
        label=label,
        pos_src=pos_src,
        pos_dst=pos_dst,
        neg_src=neg_src,
        neg_dst=neg_dst,
        neg_loss_weight=neg_loss_weight,
        negative_pool=negative_pool,
        target_route=target_route,
        node_ids=target_ids if target_kind == "node" and node_ids is None else node_ids,
        edge_ids=target_ids if target_kind == "edge" and edge_ids is None else edge_ids,
    )
    if target_kind == "edge" and int(target_ids.numel()) and int(num_negatives) > 0 and target.neg_dst is None:
        target = with_negative_samples(target, num_negatives=int(num_negatives), generator=generator)
    return replace(target, negative_pool=None) if target_kind == "edge" and target.neg_dst is not None else target


def build_window_task_target(
    labels: Any,
    window_id: int,
    *,
    target_ts: Tensor | None = None,
    negative_pool: NegativeSamplePool | None = None,
    num_negatives: int = 0,
    generator: torch.Generator | None = None,
) -> TaskTarget:
    """Slice the prepared owner-local task table into its semantic target."""

    payload = labels.task_slice(int(window_id))
    cutoff = payload.get("cutoff_ts", target_ts)
    if labels.task_kind == "node":
        nodes = payload["node_ids"].long()
        return build_task_target(
            target_kind="node",
            target_ids=nodes,
            target_ts=_align_ts(None if cutoff is None else cutoff.to(nodes.device), int(nodes.numel())),
            label=payload.get("label"),
            node_ids=nodes,
        )
    edge_ids = payload["edge_ids"].long()
    return build_task_target(
        target_kind="edge",
        target_ids=edge_ids,
        target_ts=cutoff,
        label=payload.get("label"),
        pos_src=payload["src"].long(),
        pos_dst=payload["dst"].long(),
        edge_ids=edge_ids,
        negative_pool=negative_pool,
        num_negatives=int(num_negatives),
        generator=generator,
    )


def attach_target_route(
    graph: GraphBlock,
    target: TaskTarget,
    *,
    collect_remote_endpoints: bool = False,
    lazy: bool = False,
) -> TaskTarget:
    """Attach physical embedding rows after either graph accessor completes."""

    if not collect_remote_endpoints and target.target_ts is not None and graph.dstdata.get("ts") is not None:
        roots = sampling_roots_from_target(target)
        rows = temporal_lookup_rows(graph.dst_nodes, graph.dstdata["ts"], roots.node_ids, roots.ts)
        if target.target_kind == "node":
            route = TargetRoute(target_rows=rows)
        else:
            route = TargetRoute(**{
                f"{name}_rows": rows[begin:end]
                for name, (begin, end) in roots.groups.items()
            })
        return replace(target, target_route=route)
    if target.target_kind == "node":
        nodes = target.node_ids if target.node_ids is not None else target.target_ids
        return replace(target, target_route=TargetRoute(target_rows=_graph_rows(graph, nodes)))
    if lazy:
        return target
    if collect_remote_endpoints:
        return replace(target, target_route=_endpoint_collect_route(graph, target))
    return replace(
        target,
        target_route=TargetRoute(
            pos_src_rows=_graph_rows(graph, target.pos_src),
            pos_dst_rows=_graph_rows(graph, target.pos_dst),
            neg_src_rows=None if target.neg_src is None else _graph_rows(graph, target.neg_src),
            neg_dst_rows=None if target.neg_dst is None else _graph_rows(graph, target.neg_dst),
        ),
    )


def with_negative_samples(
    target: TaskTarget,
    *,
    num_negatives: int,
    generator: torch.Generator | None = None,
) -> TaskTarget:
    from .negative import materialize_negative_samples

    return materialize_negative_samples(target, num_negatives=int(num_negatives), generator=generator)


def sampling_roots_from_target(target: TaskTarget, *, include_negative: bool = True) -> SamplingRoot:
    if target.target_kind == "node":
        nodes = target.node_ids if target.node_ids is not None else target.target_ids
        return SamplingRoot(
            node_ids=nodes.long(),
            ts=_align_ts(target.target_ts, int(nodes.numel())),
            groups={"target": (0, int(nodes.numel()))},
        )
    if target.pos_src is None or target.pos_dst is None:
        raise ValueError("edge sampling roots require TaskTarget.pos_src and pos_dst")
    pieces = [target.pos_src.long(), target.pos_dst.long()]
    groups = {
        "pos_src": (0, int(target.pos_src.numel())),
        "pos_dst": (int(target.pos_src.numel()), int(target.pos_src.numel()) + int(target.pos_dst.numel())),
    }
    cursor = int(target.pos_src.numel()) + int(target.pos_dst.numel())
    if include_negative and target.neg_src is not None:
        pieces.append(target.neg_src.long())
        groups["neg_src"] = (cursor, cursor + int(target.neg_src.numel()))
        cursor += int(target.neg_src.numel())
    if include_negative and target.neg_dst is not None:
        pieces.append(target.neg_dst.long())
        groups["neg_dst"] = (cursor, cursor + int(target.neg_dst.numel()))
    nodes = torch.cat(pieces, dim=0)
    return SamplingRoot(node_ids=nodes, ts=_repeat_edge_ts(target.target_ts, pieces), groups=groups)


def lookup_target_rows(nodes: Tensor, target_ids: Tensor) -> Tensor:
    if int(target_ids.numel()) == 0:
        return torch.empty(0, dtype=torch.long, device=target_ids.device)
    return compact_lookup_rows(nodes, target_ids)


def _graph_rows(graph: GraphBlock, nodes: Tensor | None) -> Tensor:
    if nodes is None:
        return graph.dst_nodes.new_empty((0,))
    rows = lookup_target_rows(graph.dst_nodes.long(), nodes.long())
    if bool(torch.any(rows < 0).item()) and not torch.equal(graph.src_nodes.long(), graph.dst_nodes.long()):
        rows = lookup_target_rows(graph.src_nodes.long(), nodes.long())
    return rows


def _endpoint_collect_route(graph: GraphBlock, target: TaskTarget) -> TargetRoute:
    pieces = [target.pos_src.long(), target.pos_dst.long()]
    groups = {
        "pos_src": (0, int(target.pos_src.numel())),
        "pos_dst": (int(target.pos_src.numel()), int(target.pos_src.numel()) + int(target.pos_dst.numel())),
    }
    cursor = sum(int(value.numel()) for value in pieces)
    for name, value in (("neg_src", target.neg_src), ("neg_dst", target.neg_dst)):
        if value is not None:
            pieces.append(value.long())
            groups[name] = (cursor, cursor + int(value.numel()))
            cursor += int(value.numel())
    endpoints = torch.cat(pieces, dim=0)
    rows = lookup_target_rows(graph.dst_nodes.long(), endpoints)
    local = torch.nonzero(rows >= 0, as_tuple=True)[0].long()
    remote = torch.nonzero(rows < 0, as_tuple=True)[0].long()
    return TargetRoute(
        endpoint_collect=EndpointCollectRoute(
            num_endpoints=int(endpoints.numel()),
            groups=groups,
            local_endpoint_rows=local,
            local_embedding_rows=rows.index_select(0, local),
            remote_endpoint_nodes=endpoints.index_select(0, remote),
            remote_endpoint_rows=remote,
            node_dist_index=graph.cache.get("node_dist_index"),
        )
    )


def _align_ts(ts: Tensor | None, count: int) -> Tensor | None:
    if ts is None:
        return None
    if int(ts.numel()) == int(count):
        return ts
    if int(ts.numel()) == 1:
        return ts.reshape(1).expand(int(count)).clone()
    return ts[: int(count)]


def _repeat_edge_ts(ts: Tensor | None, pieces: list[Tensor]) -> Tensor | None:
    if ts is None:
        return None
    out = []
    base_count = max(1, int(ts.numel()))
    for piece in pieces:
        repeats = max(1, (int(piece.numel()) + base_count - 1) // base_count)
        out.append(ts.repeat_interleave(repeats)[: int(piece.numel())])
    return torch.cat(out, dim=0) if out else ts.new_empty((0,))


__all__ = [
    "NegativeMode",
    "NegativeSamplePool",
    "EndpointCollectRoute",
    "SamplingRoot",
    "TargetKind",
    "TargetRoute",
    "TaskTarget",
    "attach_target_route",
    "build_task_target",
    "build_window_task_target",
    "sampling_roots_from_target",
    "lookup_target_rows",
    "with_negative_samples",
]
