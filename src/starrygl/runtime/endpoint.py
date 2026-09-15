from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import torch
from torch import Tensor

from starrygl.batch import Batch
from starrygl.model import ModelOutput
from starrygl.model._graph_ops import score_endpoint_embeddings
from starrygl.runtime.comm import Route, all_to_all_counts
from starrygl.task import EndpointCollectRoute, TaskTarget
from starrygl.utils.index import compact_lookup_rows
from starrygl.utils.route import dist_part
from starrygl.view import GraphBlock


def materialize_endpoint_output(
    model: Any,
    batch: Batch,
    output: ModelOutput,
    *,
    comm: Any | None = None,
) -> ModelOutput:
    """Fulfill a task endpoint dependency after node embeddings are produced."""

    target = _target(batch)
    collect = _collect(target)
    if collect is None:
        return output
    if output.embeddings is None:
        raise ValueError("endpoint collection requires model embeddings")
    endpoints = _materialize(batch, output.embeddings, collect, comm=comm)
    scores = score_endpoint_embeddings(
        target,
        endpoints,
        scorer=getattr(model, "edge_score", None),
        predictor=getattr(model, "edge_predictor", None),
    )
    return replace(output, aux={**output.aux, **scores}) if scores else output


def materialize_endpoint_embeddings(batch: Batch, embeddings: Tensor) -> dict[str, Tensor] | None:
    target = _target(batch)
    collect = _collect(target)
    return None if collect is None else _materialize(batch, embeddings, collect)


def _materialize(
    batch: Batch,
    embeddings: Tensor,
    collect: EndpointCollectRoute,
    *,
    comm: Any | None = None,
) -> dict[str, Tensor]:
    block = _block(batch)
    count = int(collect.num_endpoints)
    out = embeddings.new_zeros((count, *embeddings.shape[1:]))
    empty_dependency = embeddings[:0]
    filled = torch.zeros(count, dtype=torch.bool, device=embeddings.device)
    if collect.local_endpoint_rows is not None and collect.local_embedding_rows is not None:
        endpoint_rows = collect.local_endpoint_rows.to(device=embeddings.device).long()
        if int(endpoint_rows.numel()):
            out.index_copy_(
                0,
                endpoint_rows,
                embeddings.index_select(
                    0,
                    collect.local_embedding_rows.to(device=embeddings.device).long(),
                ),
            )
            filled.index_fill_(0, endpoint_rows, True)
    if collect.node_dist_index is not None:
        remote = _collect_remote(block, embeddings, collect, comm=comm)
        if remote is not None:
            empty_dependency = empty_dependency + remote[1].sum() * 0.0
        if remote is not None and int(remote[0].numel()):
            rows, values = remote
            out.index_copy_(0, rows.to(device=values.device).long(), values)
            filled.index_fill_(0, rows.to(device=filled.device).long(), True)
    if collect.route is not None and collect.recv_endpoint_rows is not None:
        comm = comm or block.cache.get("comm")
        if comm is None:
            raise RuntimeError("endpoint collection requires the runtime CommScheduler")
        route = collect.route if isinstance(collect.route, Route) else Route(
            send_sizes=tuple(int(v) for v in collect.route.get("send_sizes", ())),
            recv_sizes=tuple(int(v) for v in collect.route.get("recv_sizes", ())),
            send_index=collect.route.get("send_index"),
            recv_index=collect.route.get("recv_index"),
        )
        recv = comm.autograd_push(
            route.to(device=embeddings.device),
            embeddings,
            name="endpoint_collect:embeddings",
        )
        empty_dependency = empty_dependency + recv.sum() * 0.0
        if int(recv.numel()):
            rows = collect.recv_endpoint_rows.to(device=recv.device).long()
            out.index_copy_(0, rows, recv)
            filled.index_fill_(0, rows.to(device=filled.device), True)
    if count == 0:
        return {name: empty_dependency for name in collect.groups}
    if not bool(torch.all(filled).item()):
        raise RuntimeError(
            "endpoint route is incomplete: "
            f"expected={count}, filled={int(filled.sum().item())}"
        )
    return {
        name: out[int(begin) : int(end)]
        for name, (begin, end) in collect.groups.items()
    }


def _collect_remote(
    block: GraphBlock,
    embeddings: Tensor,
    collect: EndpointCollectRoute,
    *,
    comm: Any | None = None,
) -> tuple[Tensor, Tensor] | None:
    remote_nodes = collect.remote_endpoint_nodes
    remote_rows = collect.remote_endpoint_rows
    if remote_nodes is None or remote_rows is None:
        remote_nodes = torch.empty(0, dtype=torch.long)
        remote_rows = torch.empty(0, dtype=torch.long)
    comm = comm or block.cache.get("comm")
    if comm is None:
        if int(remote_nodes.numel()) == 0:
            return None
        raise RuntimeError("remote endpoint collection requires the runtime CommScheduler")
    world_size = int(comm.world_size)
    if world_size <= 1:
        if int(remote_nodes.numel()):
            raise RuntimeError("remote endpoint nodes require distributed execution")
        return None
    node_dist_index = collect.node_dist_index
    if node_dist_index is None:
        return None
    remote_nodes = remote_nodes.long().to(device=embeddings.device, non_blocking=True)
    remote_rows = remote_rows.long().to(device=embeddings.device, non_blocking=True)
    owners = (
        dist_part(node_dist_index.to(device=embeddings.device).long().index_select(0, remote_nodes))
        if int(remote_nodes.numel())
        else remote_nodes.new_empty((0,))
    )
    order = torch.argsort(owners, stable=True) if int(owners.numel()) else remote_nodes.new_empty((0,))
    ordered = owners.index_select(0, order) if int(owners.numel()) else owners
    send_counts = torch.bincount(ordered.long().cpu(), minlength=world_size).long()
    recv_counts = all_to_all_counts(send_counts, group=comm.group)
    request = Route(
        send_sizes=tuple(int(v) for v in send_counts.tolist()),
        recv_sizes=tuple(int(v) for v in recv_counts.tolist()),
        send_index=order,
    )
    recv_nodes = comm.finish_push(
        comm.launch_push(request, remote_nodes, name="endpoint_request:nodes")
    )
    response = _embedding_rows(block, embeddings, recv_nodes.long())
    response_route = Route(
        send_sizes=tuple(int(v) for v in recv_counts.tolist()),
        recv_sizes=tuple(int(v) for v in send_counts.tolist()),
        send_index=torch.arange(int(response.shape[0]), dtype=torch.long, device=response.device),
    )
    received = comm.autograd_push(
        response_route,
        response.contiguous(),
        name="endpoint_response:embeddings",
    )
    return remote_rows.index_select(0, order).to(device=received.device), received


def _target(batch: Batch) -> TaskTarget | None:
    value = batch.targets.get("task") if isinstance(batch.targets, Mapping) else None
    return value if isinstance(value, TaskTarget) else None


def _collect(target: TaskTarget | None) -> EndpointCollectRoute | None:
    route = None if target is None else target.target_route
    return None if route is None else route.endpoint_collect


def _block(batch: Batch) -> GraphBlock:
    if batch.graph is not None:
        return batch.graph
    if batch.blocks is not None:
        return batch.blocks[-1][-1]
    raise ValueError("endpoint collection requires a graph block")


def _embedding_rows(block: GraphBlock, embeddings: Tensor, nodes: Tensor) -> Tensor:
    if int(nodes.numel()) == 0:
        return embeddings[:0]
    rows = compact_lookup_rows(
        block.dst_nodes.to(device=embeddings.device).long(),
        nodes.to(device=embeddings.device).long(),
    )
    if bool(torch.any(rows < 0).item()):
        raise RuntimeError("requested endpoint embedding is absent from the local output")
    return embeddings.index_select(0, rows)


__all__ = ["materialize_endpoint_embeddings", "materialize_endpoint_output"]
