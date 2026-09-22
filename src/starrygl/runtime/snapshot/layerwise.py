from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
import os
import time
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from torch import Tensor

from starrygl.runtime.comm import CommScheduler, Route, RouteHandle
from starrygl.view import GraphBlock


_PROFILE_STATS: dict[str, float] = {}


def _run_flare_style_coroutines(coros: Sequence[Any]) -> list[Tensor]:
    pending: list[tuple[int, Any]] = []
    out: list[Tensor | None] = [None] * len(coros)
    for idx, coro in enumerate(coros):
        if not inspect.iscoroutine(coro):
            raise TypeError(f"expected coroutine, got {type(coro)!r}")
        pending.append((idx, coro))
    remaining = len(pending)
    while remaining > 0:
        current, pending = pending, []
        for idx, coro in current:
            try:
                coro.send(None)
            except StopIteration as exc:
                out[idx] = exc.value
                remaining -= 1
            else:
                pending.append((idx, coro))
    if any(item is None for item in out):
        raise RuntimeError("Flare-style coroutine scan left an unfinished output")
    return [item for item in out if item is not None]


def _profile_enabled() -> bool:
    return str(os.environ.get("STARRYGL_PROFILE_LAYERWISE", "")).strip().lower() in {"1", "true", "yes", "on"}


def _profile_add(key: str, value: float = 1.0) -> None:
    if not _profile_enabled():
        return
    _PROFILE_STATS[key] = float(_PROFILE_STATS.get(key, 0.0)) + float(value)


def get_layerwise_profile(*, reset: bool = False) -> dict[str, float]:
    out = {key: float(value) for key, value in sorted(_PROFILE_STATS.items())}
    if reset:
        _PROFILE_STATS.clear()
    return out


def global_snapshot_contexts(
    local_contexts: Sequence[tuple[Tensor, Tensor | None]],
    *,
    reduction: str = "sum_count",
    comm: CommScheduler | None = None,
) -> Tensor:
    """Reduce model-produced local contexts to one context per window.

    All window statistics are packed into one scheduled collective so a
    model-recurrent update observes the same context on every rank.
    """

    if not local_contexts:
        raise ValueError("snapshot contexts require at least one window")
    mode = str(reduction).strip().lower()
    if mode not in {"sum_count", "max"}:
        raise ValueError("snapshot context reduction must be 'sum_count' or 'max'")
    values = tuple(value for value, _ in local_contexts)
    if any(value.dim() != 1 or value.shape != values[0].shape for value in values):
        raise ValueError("snapshot context summaries must be aligned vectors")
    reduced = torch.stack(values, dim=0)
    if mode == "sum_count":
        counts = tuple(count for _, count in local_contexts)
        if any(count is None or count.numel() != 1 for count in counts):
            raise ValueError("sum_count snapshot context requires one count per window")
        count_tensor = torch.stack([count.reshape(()) for count in counts if count is not None]).to(reduced).reshape(-1, 1)
        payload = torch.cat((reduced, count_tensor), dim=1)
    else:
        payload = reduced

    if dist.is_available() and dist.is_initialized() and int(dist.get_world_size()) > 1:
        scheduler = comm or CommScheduler()
        start = time.perf_counter()
        scheduler.all_reduce(
            payload,
            op=dist.ReduceOp.SUM if mode == "sum_count" else dist.ReduceOp.MAX,
            name="snapshot_context:reduce",
        )
        _profile_add("snapshot_context_all_reduce_seconds", time.perf_counter() - start)
        _profile_add("snapshot_context_all_reduces")
        _profile_add("snapshot_context_payload_elements", int(payload.numel()))

    if mode == "sum_count":
        contexts = payload[:, :-1] / payload[:, -1:].clamp_min_(1.0)
    else:
        contexts = payload.nan_to_num(0.0, neginf=0.0, posinf=0.0)
    return contexts


@dataclass(frozen=True)
class _ProfiledLayerAwaitable:
    handle: Any
    kind: str
    route: Route | None

    def wait(self) -> Tensor:
        start = time.perf_counter()
        value = self.handle.wait()
        _profile_add(f"{self.kind}_wait_seconds", time.perf_counter() - start)
        _profile_add(f"{self.kind}_waits")
        return value

    def ready(self) -> bool:
        ready = getattr(self.handle, "ready", None)
        return bool(ready()) if callable(ready) else False

    def result(self) -> Tensor:
        return self.wait()

    def wait_on_stream(self, stream: torch.cuda.Stream | None = None) -> Tensor:
        wait_on_stream = getattr(self.handle, "wait_on_stream", None)
        start = time.perf_counter()
        value = wait_on_stream(stream) if callable(wait_on_stream) else self.handle.wait()
        _profile_add(f"{self.kind}_wait_seconds", time.perf_counter() - start)
        _profile_add(f"{self.kind}_waits")
        return value

    async def async_wait(self, stream: torch.cuda.Stream | None = None) -> Tensor:
        await asyncio.sleep(0.0)
        return self.wait_on_stream(stream)

    def __await__(self):
        return self.async_wait().__await__()


@dataclass(frozen=True)
class ReadyLayerAwaitable:
    value: Tensor

    def wait(self) -> Tensor:
        return self.value

    def ready(self) -> bool:
        return True

    def result(self) -> Tensor:
        return self.value

    def wait_on_stream(self, stream: torch.cuda.Stream | None = None) -> Tensor:
        del stream
        return self.value

    async def async_wait(self, stream: torch.cuda.Stream | None = None) -> Tensor:
        del stream
        await asyncio.sleep(0.0)
        return self.value

    def __await__(self):
        return self.async_wait().__await__()


def materialize_embedding_src_async(
    block: GraphBlock,
    owned: Tensor,
    *,
    comm: CommScheduler | None = None,
    name: str = "snapshot_layer",
    stream: torch.cuda.Stream | None = None,
) -> ReadyLayerAwaitable | RouteHandle:
    """Materialize next-layer embeddings in ``block.src_nodes`` layout.

    Full snapshot blocks use the prepared collective route. The caller's
    layerwise execution chain owns the collective launch order; the returned
    awaitable only controls when this window waits for the already-launched
    route. Chunk-limited blocks intentionally stay local, matching Flare-style
    chunk-decay semantics.
    """

    if _is_chunk_limited(block):
        _profile_add("local_chunk_limited")
        return ReadyLayerAwaitable(_local_src_from_owned(block, owned))
    route = _route_from_block(block)
    if route is None or route.world_size <= 1:
        _profile_add("local_no_route")
        return ReadyLayerAwaitable(_local_src_from_owned(block, owned))
    if comm is None and not (dist.is_available() and dist.is_initialized()):
        _profile_add("local_no_dist")
        return ReadyLayerAwaitable(_local_src_from_owned(block, owned))
    if route.send_index is not None and int(route.send_index.numel()) > 0:
        required_rows = block.cache.get("embedding_send_rows")
        if required_rows is None:
            # Manual blocks bind once; prepared blocks bind before H2D.
            required_rows = int(route.send_index.max().item()) + 1
            block.cache["embedding_send_rows"] = required_rows
        if required_rows > int(owned.shape[0]):
            raise RuntimeError(
                "embedding exchange route send_index exceeds owned embedding rows: "
                f"max_send_index={required_rows - 1}, owned_rows={int(owned.shape[0])}, "
                f"num_dst={int(block.num_dst or block.dst_nodes.numel())}, "
                f"num_src={int(block.num_src or block.src_nodes.numel())}, "
                f"snapshot_id={block.cache.get('snapshot_id')}, "
                f"chunk_limited={block.cache.get('chunk_limited')}"
            )
    scheduler = comm if comm is not None else CommScheduler()
    _profile_add("route_launches")
    _profile_add("route_send_rows", route.send_len)
    _profile_add("route_recv_rows", route.recv_len)
    _profile_add("route_payload_elements", (route.send_len + route.recv_len) * int(owned[0].numel() if int(owned.shape[0]) else 0))
    start = time.perf_counter()
    if bool(block.cache.get("autograd_embedding_exchange", True)):
        handle = scheduler.launch_autograd_pull(
            route.to(device=owned.device),
            owned,
            name=name,
        )
        _profile_add("autograd_route_launch_seconds", time.perf_counter() - start)
        return _ProfiledLayerAwaitable(handle, "autograd_route", route)
    handle = scheduler.launch_pull(route.to(device=owned.device), owned, name=name, stream=stream)
    _profile_add("route_launch_seconds", time.perf_counter() - start)
    return _ProfiledLayerAwaitable(handle, "route", route)


def _local_src_from_owned(block: GraphBlock, owned: Tensor) -> Tensor:
    num_src = int(block.num_src or block.src_nodes.numel())
    if int(owned.shape[0]) == num_src:
        return owned
    out = owned.new_zeros((num_src, *owned.shape[1:]))
    rows = min(int(owned.shape[0]), num_src)
    if rows > 0:
        out[:rows] = owned[:rows]
    return out


def materialize_coupled_cell(cell, blocks, x, previous):
    """Exchange a recurrent reset gate through the existing autograd Route."""
    gates = getattr(cell, "materialize_gates", None)
    if gates is None:
        return cell.materialize(blocks, {"x": x, "h_prev": previous})
    block = blocks[-1]
    update, reset = gates(block, x, previous)
    count = int(block.num_dst or block.dst_nodes.numel())
    candidate_input = reset * previous[:count]
    candidate_input_src = materialize_embedding_src_async(
        block, candidate_input, comm=block.cache.get("comm"), name="snapshot_candidate_input",
    ).wait()
    candidate = cell.materialize_candidate_input(block, x, candidate_input_src)
    return {"update": update, "candidate": candidate, "state_like": candidate,
            "candidate_input": candidate_input}, block


def _route_from_block(block: GraphBlock) -> Route | None:
    route = block.route
    if route is None:
        return None
    if isinstance(route, Route):
        return route
    if not isinstance(route, Mapping):
        return None
    send_sizes = tuple(int(v) for v in route.get("send_sizes", ()))
    recv_sizes = tuple(int(v) for v in route.get("recv_sizes", ()))
    if not send_sizes and not recv_sizes:
        return None
    send_index = _optional_long(route.get("send_index"))
    recv_index = _optional_long(route.get("recv_index", route.get("recv_src_row")))
    return Route(
        send_sizes=send_sizes,
        recv_sizes=recv_sizes,
        send_index=send_index,
        recv_index=recv_index,
        output_len=int(block.num_src or block.src_nodes.numel()),
    )


def _optional_long(value: Any) -> Tensor | None:
    if isinstance(value, Tensor):
        return value.long()
    return None


def _is_chunk_limited(block: GraphBlock) -> bool:
    value = block.cache.get("chunk_limited", False)
    if isinstance(value, Tensor):
        return bool(value.item())
    return bool(value)


__all__ = [
    "ReadyLayerAwaitable",
    "get_layerwise_profile",
    "global_snapshot_contexts",
    "materialize_embedding_src_async",
]
