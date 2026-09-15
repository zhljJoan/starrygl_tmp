from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import os
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
from torch import Tensor

from starrygl.batch import Batch
from starrygl.view import GraphBlock
from .coupled import run_coupled_window_scan
from .layerwise import _run_flare_style_coroutines, materialize_embedding_src_async
from ..state.recurrent import (
    CoupledStateMaterialization,
    materialize_coupled_neighbor_state,
    previous_state as _previous_state,
    scatter_dst_state as _scatter_dst_state,
)


class BatchLocalRecurrentCell(Protocol):
    reads_neighbor_state: bool
    state_key: str

    def materialize(self, blocks: Sequence[GraphBlock], src: Mapping[str, Tensor]) -> tuple[Mapping[str, Tensor], GraphBlock]:
        ...

    def local_forward(self, block: GraphBlock, src: Mapping[str, Tensor], dst: Mapping[str, Tensor]) -> Tensor:
        ...

    def embedding_from_state(self, state: Tensor) -> Tensor:
        ...


@dataclass(frozen=True)
class WindowScanResult:
    embeddings: Tensor
    state_embeddings: Tensor
    final_block: GraphBlock
    window_embeddings: tuple[Tensor, ...]
    state_history: tuple[tuple[Tensor, int, Tensor], ...] = ()


@dataclass(frozen=True)
class _LocalWindowInput:
    src: Mapping[str, Tensor]
    block: GraphBlock
    state_like: Tensor


def _comm_stream_for(
    values: Sequence[Tensor],
    windows: Sequence[Sequence[GraphBlock]],
    comm: Any | None = None,
) -> torch.cuda.Stream | None:
    if not torch.cuda.is_available():
        return None
    for value in values:
        if value.device.type == "cuda":
            comm = comm or next(
                (block.cache.get("comm") for blocks in windows for block in blocks if block.cache.get("comm") is not None),
                None,
            )
            stream = getattr(comm, "stream", None)
            return (
                stream(value.device)
                if callable(stream)
                else torch.cuda.Stream(device=value.device)
            )
    return None


def _current_stream_for(values: Sequence[Tensor]) -> torch.cuda.Stream | None:
    if not torch.cuda.is_available():
        return None
    for value in values:
        if value.device.type == "cuda":
            return torch.cuda.current_stream(device=value.device)
    return None


def _run_sequential_window_scan(
    batch: Batch,
    *,
    input_project: Callable[[Tensor], Tensor],
    cell: BatchLocalRecurrentCell,
    feature_key: str = "x",
    fallback_feature_key: str = "feat",
    persist_state: bool = False,
    comm: Any | None = None,
) -> WindowScanResult:
    carried: Tensor | None = None
    final_block: GraphBlock | None = None
    window_embeddings: list[Tensor] = []
    num_layers = int(cell.num_gcn_layers) if all(hasattr(cell, name) for name in (
        "num_gcn_layers", "compute_gcn_layer", "finalize_gcn")) else 0
    for window_id, blocks in enumerate(batch.iter_blocks()):
        x = input_project(_feature_for_window(batch, feature_key, window_id, fallback_feature_key).float())
        if num_layers:
            for layer in range(num_layers):
                block = blocks[min(layer, len(blocks) - 1)]
                x = cell.compute_gcn_layer(layer, blocks, _ensure_src_layout(block, x))
                if layer + 1 < num_layers:
                    handle = materialize_embedding_src_async(
                        block, x, comm=comm or block.cache.get("comm"),
                        name=f"{cell.state_key}:gcn_layer_{layer + 1}")
                    x = _wait_layer_handle_sync(handle, _current_stream_for((x,)))
            src, block = cell.finalize_gcn(x), blocks[-1]
        else:
            src, block = cell.materialize(blocks, {"x": x})
        del x
        state_like = src.get("state_like")
        if state_like is None:
            raise KeyError("batch-local recurrent cell materialize() must return 'state_like'")
        h_prev = _previous_state(
            batch_state=batch.state.get(cell.state_key),
            current=state_like,
            carried=carried,
            persist_state=bool(persist_state),
            previous_block=final_block,
            current_block=block,
        )
        carried = cell.local_forward(block, src, {"h_prev": h_prev})
        final_block = block
        window_embeddings.append(
            cell.embedding_from_state(carried) if hasattr(cell, "embedding_from_state") else carried
        )
        del state_like, h_prev
    if carried is None or final_block is None:
        raise ValueError("batch has no recurrent windows")
    return WindowScanResult(
        embeddings=window_embeddings[-1],
        state_embeddings=carried,
        final_block=final_block,
        window_embeddings=tuple(window_embeddings),
    )


def run_decoupled_window_dag_scan(
    batch: Batch,
    *,
    input_project: Callable[[Tensor], Tensor],
    cell: BatchLocalRecurrentCell,
    feature_key: str = "x",
    fallback_feature_key: str = "feat",
    persist_state: bool = False,
    comm: Any | None = None,
) -> WindowScanResult:
    """Run non-coupled window scan with runtime-visible GCN exchange handles.

    The GCN path is scheduled layer by layer. After a window finishes layer
    ``l`` local compute, the runtime immediately launches the boundary exchange
    for layer ``l + 1`` while later windows in the same layer continue local
    compute. This execution chain fixes the collective launch order by
    ``(layer, window_id)``; await only controls when a launched route is waited
    on and resumed. Recurrent updates are still scanned in window order.
    """

    if str(os.environ.get("STARRYGL_DISABLE_LAYERWISE_DAG", "")).strip().lower() in {"1", "true", "yes", "on"} or not all(
        hasattr(cell, name) for name in ("num_gcn_layers", "compute_gcn_layer", "finalize_gcn")
    ):
        return _run_sequential_window_scan(
            batch,
            input_project=input_project,
            cell=cell,
            feature_key=feature_key,
            fallback_feature_key=fallback_feature_key,
            persist_state=persist_state,
            comm=comm,
        )

    windows = [tuple(blocks) for blocks in batch.iter_blocks()]
    if not windows:
        raise ValueError("batch has no recurrent windows")
    values: list[Tensor] = [
        input_project(_feature_for_window(batch, feature_key, window_id, fallback_feature_key).float())
        for window_id in range(len(windows))
    ]
    comm_stream = _comm_stream_for(values, windows, comm)
    num_layers = int(getattr(cell, "num_gcn_layers"))
    for layer in range(num_layers):
        current_stream = _current_stream_for(values)
        schedule_order = _layer_schedule_order(windows, int(layer), route_first=layer + 1 < num_layers)

        async def run_window_layer(window_id: int, blocks: Sequence[GraphBlock]) -> Tensor:
            block = blocks[min(int(layer), len(blocks) - 1)]
            input_value = _ensure_src_layout(block, values[window_id])
            value = cell.compute_gcn_layer(layer, blocks, input_value)
            if layer + 1 < num_layers:
                handle = materialize_embedding_src_async(
                    block,
                    value,
                    comm=comm or block.cache.get("comm"),
                    name=f"{cell.state_key}:gcn_layer_{int(layer) + 1}",
                    stream=comm_stream,
                )
                if str(os.environ.get("STARRYGL_LAYERWISE_IMMEDIATE_WAIT", "")).strip().lower() in {"1", "true", "yes", "on"}:
                    return _wait_layer_handle_sync(handle, current_stream)
                return await _await_layer_handle(handle, current_stream)
            await asyncio.sleep(0.0)
            return value

        scheduled_values = _run_flare_style_coroutines(
            [run_window_layer(window_id, windows[window_id]) for window_id in schedule_order]
        )
        next_values: list[Tensor | None] = [None] * len(windows)
        for window_id, value in zip(schedule_order, scheduled_values):
            next_values[int(window_id)] = value
        if any(value is None for value in next_values):
            raise RuntimeError("layerwise window scan lost a scheduled output")
        values = [value for value in next_values if value is not None]

    materialized: list[_LocalWindowInput] = []
    for value, blocks in zip(values, windows):
        src = cell.finalize_gcn(value)
        state_like = src.get("state_like")
        if state_like is None:
            raise KeyError("decoupled recurrent cell finalize_gcn() must return 'state_like'")
        materialized.append(_LocalWindowInput(src=src, block=blocks[-1], state_like=state_like))

    carried: Tensor | None = None
    final_block: GraphBlock | None = None
    window_embeddings: list[Tensor] = []
    for window in materialized:
        h_prev = _previous_state(
            batch_state=batch.state.get(cell.state_key),
            current=window.state_like,
            carried=carried,
            persist_state=bool(persist_state),
            previous_block=final_block,
            current_block=window.block,
        )
        carried = cell.local_forward(window.block, window.src, {"h_prev": h_prev})
        final_block = window.block
        window_embeddings.append(
            cell.embedding_from_state(carried) if hasattr(cell, "embedding_from_state") else carried
        )
    if carried is None or final_block is None:
        raise ValueError("batch has no recurrent windows")
    return WindowScanResult(
        embeddings=window_embeddings[-1],
        state_embeddings=carried,
        final_block=final_block,
        window_embeddings=tuple(window_embeddings),
    )


def _ensure_src_layout(block: GraphBlock, value: Tensor) -> Tensor:
    num_src = int(block.num_src or block.src_nodes.numel())
    if int(value.shape[0]) >= num_src:
        return value
    out = value.new_zeros((num_src, *value.shape[1:]))
    rows = min(int(value.shape[0]), num_src)
    if rows > 0:
        out[:rows] = value[:rows]
    return out


def _layer_schedule_order(windows: Sequence[Sequence[GraphBlock]], layer: int, *, route_first: bool) -> list[int]:
    order = list(range(len(windows)))
    if not route_first or len(order) <= 1:
        return order
    routed: list[int] = []
    local: list[int] = []
    for window_id, blocks in enumerate(windows):
        block = blocks[min(int(layer), len(blocks) - 1)]
        if _needs_layer_exchange(block):
            routed.append(int(window_id))
        else:
            local.append(int(window_id))
    return routed + local


def _needs_layer_exchange(block: GraphBlock) -> bool:
    value = block.cache.get("chunk_limited", False)
    if isinstance(value, Tensor):
        if bool(value.item()):
            return False
    elif bool(value):
        return False
    route = block.route
    if route is None:
        return False
    if isinstance(route, Mapping):
        send_sizes = route.get("send_sizes", ())
        recv_sizes = route.get("recv_sizes", ())
        return max(len(send_sizes), len(recv_sizes)) > 1
    return int(getattr(route, "world_size", 1)) > 1


async def _await_layer_handle(handle: Any, stream: torch.cuda.Stream | None) -> Tensor:
    async_wait = getattr(handle, "async_wait", None)
    if callable(async_wait):
        return await async_wait(stream)
    wait_on_stream = getattr(handle, "wait_on_stream", None)
    if callable(wait_on_stream):
        await asyncio.sleep(0.0)
        return wait_on_stream(stream)
    await asyncio.sleep(0.0)
    return handle.wait()


def _wait_layer_handle_sync(handle: Any, stream: torch.cuda.Stream | None) -> Tensor:
    wait_on_stream = getattr(handle, "wait_on_stream", None)
    if callable(wait_on_stream):
        return wait_on_stream(stream)
    return handle.wait()



def run_model_recurrent_window_scan(
    batch: Batch,
    *,
    input_project: Callable[[Tensor], Tensor],
    cell: Any,
    comm: Any | None = None,
) -> WindowScanResult:
    """Run context reduction, model-state advance and spatial work in Runtime."""

    windows = tuple(tuple(blocks) for blocks in batch.iter_blocks())
    if not windows:
        raise ValueError("batch has no model-recurrent windows")
    values = tuple(
        input_project(_feature_for_window(batch, "x", window_id, "feat").float())
        for window_id in range(len(windows))
    )
    local_contexts = tuple(
        cell.context(blocks[-1], value)
        for blocks, value in zip(windows, values)
    )
    from .layerwise import global_snapshot_contexts

    contexts = global_snapshot_contexts(
        local_contexts,
        reduction=str(cell.context_reduction),
        comm=comm or windows[0][-1].cache.get("comm"),
    )
    state = cell.initial_state(batch)
    embeddings = []
    for blocks, value, context in zip(windows, values, contexts.unbind(0)):
        state = cell.advance_state(state, context)
        embeddings.append(cell.spatial(0, blocks[-1], value, state))
    return WindowScanResult(
        embeddings=embeddings[-1],
        state_embeddings=state,
        final_block=windows[-1][-1],
        window_embeddings=tuple(embeddings),
    )


def _feature_for_window(batch: Batch, key: str, window_id: int, fallback_key: str) -> Tensor:
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


def encode_model(
    model: Any,
    batch: Batch,
    *,
    persist_state: bool | None = None,
    comm: Any | None = None,
):
    """Run runtime-owned recurrent scans, else call the model."""

    cell = getattr(model, "runtime_cell", None)
    build_output = getattr(model, "runtime_output_from_scan", None)
    if cell is None or not callable(build_output):
        return model.encode(batch)
    scan_batch = batch
    extra_aux = {}
    prepare_scan = getattr(model, "runtime_prepare_scan", None)
    if callable(prepare_scan):
        scan_batch, extra_aux = prepare_scan(batch)
    project = getattr(model, "runtime_input_project", getattr(model, "input", _identity))
    should_persist = (
        bool(getattr(model, "runtime_persist_state", getattr(model, "persist_state", False)))
        if persist_state is None
        else bool(persist_state)
    )
    if str(getattr(cell, "state_kind", "")) == "model_recurrent":
        scan = run_model_recurrent_window_scan(
            scan_batch,
            input_project=project,
            cell=cell,
            comm=comm,
        )
    elif bool(getattr(cell, "reads_neighbor_state", False)):
        scan = run_coupled_window_scan(
            scan_batch,
            input_project=project,
            cell=cell,
            persist_state=should_persist,
            state_transform=getattr(model, "runtime_smooth_state", None),
        )
    else:
        scan = run_decoupled_window_dag_scan(
            scan_batch,
            input_project=project,
            cell=cell,
            persist_state=should_persist,
            comm=comm,
        )
    output = build_output(batch, scan)
    return replace(output, aux={**output.aux, **extra_aux}) if extra_aux else output


def _identity(value: Any) -> Any:
    return value


__all__ = [
    "BatchLocalRecurrentCell",
    "CoupledStateMaterialization",
    "WindowScanResult",
    "encode_model",
    "materialize_coupled_neighbor_state",
    "run_coupled_window_scan",
    "run_decoupled_window_dag_scan",
    "run_model_recurrent_window_scan",
]
