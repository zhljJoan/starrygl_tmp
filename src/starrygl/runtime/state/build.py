from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from starrygl.model import StarryModel
from starrygl.runtime.memory import AsyncMemoryCommitter, SharedStateRefreshFilter
from starrygl.store import MailboxManager, StateManager, StoreBundle
from starrygl.utils.route import dist_loc, dist_part


def build_state_managers(
    *,
    model: StarryModel,
    store: StoreBundle,
    kinds: Sequence[str],
    temporal_state: object,
    device: torch.device,
    comm: object,
    window_size: int = 1,
) -> Mapping[str, object]:
    config = temporal_state if isinstance(temporal_state, Mapping) else {}
    bounded = str(config.get("consistency", "exact")) == "bounded_stale"
    layouts = (store.graph.prepare or {}).get("meta", {}).get("view_plan", {}).get("required_layouts", ())
    snapshot = "neighbor_recurrent" in kinds and "snapshot_csc" in layouts and "temporal_csr" not in layouts
    if snapshot and "snapshot_hot_compute" in layouts:
        raise ValueError("owner-only snapshots require the exact owner layout; re-Prepare this hot-compute artifact")
    count, row_map, dist_index = _owner_layout(store, device)
    hot = store.graph.partition.get("hot_node_ids")
    hot = torch.empty(0, dtype=torch.long, device=device) if hot is None else hot.long().to(device=device)
    if snapshot:
        hot = hot[:0]
    shared_row_map = _shared_row_map(store.graph.num_nodes, hot, device) if bounded and hot.numel() else None
    mailbox = _mailbox(model, count, row_map, dist_index, device, comm) if "mailbox" in kinds else None
    managers: dict[str, object] = {}
    for kind in kinds:
        if kind == "mailbox":
            continue
        shape = _state_shape(model, kind)
        rows = 1 if kind == "model_recurrent" else count
        manager = StateManager(
            values=torch.zeros((rows, *shape), device=device),
            timestamps=None if kind == "model_recurrent" else torch.zeros(rows, device=device),
            row_map=None if kind == "model_recurrent" else row_map,
            node_dist_index=None if kind == "model_recurrent" else dist_index,
            kind=kind,
            comm=comm,
        )
        stale = bounded and kind in {"node_memory", "neighbor_recurrent"}
        shared = (
            StateManager(
                values=torch.zeros((int(hot.numel()), *shape), device=device),
                timestamps=torch.zeros(int(hot.numel()), device=device),
                row_map=shared_row_map,
                node_dist_index=dist_index,
                kind=kind,
                comm=comm,
            )
            if stale and shared_row_map is not None
            else None
        )
        managers[kind] = (
            AsyncMemoryCommitter(
                manager,
                shared_manager=shared,
                mailbox_manager=mailbox if kind == "node_memory" else None,
                shared_mailbox_manager=(
                    _mailbox(model, int(hot.numel()), shared_row_map, dist_index, device, comm)
                    if kind == "node_memory" and shared is not None
                    else None
                ),
                change_filter=(
                    _refresh_filter(config, shape[-1], int(hot.numel())) if shared is not None else None
                ),
                background_owner=store.graph.world_size > 1,
                bounded_stale_reads=stale,
                max_staleness=int(config.get("max_staleness", 0)) if stale else 0,
            )
            if kind == "node_memory" or stale
            else manager
        )
        if kind == "neighbor_recurrent" and snapshot and (stale or window_size > 1):
            from starrygl.runtime.memory.snapshot import bind_snapshot_cache_channel, bind_snapshot_history
            bind_snapshot_history(managers[kind], store, hot, window_size=window_size)
            if stale:
                cell = getattr(model, "runtime_cell", None)
                for channel in getattr(cell, "cache_channels", {}):
                    bind_snapshot_cache_channel(managers[kind], channel, shape[-1])
                managers[kind].change_filter = _refresh_filter(
                    config, shape[-1], int(managers[kind].snapshot_boundary_nodes.numel()))
                if managers[kind].change_filter is not None:
                    managers[kind].change_filter.max_skip = min(
                        managers[kind].change_filter.max_skip, managers[kind].max_staleness)
    return managers


def model_device(model: StarryModel, device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def _state_shape(model: StarryModel, kind: str) -> tuple[int, ...]:
    shapes = getattr(model, "state_shapes", None)
    shapes = shapes() if callable(shapes) else shapes
    if isinstance(shapes, Mapping) and kind in shapes:
        return tuple(int(value) for value in shapes[kind])
    hidden = int(getattr(model, "hidden_dim", 0) or 0)
    if kind == "mailbox" and hidden > 0:
        slots = int(getattr(model, "mailbox_size", getattr(getattr(model, "memory", None), "mailbox_size", 1)))
        return (max(1, slots), 2 * hidden + int(getattr(model, "edge_dim", 0) or 0))
    state_dim = getattr(model, "state_dim", None)
    state_dim = state_dim() if callable(state_dim) else state_dim
    size = int(state_dim or hidden)
    if size <= 0:
        raise ValueError(f"cannot infer {kind} shape; define model.state_shapes or pass state_manager")
    return (size,)


def _shared_row_map(num_nodes: int, hot: torch.Tensor, device: torch.device) -> torch.Tensor:
    rows = torch.full((int(num_nodes),), -1, dtype=torch.long, device=device)
    rows.index_copy_(0, hot, torch.arange(int(hot.numel()), dtype=torch.long, device=device))
    return rows


def _owner_layout(
    store: StoreBundle,
    device: torch.device,
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
    packed = store.graph.partition.get("node_dist_index")
    if packed is None:
        return int(store.graph.num_nodes), None, None
    packed = packed.long().to(device=device)
    owned = dist_part(packed) == int(store.graph.rank)
    row_map = torch.full((int(packed.numel()),), -1, dtype=torch.long, device=device)
    if bool(owned.any().item()):
        row_map[owned] = dist_loc(packed[owned])
    count = int(row_map.max().item()) + 1 if bool(owned.any().item()) else 0
    return count, row_map, packed


def _mailbox(model, rows, row_map, dist_index, device, comm):
    return MailboxManager(
        values=torch.zeros((int(rows), *_state_shape(model, "mailbox")), device=device),
        row_map=row_map,
        node_dist_index=dist_index,
        comm=comm,
    )


def _refresh_filter(config: Mapping[str, Any], dim: int, rows: int) -> SharedStateRefreshFilter | None:
    values = config.get("filter", {})
    if not isinstance(values, Mapping) or not bool(values.get("enabled", False)):
        return None
    max_staleness = max(1, int(config.get("max_staleness", 1)))
    return SharedStateRefreshFilter(
        dim=int(dim),
        num_rows=int(rows),
        min_change_norm=float(values.get("min_change_norm", 0.0)),
        min_cosine_distance=values.get("min_cosine_distance"),
        max_skip=min(int(values.get("max_skip", 10)), max_staleness),
    )


__all__ = ["build_state_managers", "model_device"]
