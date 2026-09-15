from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor


DIST_INDEX_LOC_BITS = 48
DIST_INDEX_LOC_MASK = (1 << DIST_INDEX_LOC_BITS) - 1


@dataclass(frozen=True)
class ExchangeRoute:
    send_sizes: tuple[int, ...]
    recv_sizes: tuple[int, ...]
    send_rows: Tensor
    recv_rows: Tensor
    send_dst_rank: Tensor
    send_dst_rows: Tensor
    recv_src_rank: Tensor
    aux: dict[str, Tensor] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "send_sizes": self.send_sizes,
            "recv_sizes": self.recv_sizes,
            "send_rows": self.send_rows,
            "recv_rows": self.recv_rows,
            "send_dst_rank": self.send_dst_rank,
            "send_dst_rows": self.send_dst_rows,
            "recv_src_rank": self.recv_src_rank,
        }
        out.update(self.aux)
        return out


@dataclass(frozen=True)
class RouteData:
    send_sizes: tuple[tuple[int, ...], ...]
    recv_sizes: tuple[tuple[int, ...], ...]
    send_index_ind: Tensor | None = None
    send_index_ptr: tuple[int, ...] | None = None
    recv_index_ind: Tensor | None = None
    recv_index_ptr: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.send_sizes) != len(self.recv_sizes):
            raise ValueError("send_sizes and recv_sizes must have the same length")
        for send, recv in zip(self.send_sizes, self.recv_sizes):
            if len(send) != len(recv):
                raise ValueError("send_sizes and recv_sizes entries must have the same world size")
        _validate_ptr(self.send_index_ind, self.send_index_ptr, "send_index")
        _validate_ptr(self.recv_index_ind, self.recv_index_ptr, "recv_index")

    def __len__(self) -> int:
        return len(self.send_sizes)

    def __getitem__(self, key: int | slice) -> RouteData:
        if isinstance(key, int):
            key = slice(key, key + 1)
        if key.step is not None and key.step != 1:
            raise ValueError("RouteData only supports contiguous slices")
        start = 0 if key.start is None else int(key.start)
        stop = len(self) if key.stop is None else int(key.stop)
        return RouteData(
            send_sizes=self.send_sizes[start:stop],
            recv_sizes=self.recv_sizes[start:stop],
            send_index_ind=_slice_ind(self.send_index_ind, self.send_index_ptr, start, stop),
            send_index_ptr=_slice_ptr(self.send_index_ptr, start, stop),
            recv_index_ind=_slice_ind(self.recv_index_ind, self.recv_index_ptr, start, stop),
            recv_index_ptr=_slice_ptr(self.recv_index_ptr, start, stop),
        )

    def to(self, *args, **kwargs) -> RouteData:
        return RouteData(
            send_sizes=self.send_sizes,
            recv_sizes=self.recv_sizes,
            send_index_ind=None if self.send_index_ind is None else self.send_index_ind.to(*args, **kwargs),
            send_index_ptr=self.send_index_ptr,
            recv_index_ind=None if self.recv_index_ind is None else self.recv_index_ind.to(*args, **kwargs),
            recv_index_ptr=self.recv_index_ptr,
        )

    def pin_memory(self, device=None) -> RouteData:
        return RouteData(
            send_sizes=self.send_sizes,
            recv_sizes=self.recv_sizes,
            send_index_ind=None if self.send_index_ind is None else self.send_index_ind.pin_memory(device=device),
            send_index_ptr=self.send_index_ptr,
            recv_index_ind=None if self.recv_index_ind is None else self.recv_index_ind.pin_memory(device=device),
            recv_index_ptr=self.recv_index_ptr,
        )

    def item(self):
        routes = self.to_routes()
        if len(routes) != 1:
            raise ValueError(f"expected one route, got {len(routes)}")
        return routes[0]

    def to_routes(self, *, group=None):
        from starrygl.runtime.comm import Route

        del group
        routes = []
        for idx in range(len(self)):
            routes.append(
                Route(
                    send_sizes=self.send_sizes[idx],
                    recv_sizes=self.recv_sizes[idx],
                    send_index=_route_index(self.send_index_ind, self.send_index_ptr, idx),
                    recv_index=_route_index(self.recv_index_ind, self.recv_index_ptr, idx),
                )
            )
        return routes

    @classmethod
    def from_routes(cls, routes: list[Any]) -> RouteData:
        send_index_parts = []
        recv_index_parts = []
        send_ptr = [0]
        recv_ptr = [0]
        send_sizes = []
        recv_sizes = []
        saw_send_none = False
        saw_recv_none = False
        for route in routes:
            send_sizes.append(tuple(int(v) for v in route.send_sizes))
            recv_sizes.append(tuple(int(v) for v in route.recv_sizes))
            if route.send_index is not None:
                if saw_send_none:
                    raise ValueError("all routes must either have send_index or all omit it")
                send_index_parts.append(route.send_index.long())
                send_ptr.append(send_ptr[-1] + int(route.send_index.numel()))
            else:
                saw_send_none = True
                if send_index_parts:
                    raise ValueError("all routes must either have send_index or all omit it")
            if route.recv_index is not None:
                if saw_recv_none:
                    raise ValueError("all routes must either have recv_index or all omit it")
                recv_index_parts.append(route.recv_index.long())
                recv_ptr.append(recv_ptr[-1] + int(route.recv_index.numel()))
            else:
                saw_recv_none = True
                if recv_index_parts:
                    raise ValueError("all routes must either have recv_index or all omit it")
        return cls(
            send_sizes=tuple(send_sizes),
            recv_sizes=tuple(recv_sizes),
            send_index_ind=torch.cat(send_index_parts, dim=0) if send_index_parts else None,
            send_index_ptr=tuple(send_ptr) if send_index_parts else None,
            recv_index_ind=torch.cat(recv_index_parts, dim=0) if recv_index_parts else None,
            recv_index_ptr=tuple(recv_ptr) if recv_index_parts else None,
        )


@dataclass(frozen=True)
class RouteBook:
    routes: Mapping[str, RouteData] = field(default_factory=dict)

    def __getitem__(self, key: str) -> RouteData:
        return self.routes[key]

    def get(self, key: str, default: RouteData | None = None) -> RouteData | None:
        return self.routes.get(key, default)

    def keys(self):
        return self.routes.keys()

    def to(self, *args, **kwargs) -> RouteBook:
        return RouteBook({name: route.to(*args, **kwargs) for name, route in self.routes.items()})

    def pin_memory(self, device=None) -> RouteBook:
        return RouteBook({name: route.pin_memory(device=device) for name, route in self.routes.items()})


def _validate_ptr(ind: Tensor | None, ptr: tuple[int, ...] | None, name: str) -> None:
    if ind is None:
        if ptr is not None:
            raise ValueError(f"{name}_ptr must be None when {name}_ind is None")
        return
    if ptr is None:
        raise ValueError(f"{name}_ptr is required when {name}_ind is set")
    if int(ptr[-1]) != int(ind.numel()):
        raise ValueError(f"{name}_ptr[-1] must equal {name}_ind.numel()")


def _route_index(ind: Tensor | None, ptr: tuple[int, ...] | None, idx: int) -> Tensor | None:
    if ind is None or ptr is None:
        return None
    return ind[int(ptr[idx]) : int(ptr[idx + 1])]


def _slice_ind(ind: Tensor | None, ptr: tuple[int, ...] | None, start: int, stop: int) -> Tensor | None:
    if ind is None or ptr is None:
        return None
    return ind[int(ptr[start]) : int(ptr[stop])]


def _slice_ptr(ptr: tuple[int, ...] | None, start: int, stop: int) -> tuple[int, ...] | None:
    if ptr is None:
        return None
    base = int(ptr[start])
    return tuple(int(v) - base for v in ptr[start : stop + 1])


def dist_part(value: Tensor) -> Tensor:
    return value.long() >> DIST_INDEX_LOC_BITS


def dist_loc(value: Tensor) -> Tensor:
    return value.long() & DIST_INDEX_LOC_MASK


def build_exchange_route(
    *,
    send_rows: Tensor,
    dst_dist_index: Tensor,
    world_size: int,
    recv_sizes: tuple[int, ...] | None = None,
    recv_rows: Tensor | None = None,
    recv_src_rank: Tensor | None = None,
    aux: dict[str, Tensor] | None = None,
) -> ExchangeRoute:
    dst_rank = dist_part(dst_dist_index)
    dst_rows = dist_loc(dst_dist_index)
    order = torch.argsort(dst_rank, stable=True)
    dst_rank = dst_rank.index_select(0, order)
    ordered_aux = {k: v.index_select(0, order) for k, v in (aux or {}).items()}
    return ExchangeRoute(
        send_sizes=tuple(int((dst_rank == peer).sum().item()) for peer in range(int(world_size))),
        recv_sizes=recv_sizes or tuple(0 for _ in range(int(world_size))),
        send_rows=send_rows.long().index_select(0, order),
        recv_rows=recv_rows if recv_rows is not None else torch.empty(0, dtype=torch.long),
        send_dst_rank=dst_rank,
        send_dst_rows=dst_rows.index_select(0, order),
        recv_src_rank=recv_src_rank if recv_src_rank is not None else torch.empty(0, dtype=torch.long),
        aux=ordered_aux,
    )


def build_state_write_mask(*, src: Tensor, dst: Tensor, time_ptr_2: Tensor, num_nodes: int) -> Tensor:
    out = torch.zeros(int(src.numel()), dtype=torch.uint8)
    last_row = torch.empty(int(num_nodes), dtype=torch.long)
    for begin, end in time_ptr_2.tolist():
        begin = int(begin)
        end = int(end)
        if end <= begin:
            continue
        rows = torch.arange(begin, end, dtype=torch.long)
        nodes = torch.cat((src[begin:end], dst[begin:end]), dim=0)
        candidate_rows = torch.cat((rows, rows), dim=0)
        unique_nodes, inverse = torch.unique(nodes, sorted=False, return_inverse=True)
        latest = torch.full((int(unique_nodes.numel()),), -1, dtype=torch.long)
        latest.scatter_reduce_(0, inverse, candidate_rows, reduce="amax", include_self=True)
        last_row[unique_nodes] = latest
        win_src = src[begin:end]
        win_dst = dst[begin:end]
        src_last = last_row.index_select(0, win_src) == rows
        dst_last = last_row.index_select(0, win_dst) == rows
        out[begin:end] = src_last.to(torch.uint8) | (dst_last.to(torch.uint8) << 1)
    return out


def build_state_write_routes_for_rank(
    *,
    rank: int,
    edge_master: Tensor,
    local_global: Tensor,
    local_pos: Tensor,
    src: Tensor,
    dst: Tensor,
    time_ptr_2: Tensor,
    state_write_mask: Tensor,
    node_dist_index: Tensor,
    node_is_hot: Tensor,
    world_size: int,
    hot: bool,
) -> list[dict[str, Any]]:
    routes = []
    if int(local_global.numel()):
        local_begin_pos = torch.searchsorted(local_global, time_ptr_2[:, 0].contiguous(), right=False)
        local_end_pos = torch.searchsorted(local_global, time_ptr_2[:, 1].contiguous(), right=False)
    else:
        local_begin_pos = torch.zeros(int(time_ptr_2.shape[0]), dtype=torch.long)
        local_end_pos = torch.zeros(int(time_ptr_2.shape[0]), dtype=torch.long)
    for window_id, (begin, end) in enumerate(time_ptr_2.tolist()):
        begin = int(begin)
        end = int(end)
        l_begin = int(local_begin_pos[window_id].item())
        l_end = int(local_end_pos[window_id].item())
        send_rows, node_ids, role = _state_write_candidates(
            edge_rows=local_global[l_begin:l_end],
            local_pos=local_pos,
            src=src,
            dst=dst,
            state_write_mask=state_write_mask,
        )
        hot_mask = node_is_hot.index_select(0, node_ids)
        select = hot_mask if hot else ~hot_mask
        send_rows = send_rows[select]
        node_ids = node_ids[select]
        role = role[select]
        recv_rows, recv_src_rank, recv_sizes = _state_write_recv_layout(
            rank=rank,
            begin=begin,
            end=end,
            edge_master=edge_master,
            src=src,
            dst=dst,
            state_write_mask=state_write_mask,
            node_dist_index=node_dist_index,
            node_is_hot=node_is_hot,
            world_size=world_size,
            hot=hot,
        )
        if int(node_ids.numel()) == 0:
            route = empty_exchange_route(world_size=world_size, recv_rows=recv_rows, recv_src_rank=recv_src_rank, recv_sizes=recv_sizes)
        else:
            route = build_exchange_route(
                send_rows=send_rows,
                dst_dist_index=node_dist_index.index_select(0, node_ids),
                world_size=world_size,
                recv_sizes=recv_sizes,
                recv_rows=recv_rows,
                recv_src_rank=recv_src_rank,
                aux={"node_ids": node_ids, "role": role},
            )
        row = route.as_dict()
        row["window_id"] = int(window_id)
        row["event_rows"] = row["send_rows"]
        row["owner_rank"] = row["send_dst_rank"]
        row["owner_rows"] = row["send_dst_rows"]
        row.setdefault("node_ids", torch.empty(0, dtype=torch.long))
        row.setdefault("role", torch.empty(0, dtype=torch.uint8))
        routes.append(row)
    return routes


def _state_write_candidates(
    *,
    edge_rows: Tensor,
    local_pos: Tensor,
    src: Tensor,
    dst: Tensor,
    state_write_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if int(edge_rows.numel()) == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.uint8)
    mask = state_write_mask.index_select(0, edge_rows)
    src_keep = edge_rows[(mask & 1) != 0]
    dst_keep = edge_rows[(mask & 2) != 0]
    return (
        torch.cat((local_pos.index_select(0, src_keep), local_pos.index_select(0, dst_keep)), dim=0),
        torch.cat((src.index_select(0, src_keep), dst.index_select(0, dst_keep)), dim=0),
        torch.cat(
            (
                torch.ones(int(src_keep.numel()), dtype=torch.uint8),
                torch.full((int(dst_keep.numel()),), 2, dtype=torch.uint8),
            ),
            dim=0,
        ),
    )


def _state_write_node_candidates(
    *,
    edge_rows: Tensor,
    src: Tensor,
    dst: Tensor,
    state_write_mask: Tensor,
) -> Tensor:
    if int(edge_rows.numel()) == 0:
        return torch.empty(0, dtype=torch.long)
    mask = state_write_mask.index_select(0, edge_rows)
    src_keep = edge_rows[(mask & 1) != 0]
    dst_keep = edge_rows[(mask & 2) != 0]
    if int(src_keep.numel()) == 0:
        return dst.index_select(0, dst_keep)
    if int(dst_keep.numel()) == 0:
        return src.index_select(0, src_keep)
    return torch.cat((src.index_select(0, src_keep), dst.index_select(0, dst_keep)), dim=0)


def _state_write_recv_layout(
    *,
    rank: int,
    begin: int,
    end: int,
    edge_master: Tensor,
    src: Tensor,
    dst: Tensor,
    state_write_mask: Tensor,
    node_dist_index: Tensor,
    node_is_hot: Tensor,
    world_size: int,
    hot: bool,
) -> tuple[Tensor, Tensor, tuple[int, ...]]:
    recv_chunks = []
    recv_peer_chunks = []
    window_rows = torch.arange(int(begin), int(end), dtype=torch.long)
    window_edge_master = edge_master[begin:end]
    for peer in range(int(world_size)):
        edge_rows = window_rows[(window_edge_master == int(peer)).nonzero(as_tuple=True)[0]]
        node_ids = _state_write_node_candidates(
            edge_rows=edge_rows,
            src=src,
            dst=dst,
            state_write_mask=state_write_mask,
        )
        if int(node_ids.numel()) == 0:
            recv_chunks.append(torch.empty(0, dtype=torch.long))
            recv_peer_chunks.append(torch.empty(0, dtype=torch.long))
            continue
        hot_mask = node_is_hot.index_select(0, node_ids)
        select = hot_mask if hot else ~hot_mask
        packed = node_dist_index.index_select(0, node_ids[select])
        owner_rank = dist_part(packed)
        owner_rows = dist_loc(packed)
        rows = owner_rows[owner_rank == int(rank)]
        recv_chunks.append(rows)
        recv_peer_chunks.append(torch.full((int(rows.numel()),), int(peer), dtype=torch.long))
    recv_rows = torch.cat(recv_chunks, dim=0) if recv_chunks else torch.empty(0, dtype=torch.long)
    recv_src_rank = torch.cat(recv_peer_chunks, dim=0) if recv_peer_chunks else torch.empty(0, dtype=torch.long)
    return recv_rows, recv_src_rank, tuple(int(v.numel()) for v in recv_chunks)


def empty_exchange_route(
    *,
    world_size: int,
    recv_rows: Tensor | None = None,
    recv_src_rank: Tensor | None = None,
    recv_sizes: tuple[int, ...] | None = None,
) -> ExchangeRoute:
    return ExchangeRoute(
        send_sizes=tuple(0 for _ in range(int(world_size))),
        recv_sizes=recv_sizes or tuple(0 for _ in range(int(world_size))),
        send_rows=torch.empty(0, dtype=torch.long),
        recv_rows=recv_rows if recv_rows is not None else torch.empty(0, dtype=torch.long),
        send_dst_rank=torch.empty(0, dtype=torch.long),
        send_dst_rows=torch.empty(0, dtype=torch.long),
        recv_src_rank=recv_src_rank if recv_src_rank is not None else torch.empty(0, dtype=torch.long),
    )


def empty_state_write_route(*, window_id: int, world_size: int) -> dict[str, Any]:
    row = empty_exchange_route(world_size=world_size).as_dict()
    row.update(
        {
            "window_id": int(window_id),
            "event_rows": row["send_rows"],
            "node_ids": torch.empty(0, dtype=torch.long),
            "owner_rank": row["send_dst_rank"],
            "owner_rows": row["send_dst_rows"],
            "role": torch.empty(0, dtype=torch.uint8),
        }
    )
    return row



__all__ = [
    "DIST_INDEX_LOC_BITS",
    "DIST_INDEX_LOC_MASK",
    "ExchangeRoute",
    "RouteBook",
    "RouteData",
    "build_exchange_route",
    "build_state_write_mask",
    "build_state_write_routes_for_rank",
    "dist_loc",
    "dist_part",
    "empty_exchange_route",
    "empty_state_write_route",
]
