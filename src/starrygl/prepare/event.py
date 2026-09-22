from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from starrygl.utils.route import build_state_write_routes_for_rank, dist_part


def build_event_view_for_rank(
    *,
    rank: int,
    src: Tensor,
    dst: Tensor,
    ts: Tensor | None,
    edge_master: Tensor,
    time_ptr_2: Tensor,
    split_time_ptr_2: dict[str, Tensor],
    state_write_mask: Tensor,
    node_dist_index: Tensor,
    node_is_hot: Tensor,
    hot_node_ids: Tensor,
    global_dst_pool: Tensor,
    world_size: int,
    include_state_write_routes: bool,
) -> dict[str, Any]:
    local_global = (edge_master == rank).nonzero(as_tuple=True)[0].long()
    local_dst = dst.index_select(0, local_global)
    local_nodes = (dist_part(node_dist_index) == rank) | node_is_hot
    local_dst_pool = global_dst_pool[local_nodes.index_select(0, global_dst_pool.long())]
    local_pos = torch.full((int(edge_master.numel()),), -1, dtype=torch.long)
    local_pos[local_global] = torch.arange(int(local_global.numel()))
    row = {
        "rank": rank,
        "edge_ids": local_global,
        "src": src.index_select(0, local_global),
        "dst": local_dst,
        "ts": ts.index_select(0, local_global) if ts is not None else torch.empty(0, dtype=torch.float32),
        "dst_pool": local_dst_pool,
        "global_dst_pool": global_dst_pool,
        "hot_node_ids": hot_node_ids.long(),
        "time_ptr_2": _localize_time_ptr_2(time_ptr_2=time_ptr_2, local_global=local_global),
        "window_end_ts": _window_end_ts(ts=ts, time_ptr_2=time_ptr_2),
        "split_time_ptr_2": {
            name: _localize_time_ptr_2(time_ptr_2=ptr, local_global=local_global)
            for name, ptr in split_time_ptr_2.items()
        },
        "split_window_end_ts": {
            name: _window_end_ts(ts=ts, time_ptr_2=ptr)
            for name, ptr in split_time_ptr_2.items()
        },
        "state_write_mask": state_write_mask.index_select(0, local_global),
    }
    if include_state_write_routes:
        common = dict(
            rank=rank,
            edge_master=edge_master,
            local_global=local_global,
            local_pos=local_pos,
            src=src,
            dst=dst,
            time_ptr_2=time_ptr_2,
            state_write_mask=state_write_mask,
            node_dist_index=node_dist_index,
            node_is_hot=node_is_hot,
            world_size=world_size,
        )
        row["state_write_routes"] = build_state_write_routes_for_rank(**common, hot=False)
        row["hot_state_write_events"] = build_state_write_routes_for_rank(**common, hot=True)
    return row


def _localize_time_ptr_2(*, time_ptr_2: Tensor, local_global: Tensor) -> Tensor:
    if not int(time_ptr_2.numel()):
        return torch.empty((0, 2), dtype=torch.long)
    if not int(local_global.numel()):
        return torch.zeros((int(time_ptr_2.shape[0]), 2), dtype=torch.long)
    bounds = time_ptr_2.long().reshape(-1)
    return torch.searchsorted(local_global.long().contiguous(), bounds).reshape(-1, 2).long().contiguous()


def _window_end_ts(*, ts: Tensor | None, time_ptr_2: Tensor) -> Tensor:
    if ts is None or not int(time_ptr_2.numel()):
        return torch.empty(int(time_ptr_2.shape[0]), dtype=torch.float32)
    ptr = time_ptr_2.long().reshape(-1, 2)
    out = torch.full((int(ptr.shape[0]),), float("nan"), dtype=ts.dtype)
    valid = ptr[:, 1] > ptr[:, 0]
    out[valid] = ts.index_select(0, ptr[valid, 1] - 1)
    return out.contiguous()


__all__ = ["build_event_view_for_rank"]
