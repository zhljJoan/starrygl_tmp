from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from starrygl.utils.route import dist_loc, dist_part


_SPLITS = ("train", "val", "test")
_NODE_TASKS = {"node_prediction", "node_classification", "node_regression"}


def build_task_shards(
    *,
    prepared: Any,
    task: Mapping[str, Any] | str,
    temporal: str,
    src: Tensor,
    dst: Tensor,
    ts: Tensor,
    edge_ids: Tensor,
    node_label: Tensor | None,
    node_label_nodes: Tensor | None,
    node_label_ts: Tensor | None,
    node_label_split: Tensor | None,
    node_label_temporal: bool,
    node_label_horizon: int,
    edge_label: Tensor | None,
) -> list[dict[str, Any]]:
    """Build one owner-local flat task table for every rank."""

    name = str(task.get("name", "") if isinstance(task, Mapping) else task).strip().lower()
    if name != "edge_prediction" and name not in _NODE_TASKS:
        raise ValueError(f"unsupported prepared task: {name!r}")
    if temporal not in {"event", "snapshot"}:
        raise ValueError("temporal must be event or snapshot")

    world_size = int(prepared.meta.get("world_size", 1))
    node_index = prepared.partition["node_dist_index"].long()
    edge_index = prepared.partition["edge_dist_index"].long()
    if name == "edge_prediction":
        return [
            _edge_table(
                prepared=prepared,
                rank=rank,
                temporal=temporal,
                edge_part=dist_part(edge_index),
                src=src,
                dst=dst,
                ts=ts,
                edge_ids=edge_ids,
                edge_label=edge_label,
            )
            for rank in range(world_size)
        ]
    return [
        _node_table(
            prepared=prepared,
            rank=rank,
            temporal=temporal,
            node_part=dist_part(node_index),
            node_loc=dist_loc(node_index),
            ts=ts,
            node_label=node_label,
            node_label_nodes=node_label_nodes,
            node_label_ts=node_label_ts,
            node_label_split=node_label_split,
            node_label_temporal=bool(node_label_temporal),
            horizon=max(0, int(node_label_horizon)),
        )
        for rank in range(world_size)
    ]


def _edge_table(
    *,
    prepared: Any,
    rank: int,
    temporal: str,
    edge_part: Tensor,
    src: Tensor,
    dst: Tensor,
    ts: Tensor,
    edge_ids: Tensor,
    edge_label: Tensor | None,
) -> dict[str, Any]:
    targets = torch.arange(int(prepared.time_ptr_2.shape[0]), dtype=torch.long)
    if temporal == "snapshot":
        targets.fill_(-1)
        for begin, end in _window_scopes(prepared):
            if end - begin > 1:
                targets[begin : end - 1] = torch.arange(begin + 1, end)

    rows = []
    for target in targets.tolist():
        if target < 0:
            physical = torch.empty(0, dtype=torch.long)
        else:
            begin, end = prepared.time_ptr_2[int(target)].long().tolist()
            physical = torch.arange(int(begin), int(end), dtype=torch.long)
            physical = physical[edge_part.index_select(0, physical) == int(rank)]
        row = {
            "src": src.index_select(0, physical),
            "dst": dst.index_select(0, physical),
            "edge_ids": edge_ids.index_select(0, physical),
            "edge_rows": physical,
        }
        if temporal == "event":
            row["cutoff_ts"] = ts.index_select(0, physical)
        if edge_label is not None:
            row["label"] = edge_label.index_select(0, physical.to(edge_label.device))
        rows.append(row)
    return {"task_kind": "edge", **_pack_rows(rows)}


def _node_table(
    *,
    prepared: Any,
    rank: int,
    temporal: str,
    node_part: Tensor,
    node_loc: Tensor,
    ts: Tensor,
    node_label: Tensor | None,
    node_label_nodes: Tensor | None,
    node_label_ts: Tensor | None,
    node_label_split: Tensor | None,
    node_label_temporal: bool,
    horizon: int,
) -> dict[str, Any]:
    count = int(prepared.time_ptr_2.shape[0])
    valid = torch.ones(count, dtype=torch.bool)
    if temporal == "snapshot" and horizon:
        for begin, end in _window_scopes(prepared):
            valid[max(begin, end - horizon) : end] = False

    rows: list[dict[str, Tensor]] = []
    if node_label_nodes is not None:
        label_nodes = node_label_nodes.long()
        owned = node_part.index_select(0, label_nodes) == int(rank)
        assigned = _label_windows(
            prepared,
            ts=ts,
            label_ts=node_label_ts,
            label_split=node_label_split,
        )
        if temporal == "event" and assigned is None:
            raise ValueError("event node tasks require timestamped node labels")
        for window_id in range(count):
            mask = owned & valid[window_id]
            if assigned is not None:
                mask &= assigned == window_id
            pos = mask.nonzero(as_tuple=True)[0]
            row = {"node_ids": label_nodes.index_select(0, pos)}
            if node_label is not None:
                row["label"] = node_label.index_select(0, pos.to(node_label.device))
            if node_label_ts is not None:
                row["cutoff_ts"] = node_label_ts.index_select(0, pos.to(node_label_ts.device))
            rows.append(row)
        return {"task_kind": "node", **_pack_rows(rows)}

    if temporal == "event":
        raise ValueError("event node tasks require node_label_nodes and node_label_ts")
    node_ids = (node_part == int(rank)).nonzero(as_tuple=True)[0].long()
    if int(node_ids.numel()):
        order = torch.argsort(node_loc.index_select(0, node_ids), stable=True)
        node_ids = node_ids.index_select(0, order)
    for window_id in range(count):
        ids = node_ids if bool(valid[window_id]) else node_ids.new_empty((0,))
        row = {"node_ids": ids}
        if node_label is not None:
            values = node_label[int(window_id)] if node_label_temporal else node_label
            row["label"] = values.index_select(0, ids.to(values.device))
        rows.append(row)
    return {"task_kind": "node", **_pack_rows(rows)}


def _label_windows(
    prepared: Any,
    *,
    ts: Tensor,
    label_ts: Tensor | None,
    label_split: Tensor | None,
) -> Tensor | None:
    if label_ts is None:
        return None
    assigned = torch.full((int(label_ts.numel()),), -1, dtype=torch.long)
    scopes = _window_scopes(prepared)
    split_values = None if label_split is None else label_split.long()
    if split_values is None:
        scopes = ((0, int(prepared.time_ptr_2.shape[0])),)
    for split_id, (begin, end) in enumerate(scopes):
        ptr = prepared.time_ptr_2[begin:end].long()
        valid = (ptr[:, 1] > ptr[:, 0]).nonzero(as_tuple=True)[0]
        if not int(valid.numel()):
            continue
        pos = (
            torch.arange(int(label_ts.numel()), dtype=torch.long)
            if split_values is None
            else (split_values == split_id).nonzero(as_tuple=True)[0]
        )
        if not int(pos.numel()):
            continue
        ends = ts.index_select(0, ptr.index_select(0, valid)[:, 1] - 1)
        values = label_ts.index_select(0, pos.to(label_ts.device)).to(ends)
        local = torch.searchsorted(ends.contiguous(), values.contiguous())
        local.clamp_max_(int(valid.numel()) - 1)
        assigned.index_copy_(0, pos, valid.index_select(0, local.cpu()) + begin)
    return assigned


def _window_scopes(prepared: Any) -> tuple[tuple[int, int], ...]:
    counts = [
        int(prepared.split_time_ptr_2[name].shape[0])
        for name in _SPLITS
        if name in prepared.split_time_ptr_2
    ]
    if not counts:
        return ((0, int(prepared.time_ptr_2.shape[0])),)
    ends = torch.tensor(counts, dtype=torch.long).cumsum(0).tolist()
    begins = [0, *ends[:-1]]
    return tuple((int(begin), int(end)) for begin, end in zip(begins, ends))


def _pack_rows(rows: list[dict[str, Tensor]]) -> dict[str, Any]:
    lengths = torch.tensor(
        [int(next(iter(row.values())).shape[0]) for row in rows], dtype=torch.long
    )
    ptr = torch.zeros(len(rows) + 1, dtype=torch.long)
    ptr[1:] = lengths.cumsum(0)
    keys = tuple(rows[0]) if rows else ()
    payload = {
        key: torch.cat([row[key] for row in rows], dim=0)
        for key in keys
    }
    return {"task_ptr": ptr, "task_payload": payload}


__all__ = ["build_task_shards"]
