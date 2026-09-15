from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from starrygl.utils.index import compact_lookup_rows
from starrygl.view import GraphBlock


def event_rows_to_graph_block(view: Mapping[str, Any], edge_rows: Tensor, extra_nodes: Tensor | None = None) -> GraphBlock:
    src = view["src"].index_select(0, edge_rows).long()
    dst = view["dst"].index_select(0, edge_rows).long()
    edge_ids = view["edge_ids"].index_select(0, edge_rows).long()
    node_pieces = [src, dst]
    if extra_nodes is not None and int(extra_nodes.numel()) > 0:
        node_pieces.append(extra_nodes.long())
    nodes = torch.unique(torch.cat(node_pieces, dim=0), sorted=True) if node_pieces else torch.empty(0, dtype=torch.long)
    local_src, local_dst = local_edge_index(nodes, src, dst)
    edata = {}
    if "ts" in view:
        edata["ts"] = view["ts"].index_select(0, edge_rows)
    return GraphBlock(
        src_nodes=nodes,
        dst_nodes=nodes,
        edge_ids=edge_ids,
        format="coo",
        row=local_src,
        col=local_dst,
        edge_index=(
            torch.stack((local_src, local_dst), dim=0)
            if int(local_src.numel())
            else torch.empty((2, 0), dtype=torch.long, device=nodes.device)
        ),
        num_src=int(nodes.numel()),
        num_dst=int(nodes.numel()),
        edata=edata,
        exec_mode="LOCAL_EVENT",
    )


def root_nodes_to_graph_block(root_nodes: Tensor, root_ts: Tensor | None = None) -> GraphBlock:
    nodes = torch.unique(root_nodes.long(), sorted=True) if int(root_nodes.numel()) else root_nodes.new_empty((0,))
    empty = nodes.new_empty((0,))
    srcdata = {}
    if root_ts is not None and int(root_nodes.numel()) > 0:
        unique_nodes, inverse = torch.unique(root_nodes.long(), sorted=True, return_inverse=True)
        pos = torch.arange(int(root_nodes.numel()), dtype=torch.long, device=root_nodes.device)
        latest = torch.full((int(unique_nodes.numel()),), -1, dtype=torch.long, device=root_nodes.device)
        latest.scatter_reduce_(0, inverse, pos, reduce="amax", include_self=True)
        srcdata["ts"] = root_ts.to(device=root_nodes.device).reshape(-1).index_select(0, latest.to(device=root_nodes.device))
        nodes = unique_nodes
    return GraphBlock(
        src_nodes=nodes,
        dst_nodes=nodes,
        edge_ids=empty,
        format="coo",
        row=empty,
        col=empty,
        edge_index=torch.empty((2, 0), dtype=torch.long, device=nodes.device),
        num_src=int(nodes.numel()),
        num_dst=int(nodes.numel()),
        srcdata=srcdata,
        edata={},
        exec_mode="LOCAL_EVENT",
    )


def snapshot_row_to_graph_block(
    row: Mapping[str, Any],
    *,
    attach_reverse: bool = True,
    precompute_edge_rows: bool = False,
    use_sparse_gcn: bool = False,
    use_dgl_gcn: bool = False,
) -> GraphBlock:
    edge_row = edge_col = None
    if bool(precompute_edge_rows) and "indptr" in row and "indices" in row:
        counts = row["indptr"].long()[1:] - row["indptr"].long()[:-1]
        edge_col = torch.repeat_interleave(
            torch.arange(int(counts.numel()), dtype=torch.long, device=row["indices"].device),
            counts,
        )
        edge_row = row["indices"].long()
    block = GraphBlock(
        src_nodes=row["src_nodes"].long(),
        dst_nodes=row["dst_nodes"].long(),
        edge_ids=row["edge_ids"].long(),
        format="csc",
        indptr=row["indptr"].long(),
        indices=row["indices"].long(),
        row=edge_row,
        col=edge_col,
        num_src=int(row["src_nodes"].numel()),
        num_dst=int(row["dst_nodes"].numel()),
        edata={"ts": row["ts"]} if "ts" in row else {},
        route=row.get("route"),
        exec_mode="DISTRIBUTED_FULL" if store_world_size(row) > 1 else "LOCAL_FULL",
    )
    if "edge_gcn_norm" in row:
        block.edata["gcn_norm"] = row["edge_gcn_norm"].float()
    if "self_gcn_norm" in row:
        block.edata["self_gcn_norm"] = row["self_gcn_norm"].float()
    block.cache["snapshot_id"] = int(row.get("snapshot_id", 0))
    block.cache["chunk_limited"] = bool(row.get("chunk_limited", False))
    route = row.get("route")
    send_index = route.get("send_index") if isinstance(route, Mapping) else getattr(route, "send_index", None)
    if isinstance(send_index, Tensor):
        # Prepared rows are on CPU here; preserve this bound across H2D.
        block.cache["embedding_send_rows"] = int(send_index.max().item()) + 1 if send_index.numel() else 0
    if "diffusion" in row:
        block.cache["diffusion"] = row["diffusion"]
    if "edge_feature_ids" in row:
        block.cache["edge_feature_ids"] = row["edge_feature_ids"].long()
    if "src_feature_row" in row:
        block.cache["src_feature_row"] = row["src_feature_row"].long()
    if "src_data_row" in row:
        block.cache["src_data_row"] = row["src_data_row"].long()
    if isinstance(row.get("node_data"), Mapping):
        block.cache["node_data"] = row["node_data"]
    if bool(use_sparse_gcn):
        block.cache["use_sparse_tensor_gcn"] = True
    if bool(use_dgl_gcn):
        block.cache["use_dgl_gcn"] = True
    src_state_rows = row.get("src_feature_row")
    block.cache["src_state_rows"] = (
        src_state_rows.long()
        if isinstance(src_state_rows, Tensor) and int(src_state_rows.numel()) == int(block.num_src or 0)
        else row["src_nodes"].long()
    )
    dst_state_rows = row.get("dst_feature_row")
    block.cache["dst_state_rows"] = (
        dst_state_rows.long()
        if isinstance(dst_state_rows, Tensor) and int(dst_state_rows.numel()) == int(block.num_dst or 0)
        else row["dst_nodes"].long()
    )
    if bool(attach_reverse):
        attach_reverse_direction(block)
    return block


def snapshot_graph_block(
    row: Mapping[str, Any],
    *,
    attach_reverse: bool,
    precompute_edge_rows: bool = False,
    use_sparse_gcn: bool = False,
    use_dgl_gcn: bool = False,
) -> GraphBlock:
    return snapshot_row_to_graph_block(
        row,
        attach_reverse=bool(attach_reverse),
        precompute_edge_rows=bool(precompute_edge_rows),
        use_sparse_gcn=bool(use_sparse_gcn),
        use_dgl_gcn=bool(use_dgl_gcn),
    )


def move_graph_block(
    block: GraphBlock,
    device: torch.device,
    memo: dict[int, GraphBlock] | None = None,
    *,
    non_blocking: bool = False,
    pin_memory: bool = False,
) -> GraphBlock:
    memo = {} if memo is None else memo
    key = id(block)
    if key in memo:
        return memo[key]
    if _graph_block_on_device(block, device):
        memo[key] = block
        return block
    moved = GraphBlock(
        src_nodes=_move_tensor(block.src_nodes, device, non_blocking, pin_memory),
        dst_nodes=_move_tensor(block.dst_nodes, device, non_blocking, pin_memory),
        edge_ids=_move_tensor(block.edge_ids, device, non_blocking, pin_memory),
        format=block.format,
        indptr=None if block.indptr is None else _move_tensor(block.indptr, device, non_blocking, pin_memory),
        indices=None if block.indices is None else _move_tensor(block.indices, device, non_blocking, pin_memory),
        row=None if block.row is None else _move_tensor(block.row, device, non_blocking, pin_memory),
        col=None if block.col is None else _move_tensor(block.col, device, non_blocking, pin_memory),
        edge_index=None if block.edge_index is None else _move_tensor(block.edge_index, device, non_blocking, pin_memory),
        num_src=block.num_src,
        num_dst=block.num_dst,
        srcdata=_move_graph_value(block.srcdata, device, non_blocking, pin_memory),
        dstdata=_move_graph_value(block.dstdata, device, non_blocking, pin_memory),
        edata=_move_graph_value(block.edata, device, non_blocking, pin_memory),
        cache={
            name: value
            if name == "node_dist_index"
            else _move_graph_value(value, device, non_blocking, pin_memory)
            for name, value in block.cache.items()
            if not str(name).startswith("node_row_map:")
        },
        route=_move_graph_value(block.route, device, non_blocking, pin_memory),
        exec_mode=block.exec_mode,
    )
    memo[key] = moved
    return moved


def attach_reverse_direction(block: GraphBlock) -> None:
    if block.indptr is None or block.indices is None:
        return
    num_dst = int(block.num_dst or block.dst_nodes.numel())
    if num_dst <= 0 or int(block.indices.numel()) == 0:
        empty = block.indices.new_empty((0,))
        block.cache["reverse_row"] = empty
        block.cache["reverse_col"] = empty
        return
    counts = block.indptr[1:] - block.indptr[:-1]
    forward_dst = torch.repeat_interleave(torch.arange(num_dst, dtype=torch.long, device=block.indices.device), counts)
    forward_src = block.indices.long()
    keep = forward_src < num_dst
    if bool(torch.any(keep).item()):
        keep_index = torch.nonzero(keep, as_tuple=True)[0]
        reverse_row = forward_dst.index_select(0, keep_index).long()
        reverse_col = forward_src.index_select(0, keep_index).long()
        block.cache["reverse_row"] = reverse_row
        block.cache["reverse_col"] = reverse_col
        deg = torch.bincount(reverse_col.cpu(), minlength=num_dst).clamp_min_(1).to(dtype=torch.float32)
        block.edata["reverse_gcn_norm"] = deg.index_select(0, reverse_col.cpu()).reciprocal()
    else:
        empty = block.indices.new_empty((0,))
        block.cache["reverse_row"] = empty
        block.cache["reverse_col"] = empty
        block.edata["reverse_gcn_norm"] = torch.empty(0, dtype=torch.float32)


def local_edge_index(nodes: Tensor, src: Tensor, dst: Tensor) -> tuple[Tensor, Tensor]:
    if int(nodes.numel()) == 0:
        empty = torch.empty(0, dtype=torch.long, device=nodes.device)
        return empty, empty
    return compact_lookup_rows(nodes, src), compact_lookup_rows(nodes, dst)


def store_world_size(row: Mapping[str, Any]) -> int:
    route = row.get("route")
    if isinstance(route, Mapping) and "send_sizes" in route:
        return len(route["send_sizes"])
    return 1


def move_cache_tensor(key: object) -> bool:
    name = str(key)
    return not (name.startswith("node_row_map:") or name == "node_dist_index")


def _move_graph_value(
    value: Any,
    device: torch.device,
    non_blocking: bool = False,
    pin_memory: bool = False,
) -> Any:
    if isinstance(value, Tensor):
        return _move_tensor(value, device, non_blocking, pin_memory)
    if isinstance(value, Mapping):
        return {
            key: _move_graph_value(item, device, non_blocking, pin_memory)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_move_graph_value(item, device, non_blocking, pin_memory) for item in value)
    if isinstance(value, list):
        return [_move_graph_value(item, device, non_blocking, pin_memory) for item in value]
    move = getattr(value, "to", None)
    if callable(move):
        return move(device=device, non_blocking=non_blocking)
    return value


def _move_tensor(
    value: Tensor,
    device: torch.device,
    non_blocking: bool,
    pin_memory: bool,
) -> Tensor:
    source = (
        value.pin_memory()
        if pin_memory and value.device.type == "cpu" and not value.is_pinned()
        else value
    )
    return source.to(device=device, non_blocking=non_blocking)


def _graph_block_on_device(block: GraphBlock, device: torch.device) -> bool:
    tensors = (
        block.src_nodes,
        block.dst_nodes,
        block.edge_ids,
        block.indptr,
        block.indices,
        block.row,
        block.col,
        block.edge_index,
    )
    if any(isinstance(value, Tensor) and value.device != device for value in tensors):
        return False
    values = (*block.srcdata.values(), *block.dstdata.values(), *block.edata.values())
    return all(not isinstance(value, Tensor) or value.device == device for value in values)


__all__ = [
    "attach_reverse_direction",
    "event_rows_to_graph_block",
    "local_edge_index",
    "move_cache_tensor",
    "move_graph_block",
    "root_nodes_to_graph_block",
    "snapshot_graph_block",
    "snapshot_row_to_graph_block",
    "store_world_size",
]
