from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from starrygl.prepare.temporal_csr import compressed
from starrygl.utils.route import dist_loc


def build_snapshot_csc_views(
    *,
    src: Tensor,
    dst: Tensor,
    ts: Tensor | None,
    edge_ids: Tensor,
    edge_dist_index: Tensor,
    node_master: Tensor,
    hot_node_ids: Tensor,
    node_is_hot: Tensor,
    node_to_chunk: Tensor,
    time_ptr_2: Tensor,
    num_nodes: int,
    world_size: int,
    edge_weight: Tensor | None = None,
    diffusion: bool = False,
    hot_compute: bool = False,
) -> list[dict[str, Any]]:
    views = [
        {"rank": rank, "format": "snapshot_csc", "slices": []}
        for rank in range(world_size)
    ]
    dst_nodes_by_rank = [(node_master == rank).nonzero(as_tuple=True)[0].long() for rank in range(world_size)]
    if hot_compute:
        dst_nodes_by_rank = [torch.cat((nodes, hot_node_ids[node_master[hot_node_ids] != rank]))
                             for rank, nodes in enumerate(dst_nodes_by_rank)]
    owned_nodes = [nodes[~node_is_hot.index_select(0, nodes)] if int(nodes.numel()) else nodes for nodes in dst_nodes_by_rank]
    dst_rows = []
    dst_feature_rows = []
    hot_row = torch.full((num_nodes,), -1, dtype=torch.long)
    if int(hot_node_ids.numel()):
        hot_row[hot_node_ids] = torch.arange(int(hot_node_ids.numel()))
    for rank, dst_nodes in enumerate(dst_nodes_by_rank):
        row = torch.full((num_nodes,), -1, dtype=torch.long)
        if int(dst_nodes.numel()):
            row[dst_nodes] = torch.arange(int(dst_nodes.numel()))
        dst_rows.append(row)
        feature_rows = hot_row.index_select(0, dst_nodes) if int(dst_nodes.numel()) else torch.empty(0, dtype=torch.long)
        if int(owned_nodes[rank].numel()):
            feature_rows[row.index_select(0, owned_nodes[rank])] = torch.arange(
                int(hot_node_ids.numel()), int(hot_node_ids.numel()) + int(owned_nodes[rank].numel())
            )
        dst_feature_rows.append(feature_rows)

    for snapshot_id, (begin, end) in enumerate(time_ptr_2.tolist()):
        edge_range = torch.arange(begin, end, dtype=torch.long)
        edge_norm, self_norm = _gcn_norm(
            src=src,
            dst=dst,
            edge_range=edge_range,
            edge_weight=edge_weight,
            num_nodes=num_nodes,
        )
        rows = []
        recv_nodes: list[list[Tensor]] = [[] for _ in range(world_size)]
        if diffusion:
            snap_src, snap_dst = src[edge_range], dst[edge_range]
            weight = torch.ones(len(edge_range)) if edge_weight is None else edge_weight[edge_range].float()
            weight = weight * (snap_src != snap_dst)
            out_degree = torch.ones(num_nodes).index_add_(0, snap_src, weight)
            in_degree = torch.ones(num_nodes).index_add_(0, snap_dst, weight)
        for rank in range(world_size):
            local_edges = edge_range[dst_rows[rank][dst[edge_range]] >= 0]
            local_src = src.index_select(0, local_edges)
            local_dst = dst.index_select(0, local_edges)
            dst_nodes = dst_nodes_by_rank[rank]
            remote_src = torch.unique(local_src[dst_rows[rank][local_src] < 0], sorted=True)
            if diffusion:
                outgoing = edge_range[dst_rows[rank][src[edge_range]] >= 0]
                reverse_src = dst[outgoing]
                remote_src = torch.unique(torch.cat((remote_src, reverse_src[dst_rows[rank][reverse_src] < 0])), sorted=True)
            nonhot_remote = remote_src[~node_is_hot.index_select(0, remote_src)] if int(remote_src.numel()) else remote_src
            node_feature_ids = torch.cat((owned_nodes[rank], nonhot_remote))
            src_nodes = torch.cat((dst_nodes, remote_src))
            local_dst_rows = dst_rows[rank].index_select(0, local_dst) if int(local_dst.numel()) else local_dst
            local_src_rows = dst_rows[rank].index_select(0, local_src) if int(local_src.numel()) else local_src
            if int(remote_src.numel()) and int(local_src.numel()):
                remote = local_src_rows < 0
                local_src_rows[remote] = int(dst_nodes.numel()) + torch.searchsorted(remote_src, local_src[remote])
            src_feature_row = dst_feature_rows[rank]
            if int(remote_src.numel()):
                remote_hot = node_is_hot.index_select(0, remote_src)
                remote_rows = torch.empty(int(remote_src.numel()), dtype=torch.long)
                if bool(remote_hot.any()):
                    remote_rows[remote_hot] = hot_row.index_select(0, remote_src[remote_hot])
                if bool((~remote_hot).any()):
                    remote_rows[~remote_hot] = (
                        int(hot_node_ids.numel()) + int(owned_nodes[rank].numel())
                        + torch.searchsorted(nonhot_remote, remote_src[~remote_hot])
                    )
                src_feature_row = torch.cat((src_feature_row, remote_rows))
            node_feature_row = torch.arange(
                int(hot_node_ids.numel()), int(hot_node_ids.numel()) + int(owned_nodes[rank].numel())
            )
            if int(nonhot_remote.numel()):
                node_feature_row = torch.cat((node_feature_row, torch.arange(
                    int(hot_node_ids.numel()) + int(owned_nodes[rank].numel()),
                    int(hot_node_ids.numel()) + int(owned_nodes[rank].numel()) + int(nonhot_remote.numel()),
                )))
            indptr, indices, order = compressed(
                primary=local_dst_rows,
                indices=local_src_rows,
                num_nodes=int(dst_nodes.numel()),
                sort_keys=(node_to_chunk.index_select(0, local_src), ts.index_select(0, local_edges) if ts is not None else local_edges),
            )
            for peer in range(world_size):
                recv_nodes[rank].append(remote_src[node_master.index_select(0, remote_src) == peer])
            ordered_edges = local_edges.index_select(0, order)
            rows.append({
                "snapshot_id": snapshot_id,
                "src_nodes": src_nodes,
                "dst_nodes": dst_nodes,
                "edge_ids": edge_ids.index_select(0, ordered_edges) if int(local_edges.numel()) else local_edges,
                "ts": ts.index_select(0, ordered_edges) if ts is not None else torch.empty(0, dtype=torch.float32),
                "indptr": indptr,
                "indices": indices,
                "node_chunk": node_to_chunk.index_select(0, src_nodes) if int(src_nodes.numel()) else torch.empty(0, dtype=torch.long),
                "node_feature_ids": node_feature_ids,
                "node_feature_row": node_feature_row,
                "src_feature_row": src_feature_row,
                "edge_feature_ids": ordered_edges,
                "edge_feature_row": dist_loc(edge_dist_index.index_select(0, ordered_edges)) if int(local_edges.numel()) else local_edges,
                "edge_gcn_norm": edge_norm.index_select(0, ordered_edges - begin) if int(local_edges.numel()) else torch.empty(0),
                "self_gcn_norm": self_norm.index_select(0, dst_nodes) if int(dst_nodes.numel()) else torch.empty(0),
            })
            if diffusion:
                source_rows = torch.full((num_nodes,), -1, dtype=torch.long)
                source_rows[src_nodes] = torch.arange(len(src_nodes))
                rows[-1]["diffusion"] = {
                    "forward_norm": weight[ordered_edges - begin] / out_degree[src[ordered_edges]],
                    "reverse_row": source_rows[dst[outgoing]],
                    "reverse_col": dst_rows[rank][src[outgoing]],
                    "reverse_norm": weight[outgoing - begin] / in_degree[dst[outgoing]],
                    "forward_self": out_degree[dst_nodes].reciprocal(),
                    "reverse_self": in_degree[dst_nodes].reciprocal(),
                }
        for rank in range(world_size):
            row = dict(rows[rank])
            row["route"] = _collective_route(rank=rank, src_nodes=rows[rank]["src_nodes"], recv_nodes=recv_nodes, world_size=world_size)
            views[rank]["slices"].append(row)
    return views


def _gcn_norm(
    *,
    src: Tensor,
    dst: Tensor,
    edge_range: Tensor,
    edge_weight: Tensor | None,
    num_nodes: int,
) -> tuple[Tensor, Tensor]:
    if not int(edge_range.numel()):
        return torch.empty(0), torch.ones(num_nodes).reciprocal()
    local_src, local_dst = src[edge_range].long(), dst[edge_range].long()
    out_degree, in_degree = torch.ones(num_nodes), torch.ones(num_nodes)
    weight = (
        torch.ones(int(edge_range.numel()))
        if edge_weight is None
        else edge_weight.index_select(0, edge_range).float()
    )
    out_degree.index_add_(0, local_src, weight)
    in_degree.index_add_(0, local_dst, weight)
    return (
        (weight * torch.rsqrt(out_degree[local_src] * in_degree[local_dst])).nan_to_num_(0.0),
        torch.rsqrt(out_degree * in_degree).nan_to_num_(0.0),
    )


def _collective_route(
    *, rank: int, src_nodes: Tensor, recv_nodes: list[list[Tensor]], world_size: int
) -> dict[str, Any]:
    row = torch.full((int(src_nodes.max()) + 1,), -1, dtype=torch.long) if int(src_nodes.numel()) else torch.empty(0, dtype=torch.long)
    if int(src_nodes.numel()):
        row[src_nodes] = torch.arange(int(src_nodes.numel()))
    send = []
    for peer in range(world_size):
        requested = recv_nodes[peer][rank]
        if int(requested.numel()) and int(requested.max()) >= int(row.numel()):
            raise ValueError("route requested a node missing from provider src layout")
        send.append(row.index_select(0, requested.long()) if int(requested.numel()) else torch.empty(0, dtype=torch.long))
    recv = recv_nodes[rank]
    recv_rows = [row.index_select(0, nodes.long()) if int(nodes.numel()) else torch.empty(0, dtype=torch.long) for nodes in recv]
    return {
        "send_sizes": [int(value.numel()) for value in send],
        "recv_sizes": [int(value.numel()) for value in recv],
        "send_index": torch.cat(send).long() if send else torch.empty(0, dtype=torch.long),
        "recv_src_row": torch.cat(recv_rows).long() if recv_rows else torch.empty(0, dtype=torch.long),
    }


__all__ = ["build_snapshot_csc_views"]
