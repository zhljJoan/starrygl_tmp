from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


DIST_INDEX_LOC_BITS = 48
DIST_INDEX_LOC_MASK = (1 << DIST_INDEX_LOC_BITS) - 1


def build_static_feature_shards(
    *,
    prepared: Any,
    node_feat: Tensor | None = None,
    edge_feat: Tensor | None = None,
    include_static_one_hop: bool = False,
    replicate_node_features: bool = False,
) -> list[dict[str, Tensor]]:
    partition = prepared.partition
    world_size = int(prepared.meta.get("world_size", 1))
    hot_node_ids = partition.get("hot_node_ids", torch.empty(0, dtype=torch.long)).long()
    hot_count = int(partition.get("hot_count", torch.tensor(int(hot_node_ids.numel()))).item())
    node_index = partition["node_dist_index"].long()
    edge_index = partition["edge_dist_index"].long()
    node_part = _dist_part(node_index)
    node_loc = _dist_loc(node_index)
    edge_part = _dist_part(edge_index)
    edge_loc = _dist_loc(edge_index)

    out = []
    for rank in range(world_size):
        if bool(replicate_node_features):
            node_ids = torch.arange(int(node_index.numel()), dtype=torch.long)
        else:
            node_ids = _node_feature_ids_for_rank(
                rank=rank,
                hot_node_ids=hot_node_ids,
                hot_count=hot_count,
                node_part=node_part,
                node_loc=node_loc,
                prepared=prepared,
                include_static_one_hop=bool(include_static_one_hop),
            )
        edge_ids = _edge_feature_ids_for_rank(
            rank=rank,
            edge_part=edge_part,
            edge_loc=edge_loc,
            prepared=prepared,
            include_snapshot_edges=bool(include_static_one_hop),
        )
        row = {
            "rank": torch.tensor(rank, dtype=torch.long),
            "node_ids": node_ids,
            "edge_ids": edge_ids,
        }
        if bool(replicate_node_features):
            row["node_features_replicated"] = torch.tensor(True, dtype=torch.bool)
        row["node_row_map"] = _dense_row_map(row["node_ids"], size=int(node_index.numel()))
        row["edge_row_map"] = _dense_row_map(row["edge_ids"], size=int(edge_index.numel()))
        if node_feat is not None:
            row["node_feat"] = (
                node_feat.index_select(1, node_ids)
                if node_feat.dim() >= 3
                else node_feat.index_select(0, node_ids)
            )
        if edge_feat is not None:
            row["edge_feat"] = edge_feat.index_select(0, edge_ids)
        out.append(row)
    return out


def write_prepare_artifacts(
    root: str | Any,
    *,
    prepared: Any,
    feature_shards: list[dict[str, Tensor]] | None = None,
    label_shards: list[dict[str, Any]] | None = None,
    feature_layout: str = "separate",
) -> None:
    from pathlib import Path

    feature_layout = str(feature_layout).strip().lower()
    if feature_layout not in {"separate", "snapshot_csc"}:
        raise ValueError("feature_layout must be separate or snapshot_csc")

    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    prepare, graph_shards = _split_prepare_runtime_dict(prepared)
    prepare.setdefault("meta", {})["feature_layout"] = feature_layout
    features = {
        int(shard["rank"].item()): shard
        for shard in feature_shards or []
    }
    torch.save(prepare, path / "prepare.pt")
    temporal_csr = getattr(prepared, "temporal_csr_view", None)
    if isinstance(temporal_csr, dict) and temporal_csr:
        torch.save({"temporal_csr_view": temporal_csr}, path / "temporal_csr.pt")
    for rank, shard in enumerate(graph_shards):
        feature = features.get(rank)
        embedded = False
        if feature_layout == "snapshot_csc" and feature is not None:
            views = getattr(prepared, "snapshot_csc_views", ())
            view = views[rank] if rank < len(views) else {}
            embedded = _embed_snapshot_node_features(shard, view, feature)
        torch.save(shard, path / f"graph_{rank:03d}.pt")
        if feature is not None:
            separate = dict(feature)
            if embedded:
                separate.pop("node_feat", None)
            torch.save(separate, path / f"feature_{rank:03d}.pt")
    if label_shards is not None:
        for shard in label_shards:
            rank = int(shard["rank"].item())
            torch.save(shard, path / f"label_{rank:03d}.pt")


def _embed_snapshot_node_features(
    graph_shard: dict[str, Any],
    view: Any,
    feature_shard: dict[str, Tensor],
) -> bool:
    node_feat = feature_shard.get("node_feat")
    if not isinstance(node_feat, Tensor) or node_feat.dim() != 3:
        return False
    slices = view.get("slices", ()) if isinstance(view, dict) else ()
    if not slices:
        raise ValueError("snapshot_csc feature layout requires Snapshot-CSC slices")
    if int(node_feat.shape[0]) != len(slices):
        raise ValueError("temporal node features must have one row group per Snapshot-CSC slice")
    row_map = feature_shard.get("node_row_map")
    if not isinstance(row_map, Tensor):
        raise ValueError("snapshot_csc feature layout requires node_row_map")

    parts = []
    for snapshot_id, row in enumerate(slices):
        src_nodes = row["src_nodes"].long()
        if int(src_nodes.numel()) and int(src_nodes.max().item()) >= int(row_map.numel()):
            raise ValueError("Snapshot-CSC source node is outside node_row_map")
        local_rows = row_map.index_select(0, src_nodes.to(device=row_map.device))
        if bool((local_rows < 0).any().item()):
            raise ValueError("Snapshot-CSC source feature is missing from the rank feature shard")
        parts.append(node_feat[snapshot_id].index_select(0, local_rows.to(device=node_feat.device)))

    snapshot = dict(graph_shard.get("snapshot_csc_view", {}))
    lengths = torch.tensor([int(value.shape[0]) for value in parts], dtype=torch.long)
    ptr = torch.zeros(len(parts) + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(lengths, dim=0)
    snapshot["node_data"] = {"x": {"data": torch.cat(parts, dim=0), "ptr": ptr}}
    graph_shard["snapshot_csc_view"] = snapshot
    return True


def _split_prepare_runtime_dict(prepared: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row = prepared.as_dict()
    event_views = list(row.pop("event_views", []))
    snapshot_views = list(row.pop("snapshot_csc_views", []))
    row.pop("temporal_csr_view", {})
    world_size = int(row.get("meta", {}).get("world_size", max(len(event_views), len(snapshot_views), 1)))
    row["event_views"] = []
    row["snapshot_csc_views"] = []
    row["temporal_csr_view"] = {}
    row.setdefault("meta", {})["graph_shards"] = True
    graph_shards: list[dict[str, Any]] = []
    for rank in range(world_size):
        graph_shards.append(
            {
                "rank": torch.tensor(rank, dtype=torch.long),
                "event_view": event_views[rank] if rank < len(event_views) else {},
                "snapshot_csc_view": _slim_snapshot_view(snapshot_views[rank] if rank < len(snapshot_views) else {}),
                "temporal_csr_view": {},
            }
        )
    return row, graph_shards


def _slim_snapshot_view(view: Any) -> Any:
    if not isinstance(view, dict):
        return view
    out = dict(view)
    out.pop("routes", None)
    slices = out.get("slices", [])
    if not slices:
        out["slices"] = []
        return out
    first = slices[0] if isinstance(slices[0], dict) else {}
    dst_nodes = first.get("dst_nodes")
    dst_count = int(dst_nodes.numel()) if isinstance(dst_nodes, Tensor) else 0
    if dst_count > 0:
        out["dst_nodes"] = dst_nodes
        node_chunk = first.get("node_chunk")
        if isinstance(node_chunk, Tensor) and int(node_chunk.numel()) >= dst_count:
            out["dst_node_chunk"] = node_chunk.long()[:dst_count]
        src_feature_row = first.get("src_feature_row")
        if isinstance(src_feature_row, Tensor) and int(src_feature_row.numel()) >= dst_count:
            out["dst_feature_row"] = src_feature_row.long()[:dst_count]
    slim_slices = []
    for item in slices:
        if not isinstance(item, dict):
            slim_slices.append(item)
            continue
        row = dict(item)
        row.pop("node_feature_ids", None)
        row.pop("one_hop_src_nodes", None)
        row.pop("edge_feature_row", None)
        if dst_count > 0:
            src_nodes = row.get("src_nodes")
            if isinstance(src_nodes, Tensor) and int(src_nodes.numel()) >= dst_count:
                row["extra_src_nodes"] = src_nodes.long()[dst_count:]
                row.pop("src_nodes", None)
            row.pop("dst_nodes", None)
            node_chunk = row.get("node_chunk")
            if isinstance(node_chunk, Tensor) and int(node_chunk.numel()) >= dst_count:
                row["extra_node_chunk"] = node_chunk.long()[dst_count:]
                row.pop("node_chunk", None)
            src_feature_row = row.get("src_feature_row")
            if isinstance(src_feature_row, Tensor) and int(src_feature_row.numel()) >= dst_count:
                row["extra_src_feature_row"] = src_feature_row.long()[dst_count:]
                row.pop("src_feature_row", None)
        slim_slices.append(row)
    out["slices"] = slim_slices
    return _pack_snapshot_slices(out)


_SNAPSHOT_SLICE_TENSOR_FIELDS = (
    "extra_src_nodes",
    "extra_node_chunk",
    "extra_src_feature_row",
    "edge_ids",
    "ts",
    "indptr",
    "indices",
    "node_feature_row",
    "edge_feature_ids",
    "edge_gcn_norm",
    "self_gcn_norm",
)


def _pack_snapshot_slices(view: dict[str, Any]) -> dict[str, Any]:
    slices = view.get("slices", [])
    if not slices or not all(isinstance(item, dict) for item in slices):
        return view
    out = dict(view)
    rows = list(slices)
    out["slice_format"] = "columnar_v1"
    out["slice_count"] = torch.tensor(len(rows), dtype=torch.long)
    out["snapshot_ids"] = torch.tensor([int(row.get("snapshot_id", idx)) for idx, row in enumerate(rows)], dtype=torch.long)

    tensors: dict[str, Tensor] = {}
    ptrs: dict[str, Tensor] = {}
    for field in _SNAPSHOT_SLICE_TENSOR_FIELDS:
        values = [row.get(field) for row in rows]
        first_tensor = next((value for value in values if isinstance(value, Tensor)), None)
        if first_tensor is None:
            continue
        empty = first_tensor.reshape(-1).new_empty((0,))
        parts = [value.reshape(-1) if isinstance(value, Tensor) else empty for value in values]
        ptr = torch.empty(len(parts) + 1, dtype=torch.long)
        ptr[0] = 0
        lengths = torch.tensor([int(part.numel()) for part in parts], dtype=torch.long)
        ptr[1:] = torch.cumsum(lengths, dim=0)
        tensors[field] = torch.cat(parts, dim=0) if int(ptr[-1].item()) else parts[0].new_empty((0,))
        ptrs[field] = ptr
    out["slice_tensors"] = tensors
    out["slice_ptrs"] = ptrs
    if "diffusion" in rows[0]:
        out["diffusion"] = [row["diffusion"] for row in rows]

    routes = [row.get("route") for row in rows]
    if any(isinstance(route, dict) for route in routes):
        out["slice_routes"] = _pack_snapshot_routes(routes)
    out["slices"] = []
    return out


def _pack_snapshot_routes(routes: list[Any]) -> dict[str, Any]:
    world_size = 0
    for route in routes:
        if isinstance(route, dict):
            world_size = max(world_size, len(route.get("send_sizes", ())), len(route.get("recv_sizes", ())))
    send_sizes = torch.zeros((len(routes), world_size), dtype=torch.long)
    recv_sizes = torch.zeros((len(routes), world_size), dtype=torch.long)
    send_parts = []
    recv_parts = []
    send_ptr = torch.zeros(len(routes) + 1, dtype=torch.long)
    recv_ptr = torch.zeros(len(routes) + 1, dtype=torch.long)
    for idx, route in enumerate(routes):
        if not isinstance(route, dict):
            send_ptr[idx + 1] = send_ptr[idx]
            recv_ptr[idx + 1] = recv_ptr[idx]
            continue
        send = torch.as_tensor(route.get("send_sizes", ()), dtype=torch.long)
        recv = torch.as_tensor(route.get("recv_sizes", ()), dtype=torch.long)
        send_sizes[idx, : int(send.numel())] = send
        recv_sizes[idx, : int(recv.numel())] = recv
        send_index = route.get("send_index")
        recv_src_row = route.get("recv_src_row")
        send_tensor = send_index.long().reshape(-1) if isinstance(send_index, Tensor) else torch.empty(0, dtype=torch.long)
        recv_tensor = recv_src_row.long().reshape(-1) if isinstance(recv_src_row, Tensor) else torch.empty(0, dtype=torch.long)
        send_parts.append(send_tensor)
        recv_parts.append(recv_tensor)
        send_ptr[idx + 1] = send_ptr[idx] + int(send_tensor.numel())
        recv_ptr[idx + 1] = recv_ptr[idx] + int(recv_tensor.numel())
    return {
        "send_sizes": send_sizes,
        "recv_sizes": recv_sizes,
        "send_index": torch.cat(send_parts, dim=0) if send_parts and int(send_ptr[-1].item()) else torch.empty(0, dtype=torch.long),
        "send_index_ptr": send_ptr,
        "recv_src_row": torch.cat(recv_parts, dim=0) if recv_parts and int(recv_ptr[-1].item()) else torch.empty(0, dtype=torch.long),
        "recv_src_row_ptr": recv_ptr,
    }


def _node_feature_ids_for_rank(
    *,
    rank: int,
    hot_node_ids: Tensor,
    hot_count: int,
    node_part: Tensor,
    node_loc: Tensor,
    prepared: Any,
    include_static_one_hop: bool,
) -> Tensor:
    owned = _owned_ids_by_part(part=node_part, loc=node_loc, rank=rank)
    owned = owned[node_loc.index_select(0, owned) >= int(hot_count)] if int(owned.numel()) else owned
    one_hop = torch.empty(0, dtype=torch.long)
    if include_static_one_hop:
        one_hop = _collect_snapshot_ids(
            prepared=prepared,
            rank=rank,
            field="node_feature_ids",
            owner_part=node_part,
        )
        one_hop = one_hop[node_part.index_select(0, one_hop) != int(rank)] if int(one_hop.numel()) else one_hop
    return torch.cat((hot_node_ids, owned, torch.unique(one_hop, sorted=True)), dim=0)


def _edge_feature_ids_for_rank(
    *,
    rank: int,
    edge_part: Tensor,
    edge_loc: Tensor,
    prepared: Any,
    include_snapshot_edges: bool,
) -> Tensor:
    owned = _owned_ids_by_part(part=edge_part, loc=edge_loc, rank=rank)
    if not bool(include_snapshot_edges):
        return owned
    snapshot_edges = _collect_snapshot_ids(
        prepared=prepared,
        rank=rank,
        field="edge_feature_ids",
        owner_part=edge_part,
    )
    if int(snapshot_edges.numel()) == 0:
        return owned
    return torch.unique(torch.cat((owned, snapshot_edges.long()), dim=0), sorted=True)


def _owned_ids_by_part(*, part: Tensor, loc: Tensor, rank: int) -> Tensor:
    owned = (part == int(rank)).nonzero(as_tuple=True)[0].long()
    owned = owned[torch.argsort(loc.index_select(0, owned), stable=True)] if int(owned.numel()) else owned
    return owned


def _collect_snapshot_ids(*, prepared: Any, rank: int, field: str, owner_part: Tensor) -> Tensor:
    if rank >= len(prepared.snapshot_csc_views):
        return torch.empty(0, dtype=torch.long)
    chunks = []
    for row in prepared.snapshot_csc_views[rank].get("slices", []):
        ids = row.get(field)
        if ids is not None and int(ids.numel()) > 0:
            chunks.append(ids.long())
    if not chunks:
        return torch.empty(0, dtype=torch.long)
    ids = torch.cat(chunks, dim=0)
    ids = ids[(ids >= 0) & (ids < int(owner_part.numel()))]
    return torch.unique(ids, sorted=True)


def _dense_row_map(ids: Tensor, *, size: int) -> Tensor:
    out = torch.full((int(size),), -1, dtype=torch.long)
    if int(ids.numel()) > 0:
        out[ids.long()] = torch.arange(int(ids.numel()), dtype=torch.long)
    return out


def _scatter_by_loc(*, values: Tensor, loc: Tensor) -> Tensor:
    if int(loc.numel()) == 0:
        return values.new_empty((0, *values.shape[1:]))
    out = values.new_zeros((int(loc.max().item()) + 1, *values.shape[1:]))
    out[loc.long()] = values
    return out


def _scatter_temporal_by_loc(*, values: Tensor, loc: Tensor) -> Tensor:
    if int(loc.numel()) == 0:
        return values.new_empty((int(values.shape[0]), 0, *values.shape[2:]))
    out = values.new_zeros((int(values.shape[0]), int(loc.max().item()) + 1, *values.shape[2:]))
    out[:, loc.long()] = values
    return out


def _dist_part(value: Tensor) -> Tensor:
    return value.long() >> DIST_INDEX_LOC_BITS


def _dist_loc(value: Tensor) -> Tensor:
    return value.long() & DIST_INDEX_LOC_MASK


__all__ = ["build_static_feature_shards", "write_prepare_artifacts"]
