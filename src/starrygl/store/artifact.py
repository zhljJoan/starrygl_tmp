from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from .graph import FeatureManager, GraphStore, LabelStore, StoreBundle


def artifact_fingerprint(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def _hash_value(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"{")
        for key in sorted(value, key=str):
            _hash_value(digest, str(key))
            _hash_value(digest, value[key])
        digest.update(b"}")
    elif isinstance(value, (list, tuple)):
        digest.update(b"[")
        for item in value:
            _hash_value(digest, item)
        digest.update(b"]")
    elif isinstance(value, (str, Path)):
        text = str(value)
        digest.update(f"str:{text}".encode())
        try:
            path = Path(text).expanduser()
            if path.is_dir():
                path = path / ("graph.pt" if (path / "graph.pt").exists() else "edges.csv")
            if path.is_file():
                stat = path.stat()
                digest.update(f":file:{stat.st_size}:{stat.st_mtime_ns}".encode())
        except OSError:
            pass
    elif value is None or isinstance(value, (bool, int, float)):
        digest.update(f"{type(value).__name__}:{value!r}".encode())
    else:
        raise TypeError(f"unsupported prepare fingerprint value: {type(value).__name__}")


def load_starrygl_store(
    root: str | Path,
    *,
    rank: int = 0,
    map_location: str | torch.device = "cpu",
    mmap: bool = False,
    feature_mmap: bool | None = None,
    require_label: bool = False,
    load_temporal_csr: bool = True,
) -> StoreBundle:
    path = Path(root)
    feature_mmap = bool(mmap) if feature_mmap is None else bool(feature_mmap)
    prepare = _load_pt(path / "prepare.pt", map_location=map_location, mmap=bool(mmap))
    feature_layout = str(prepare.get("meta", {}).get("feature_layout", "separate"))
    if feature_layout not in {"separate", "snapshot_csc"}:
        raise ValueError(f"unknown feature layout: {feature_layout!r}")
    graph_mmap = bool(mmap) or (feature_layout == "snapshot_csc" and feature_mmap)
    graph_shard = _load_optional_graph_shard(
        path,
        rank=int(rank),
        map_location=map_location,
        mmap=graph_mmap,
        slim=not bool(load_temporal_csr),
    )
    prepare = _merge_graph_shard(
        prepare,
        graph_shard,
        rank=int(rank),
        root=path,
        map_location=map_location,
        mmap=bool(mmap),
        load_temporal_csr=bool(load_temporal_csr),
    )
    feature_path = path / f"feature_{int(rank):03d}.pt"
    feature = _load_optional_pt(feature_path, map_location=map_location, mmap=bool(feature_mmap))
    if feature is None:
        raise FileNotFoundError(f"feature data missing from {feature_path}")
    label_path = path / f"label_{int(rank):03d}.pt"
    if label_path.exists():
        label = _load_pt(label_path, map_location=map_location, mmap=bool(mmap))
    elif require_label:
        raise FileNotFoundError(label_path)
    else:
        label = None
    return StoreBundle(
        graph=GraphStore.from_prepare(prepare, rank=int(rank)),
        features=FeatureManager.from_shard(feature),
        labels=LabelStore.from_shard(label),
    )


def _load_pt(path: Path, *, map_location: str | torch.device, mmap: bool) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(
        str(path) if mmap else path,
        map_location=map_location,
        weights_only=False,
        mmap=bool(mmap),
    )


def _load_optional_pt(path: Path, *, map_location: str | torch.device, mmap: bool) -> Any | None:
    if not path.exists():
        return None
    return _load_pt(path, map_location=map_location, mmap=bool(mmap))


def _load_optional_graph_shard(
    root: Path,
    *,
    rank: int,
    map_location: str | torch.device,
    mmap: bool,
    slim: bool,
) -> Any | None:
    if slim:
        for name in (f"graph_{int(rank):03d}.event.pt", f"graph_{int(rank):03d}.slim.pt"):
            value = _load_optional_pt(root / name, map_location=map_location, mmap=bool(mmap))
            if value is not None:
                return value
    return _load_optional_pt(root / f"graph_{int(rank):03d}.pt", map_location=map_location, mmap=bool(mmap))


def _merge_graph_shard(
    prepare: Any,
    shard: Any | None,
    *,
    rank: int,
    root: Path,
    map_location: str | torch.device,
    mmap: bool,
    load_temporal_csr: bool,
) -> Any:
    if shard is None or not isinstance(prepare, dict) or not isinstance(shard, dict):
        return prepare
    out = dict(prepare)
    world_size = int(out.get("meta", {}).get("world_size", int(rank) + 1))
    event_views = [{} for _ in range(max(world_size, int(rank) + 1))]
    snapshot_views = [{} for _ in range(max(world_size, int(rank) + 1))]
    event_views[int(rank)] = shard.get("event_view", {})
    snapshot_views[int(rank)] = _restore_snapshot_view(shard.get("snapshot_csc_view", {}))
    out["event_views"] = event_views
    out["snapshot_csc_views"] = snapshot_views
    if bool(load_temporal_csr):
        temporal_csr = shard.get("temporal_csr_view", out.get("temporal_csr_view", {}))
        if not temporal_csr:
            standalone = _load_optional_pt(root / "temporal_csr.pt", map_location=map_location, mmap=bool(mmap))
            if isinstance(standalone, dict):
                temporal_csr = standalone.get("temporal_csr_view", standalone)
        if not temporal_csr and int(rank) != 0:
            graph0 = _load_optional_graph_shard(root, rank=0, map_location=map_location, mmap=bool(mmap), slim=False)
            if isinstance(graph0, dict):
                temporal_csr = graph0.get("temporal_csr_view", temporal_csr)
        out["temporal_csr_view"] = temporal_csr
    else:
        out["temporal_csr_view"] = {}
    return out


def _restore_snapshot_view(view: Any) -> Any:
    if not isinstance(view, dict):
        return view
    if view.get("slice_format") == "columnar_v1":
        out = dict(view)
        out["slices"] = _LazySnapshotSlices(view)
        for key in (
            "slice_format",
            "slice_count",
            "snapshot_ids",
            "slice_tensors",
            "slice_ptrs",
            "slice_routes",
            "node_data",
        ):
            out.pop(key, None)
        return out
    out = dict(view)
    out["slices"] = [
        _restore_snapshot_row(out, item) if isinstance(item, dict) else item
        for item in out.get("slices", []) or []
    ]
    return out


class _LazySnapshotSlices:
    _by_snapshot_id = True

    def __init__(self, view: dict[str, Any]) -> None:
        self._view = view
        count = int(view.get("slice_count", torch.tensor(0)).item())
        snapshot_ids = view.get("snapshot_ids")
        self._ids = [
            int(snapshot_ids[idx].item()) if isinstance(snapshot_ids, torch.Tensor) else idx
            for idx in range(count)
        ]
        self._positions = {snapshot_id: idx for idx, snapshot_id in enumerate(self._ids)}

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self):
        for idx in range(len(self)):
            yield self._row(idx)

    def __contains__(self, snapshot_id: object) -> bool:
        try:
            return int(snapshot_id) in self._positions
        except (TypeError, ValueError):
            return False

    def __getitem__(self, snapshot_id: int | slice) -> Any:
        if isinstance(snapshot_id, slice):
            return [self._row(idx) for idx in range(*snapshot_id.indices(len(self)))]
        key = int(snapshot_id)
        position = self._positions.get(key)
        if position is None and key < 0:
            position = len(self) + key
        if position is None or not 0 <= position < len(self):
            raise IndexError(f"snapshot index out of range: {key}")
        return self._row(position)

    def get(self, snapshot_id: int, default: Any = None) -> Any:
        return self[snapshot_id] if snapshot_id in self else default

    def _row(self, position: int) -> dict[str, Any]:
        return _restore_snapshot_row(self._view, _unpack_snapshot_row(self._view, position))


def _restore_snapshot_row(view: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    dst_nodes = view.get("dst_nodes")
    dst_node_chunk = view.get("dst_node_chunk")
    dst_feature_row = view.get("dst_feature_row")
    if dst_nodes is None:
        return item
    row = dict(item)
    extra_src = row.pop("extra_src_nodes", None)
    row["dst_nodes"] = dst_nodes
    row["src_nodes"] = (
        torch.cat((dst_nodes, extra_src.long()), dim=0)
        if isinstance(extra_src, torch.Tensor) and int(extra_src.numel()) > 0
        else dst_nodes
    )
    extra_chunk = row.pop("extra_node_chunk", None)
    if isinstance(dst_node_chunk, torch.Tensor):
        row["node_chunk"] = (
            torch.cat((dst_node_chunk, extra_chunk.long()), dim=0)
            if isinstance(extra_chunk, torch.Tensor) and int(extra_chunk.numel()) > 0
            else dst_node_chunk
        )
    extra_feature_row = row.pop("extra_src_feature_row", None)
    if isinstance(dst_feature_row, torch.Tensor):
        row["src_feature_row"] = (
            torch.cat((dst_feature_row, extra_feature_row.long()), dim=0)
            if isinstance(extra_feature_row, torch.Tensor) and int(extra_feature_row.numel()) > 0
            else dst_feature_row
        )
    return row


def _unpack_snapshot_row(view: dict[str, Any], idx: int) -> dict[str, Any]:
    snapshot_ids = view.get("snapshot_ids")
    row: dict[str, Any] = {
        "snapshot_id": int(snapshot_ids[idx].item()) if isinstance(snapshot_ids, torch.Tensor) else idx
    }
    if "diffusion" in view:
        row["diffusion"] = view["diffusion"][idx]
    tensors = view.get("slice_tensors", {})
    ptrs = view.get("slice_ptrs", {})
    if isinstance(tensors, dict) and isinstance(ptrs, dict):
        for field, values in tensors.items():
            ptr = ptrs.get(field)
            if isinstance(values, torch.Tensor) and isinstance(ptr, torch.Tensor):
                row[field] = values[int(ptr[idx].item()) : int(ptr[idx + 1].item())]
    routes = view.get("slice_routes")
    if isinstance(routes, dict):
        row["route"] = _unpack_snapshot_route(routes, idx)
    node_data = view.get("node_data", {})
    if isinstance(node_data, dict):
        values = {
            str(name): packed["data"][int(packed["ptr"][idx].item()) : int(packed["ptr"][idx + 1].item())]
            for name, packed in node_data.items()
            if isinstance(packed, dict)
            and isinstance(packed.get("data"), torch.Tensor)
            and isinstance(packed.get("ptr"), torch.Tensor)
        }
        if values:
            row["node_data"] = values
    return row


def _unpack_snapshot_route(routes: dict[str, Any], idx: int) -> dict[str, Any]:
    send_sizes = routes.get("send_sizes")
    recv_sizes = routes.get("recv_sizes")
    send_index = _slice_route_tensor(routes.get("send_index"), routes.get("send_index_ptr"), idx)
    recv_src_row = _slice_route_tensor(routes.get("recv_src_row"), routes.get("recv_src_row_ptr"), idx)
    return {
        "send_sizes": send_sizes[idx].tolist() if isinstance(send_sizes, torch.Tensor) else [],
        "recv_sizes": recv_sizes[idx].tolist() if isinstance(recv_sizes, torch.Tensor) else [],
        "send_index": send_index,
        "recv_src_row": recv_src_row,
    }


def _slice_route_tensor(values: Any, ptr: Any, idx: int) -> torch.Tensor | None:
    if not isinstance(values, torch.Tensor) or not isinstance(ptr, torch.Tensor):
        return None
    begin = int(ptr[idx].item())
    end = int(ptr[idx + 1].item())
    if begin == end:
        return None
    return values[begin:end]


__all__ = ["load_starrygl_store"]
