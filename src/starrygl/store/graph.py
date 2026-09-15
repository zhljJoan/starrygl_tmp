from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


SPLIT_NAMES = ("train", "val", "test")


def split_window_range(split_ptrs: Mapping[str, Tensor], split: str) -> range:
    """Return the split's global row range in the concatenated time_ptr_2 table."""
    if split not in SPLIT_NAMES or split not in split_ptrs:
        return range(0)
    preceding = SPLIT_NAMES[: SPLIT_NAMES.index(split)]
    start = sum(int(split_ptrs[name].shape[0]) for name in preceding if name in split_ptrs)
    return range(start, start + int(split_ptrs[split].shape[0]))


@dataclass(frozen=True)
class GraphStore:
    num_nodes: int
    src: Tensor | None = None
    dst: Tensor | None = None
    edge_ids: Tensor | None = None
    timestamps: Tensor | None = None
    snapshot_ptr: Tensor | None = None
    edge_weight: Tensor | None = None
    rank: int = 0
    prepare: Mapping[str, Any] | None = None
    runtime_cache: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GraphStore":
        src = data.get("src")
        dst = data.get("dst")
        if src is not None:
            src = src.long()
        if dst is not None:
            dst = dst.long()
        num_nodes = data.get("num_nodes")
        if num_nodes is None:
            max_node = int(torch.cat((src, dst)).max().item()) if src is not None and dst is not None and int(src.numel()) else -1
            num_nodes = max_node + 1
        edge_ids = data.get("edge_ids")
        timestamps = data.get("ts", data.get("timestamps"))
        return cls(
            num_nodes=int(num_nodes),
            src=src,
            dst=dst,
            edge_ids=None if edge_ids is None else edge_ids.long(),
            timestamps=None if timestamps is None else torch.as_tensor(timestamps),
            snapshot_ptr=None if data.get("snapshot_ptr") is None else data["snapshot_ptr"].long(),
            edge_weight=data.get("edge_weight"),
        )

    @classmethod
    def from_prepare(cls, prepare: Mapping[str, Any], *, rank: int) -> "GraphStore":
        meta = prepare.get("meta", {})
        return cls(num_nodes=int(meta.get("num_nodes", 0)), rank=int(rank), prepare=prepare)

    @property
    def num_edges(self) -> int:
        if self.src is not None:
            return int(self.src.numel())
        if self.edge_ids is not None:
            return int(self.edge_ids.numel())
        if self.prepare is not None:
            return int(self.prepare.get("meta", {}).get("num_edges", 0))
        return 0

    @property
    def world_size(self) -> int:
        if self.prepare is None:
            return 1
        return int(self.prepare.get("meta", {}).get("world_size", len(self.prepare.get("event_views", [])) or 1))

    @property
    def partition(self) -> Mapping[str, Any]:
        if self.prepare is None:
            return {}
        return self.prepare["partition"]

    @property
    def time_ptr_2(self) -> Tensor:
        if self.prepare is None:
            return torch.empty((0, 2), dtype=torch.long)
        return self.prepare["time_ptr_2"]

    @property
    def split_time_ptr_2(self) -> Mapping[str, Tensor]:
        if self.prepare is None:
            return {}
        return self.prepare.get("split_time_ptr_2", {"train": self.time_ptr_2})

    @property
    def split_ranges(self) -> Mapping[str, Any]:
        if self.prepare is None:
            return {}
        return self.prepare.get("split_ranges", {})

    @property
    def split_masks(self) -> Mapping[str, Tensor]:
        if self.prepare is None:
            return {}
        return self.prepare.get("split_masks", {})

    @property
    def event_view(self) -> Mapping[str, Any]:
        if self.prepare is None:
            return {}
        view = self.prepare["event_views"][int(self.rank)]
        if "hot_node_ids" in view:
            return view
        hot = self.partition.get("hot_node_ids")
        if hot is None:
            return view
        out = dict(view)
        out["hot_node_ids"] = hot.long()
        return out

    @property
    def snapshot_csc_view(self) -> Mapping[str, Any]:
        if self.prepare is None:
            return {}
        return self.prepare["snapshot_csc_views"][int(self.rank)]

    @property
    def temporal_csr_view(self) -> Mapping[str, Any]:
        if self.prepare is None:
            return {}
        return self.prepare.get("temporal_csr_view", {})

    def edges(self, edge_ids: Tensor) -> tuple[Tensor, Tensor]:
        if self.src is None or self.dst is None:
            view = self.event_view
            return view["src"].index_select(0, edge_ids.long()), view["dst"].index_select(0, edge_ids.long())
        return self.src.index_select(0, edge_ids.long()), self.dst.index_select(0, edge_ids.long())

    def snapshot(self, index: int) -> "GraphStore":
        if self.snapshot_ptr is None:
            return self
        if self.src is None or self.dst is None:
            raise ValueError("snapshot slicing requires src and dst tensors")
        sid = int(index)
        if sid < 0 or sid + 1 >= int(self.snapshot_ptr.numel()):
            raise IndexError(f"snapshot index out of range: {sid}")
        begin = int(self.snapshot_ptr[sid].item())
        end = int(self.snapshot_ptr[sid + 1].item())
        rows = torch.arange(begin, end, dtype=torch.long, device=self.src.device)
        edge_ids = self.edge_ids
        if edge_ids is None:
            edge_ids = torch.arange(int(self.src.numel()), dtype=torch.long, device=self.src.device)
        return GraphStore(
            num_nodes=int(self.num_nodes),
            src=self.src.index_select(0, rows),
            dst=self.dst.index_select(0, rows),
            edge_ids=edge_ids.index_select(0, rows.to(device=edge_ids.device)),
            timestamps=None if self.timestamps is None else self.timestamps.index_select(0, rows.to(device=self.timestamps.device)),
            snapshot_ptr=torch.tensor([0, int(rows.numel())], dtype=torch.long, device=self.src.device),
            edge_weight=None if self.edge_weight is None else self.edge_weight.index_select(0, rows.to(device=self.edge_weight.device)),
            rank=int(self.rank),
            prepare=self.prepare,
        )


class FeatureManager:
    def __init__(
        self,
        features: Mapping[str, Tensor] | None = None,
        *,
        node_features: Mapping[str, Tensor] | None = None,
        edge_features: Mapping[str, Tensor] | None = None,
        node_row_map: Tensor | None = None,
        edge_row_map: Tensor | None = None,
        node_ids: Tensor | None = None,
        edge_ids: Tensor | None = None,
        node_features_replicated: bool = False,
    ) -> None:
        self.node_features = dict(node_features or features or {})
        self.edge_features = dict(edge_features or {})
        self.node_row_map = node_row_map.long() if node_row_map is not None else None
        self.edge_row_map = edge_row_map.long() if edge_row_map is not None else None
        self.node_ids = node_ids.long() if node_ids is not None else None
        self.edge_ids = edge_ids.long() if edge_ids is not None else None
        self.node_features_replicated = bool(node_features_replicated) or _row_map_all_present(self.node_row_map)
        self.node_row_map_is_identity = _row_map_is_identity(self.node_row_map)
        self.edge_row_map_is_identity = _row_map_is_identity(self.edge_row_map)
        self._device_lock = Lock()
        self._feature_device = _feature_device(self.node_features, self.edge_features)

    @classmethod
    def from_shard(cls, shard: Mapping[str, Any]) -> "FeatureManager":
        node = {"x": shard["node_feat"]} if "node_feat" in shard else {}
        edge = {"edge": shard["edge_feat"]} if "edge_feat" in shard else {}
        return cls(
            node_features=node,
            edge_features=edge,
            node_row_map=shard.get("node_row_map"),
            edge_row_map=shard.get("edge_row_map"),
            node_ids=shard.get("node_ids"),
            edge_ids=shard.get("edge_ids"),
            node_features_replicated=bool(shard.get("node_features_replicated", False)),
        )

    def read_nodes(self, ids: Tensor, names: Sequence[str] | None = None) -> dict[str, Tensor]:
        rows = ids.long() if self.node_row_map_is_identity else self._rows(ids.long(), self.node_row_map)
        return self._read(self.node_features, rows, names)

    def read_nodes_at(self, ids: Tensor, snapshot_id: int, names: Sequence[str] | None = None) -> dict[str, Tensor]:
        rows = ids.long() if self.node_row_map_is_identity else self._rows(ids.long(), self.node_row_map)
        return self._read_at(self.node_features, rows, int(snapshot_id), names)

    def read_edges(self, ids: Tensor, names: Sequence[str] | None = None) -> dict[str, Tensor]:
        rows = ids.long() if self.edge_row_map_is_identity else self._rows(ids.long(), self.edge_row_map)
        return self._read(self.edge_features, rows, names)

    def read_node_rows(self, rows: Tensor, names: Sequence[str] | None = None) -> dict[str, Tensor]:
        return self._read(self.node_features, rows.long(), names)

    def read_node_rows_at(self, rows: Tensor, snapshot_id: int, names: Sequence[str] | None = None) -> dict[str, Tensor]:
        return self._read_at(self.node_features, rows.long(), int(snapshot_id), names)

    def read_edge_rows(self, rows: Tensor, names: Sequence[str] | None = None) -> dict[str, Tensor]:
        return self._read(self.edge_features, rows.long(), names)

    def to(self, device: str | torch.device) -> "FeatureManager":
        target = torch.device(device)
        with self._device_lock:
            if self._feature_device == target:
                return self
            self.node_features = {name: value.to(device=target) for name, value in self.node_features.items()}
            self.edge_features = {name: value.to(device=target) for name, value in self.edge_features.items()}
            self.node_row_map = None if self.node_row_map is None else self.node_row_map.to(device=target)
            self.edge_row_map = None if self.edge_row_map is None else self.edge_row_map.to(device=target)
            self.node_ids = None if self.node_ids is None else self.node_ids.to(device=target)
            self.edge_ids = None if self.edge_ids is None else self.edge_ids.to(device=target)
            self._feature_device = target
        return self

    @staticmethod
    def _rows(ids: Tensor, row_map: Tensor | None) -> Tensor:
        if row_map is None:
            return ids
        return row_map.index_select(0, ids.to(device=row_map.device))

    @staticmethod
    def _read(features: Mapping[str, Tensor], rows: Tensor, names: Sequence[str] | None) -> dict[str, Tensor]:
        keys = tuple(features.keys()) if names is None else tuple(names)
        return {name: features[name].index_select(0, rows.to(device=features[name].device)) for name in keys}

    @staticmethod
    def _read_at(features: Mapping[str, Tensor], rows: Tensor, snapshot_id: int, names: Sequence[str] | None) -> dict[str, Tensor]:
        keys = tuple(features.keys()) if names is None else tuple(names)
        out = {}
        for name in keys:
            value = features[name]
            if value.dim() >= 3:
                sid = max(0, min(int(snapshot_id), int(value.shape[0]) - 1))
                value = value[int(sid)]
            out[name] = value.index_select(0, rows.to(device=value.device))
        return out


class LabelStore:
    def __init__(
        self,
        *,
        node_label: Tensor | None = None,
        edge_label: Tensor | None = None,
        node_label_ids: Tensor | None = None,
        node_label_temporal: bool = False,
        task_kind: str | None = None,
        task_ptr: Tensor | None = None,
        task_payload: Mapping[str, Tensor] | None = None,
    ) -> None:
        self.node_label = node_label
        self.edge_label = edge_label
        self.node_label_ids = node_label_ids
        self.node_label_temporal = bool(node_label_temporal)
        self.task_kind = None if task_kind is None else str(task_kind)
        self.task_ptr = None if task_ptr is None else task_ptr.long()
        self.task_payload = dict(task_payload or {})
        if self.task_kind not in {None, "node", "edge"}:
            raise ValueError("task_kind must be node or edge")
        if (self.task_kind is None) != (self.task_ptr is None):
            raise ValueError("task_kind and task_ptr must be provided together")
        if self.task_ptr is not None:
            if self.task_ptr.dim() != 1 or not int(self.task_ptr.numel()):
                raise ValueError("task_ptr must be a non-empty 1-D tensor")
            if int(self.task_ptr[0].item()) != 0:
                raise ValueError("task_ptr must start at zero")
            if bool(torch.any(self.task_ptr[1:] < self.task_ptr[:-1]).item()):
                raise ValueError("task_ptr must be monotonic")
            rows = int(self.task_ptr[-1].item())
            if any(int(value.shape[0]) != rows for value in self.task_payload.values()):
                raise ValueError("task payload columns must align with task_ptr")
            required = (
                ("node_ids",)
                if self.task_kind == "node"
                else ("src", "dst", "edge_ids", "edge_rows")
            )
            if int(self.task_ptr.numel()) > 1 and any(
                name not in self.task_payload for name in required
            ):
                raise ValueError(f"{self.task_kind} task payload is incomplete")

    @classmethod
    def from_shard(cls, shard: Mapping[str, Any] | None) -> "LabelStore":
        shard = {} if shard is None else shard
        return cls(
            node_label=shard.get("node_label"),
            edge_label=shard.get("edge_label"),
            node_label_ids=shard.get("node_label_ids"),
            node_label_temporal=bool(shard.get("node_label_temporal", False)),
            task_kind=shard.get("task_kind"),
            task_ptr=shard.get("task_ptr"),
            task_payload=shard.get("task_payload"),
        )

    def task_slice(self, window_id: int) -> dict[str, Tensor]:
        if self.task_ptr is None or self.task_kind is None:
            raise ValueError("label shard has no prepared task table; re-run prepare")
        if not 0 <= int(window_id) < int(self.task_ptr.numel()) - 1:
            raise IndexError(window_id)
        begin, end = self.task_ptr[int(window_id) : int(window_id) + 2].tolist()
        return {name: value[int(begin) : int(end)] for name, value in self.task_payload.items()}

    def read_node_rows(self, rows: Tensor) -> Tensor:
        if self.node_label is None:
            raise KeyError("node_label")
        return self.node_label.index_select(0, rows.long())

    def read_node_rows_at(self, rows: Tensor, snapshot_id: int) -> Tensor:
        if self.node_label is None:
            raise KeyError("node_label")
        value = self.node_label
        if self.node_label_temporal:
            sid = max(0, min(int(snapshot_id), int(value.shape[0]) - 1))
            value = value[int(sid)]
        return value.index_select(0, rows.long())

    def read_edge_rows(self, rows: Tensor) -> Tensor:
        if self.edge_label is None:
            raise KeyError("edge_label")
        return self.edge_label.index_select(0, rows.long())


@dataclass(frozen=True)
class StoreBundle:
    graph: GraphStore
    features: FeatureManager
    labels: LabelStore


def _feature_device(*groups: Mapping[str, Tensor]) -> torch.device | None:
    for group in groups:
        for value in group.values():
            return value.device
    return None


def _row_map_all_present(row_map: Tensor | None) -> bool:
    return row_map is not None and int(row_map.numel()) > 0 and bool(torch.all(row_map >= 0).item())


def _row_map_is_identity(row_map: Tensor | None) -> bool:
    return row_map is not None and int(row_map.numel()) > 0 and bool(torch.equal(row_map.cpu(), torch.arange(int(row_map.numel()), dtype=row_map.dtype)))


__all__ = ["FeatureManager", "GraphStore", "LabelStore", "StoreBundle"]
