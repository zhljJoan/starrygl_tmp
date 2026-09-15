from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
from torch import Tensor

from starrygl.partition import PartitionConfig, PartitionPlan, partition_graph
from starrygl.plan import ViewPlan

from .build_part_graph import PrepareConfig, PreparedViews, materialize_graph_views


_SIDECAR_FILES = {
    "node_feat": ("node_feat.pt", "node_features.pt"),
    "edge_feat": ("edge_feat.pt", "edge_features.pt"),
    "node_label": ("node_label.pt",),
    "edge_label": ("edge_label.pt",),
}


@dataclass(frozen=True)
class GraphData:
    src: Tensor
    dst: Tensor
    ts: Tensor
    edge_ids: Tensor
    num_nodes: int
    time_ptr_2: Tensor | None = None
    node_feat: Tensor | None = None
    edge_weight: Tensor | None = None
    edge_feat: Tensor | None = None
    node_label: Tensor | None = None
    node_label_nodes: Tensor | None = None
    node_label_ts: Tensor | None = None
    node_label_split: Tensor | None = None
    node_label_temporal: bool = False
    node_label_horizon: int = 0
    edge_label: Tensor | None = None
    split_labels: Tensor | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


def load_graph_data(data: Mapping[str, Any] | str | Path) -> GraphData:
    graph = _load(data)
    src = _tensor(graph, "src", torch.long, required=True)
    dst = _tensor(graph, "dst", torch.long, required=True)
    assert src is not None and dst is not None
    edge_count = int(src.numel())
    if int(dst.numel()) != edge_count:
        raise ValueError("src and dst must have the same length")

    ts = _tensor(graph, "ts", torch.float32)
    edge_ids = _tensor(graph, "edge_ids", torch.long)
    ts = torch.arange(edge_count, dtype=torch.float32) if ts is None else ts
    edge_ids = torch.arange(edge_count, dtype=torch.long) if edge_ids is None else edge_ids
    if int(ts.numel()) != edge_count or int(edge_ids.numel()) != edge_count:
        raise ValueError("ts and edge_ids must have one value per edge")

    order = torch.argsort(ts, stable=True)
    edge_weight = _edge_tensor(graph, "edge_weight", order, edge_count, dtype=torch.float32)
    edge_feat = _edge_tensor(graph, "edge_feat", order, edge_count)
    edge_label = _edge_tensor(graph, "edge_label", order, edge_count)
    split_labels = _edge_tensor(graph, "split_labels", order, edge_count, dtype=torch.long)
    src = src.index_select(0, order)
    dst = dst.index_select(0, order)
    ts = ts.index_select(0, order)
    edge_ids = edge_ids.index_select(0, order)

    node_feat = _tensor(graph, "node_feat")
    time_ptr_2 = _tensor(graph, "time_ptr_2", torch.long)
    if time_ptr_2 is not None:
        time_ptr_2 = time_ptr_2.reshape(-1, 2)
    num_nodes = int(graph.get("num_nodes", _infer_num_nodes(src, dst)))
    if node_feat is not None:
        node_axis = 1 if node_feat.dim() >= 3 else 0
        num_nodes = max(num_nodes, int(node_feat.shape[node_axis]))
    node_labels = _node_labels(graph, edge_ts=ts, edge_split=split_labels)
    node_label = _tensor(node_labels, "node_label")
    node_label_nodes = _tensor(node_labels, "node_label_nodes", torch.long)
    node_label_ts = _tensor(node_labels, "node_label_ts", torch.float32)
    node_label_split = _tensor(node_labels, "node_label_split", torch.uint8)
    if node_label_nodes is not None and int(node_label_nodes.numel()):
        num_nodes = max(num_nodes, int(node_label_nodes.max().item()) + 1)
    node_label_temporal = _node_label_is_temporal(
        node_label,
        node_label_nodes=node_label_nodes,
        time_ptr_2=time_ptr_2,
        num_nodes=num_nodes,
        declared=graph.get("node_label_temporal"),
    )

    return GraphData(
        src=src,
        dst=dst,
        ts=ts,
        edge_ids=edge_ids,
        num_nodes=num_nodes,
        time_ptr_2=time_ptr_2,
        node_feat=node_feat,
        edge_weight=edge_weight,
        edge_feat=edge_feat,
        node_label=node_label,
        node_label_nodes=node_label_nodes,
        node_label_ts=node_label_ts,
        node_label_split=node_label_split,
        node_label_temporal=node_label_temporal,
        node_label_horizon=max(
            0,
            int(
                graph.get(
                    "node_label_horizon",
                    1 if graph.get("node_label_source") else 0,
                )
            ),
        ),
        edge_label=edge_label,
        split_labels=split_labels,
        meta={key: value for key, value in graph.items() if key not in _TENSOR_KEYS},
    )


def partition_graph_data(
    graph: GraphData,
    *,
    config: PrepareConfig,
    node_master: Tensor | None = None,
    edge_master: Tensor | None = None,
    hot_node_ids: Tensor | None = None,
    node_to_chunk: Tensor | None = None,
) -> PartitionPlan:
    return partition_graph(
        src=graph.src,
        dst=graph.dst,
        ts=graph.ts,
        num_nodes=graph.num_nodes,
        config=PartitionConfig(
            num_parts=int(config.world_size),
            chunks_per_rank=int(config.chunks_per_rank),
            backend=str(config.partition_backend),
            speed_beta=float(config.speed_partition_beta),
            hot_node_ratio=float(config.speed_partition_topk_ratio),
            hot_node_type=str(config.speed_partition_topk_type),
        ),
        node_master=node_master,
        edge_master=edge_master,
        hot_node_ids=hot_node_ids,
        node_to_chunk=node_to_chunk,
    )


def materialize_graph_data(
    graph: GraphData,
    *,
    config: PrepareConfig,
    partition_plan: PartitionPlan,
    view_plan: ViewPlan,
) -> PreparedViews:
    if config.time_ptr_2 is None and graph.time_ptr_2 is not None:
        config = replace(config, time_ptr_2=graph.time_ptr_2)
    return materialize_graph_views(
        src=graph.src,
        dst=graph.dst,
        ts=graph.ts,
        num_nodes=graph.num_nodes,
        config=config,
        edge_ids=graph.edge_ids,
        edge_weight=graph.edge_weight,
        split_labels=graph.split_labels,
        partition_plan=partition_plan,
        view_plan=view_plan,
    )


def _load(data: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(data, Mapping):
        graph = dict(data)
        source = graph.pop("source", None)
        if source is None:
            return graph
        loaded = _load(source)
        loaded.update(graph)
        return loaded

    path = Path(data).expanduser()
    if path.is_dir():
        root = path
        pt = root / "graph.pt"
        csv_path = root / "edges.csv"
        graph = _load(pt if pt.exists() else csv_path)
        for key, names in _SIDECAR_FILES.items():
            if key in graph:
                continue
            sidecar = next((root / name for name in names if (root / name).exists()), None)
            if sidecar is not None:
                graph[key] = torch.load(sidecar, map_location="cpu", weights_only=False)
        return graph
    if path.suffix in {".pt", ".pth"}:
        value = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(value, Mapping):
            raise ValueError("graph tensor file must contain a mapping")
        return dict(value)
    if path.suffix == ".csv":
        return _load_csv(path)
    raise ValueError(f"unsupported graph data source: {path}")


def _load_csv(path: Path) -> dict[str, Tensor]:
    frame = pd.read_csv(path)
    fields = set(frame.columns)
    if not {"src", "dst"}.issubset(fields):
        raise ValueError("CSV graph data requires src and dst columns")

    def column(name: str, dtype: torch.dtype) -> Tensor | None:
        if name not in fields:
            return None
        return torch.as_tensor(frame[name].to_numpy(copy=False), dtype=dtype).contiguous()

    graph = {
        "src": column("src", torch.long),
        "dst": column("dst", torch.long),
    }
    for name, source, dtype in (
        ("ts", "ts" if "ts" in fields else "time", torch.float32),
        ("edge_ids", "edge_ids", torch.long),
        ("edge_weight", "edge_weight", torch.float32),
        ("edge_label", "edge_label", torch.float32),
        ("split_labels", "split_labels" if "split_labels" in fields else "ext_roll", torch.long),
    ):
        value = column(source, dtype)
        if value is not None:
            graph[name] = value
    return graph  # type: ignore[return-value]


def _tensor(
    graph: Mapping[str, Any],
    name: str,
    dtype: torch.dtype | None = None,
    *,
    required: bool = False,
) -> Tensor | None:
    value = graph.get(name)
    if value is None:
        if required:
            raise ValueError(f"graph data missing required field {name!r}")
        return None
    return torch.as_tensor(value, dtype=dtype).cpu().contiguous()


def _edge_tensor(
    graph: Mapping[str, Any],
    name: str,
    order: Tensor,
    edge_count: int,
    *,
    dtype: torch.dtype | None = None,
) -> Tensor | None:
    value = _tensor(graph, name, dtype)
    if value is None:
        return None
    if int(value.shape[0]) != int(edge_count):
        raise ValueError(f"{name} must have one row per edge")
    return value.index_select(0, order).contiguous()


def _infer_num_nodes(src: Tensor, dst: Tensor) -> int:
    return int(torch.cat((src, dst)).max().item()) + 1 if int(src.numel()) else 0


def _node_label_is_temporal(
    value: Tensor | None,
    *,
    node_label_nodes: Tensor | None,
    time_ptr_2: Tensor | None,
    num_nodes: int,
    declared: Any,
) -> bool:
    if declared is not None and not isinstance(declared, bool):
        raise TypeError("node_label_temporal must be a boolean")
    inferred = bool(
        value is not None
        and node_label_nodes is None
        and time_ptr_2 is not None
        and value.dim() >= 2
        and int(value.shape[0]) == int(time_ptr_2.shape[0])
        and int(value.shape[1]) == int(num_nodes)
    )
    temporal = inferred if declared is None else bool(declared)
    if not temporal:
        return False
    if value is None or node_label_nodes is not None or time_ptr_2 is None or value.dim() < 2:
        raise ValueError("temporal node labels require dense [snapshot, node, ...] data and time_ptr_2")
    if int(value.shape[0]) != int(time_ptr_2.shape[0]) or int(value.shape[1]) != int(num_nodes):
        raise ValueError("temporal node labels must have shape [num_snapshots, num_nodes, ...]")
    return True


def _node_labels(graph: Mapping[str, Any], *, edge_ts: Tensor, edge_split: Tensor | None) -> Mapping[str, Any]:
    if graph.get("node_label") is not None or not isinstance(graph.get("node_label_dict"), Mapping):
        return graph
    rows = [
        (float(timestamp), int(node), torch.as_tensor(label).detach().cpu())
        for timestamp, labels in sorted(graph["node_label_dict"].items(), key=lambda item: float(item[0]))
        if isinstance(labels, Mapping)
        for node, label in labels.items()
    ]
    if not rows:
        return graph
    shapes = {tuple(label.shape) for _, _, label in rows}
    if len(shapes) != 1:
        raise ValueError("node_label_dict contains labels with inconsistent shapes")
    out = dict(graph)
    out["node_label_ts"] = torch.tensor([timestamp for timestamp, _, _ in rows], dtype=torch.float32)
    out["node_label_nodes"] = torch.tensor([node for _, node, _ in rows], dtype=torch.long)
    out["node_label"] = torch.stack([label for _, _, label in rows]).contiguous()
    split = _node_label_splits(out["node_label_ts"], edge_ts=edge_ts, edge_split=edge_split)
    if split is not None:
        out["node_label_split"] = split
    return out


def _node_label_splits(node_ts: Tensor, *, edge_ts: Tensor, edge_split: Tensor | None) -> Tensor | None:
    if edge_split is None or not int(edge_ts.numel()) or int(edge_ts.numel()) != int(edge_split.numel()):
        return None
    result = torch.full((int(node_ts.numel()),), 2, dtype=torch.uint8)
    assigned = torch.zeros_like(result, dtype=torch.bool)
    for split_id in (0, 1, 2):
        split_ts = edge_ts[edge_split == split_id]
        if not int(split_ts.numel()):
            continue
        mask = (node_ts >= split_ts.min() - 1.0) & (node_ts < split_ts.max())
        result[mask] = split_id
        assigned |= mask
    return result if bool(assigned.any()) else None


_TENSOR_KEYS = {
    "src",
    "dst",
    "ts",
    "edge_ids",
    "time_ptr_2",
    "node_feat",
    "edge_weight",
    "edge_feat",
    "node_label",
    "node_label_nodes",
    "node_label_ts",
    "node_label_split",
    "node_label_temporal",
    "edge_label",
    "split_labels",
    "node_label_dict",
}


__all__ = ["GraphData", "load_graph_data", "materialize_graph_data", "partition_graph_data"]
