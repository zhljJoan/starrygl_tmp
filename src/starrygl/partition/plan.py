from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from torch import Tensor

from starrygl.utils.route import dist_part


@dataclass(frozen=True)
class PartitionContext:
    graph: Any
    num_parts: int = 1
    share_nodes: bool = True
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_parts < 1:
            raise ValueError("num_parts must be >= 1")


@dataclass(frozen=True)
class ReplicaMap:
    owner: Tensor
    replicas: Mapping[int, Tensor] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "owner": _tensor_summary(self.owner),
            "replicas": {int(rank): _tensor_summary(nodes) for rank, nodes in self.replicas.items()},
        }


@dataclass(frozen=True)
class ChunkTable:
    node_ids: Tensor | None = None
    time_ptr: Tensor | None = None
    chunk_ids: Tensor | None = None
    values: Mapping[str, Tensor] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_ids": None if self.node_ids is None else _tensor_summary(self.node_ids),
            "time_ptr": None if self.time_ptr is None else _tensor_summary(self.time_ptr),
            "chunk_ids": None if self.chunk_ids is None else _tensor_summary(self.chunk_ids),
            "values": {name: _tensor_summary(value) for name, value in self.values.items()},
        }


@dataclass(frozen=True)
class PartitionPlan:
    node_master: Tensor | None = None
    edge_master: Tensor | None = None
    node_replicas: ReplicaMap | None = None
    edge_replicas: ReplicaMap | None = None
    shared_nodes: Tensor | None = None
    chunk_table: ChunkTable = field(default_factory=ChunkTable)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_artifact(cls, partition: Mapping[str, Any]) -> "PartitionPlan":
        node_index = partition.get("node_dist_index")
        edge_index = partition.get("edge_dist_index")
        node_master = partition.get("node_master")
        edge_master = partition.get("edge_master")
        if node_master is None and isinstance(node_index, Tensor):
            node_master = dist_part(node_index.long())
        if edge_master is None and isinstance(edge_index, Tensor):
            edge_master = dist_part(edge_index.long())
        values = {
            name: value
            for name in ("node_to_chunk", "edge_chunk")
            if isinstance((value := partition.get(name)), Tensor)
        }
        return cls(
            node_master=node_master if isinstance(node_master, Tensor) else None,
            edge_master=edge_master if isinstance(edge_master, Tensor) else None,
            shared_nodes=partition.get("hot_node_ids") if isinstance(partition.get("hot_node_ids"), Tensor) else None,
            chunk_table=ChunkTable(values=values),
            metadata={"source": "prepared_artifact"},
        )

    def explain(self) -> str:
        node_count = 0 if self.node_master is None else int(self.node_master.numel())
        edge_count = 0 if self.edge_master is None else int(self.edge_master.numel())
        shared = "enabled" if self.shared_nodes is not None and int(self.shared_nodes.numel()) > 0 else "disabled"
        edge_replica = "enabled" if self.edge_replicas is not None else "disabled"
        return (
            "PartitionPlan("
            f"nodes={node_count}, edges={edge_count}, "
            f"shared_nodes={shared}, edge_replicas={edge_replica})"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.explain(),
            "node_master": None if self.node_master is None else _tensor_summary(self.node_master),
            "edge_master": None if self.edge_master is None else _tensor_summary(self.edge_master),
            "node_replicas": None if self.node_replicas is None else self.node_replicas.as_dict(),
            "edge_replicas": None if self.edge_replicas is None else self.edge_replicas.as_dict(),
            "shared_nodes": None if self.shared_nodes is None else _tensor_summary(self.shared_nodes),
            "chunk_table": self.chunk_table.as_dict(),
            "metadata": dict(self.metadata),
        }


def _tensor_summary(value: Tensor, *, value_limit: int = 32) -> dict[str, Any]:
    flat = value.detach().cpu().reshape(-1)
    result: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "device": str(value.device),
        "numel": int(value.numel()),
    }
    if int(flat.numel()) <= value_limit:
        result["values"] = flat.tolist()
    else:
        result["head"] = flat[:value_limit].tolist()
    return result


__all__ = ["ChunkTable", "PartitionContext", "PartitionPlan", "ReplicaMap"]
