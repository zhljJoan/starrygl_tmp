from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

import torch
from torch import Tensor

from starrygl.view.base import GraphBlock, GraphFormat, graph_block_from_coo

if TYPE_CHECKING:
    from starrygl.batch import Batch
    from starrygl.store import GraphStore


SnapshotGraphCacheKey = tuple[int, int | None, str, bool]


@dataclass(frozen=True)
class SnapshotWindowBlob:
    snapshot_indices: tuple[int, ...]
    chunk_decay: tuple[int, ...]
    num_full_snapshots: int
    chunk_thresholds: tuple[int | None, ...]
    graph_formats: tuple[str, ...]
    edge_counts: tuple[int, ...]
    materialized_coo: tuple[bool, ...]
    source_edge_counts: tuple[int, ...] = ()

    @property
    def is_fixed_csr(self) -> bool:
        return all(fmt == "csr" for fmt in self.graph_formats)

    @property
    def dropped_edge_counts(self) -> tuple[int, ...]:
        sources = self.source_edge_counts if self.source_edge_counts else self.edge_counts
        return tuple(max(0, int(source) - int(kept)) for source, kept in zip(sources, self.edge_counts))

    def as_dict(self) -> dict[str, Any]:
        sources = self.source_edge_counts if self.source_edge_counts else self.edge_counts
        return {
            "snapshot_indices": list(self.snapshot_indices),
            "chunk_decay": list(self.chunk_decay),
            "num_full_snapshots": int(self.num_full_snapshots),
            "chunk_thresholds": [None if value is None else int(value) for value in self.chunk_thresholds],
            "graph_formats": list(self.graph_formats),
            "edge_counts": [int(value) for value in self.edge_counts],
            "source_edge_counts": [int(value) for value in sources],
            "dropped_edge_counts": [int(value) for value in self.dropped_edge_counts],
            "materialized_coo": [bool(value) for value in self.materialized_coo],
            "is_fixed_csr": self.is_fixed_csr,
        }


@dataclass(frozen=True)
class SnapshotWindow:
    graphs: tuple[GraphBlock, ...]
    snapshot_indices: tuple[int, ...] = ()
    chunk_decay: tuple[int, ...] = ()
    num_full_snapshots: int = 1
    source_edge_counts: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.graphs:
            raise ValueError("SnapshotWindow requires at least one graph")
        if not self.snapshot_indices:
            object.__setattr__(self, "snapshot_indices", tuple(range(len(self.graphs))))
        elif len(self.snapshot_indices) != len(self.graphs):
            raise ValueError("SnapshotWindow snapshot_indices must match graphs")

    def __len__(self) -> int:
        return len(self.graphs)

    def __iter__(self) -> Iterator[GraphBlock]:
        return iter(self.graphs)

    def __getitem__(self, index: int) -> GraphBlock:
        return self.graphs[index]

    @property
    def latest_graph(self) -> GraphBlock:
        return self.graphs[-1]

    @property
    def blob(self) -> SnapshotWindowBlob:
        return SnapshotWindowBlob(
            snapshot_indices=self.snapshot_indices,
            chunk_decay=self.chunk_decay,
            num_full_snapshots=int(self.num_full_snapshots),
            chunk_thresholds=_window_decay_thresholds(
                window_size=len(self.graphs),
                chunk_decay=self.chunk_decay,
                num_full_snapshots=int(self.num_full_snapshots),
            ),
            graph_formats=tuple(str(graph.format) for graph in self.graphs),
            edge_counts=tuple(int(graph.num_edges) for graph in self.graphs),
            materialized_coo=tuple(graph.edge_index is not None for graph in self.graphs),
            source_edge_counts=(
                self.source_edge_counts
                if self.source_edge_counts
                else tuple(int(graph.num_edges) for graph in self.graphs)
            ),
        )


@dataclass(frozen=True)
class SnapshotBlockView:
    graph: GraphBlock | SnapshotWindow

    @classmethod
    def from_store(
        cls,
        store: "GraphStore",
        *,
        format: GraphFormat = "csr",
        snapshot_index: int | None = None,
        materialize_coo: bool = True,
    ) -> "SnapshotBlockView":
        snapshot = store if snapshot_index is None else store.snapshot(int(snapshot_index))
        src = _required_tensor(snapshot.src, "src")
        dst = _required_tensor(snapshot.dst, "dst")
        edge_ids = snapshot.edge_ids
        if edge_ids is None:
            edge_ids = torch.arange(int(src.numel()), dtype=torch.long, device=src.device)
        return cls(
            graph=graph_block_from_coo(
                src=src,
                dst=dst,
                edge_ids=edge_ids,
                num_nodes=int(snapshot.num_nodes),
                format=format,
                materialize_coo=materialize_coo,
                edge_weight=snapshot.edge_weight,
            )
        )

    @classmethod
    def window_from_store(
        cls,
        store: "GraphStore",
        *,
        snapshot_indices: Sequence[int],
        format: GraphFormat = "csr",
        chunk_decay: Sequence[Any] | None = None,
        num_full_snapshots: int = 1,
        node_chunk_order: Tensor | None = None,
        materialize_coo: bool = True,
    ) -> "SnapshotBlockView":
        return _snapshot_window_view_from_cache(
            store,
            snapshot_indices=snapshot_indices,
            chunk_decay=chunk_decay,
            num_full_snapshots=num_full_snapshots,
            node_chunk_order=node_chunk_order,
            graph_cache={},
            format=format,
            materialize_coo=materialize_coo,
        )[0]

    def materialize(
        self,
        *,
        num_layers: int = 1,
        features: Mapping[str, Tensor] | None = None,
        state: Mapping[str, Tensor] | None = None,
        targets: Mapping[str, Tensor] | None = None,
    ) -> "Batch":
        from starrygl.batch import Batch

        return Batch(
            mode="snapshot",
            graph=self.graph,
            num_layers=int(num_layers),
            features={} if features is None else features,
            state={} if state is None else state,
            targets={} if targets is None else targets,
        )


def _snapshot_window_view_from_cache(
    store: "GraphStore",
    *,
    snapshot_indices: Sequence[int],
    chunk_decay: Sequence[Any] | None,
    num_full_snapshots: int,
    node_chunk_order: Tensor | None,
    graph_cache: dict[SnapshotGraphCacheKey, GraphBlock],
    format: GraphFormat = "csr",
    materialize_coo: bool = False,
) -> tuple[SnapshotBlockView, int, int]:
    indices = tuple(int(index) for index in snapshot_indices)
    if not indices:
        raise ValueError("snapshot window requires at least one snapshot index")
    full_count = max(1, int(num_full_snapshots))
    decays = _normalize_chunk_decay(
        chunk_decay,
        node_chunk_order=node_chunk_order,
        window_size=len(indices),
        num_full_snapshots=full_count,
    )
    thresholds = _window_decay_thresholds(window_size=len(indices), chunk_decay=decays, num_full_snapshots=full_count)
    graphs: list[GraphBlock] = []
    source_edge_counts: list[int] = []
    hits = 0
    misses = 0
    for index, threshold in zip(indices, thresholds):
        source_edge_counts.append(_snapshot_source_edge_count(store, int(index)))
        key = (int(index), None if threshold is None else int(threshold), str(format), bool(materialize_coo))
        graph = graph_cache.get(key)
        if graph is None:
            filtered = _filter_snapshot_store_by_dst_chunk(
                store,
                index=int(index),
                max_order=threshold,
                node_chunk_order=node_chunk_order,
            )
            graph = SnapshotBlockView.from_store(filtered, format=format, materialize_coo=materialize_coo).graph
            graph_cache[key] = graph
            misses += 1
        else:
            hits += 1
        graphs.append(graph)
    if len(graphs) == 1:
        return SnapshotBlockView(graphs[0]), hits, misses
    return (
        SnapshotBlockView(
            SnapshotWindow(
                graphs=tuple(graphs),
                snapshot_indices=indices,
                chunk_decay=decays,
                num_full_snapshots=full_count,
                source_edge_counts=tuple(source_edge_counts),
            )
        ),
        hits,
        misses,
    )


def _window_decay_thresholds(
    *,
    window_size: int,
    chunk_decay: tuple[int, ...],
    num_full_snapshots: int,
) -> tuple[int | None, ...]:
    thresholds: list[int | None] = []
    for pos in range(int(window_size)):
        age = int(window_size) - int(pos) - 1
        if age < int(num_full_snapshots):
            thresholds.append(None)
            continue
        decay_index = age - int(num_full_snapshots)
        thresholds.append(int(chunk_decay[decay_index]) if decay_index < len(chunk_decay) else None)
    return tuple(thresholds)


def _normalize_chunk_decay(
    chunk_decay: Sequence[Any] | None,
    *,
    node_chunk_order: Tensor | None,
    window_size: int,
    num_full_snapshots: int,
) -> tuple[int, ...]:
    values = tuple(() if chunk_decay is None else chunk_decay)
    if not values:
        return ()
    chunk_count = _chunk_count(node_chunk_order)
    decay_count = max(0, int(window_size) - int(num_full_snapshots))
    return tuple(
        _chunk_decay_threshold(value, chunk_count=chunk_count, decay_count=decay_count, pos=pos)
        for pos, value in enumerate(values)
    )


def _chunk_count(node_chunk_order: Tensor | None) -> int | None:
    if node_chunk_order is None or int(node_chunk_order.numel()) == 0:
        return None
    return int(node_chunk_order.max().item()) + 1


def _chunk_decay_threshold(value: Any, *, chunk_count: int | None, decay_count: int, pos: int) -> int:
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "half":
            return _ratio_to_chunk_threshold(0.5 ** (int(pos) + 1), chunk_count)
        if text.startswith("auto:"):
            ratio = float(text.split(":", 1)[1])
            if decay_count > 0:
                ratio = ratio ** ((int(pos) + 1) / float(decay_count))
            return _ratio_to_chunk_threshold(ratio, chunk_count)
        value = float(text)
    if isinstance(value, float) and 0.0 <= float(value) <= 1.0:
        return _ratio_to_chunk_threshold(float(value), chunk_count)
    return int(value)


def _ratio_to_chunk_threshold(ratio: float, chunk_count: int | None) -> int:
    if chunk_count is None:
        return int(ratio)
    bounded = max(0.0, min(float(ratio), 1.0))
    return max(0, min(int(round(float(chunk_count) * bounded)), int(chunk_count)))


def _snapshot_source_edge_count(store: GraphStore, index: int) -> int:
    if store.snapshot_ptr is None:
        return int(store.num_edges)
    sid = int(index)
    if sid < 0 or sid + 1 >= int(store.snapshot_ptr.numel()):
        raise IndexError(f"snapshot index out of range: {sid}")
    begin = int(store.snapshot_ptr[sid].item())
    end = int(store.snapshot_ptr[sid + 1].item())
    return max(0, end - begin)


def _filter_snapshot_store_by_dst_chunk(
    store: "GraphStore",
    *,
    index: int | None = None,
    max_order: int | None,
    node_chunk_order: Tensor | None,
) -> "GraphStore":
    from starrygl.store import GraphStore

    snapshot = store if index is None else store.snapshot(int(index))
    if max_order is None or node_chunk_order is None or snapshot.dst is None or int(snapshot.dst.numel()) == 0:
        return snapshot
    order = node_chunk_order.to(device=snapshot.dst.device, dtype=torch.long)
    dst_order = order.index_select(0, snapshot.dst.to(device=order.device, dtype=torch.long))
    keep = (dst_order < int(max_order)).nonzero(as_tuple=True)[0].to(device=snapshot.dst.device, dtype=torch.long)
    return GraphStore(
        num_nodes=int(snapshot.num_nodes),
        src=_required_tensor(snapshot.src, "src").index_select(0, keep),
        dst=snapshot.dst.index_select(0, keep),
        edge_ids=None if snapshot.edge_ids is None else snapshot.edge_ids.index_select(0, keep.to(device=snapshot.edge_ids.device)),
        timestamps=None if snapshot.timestamps is None else snapshot.timestamps.index_select(0, keep.to(device=snapshot.timestamps.device)),
        snapshot_ptr=torch.tensor([0, int(keep.numel())], dtype=torch.long, device=snapshot.dst.device),
        edge_weight=None if snapshot.edge_weight is None else snapshot.edge_weight.index_select(0, keep.to(device=snapshot.edge_weight.device)),
    )


def _required_tensor(value: Tensor | None, name: str) -> Tensor:
    if value is None:
        raise ValueError(f"snapshot view requires {name}")
    return value


__all__ = ["SnapshotBlockView", "SnapshotWindow", "SnapshotWindowBlob"]
