from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

from torch import Tensor

from starrygl.view.base import GraphBlock


BatchMode = Literal["event", "snapshot"]
WindowPolicy = Literal["event_window", "full_snapshot", "chunk_decay"]
SamplingPolicy = Literal["full", "neighbor"]


@dataclass(frozen=True)
class EventRows:
    """Raw, ordered event rows; consumers apply their own reduction."""

    src: Tensor
    dst: Tensor
    edge_ids: Tensor
    ts: Tensor | None = None
    state_write_mask: Tensor | None = None


@dataclass
class Batch:
    """One model input shared by event and snapshot execution."""

    mode: BatchMode
    features: Mapping[str, Any] = field(default_factory=dict)
    state: Mapping[str, Any] = field(default_factory=dict)
    targets: Mapping[str, Any] = field(default_factory=dict)
    blocks: Sequence[Sequence[GraphBlock]] | None = None
    graph: GraphBlock | None = None
    num_layers: int = 1

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.blocks is None and self.graph is None:
            raise ValueError("Batch requires blocks or graph")
        if self.blocks is not None:
            if len(self.blocks) < 1:
                raise ValueError("blocks requires at least one window")
            if any(len(window) < 1 for window in self.blocks):
                raise ValueError("each blocks window requires at least one layer")

    def layer_blocks(self) -> Iterable[GraphBlock]:
        if self.blocks is not None:
            return iter(self.blocks[0])
        return (self.graph for _ in range(self.num_layers) if self.graph is not None)

    def window_blocks(self, window_id: int) -> Sequence[GraphBlock]:
        if self.blocks is not None:
            return self.blocks[int(window_id)]
        if int(window_id) != 0:
            raise IndexError(window_id)
        if self.graph is not None:
            return tuple(self.graph for _ in range(self.num_layers))
        raise IndexError(window_id)

    def iter_blocks(self) -> Iterator[Sequence[GraphBlock]]:
        if self.blocks is not None:
            return iter(self.blocks)
        return iter((self.window_blocks(0),))

__all__ = [
    "Batch",
    "BatchMode",
    "EventRows",
    "SamplingPolicy",
    "WindowPolicy",
]
