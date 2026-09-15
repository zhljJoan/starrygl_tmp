from __future__ import annotations

from typing import Any, Mapping, Sequence

from torch import Tensor

from starrygl.batch import Batch
from starrygl.view import GraphBlock


AccessedWindow = tuple[
    int,
    Mapping[str, Any],
    tuple[tuple[GraphBlock, ...], ...],
    tuple[Tensor, ...],
    tuple[Tensor, ...],
]


def attach_comm_to_blocks(blocks: Sequence[GraphBlock], comm: Any | None) -> None:
    if comm is None:
        return
    for block in blocks:
        block.cache["comm"] = comm


def available_feature_names(features: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(name)
        for name, value in features.items()
        if value.dim() > 0 and (value.dim() == 1 or int(value.shape[-1]) > 0)
    )


def materialized_batch(
    *,
    mode: str,
    features: Mapping[str, Any],
    targets: Mapping[str, Any],
    blocks: Sequence[Sequence[GraphBlock]] | None = None,
    graph: GraphBlock | None = None,
    num_layers: int = 1,
    state: Mapping[str, Any] | None = None,
) -> Batch:
    return Batch(
        mode=mode,  # type: ignore[arg-type]
        features=features,
        targets=targets,
        state={} if state is None else state,
        blocks=blocks,
        graph=graph,
        num_layers=int(num_layers),
    )


def materialize_accessed_window(
    accessed: AccessedWindow,
    *,
    mode: str,
    num_layers: int,
) -> Batch:
    """Turn either graph accessor's short tuple into the one model Batch."""

    _, targets, blocks, _, _ = accessed
    return materialized_batch(
        mode=mode,
        features={},
        targets=targets,
        blocks=blocks,
        graph=blocks[-1][-1],
        num_layers=int(num_layers),
    )


__all__ = [
    "AccessedWindow",
    "attach_comm_to_blocks",
    "available_feature_names",
    "materialize_accessed_window",
    "materialized_batch",
]
