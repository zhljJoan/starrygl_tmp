from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from torch import Tensor, nn

from starrygl.batch import Batch


@dataclass(frozen=True)
class ModelOutput:
    embeddings: Tensor | None = None
    logits: Tensor | None = None
    predictions: Tensor | None = None
    state_embeddings: Tensor | None = None
    aux: Mapping[str, Any] = field(default_factory=dict)

    @property
    def commit_embeddings(self) -> Tensor | None:
        if self.state_embeddings is not None:
            return self.state_embeddings
        return self.embeddings


@dataclass(frozen=True)
class StateDelta:
    node_ids: Tensor
    values: Tensor
    kind: str = "state"
    timestamps: Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class StarryModel(nn.Module):
    """Unified model contract for event and snapshot execution."""

    def encode(self, batch: Batch) -> ModelOutput:
        raise NotImplementedError

    def forward(self, batch: Batch) -> ModelOutput:
        return self.encode(batch)

    def state_update(self, batch: Batch, output: ModelOutput) -> StateDelta | None:
        del batch, output
        return None


__all__ = ["ModelOutput", "StarryModel", "StateDelta"]
