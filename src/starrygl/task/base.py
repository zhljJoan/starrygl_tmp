from __future__ import annotations

from typing import Mapping

from torch import Tensor

from starrygl.batch import Batch
from starrygl.model import ModelOutput


class StarryTask:
    target_owner: str
    output_owner: str
    name: str

    def supervision(self, batch: Batch) -> Batch:
        return batch

    def compute_loss(self, output: ModelOutput, batch: Batch) -> Tensor:
        raise NotImplementedError

    def compute_metrics(self, output: ModelOutput, batch: Batch) -> Mapping[str, Tensor]:
        del output, batch
        return {}


__all__ = ["StarryTask"]
