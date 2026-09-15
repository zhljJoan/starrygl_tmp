from __future__ import annotations

import torch
from torch import Tensor, nn


class EdgePredictor(nn.Module):
    """MemShare-style edge predictor for positive and negative endpoint pairs."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.src_fc = nn.Linear(self.dim, self.dim)
        self.dst_fc = nn.Linear(self.dim, self.dim)
        self.out_fc = nn.Linear(self.dim, 1)

    def forward(
        self,
        h_pos_src: Tensor,
        h_pos_dst: Tensor,
        h_neg_src: Tensor | None = None,
        h_neg_dst: Tensor | None = None,
        *,
        neg_samples: int = 1,
        mode: str = "triplet",
    ) -> tuple[Tensor, Tensor | None]:
        h_pos_src = self.src_fc(h_pos_src)
        h_pos_dst = self.dst_fc(h_pos_dst)
        h_pos_edge = torch.relu(h_pos_src + h_pos_dst)
        pos_score = self.out_fc(h_pos_edge).squeeze(-1)
        if h_neg_dst is None:
            return pos_score, None
        h_neg_dst = self.dst_fc(h_neg_dst)
        if mode == "triplet":
            src = h_pos_src.tile((max(1, int(neg_samples)), 1))
            if int(src.shape[0]) != int(h_neg_dst.shape[0]):
                repeat = max(1, int(h_neg_dst.shape[0]) // max(1, int(h_pos_src.shape[0])))
                src = h_pos_src.tile((repeat, 1))[: int(h_neg_dst.shape[0])]
            h_neg_edge = torch.relu(src + h_neg_dst)
        else:
            if h_neg_src is None:
                raise ValueError("h_neg_src is required when mode is not 'triplet'")
            h_neg_edge = torch.relu(self.src_fc(h_neg_src) + h_neg_dst)
        return pos_score, self.out_fc(h_neg_edge).squeeze(-1)


__all__ = ["EdgePredictor"]
