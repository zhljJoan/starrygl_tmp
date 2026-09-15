from __future__ import annotations

import torch
from torch import Tensor


class SnapshotHistory:
    """Sliding-window slots with cumulative per-step increment statistics.

    Packet columns are state, mean increment, observation count, producer version.
    W circular output slots share one predecessor slot. Producer versions are
    metadata, not allocation indices. Snapshot s produces version s+1.
    """

    def __init__(self, node_ids: Tensor, num_nodes: int, window_size: int, dim: int):
        if window_size < 1:
            raise ValueError("snapshot history window_size must be positive")
        self.node_ids = node_ids.long()
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.window_start = 0
        self.row_map = node_ids.new_full((num_nodes,), -1)
        self.row_map[node_ids] = torch.arange(node_ids.numel(), device=node_ids.device)
        self.packets = torch.zeros(
            window_size + 1, node_ids.numel(), 2 * dim + 2, device=node_ids.device,
        )
        self.valid = torch.zeros(window_size + 1, node_ids.numel(), dtype=torch.bool, device=node_ids.device)
        self.valid[0] = True

    @property
    def values(self) -> Tensor:
        return self.packets[..., :self.dim]

    @torch.no_grad()
    def advance(self, window_start: int) -> None:
        if window_start < self.window_start:
            raise ValueError("reset snapshot history before replaying an earlier window")
        if window_start == self.window_start:
            return
        predecessor = self.read(self.node_ids, window_start)
        self.packets[0].copy_(predecessor)
        self.valid[0] = True
        self.valid[1:] &= self.packets[1:, :, -1] > window_start
        self.window_start = int(window_start)

    def read(self, node_ids: Tensor, version: int | Tensor) -> Tensor:
        rows = self.row_map[node_ids]
        versions = self.packets[:, rows, -1]
        valid = self.valid[:, rows] & (versions <= version)
        selected = torch.where(valid, versions, -1).argmax(0)
        packets = self.packets[selected, rows]
        return packets.masked_fill(~valid.any(0)[:, None], 0)

    @torch.no_grad()
    def update(self, node_ids: Tensor, version: int, values: Tensor) -> None:
        if not self.window_start < version <= self.window_start + self.window_size:
            raise ValueError("snapshot output is outside the active history window")
        previous = self.read(node_ids, version - 1)
        count = previous[:, -2:-1] + 1
        gap = version - previous[:, -1:]
        change = (values.detach() - previous[:, :self.dim]) / gap
        mean = (previous[:, self.dim:2 * self.dim] * (count - 1) + change) / count
        packet = torch.cat((values.detach(), mean, count, torch.full_like(count, version)), 1)
        self.install(node_ids, packet)

    @torch.no_grad()
    def install(self, node_ids: Tensor, packets: Tensor) -> None:
        rows = self.row_map[node_ids]
        versions = packets[:, -1].long()
        slots = torch.where(versions <= self.window_start, 0, 1 + (versions - 1) % self.window_size)
        keep = (versions >= 0) & (versions <= self.window_start + self.window_size)
        keep &= ~self.valid[slots, rows] | (versions >= self.packets[slots, rows, -1])
        slots, rows = slots[keep], rows[keep]
        self.packets[slots, rows] = packets[keep]
        self.valid[slots, rows] = True

    def reset(self) -> None:
        self.packets.zero_()
        self.valid.zero_()
        self.valid[0] = True
        self.window_start = 0
