from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


SPLIT_NAMES = ("train", "val", "test")


def resolve_time_ptr_2(*, edge_count: int, ts: Tensor | None, config: Any) -> Tensor:
    if config.time_ptr_2 is not None:
        ptr = config.time_ptr_2.long().cpu().contiguous()
        if ptr.dim() != 2 or int(ptr.size(1)) != 2:
            raise ValueError("time_ptr_2 must have shape [num_slices, 2]")
        return ptr
    if config.time_split == "batch":
        return _batch_time_ptr_2(edge_count=edge_count, target_batch_size=int(config.target_batch_size))
    if config.time_split == "adaptive_batch":
        if ts is None:
            raise ValueError("adaptive_batch time splitting requires ts")
        return _adaptive_batch_time_ptr_2(
            edge_count=edge_count, ts=ts, target_batch_size=int(config.target_batch_size)
        )
    slices = int(config.num_time_slices or 1)
    if slices < 1:
        raise ValueError("num_time_slices must be >= 1")
    if edge_count == 0:
        return torch.zeros((slices, 2), dtype=torch.long)
    bounds = torch.linspace(0, edge_count, steps=slices + 1).round().long()
    return torch.stack((bounds[:-1], bounds[1:]), dim=1).long().contiguous()


def resolve_split_ranges(
    *, edge_count: int, ts: Tensor | None, config: Any, split_labels: Tensor | None = None
) -> dict[str, tuple[int, int]]:
    edge_count = int(edge_count)
    if split_labels is not None:
        return _split_ranges_from_labels(split_labels.long().cpu().contiguous(), edge_count=edge_count)
    ratios = torch.tensor(tuple(float(v) for v in config.split_ratios), dtype=torch.float64)
    ratios = ratios / ratios.sum()
    if edge_count == 0:
        return {name: (0, 0) for name in SPLIT_NAMES}
    first = int(round(float(ratios[0].item()) * edge_count))
    second = int(round(float((ratios[0] + ratios[1]).item()) * edge_count))
    first = max(0, min(edge_count, first))
    second = max(first, min(edge_count, second))
    if ts is not None:
        first = _move_boundary_off_equal_ts(ts=ts, boundary=first, low=0, high=edge_count)
        second = _move_boundary_off_equal_ts(ts=ts, boundary=second, low=first, high=edge_count)
    return {"train": (0, first), "val": (first, second), "test": (second, edge_count)}


def split_masks(
    *, edge_count: int, split_ranges: dict[str, tuple[int, int]], split_labels: Tensor | None
) -> dict[str, Tensor]:
    if split_labels is not None:
        return {name: split_labels == index for index, name in enumerate(SPLIT_NAMES)}
    result = {}
    for name, (begin, end) in split_ranges.items():
        result[name] = torch.zeros(edge_count, dtype=torch.bool)
        result[name][int(begin) : int(end)] = True
    return result


def resolve_split_time_ptr_2(
    *, edge_count: int, ts: Tensor | None, config: Any, split_ranges: dict[str, tuple[int, int]]
) -> dict[str, Tensor]:
    if config.time_ptr_2 is not None:
        ptr = resolve_time_ptr_2(edge_count=edge_count, ts=ts, config=config)
        return _split_time_ptr_2_by_rows(ptr=ptr, split_ratios=config.split_ratios)
    return {
        name: _resolve_range_time_ptr_2(split=name, begin=begin, end=end, ts=ts, config=config)
        for name, (begin, end) in split_ranges.items()
    }


def concat_time_ptr_2(split_time_ptr_2: dict[str, Tensor]) -> Tensor:
    chunks = [split_time_ptr_2[name] for name in SPLIT_NAMES if int(split_time_ptr_2[name].numel())]
    return torch.cat(chunks).long().contiguous() if chunks else torch.empty((0, 2), dtype=torch.long)


def _split_ranges_from_labels(labels: Tensor, *, edge_count: int) -> dict[str, tuple[int, int]]:
    if int(labels.numel()) != edge_count:
        raise ValueError("split_labels must have one value per edge")
    counts = [int((labels == index).sum()) for index in range(3)]
    if sum(counts) != edge_count:
        raise ValueError("split_labels may only contain 0=train, 1=val, 2=test")
    expected = torch.repeat_interleave(torch.arange(3, dtype=labels.dtype), torch.tensor(counts))
    if not torch.equal(labels, expected):
        raise ValueError("split_labels must be contiguous in train/val/test order after timestamp sorting")
    train_end, val_end = counts[0], counts[0] + counts[1]
    return {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, edge_count)}


def _move_boundary_off_equal_ts(*, ts: Tensor, boundary: int, low: int, high: int) -> int:
    boundary, low, high = int(boundary), int(low), int(high)
    if boundary <= low or boundary >= high:
        return max(low, min(high, boundary))
    if ts[boundary - 1].item() != ts[boundary].item():
        return boundary
    value = ts[boundary].item()
    window = ts[low:high]
    left = low + int(torch.searchsorted(window, torch.as_tensor(value, dtype=window.dtype)).item())
    right = low + int(torch.searchsorted(window, torch.as_tensor(value, dtype=window.dtype), right=True).item())
    if left == low:
        return right
    if right == high:
        return left
    return left if boundary - left <= right - boundary else right


def _resolve_range_time_ptr_2(*, split: str, begin: int, end: int, ts: Tensor | None, config: Any) -> Tensor:
    edge_count = int(end) - int(begin)
    if edge_count <= 0:
        return torch.empty((0, 2), dtype=torch.long)
    if config.time_split == "adaptive_batch" and split == "train":
        if ts is None:
            raise ValueError("adaptive_batch time splitting requires ts")
        local = _adaptive_batch_time_ptr_2(edge_count=edge_count, ts=ts[begin:end], target_batch_size=int(config.target_batch_size))
    elif config.time_split in {"batch", "adaptive_batch"}:
        local = _batch_time_ptr_2(edge_count=edge_count, target_batch_size=int(config.target_batch_size))
    else:
        local = _equal_edge_time_ptr_2(edge_count=edge_count, slices=int(config.num_time_slices or 1))
    return local + int(begin)


def _equal_edge_time_ptr_2(*, edge_count: int, slices: int) -> Tensor:
    if edge_count <= 0:
        return torch.empty((0, 2), dtype=torch.long)
    bounds = torch.linspace(0, edge_count, steps=slices + 1).round().long()
    return torch.stack((bounds[:-1], bounds[1:]), dim=1).long().contiguous()


def _split_time_ptr_2_by_rows(*, ptr: Tensor, split_ratios: tuple[float, float, float]) -> dict[str, Tensor]:
    ptr = ptr.long().cpu().contiguous()
    count = int(ptr.shape[0])
    if count == 0:
        return {name: torch.empty((0, 2), dtype=torch.long) for name in SPLIT_NAMES}
    ratios = torch.tensor(split_ratios, dtype=torch.float64)
    ratios /= ratios.sum()
    first = max(0, min(count, round(float(ratios[0]) * count)))
    second = max(first, min(count, round(float(ratios[:2].sum()) * count)))
    return {"train": ptr[:first].clone(), "val": ptr[first:second].clone(), "test": ptr[second:].clone()}


def _batch_time_ptr_2(*, edge_count: int, target_batch_size: int) -> Tensor:
    if edge_count == 0:
        return torch.zeros((0, 2), dtype=torch.long)
    bounds = torch.arange(0, edge_count, target_batch_size, dtype=torch.long)
    if int(bounds[-1]) != edge_count:
        bounds = torch.cat((bounds, torch.tensor([edge_count])))
    return torch.stack((bounds[:-1], bounds[1:]), dim=1).long().contiguous()


def _adaptive_batch_time_ptr_2(*, edge_count: int, ts: Tensor, target_batch_size: int) -> Tensor:
    if edge_count == 0:
        return torch.zeros((0, 2), dtype=torch.long)
    ts = ts.cpu().contiguous()
    if int(ts.numel()) != edge_count:
        raise ValueError("ts must have one value per edge")
    rows = []
    begin = 0
    while begin < edge_count:
        target = min(begin + target_batch_size, edge_count)
        left = target
        while left > begin and target < edge_count and ts[left - 1].item() == ts[left].item():
            left -= 1
        right = target
        while right < edge_count and ts[right - 1].item() == ts[right].item():
            right += 1
        end = target if target == edge_count else right if left == begin else left if right == edge_count else left if target - left < right - target else right
        rows.append((begin, end))
        begin = end
    return torch.tensor(rows, dtype=torch.long)


__all__ = ["concat_time_ptr_2", "resolve_split_ranges", "resolve_split_time_ptr_2", "resolve_time_ptr_2", "split_masks"]
