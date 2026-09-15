"""Epoch-boundary measurements shared by config-driven component experiments."""
from __future__ import annotations

import time

import torch
import torch.distributed as dist

from starrygl.store.graph import split_window_range


def synchronized_start(device, comm):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    if dist.is_initialized():
        dist.barrier(group=comm.group)
    return time.perf_counter()


def elapsed_rankmax(start, device, comm):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = torch.tensor(time.perf_counter() - start, dtype=torch.float64, device=device)
    if dist.is_initialized():
        comm.all_reduce(elapsed, op=dist.ReduceOp.MAX, name="experiment_elapsed")
    return float(elapsed)


def prepared_workload(store, device, comm):
    """Count final-window owner targets, excluding negatives and history reuse."""
    windows = split_window_range(store.graph.split_time_ptr_2, "train")
    ptr = store.labels.task_ptr
    counts = (ptr[1:] - ptr[:-1])[windows.start:windows.stop].to(device)
    if dist.is_initialized():
        comm.all_reduce(counts, name="experiment_target_counts")
    return {"train_prepared_targets": int(counts.sum()),
            "train_supervised_windows": int((counts > 0).sum()),
            "train_windows": len(windows)}


def global_counters(runtime, device, comm):
    counts = runtime.profile_counters() if hasattr(runtime, "profile_counters") else {}
    if not dist.is_initialized():
        return counts
    rows = [None] * comm.world_size
    dist.all_gather_object(rows, counts, group=comm.group)
    return {key: sum(row.get(key, 0) for row in rows)
            for key in sorted(set().union(*(row.keys() for row in rows)))}
