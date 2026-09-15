from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from starrygl.batch import BatchMode, SamplingPolicy, WindowPolicy
from starrygl.model import ModelOutput, StarryModel
from starrygl.store import StoreBundle
from starrygl.task import StarryTask

from .builders import build_task_from_config
from .comm import CommScheduler
from .loop import EpochResult, run_epoch


def fit(
    self,
    *,
    store: StoreBundle | None = None,
    model: StarryModel | None = None,
    task: StarryTask | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    epochs: int | None = None,
    artifact_root: str | Path | None = None,
    rank: int = 0,
    map_location: str | torch.device = "cpu",
    mmap: bool = False,
    mode: BatchMode | None = None,
    split: str = "train",
    window_policy: WindowPolicy | None = None,
    sampling_policy: SamplingPolicy | None = None,
    chunk_decay: Sequence[int] | torch.Tensor | None = None,
    num_full_snapshots: int | None = None,
    num_layers: int | None = None,
    fanouts: Sequence[int] | torch.Tensor | None = None,
    sampler_options: Mapping[str, Any] | None = None,
    num_negatives: int | None = None,
    state_manager: object | Mapping[str, object] | None = None,
    generator: torch.Generator | None = None,
    comm: CommScheduler | None = None,
    device: str | torch.device | None = None,
    compute_metrics: bool | None = None,
    epoch_callback: Any | None = None,
) -> list[EpochResult]:
    store_obj = self._store(store, artifact_root=artifact_root, rank=rank, map_location=map_location, mmap=mmap)
    model_obj = self._model(model, store_obj)
    device = self._device(device)
    if device is not None:
        model_obj = model_obj.to(device)
    task_obj = task if task is not None else build_task_from_config(self.task)
    optimizer_obj = optimizer if optimizer is not None else self._optimizer(model_obj)
    scheduler = comm or CommScheduler()
    state_obj = self._state_manager(
        state_manager, model=model_obj, store=store_obj, device=device, comm=scheduler,
        window_policy=window_policy, num_full_snapshots=num_full_snapshots, chunk_decay=chunk_decay,
    )
    epoch_count = int(epochs if epochs is not None else self.train_config.get("epochs", 1))
    results = []
    cumulative = 0.0
    for epoch in range(1, max(1, epoch_count) + 1):
        _reset_epoch_state(state_obj, model_obj)
        started = time.perf_counter()
        result = run_epoch(
            store=store_obj,
            model=model_obj,
            task=task_obj,
            optimizer=optimizer_obj,
            mode=self._batch_mode(mode),
            training=True,
            split=split,
            window_policy=self._window_policy(window_policy),
            sampling_policy=self._sampling_policy(sampling_policy),
            chunk_decay=self._chunk_decay(chunk_decay),
            num_full_snapshots=self._num_full_snapshots(num_full_snapshots),
            num_layers=self._num_layers(num_layers),
            fanouts=self._fanouts(fanouts),
            sampler_options=self._sampler_options(sampler_options, fanouts=fanouts, train=True, split=split),
            num_negatives=self._num_negatives(num_negatives),
            state_manager=state_obj,
            generator=generator,
            comm=scheduler,
            device=device,
            compute_metrics=(
                bool(self.runtime_config.get("train_compute_metrics", True))
                if compute_metrics is None
                else bool(compute_metrics)
            ),
            gradient_sync=self._gradient_sync(),
            wait_policy=self.plan.wait_policy,
        )
        elapsed = time.perf_counter() - started
        cumulative += elapsed
        results.append(result)
        if epoch_callback is not None:
            epoch_callback(
                {
                    "epoch": epoch,
                    "train_seconds": elapsed,
                    "train_seconds_cumulative": cumulative,
                    "train": result,
                }
            )
    return results


def evaluate(
    self,
    *,
    store: StoreBundle | None = None,
    model: StarryModel | None = None,
    task: StarryTask | None = None,
    artifact_root: str | Path | None = None,
    rank: int = 0,
    map_location: str | torch.device = "cpu",
    mmap: bool = False,
    mode: BatchMode | None = None,
    split: str | None = None,
    window_policy: WindowPolicy | None = None,
    sampling_policy: SamplingPolicy | None = None,
    chunk_decay: Sequence[int] | torch.Tensor | None = None,
    num_full_snapshots: int | None = None,
    num_layers: int | None = None,
    fanouts: Sequence[int] | torch.Tensor | None = None,
    sampler_options: Mapping[str, Any] | None = None,
    num_negatives: int | None = None,
    state_manager: object | Mapping[str, object] | None = None,
    generator: torch.Generator | None = None,
    comm: CommScheduler | None = None,
    device: str | torch.device | None = None,
    commit_state: bool = True,
    compute_metrics: bool = True,
) -> EpochResult:
    store_obj = self._store(store, artifact_root=artifact_root, rank=rank, map_location=map_location, mmap=mmap)
    split = _evaluation_split(store_obj, split)
    model_obj = self._model(model, store_obj)
    device = self._device(device)
    if device is not None:
        model_obj = model_obj.to(device)
    scheduler = comm or CommScheduler()
    state_obj = self._state_manager(
        state_manager, model=model_obj, store=store_obj, device=device, comm=scheduler,
        window_policy=window_policy, num_full_snapshots=num_full_snapshots, chunk_decay=chunk_decay,
    )
    task_obj = task if task is not None else build_task_from_config(self.task)
    mode_obj = self._batch_mode(mode)
    sampling_obj = self._sampling_policy(sampling_policy)
    options = self._sampler_options(sampler_options, fanouts=fanouts, train=False, split=split)
    common = dict(
        store=store_obj,
        model=model_obj,
        task=task_obj,
        mode=mode_obj,
        training=False,
        num_layers=self._num_layers(num_layers),
        fanouts=self._fanouts(fanouts),
        sampler_options=options,
        num_negatives=self._num_negatives(num_negatives),
        state_manager=state_obj,
        generator=generator,
        comm=scheduler,
        device=device,
        wait_policy=self.plan.wait_policy,
    )
    if _uses_exact_snapshot_replay(model_obj, task_obj, mode_obj, sampling_obj, state_obj):
        _reset_epoch_state(state_obj, model_obj)
        exact = dict(
            common,
            window_policy="full_snapshot",
            sampling_policy="full",
            chunk_decay=None,
            num_full_snapshots=1,
        )
        _warm_snapshot_history(exact, split)
        return run_epoch(split=split, commit_state=commit_state, compute_metrics=compute_metrics, **exact)
    return run_epoch(
        split=split,
        window_policy=self._window_policy(window_policy),
        sampling_policy=sampling_obj,
        chunk_decay=self._chunk_decay(chunk_decay),
        num_full_snapshots=self._num_full_snapshots(num_full_snapshots),
        commit_state=commit_state,
        compute_metrics=compute_metrics,
        **common,
    )


@torch.no_grad()
def predict(
    self,
    *,
    store: StoreBundle | None = None,
    model: StarryModel | None = None,
    task: StarryTask | None = None,
    artifact_root: str | Path | None = None,
    rank: int = 0,
    map_location: str | torch.device = "cpu",
    mmap: bool = False,
    mode: BatchMode | None = None,
    split: str = "test",
    window_policy: WindowPolicy | None = None,
    sampling_policy: SamplingPolicy | None = None,
    chunk_decay: Sequence[int] | torch.Tensor | None = None,
    num_full_snapshots: int | None = None,
    num_layers: int | None = None,
    fanouts: Sequence[int] | torch.Tensor | None = None,
    sampler_options: Mapping[str, Any] | None = None,
    state_manager: object | Mapping[str, object] | None = None,
    comm: CommScheduler | None = None,
    device: str | torch.device | None = None,
) -> list[ModelOutput]:
    store_obj = self._store(store, artifact_root=artifact_root, rank=rank, map_location=map_location, mmap=mmap)
    model_obj = self._model(model, store_obj)
    device = self._device(device)
    if device is not None:
        model_obj = model_obj.to(device)
    scheduler = comm or CommScheduler()
    state_obj = self._state_manager(
        state_manager, model=model_obj, store=store_obj, device=device, comm=scheduler,
        window_policy=window_policy, num_full_snapshots=num_full_snapshots, chunk_decay=chunk_decay,
    )
    task_obj = task if task is not None else build_task_from_config(self.task)
    mode_obj = self._batch_mode(mode)
    sampling_obj = self._sampling_policy(sampling_policy)
    options = self._sampler_options(sampler_options, fanouts=fanouts, train=False, split=split)
    outputs: list[ModelOutput] = []
    common = dict(
        store=store_obj,
        model=model_obj,
        task=task_obj,
        mode=mode_obj,
        training=False,
        num_layers=self._num_layers(num_layers),
        fanouts=self._fanouts(fanouts),
        sampler_options=options,
        num_negatives=0,
        state_manager=state_obj,
        comm=scheduler,
        device=device,
        wait_policy=self.plan.wait_policy,
    )
    if _uses_exact_snapshot_replay(model_obj, task_obj, mode_obj, sampling_obj, state_obj):
        _reset_epoch_state(state_obj, model_obj)
        exact = dict(
            common,
            window_policy="full_snapshot",
            sampling_policy="full",
            chunk_decay=None,
            num_full_snapshots=1,
        )
        _warm_snapshot_history(exact, split)
        run_epoch(
            split=split,
            commit_state=True,
            compute_metrics=False,
            output_callback=outputs.append,
            **exact,
        )
    else:
        run_epoch(
            split=split,
            window_policy=self._window_policy(window_policy),
            sampling_policy=sampling_obj,
            chunk_decay=self._chunk_decay(chunk_decay),
            num_full_snapshots=self._num_full_snapshots(num_full_snapshots),
            commit_state=True,
            compute_metrics=False,
            output_callback=outputs.append,
            **common,
        )
    return outputs


def _reset_epoch_state(
    state_manager: object | Mapping[str, object] | None,
    model: StarryModel,
) -> None:
    managers = state_manager.values() if isinstance(state_manager, Mapping) else (() if state_manager is None else (state_manager,))
    for manager in managers:
        reset = getattr(manager, "reset", None)
        if callable(reset):
            reset()
    clear = getattr(model, "clear_state_compensation", None)
    if callable(clear):
        clear()


def _uses_exact_snapshot_replay(model, task, mode, sampling, state_manager) -> bool:
    cell = getattr(model, "runtime_cell", None)
    return bool(
        mode == "snapshot"
        and sampling == "full"
        and state_manager is not None
        and getattr(task, "target_owner", None) == "node_master"
        and cell is not None
        and not bool(getattr(cell, "reads_neighbor_state", False))
    )


def _warm_snapshot_history(common: Mapping[str, Any], split: str) -> None:
    order = ("train", "val", "test")
    history = order[: order.index(split)] if split in order else ()
    store = common.get("store")
    for history_split in history:
        ptr = store.graph.split_time_ptr_2.get(history_split) if isinstance(store, StoreBundle) else None
        if isinstance(ptr, torch.Tensor) and int(ptr.numel()) == 0:
            continue
        run_epoch(split=history_split, commit_state=True, compute_metrics=False, **common)


def _evaluation_split(store: StoreBundle, split: str | None) -> str:
    if split is not None:
        return str(split)
    val = store.graph.split_time_ptr_2.get("val")
    return "test" if isinstance(val, torch.Tensor) and int(val.numel()) == 0 else "val"


__all__ = ["evaluate", "fit", "predict"]
