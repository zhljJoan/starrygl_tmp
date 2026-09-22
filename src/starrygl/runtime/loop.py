from __future__ import annotations

from functools import partial
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor

from starrygl.model import ModelOutput, StarryModel
from starrygl.store import StoreBundle
from starrygl.task import NodePredictionTask, StarryTask

from .dataloader.pipeline import state_prefetch_enabled
from .dataloader.loader import DataLoader, with_materialize_device
from .comm import CommScheduler
from .epoch import (
    EpochResult,
    accumulate,
    bind_state_comm,
    dist_world_size,
    empty_supervision,
    eval_autocast,
    is_node_regression_mse,
    needs_empty_collective_encode,
    needs_empty_state_encode,
    node_regression_scales,
    prepare_supervision,
    result,
    sum_scalar,
    step_optimizer,
    sync_model_parameters,
    with_epoch_chunk_order,
    with_model_snapshot_options,
    zero_output_loss,
)
from .endpoint import materialize_endpoint_output
from .state import (
    finish_state_update,
    launch_state_update,
    poll_state_update,
)
from .snapshot.scan import encode_model
from starrygl.batch import BatchMode, SamplingPolicy, WindowPolicy
from .state.access import finish_hydrate_state, submit_hydrate_state


def run_epoch(
    *,
    store: StoreBundle,
    model: StarryModel,
    task: StarryTask,
    mode: BatchMode,
    training: bool,
    window_policy: WindowPolicy,
    sampling_policy: SamplingPolicy,
    optimizer: torch.optim.Optimizer | None = None,
    split: str = "train",
    chunk_decay: Sequence[int] | Tensor | None = None,
    num_full_snapshots: int = 1,
    num_layers: int = 1,
    fanouts: Sequence[int] | Tensor | None = None,
    sampler_options: Mapping[str, Any] | None = None,
    num_negatives: int = 0,
    state_manager: object | Mapping[str, object] | None = None,
    generator: torch.Generator | None = None,
    comm: CommScheduler | None = None,
    device: str | torch.device | None = None,
    compute_metrics: bool = True,
    commit_state: bool = False,
    gradient_sync: str | None = None,
    output_callback: Callable[[ModelOutput], None] | None = None,
    wait_policy: str = "block",
    batch_callback: Callable | None = None,
) -> EpochResult:
    """Execute one train or evaluation epoch through the shared runtime spine."""

    if wait_policy != "block":
        raise NotImplementedError("the ready-batch pipeline currently implements wait_policy='block' only")
    model.train(training)
    if training:
        sync_model_parameters(model, gradient_sync)
    options = with_materialize_device(sampler_options, device)
    options = dict(with_epoch_chunk_order(store, options, split=split, generator=generator) or {})
    options = dict(with_model_snapshot_options(options, model))
    window_mean = getattr(task, "train_loss_mode", "last_only") == "window_mean"
    if window_mean and (mode != "snapshot" or sampling_policy == "neighbor"
                        or task.target_owner != "node_master"):
        raise ValueError("window_mean requires full/chunk snapshot node prediction, not Event/edge/neighbor sampling")
    options["_snapshot_train_loss_mode"] = "window_mean" if training and window_mean else "last_only"
    batch_local_state = _batch_local_snapshot_state(
        model,
        mode=mode,
        training=training,
        window_policy=window_policy,
    )
    active_state_manager = None if batch_local_state else state_manager
    pipeline_enabled = bool(options.get("access_pipeline", True))
    if pipeline_enabled:
        options.setdefault("defer_node_feature_finish", True)
        options.setdefault("defer_feature_launch", True)
    scheduler = comm or CommScheduler()
    bind_state_comm(active_state_manager, scheduler)
    report_scale, backward_scale = node_regression_scales(
        store,
        task=task,
        device=device,
        gradient_sync=gradient_sync,
    )
    if device is not None and options.get("feature_cache_on_device", False):
        store.features.to(device)
    task_device = device if device is not None and options.get("snapshot_materialize_on_device") is True else "cpu"
    store.labels.task_payload = {
        name: value.to(device=task_device) for name, value in store.labels.task_payload.items()
    }

    skip = int(options.get("skip_batches_per_epoch", 0) or 0)
    maximum = int(options.get("max_batches_per_epoch", 0) or 0)
    prefetch_state = (
        partial(submit_hydrate_state, state_manager=active_state_manager)
        if state_prefetch_enabled(active_state_manager, pipeline_enabled=pipeline_enabled)
        else None
    )
    batches = DataLoader(
        store,
        mode=mode,
        split=split,
        window_policy=window_policy,
        sampling_policy=sampling_policy,
        chunk_decay=chunk_decay,
        num_full_snapshots=num_full_snapshots,
        num_layers=num_layers,
        fanouts=fanouts,
        sampler_options=options,
        num_negatives=num_negatives,
        generator=generator,
        comm=scheduler,
        enabled=pipeline_enabled,
        device=device,
        prefetch_state=prefetch_state,
        skip=skip,
        maximum=maximum,
    )
    supervision_schedule = (
        _prepared_supervision_schedule(
            store,
            task=task,
            batches=batches,
            mode=mode,
            window_policy=window_policy,
            sampling_policy=sampling_policy,
            batch_callback=batch_callback,
            comm=scheduler,
            device=next(model.parameters()).device,
        )
        if training and optimizer is not None and gradient_sync in {"all_reduce", "ddp", "mean"}
        else None
    )

    total_loss: Tensor | float = 0.0
    total_steps = 0
    metrics: dict[str, Tensor | float] = {}
    should_commit_state = bool((training or commit_state) and active_state_manager is not None)
    progress_every = int(options.get("progress_every", 0) or 0)
    force_empty_step = bool(
        training
        and dist_world_size() > 1
        and gradient_sync in {"all_reduce", "ddp", "mean"}
    )
    for batch_id, batch in enumerate(batches):
        globally_active = None if supervision_schedule is None else supervision_schedule[batch_id]
        poll_state_update(active_state_manager)
        if prefetch_state is None:
            finish_state_update(active_state_manager)
            pending = submit_hydrate_state(batch, active_state_manager)
            batch = finish_hydrate_state(batch, pending)
        if _skip_empty_batch(batch) and not force_empty_step:
            if should_commit_state:
                launch_state_update(active_state_manager, None)
            continue

        batch = prepare_supervision(task.supervision(batch), num_negatives=num_negatives, generator=generator)
        if batch_callback is not None:
            batch_callback(batch)
        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with eval_autocast(device, options):
                output = encode_model(
                    model,
                    batch,
                    persist_state=False if batch_local_state else None,
                    comm=scheduler,
                )
            output = materialize_endpoint_output(model, batch, output, comm=scheduler)
            if output_callback is not None:
                output_callback(output)
            if empty_supervision(batch):
                if training:
                    zero_output_loss(output, model).backward()
                    if optimizer is not None:
                        step_optimizer(
                            model, optimizer, gradient_sync, scheduler,
                            has_supervision=False, global_has_supervision=globally_active,
                        )
                if should_commit_state:
                    with torch.no_grad():
                        launch_state_update(active_state_manager, model.state_update(batch, output))
                continue

            if not training and not compute_metrics:
                if should_commit_state:
                    with torch.no_grad():
                        launch_state_update(active_state_manager, model.state_update(batch, output))
                continue
            loss = task.compute_loss(output, batch)
            if training:
                backward_loss = loss if backward_scale is None else loss * backward_scale.to(loss)
                backward_loss.backward()
                if optimizer is not None:
                    step_optimizer(
                        model, optimizer, gradient_sync, scheduler,
                        has_supervision=True, global_has_supervision=globally_active,
                    )
            if should_commit_state:
                with torch.no_grad():
                    launch_state_update(active_state_manager, model.state_update(batch, output))
        report_loss = loss if report_scale is None else loss * report_scale.to(loss)
        total_loss = sum_scalar(total_loss, report_loss.detach())
        total_steps += 1
        if progress_every > 0 and total_steps % progress_every == 0 and store.graph.rank <= 0:
            print(
                f"[starrygl.{'train' if training else 'eval'}] "
                f"split={split} step={total_steps}",
                flush=True,
            )
        if compute_metrics:
            values = task.compute_metrics(output, batch)
            if report_scale is not None:
                values = {name: value * report_scale.to(value) for name, value in values.items()}
            accumulate(metrics, values)

    finish_state_update(
        active_state_manager,
        final=bool(options.get("final_shared_flush", True)),
    )
    return result(
        total_loss=total_loss,
        total_steps=total_steps,
        metric_sums=metrics,
        distributed_sum=is_node_regression_mse(task) and dist_world_size() > 1,
    )


def _skip_empty_batch(batch) -> bool:
    return (
        empty_supervision(batch)
        and not needs_empty_collective_encode(batch)
        and not needs_empty_state_encode(batch)
        and not (batch.mode == "snapshot" and batch.state)
    )


def _prepared_supervision_schedule(
    store,
    *,
    task,
    batches,
    mode,
    window_policy,
    sampling_policy,
    batch_callback,
    comm,
    device,
):
    """Resolve static full-snapshot node activity with one collective."""

    ptr = getattr(store.labels, "task_ptr", None)
    cache = getattr(store.graph, "runtime_cache", None)
    if (
        dist_world_size() <= 1
        or type(task) is not NodePredictionTask
        or getattr(task, "train_loss_mode", "last_only") != "last_only"
        or batch_callback is not None
        or mode != "snapshot"
        or window_policy != "full_snapshot"
        or sampling_policy != "full"
        or not isinstance(ptr, Tensor)
        or not isinstance(cache, dict)
    ):
        return None
    window_ids = batches.window_ids[batches.skip :]
    if batches.maximum:
        window_ids = window_ids[: batches.maximum]
    key = ("global_supervision", batches.split, window_ids.start, window_ids.stop)
    if key not in cache:
        active = (ptr[window_ids.start + 1 : window_ids.stop + 1]
                  > ptr[window_ids.start : window_ids.stop]).to(device=device, dtype=torch.int32)
        comm.all_reduce(active, op=torch.distributed.ReduceOp.MAX, name="prepared_supervision")
        cache[key] = tuple(bool(value) for value in active.cpu().tolist())
    return cache[key]


def _batch_local_snapshot_state(
    model: StarryModel,
    *,
    mode: BatchMode,
    training: bool,
    window_policy: WindowPolicy = "full_snapshot",
) -> bool:
    if not training or mode != "snapshot":
        return False
    if bool(getattr(model, "runtime_batch_local_state", False)):
        return True
    cell = getattr(model, "runtime_cell", None)
    return cell is not None and not bool(getattr(cell, "reads_neighbor_state", False))


__all__ = ["EpochResult", "run_epoch"]
