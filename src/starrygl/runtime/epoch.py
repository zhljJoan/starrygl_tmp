from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch import Tensor

from starrygl.model import (
    DCRNNModel, EvolveGCNModel, GConvGRUModel, MPNNLSTMModel, StarryModel, TGCNModel,
)
from starrygl.batch import EventRows
from starrygl.store import StoreBundle
from starrygl.task import StarryTask, TaskTarget, materialize_negative_samples

from .comm import CommScheduler


@dataclass(frozen=True)
class EpochResult:
    loss: float
    steps: int
    metrics: Mapping[str, float] = field(default_factory=dict)


def with_epoch_chunk_order(
    store: StoreBundle,
    options: Mapping[str, Any] | None,
    *,
    split: str,
    generator: torch.Generator | None,
) -> Mapping[str, Any] | None:
    if not options or "chunk_order" not in options or isinstance(options["chunk_order"], Tensor):
        return options
    count = int(options.get("chunks_per_rank", 0) or snapshot_chunk_count(store))
    if count <= 0:
        return options
    policy = str(options["chunk_order"]).strip().lower()
    if policy in {"identity", "none"} or (policy == "rand" and split != "train"):
        order = torch.arange(count)
    elif policy in {"rand", "random"}:
        order = torch.randperm(count, generator=generator if _cpu_generator(generator) else None)
        if dist.is_available() and dist.is_initialized():
            payload = [order.cpu() if dist.get_rank() == 0 else torch.empty_like(order.cpu())]
            dist.broadcast_object_list(payload, src=0)
            order = payload[0].long()
    else:
        return options
    return {**options, "chunk_order": order}


def with_model_snapshot_options(
    options: Mapping[str, Any] | None,
    model: StarryModel,
) -> Mapping[str, Any]:
    out = dict(options or {})
    # These cells cache only static graph layouts, never hidden/autograd state.
    known_cell = type(model) in (GConvGRUModel, DCRNNModel)
    known_gcn = type(model) in (TGCNModel, MPNNLSTMModel, EvolveGCNModel)
    if out.get("snapshot_materialize_on_device") is True:
        if not known_gcn or bool(out.get("snapshot_reverse_direction", False)):
            raise ValueError("snapshot_materialize_on_device requires built-in TGCN/MPNN-LSTM/EvolveGCN without reverse direction")
        out["snapshot_reverse_direction"] = False
    out["_reuse_static_snapshot_graph"] = known_cell or (
        known_gcn and bool(out.get("snapshot_dgl_gcn", False))
        and not bool(out.get("snapshot_sparse_gcn", False))
    )
    out["_snapshot_edge_feature_names"] = ("w",) if known_cell or known_gcn else None
    cell = getattr(model, "runtime_cell", None)
    if cell is not None:
        out.setdefault("snapshot_reverse_direction", bool(getattr(cell, "reads_neighbor_state", False)))
    return out


def prepare_supervision(batch, *, num_negatives: int, generator: torch.Generator | None):
    if num_negatives < 1:
        return batch
    target = batch.targets.get("task") if isinstance(batch.targets, Mapping) else None
    if not isinstance(target, TaskTarget) or target.neg_dst is not None:
        return batch
    targets = dict(batch.targets)
    targets["task"] = materialize_negative_samples(
        target,
        num_negatives=num_negatives,
        generator=generator,
    )
    batch.targets = targets
    return batch


def sync_gradients(model: StarryModel, mode: str | None) -> None:
    if not _distributed_sync(mode):
        return
    scale = float(dist.get_world_size())
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad)
        parameter.grad.div_(scale)


def step_optimizer(
    model,
    optimizer,
    mode,
    comm: CommScheduler,
    *,
    has_supervision: bool,
    global_has_supervision: bool | None = None,
) -> None:
    """Keep empty ranks in gradient collectives without taking an empty Adam step."""

    sync_gradients(model, mode)
    if _distributed_sync(mode):
        if global_has_supervision is None:
            active = torch.full(
                (), int(has_supervision), dtype=torch.long,
                device=next(model.parameters()).device,
            )
            comm.all_reduce(active, op=dist.ReduceOp.MAX, name="optimizer_supervision")
            has_supervision = has_supervision or bool(active.item())
        else:
            has_supervision = bool(global_has_supervision)
    if has_supervision:
        optimizer.step()


def zero_output_loss(output: Any, model: StarryModel) -> Tensor:
    """Connect an empty-owner step to every launched autograd collective."""

    values = [output.embeddings, output.logits, output.predictions, output.aux]
    loss = None
    while values:
        value = values.pop()
        if isinstance(value, Tensor) and value.requires_grad:
            term = value.sum() * 0.0
            loss = term if loss is None else loss + term
        elif isinstance(value, Mapping):
            values.extend(value.values())
        elif isinstance(value, (tuple, list)):
            values.extend(value)
    if loss is not None:
        return loss
    parameters = [parameter.sum() * 0.0 for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("training an empty distributed batch requires a differentiable model output")
    return sum(parameters[1:], parameters[0])


def sync_model_parameters(model: StarryModel, mode: str | None) -> None:
    if not _distributed_sync(mode):
        return
    with torch.no_grad():
        for value in model.state_dict().values():
            if isinstance(value, Tensor):
                dist.broadcast(value, src=0)


def eval_autocast(device: str | torch.device | None, options: Mapping[str, Any] | None):
    dtype_name = str((options or {}).get("eval_autocast_dtype", "")).strip().lower()
    if dtype_name in {"", "none", "false", "off"}:
        return nullcontext()
    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if resolved.type != "cuda":
        return nullcontext()
    dtypes = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in dtypes:
        raise ValueError(f"unsupported eval_autocast_dtype: {dtype_name}")
    return torch.autocast(device_type="cuda", dtype=dtypes[dtype_name])


def empty_supervision(batch) -> bool:
    windows = batch.targets.get("window_tasks") if isinstance(batch.targets, Mapping) else None
    if windows is not None:
        return all(target.target_ids.numel() == 0 for target in windows)
    target = batch.targets.get("task") if isinstance(batch.targets, Mapping) else None
    ids = getattr(target, "target_ids", None)
    return ids is not None and ids.numel() == 0


def needs_empty_collective_encode(batch) -> bool:
    target = batch.targets.get("task") if isinstance(batch.targets, Mapping) else None
    collect = getattr(getattr(target, "target_route", None), "endpoint_collect", None)
    return bool(collect is not None and getattr(collect, "node_dist_index", None) is not None)


def needs_empty_state_encode(batch) -> bool:
    events = batch.targets.get("events") if isinstance(batch.targets, Mapping) else None
    return bool(
        isinstance(events, EventRows)
        and events.src.numel()
        and events.dst.numel()
    )


def bind_state_comm(
    state_manager: object | Mapping[str, object] | None,
    comm: CommScheduler,
) -> None:
    managers = state_manager.values() if isinstance(state_manager, Mapping) else (state_manager,)
    for manager in managers:
        if manager is None:
            continue
        if hasattr(manager, "comm") and getattr(manager, "comm") is None:
            manager.comm = comm
        for name in ("memory_manager", "shared_manager", "mailbox_manager", "shared_mailbox_manager"):
            child = getattr(manager, name, None)
            if child is not None and hasattr(child, "comm") and getattr(child, "comm") is None:
                child.comm = comm


def accumulate(out: dict[str, Tensor | float], metrics: Mapping[str, Tensor]) -> None:
    weight = metrics.get("num_examples")
    for name, value in metrics.items():
        value = value.detach()
        if weight is not None and name not in {"num_examples", "_classification_confusion"}:
            value = value * weight.to(value)
        if int(value.numel()) == 1:
            out[name] = sum_scalar(out.get(name, 0.0), value)
        else:
            current = out.get(name)
            out[name] = value.clone() if current is None else current + value.to(current.device)


def sum_scalar(current: Tensor | float, value: Tensor) -> Tensor | float:
    scalar = value.reshape(()).detach()
    if isinstance(current, Tensor):
        return current + scalar.to(current.device)
    return scalar if current == 0.0 else float(current) + float(scalar.cpu())


def result(
    *,
    total_loss: Tensor | float,
    total_steps: int,
    metric_sums: Mapping[str, Tensor | float],
    distributed_sum: bool = False,
) -> EpochResult:
    metric_sums = dict(metric_sums)
    loss_steps = total_steps
    if dist_world_size() > 1:
        payload = (scalar_float(total_loss), total_steps, _cpu_metrics(metric_sums))
        gathered: list[tuple[float, int, dict[str, Tensor | float]] | None] = [
            None
        ] * dist_world_size()
        dist.all_gather_object(gathered, payload)
        rows = [row for row in gathered if row is not None]
        total_loss = sum(row[0] for row in rows)
        total_steps = max((row[1] for row in rows), default=0)
        loss_steps = total_steps if distributed_sum else sum(row[1] for row in rows)
        metric_sums = _sum_metric_rows(row[2] for row in rows)
    if total_steps == 0:
        return EpochResult(0.0, 0, {name: 0.0 for name in metric_sums if not name.startswith("_")})
    denominator = metric_sums.get("num_examples")
    metric_steps = (
        max(1.0, scalar_float(denominator))
        if denominator is not None
        else float(loss_steps)
    )
    confusion = metric_sums.pop("_classification_confusion", None)
    metrics = {
        name: scalar_float(value) / metric_steps
        for name, value in metric_sums.items()
        if name != "num_examples"
    }
    if isinstance(confusion, Tensor):
        true_positive = confusion.diagonal()
        f1_denominator = confusion.sum(dim=0) + confusion.sum(dim=1)
        valid = f1_denominator > 0
        metrics["f1_macro"] = scalar_float(
            (2 * true_positive[valid] / f1_denominator[valid]).mean()
            if bool(valid.any().item())
            else confusion.sum() * 0.0
        )
    return EpochResult(
        scalar_float(total_loss) / max(1, loss_steps),
        total_steps,
        metrics,
    )


def _cpu_metrics(values: Mapping[str, Tensor | float]) -> dict[str, Tensor | float]:
    return {
        name: value.detach().cpu() if isinstance(value, Tensor) else float(value)
        for name, value in values.items()
    }


def _sum_metric_rows(rows) -> dict[str, Tensor | float]:
    out: dict[str, Tensor | float] = {}
    for row in rows:
        for name, value in row.items():
            if isinstance(value, Tensor):
                current = out.get(name)
                out[name] = value.clone() if current is None else current + value
            else:
                out[name] = float(out.get(name, 0.0)) + float(value)
    return out


def node_regression_scales(
    store: StoreBundle,
    *,
    task: StarryTask,
    device: str | torch.device | None,
    gradient_sync: str | None,
) -> tuple[Tensor | None, Tensor | None]:
    if not is_node_regression_mse(task) or dist_world_size() <= 1:
        return None, None
    dev = torch.device(device or (f"cuda:{os.environ.get('LOCAL_RANK', 0)}" if torch.cuda.is_available() else "cpu"))
    local = torch.tensor(float(_local_node_owner_count(store)), device=dev)
    total = local.clone()
    dist.all_reduce(total)
    if total.item() <= 0:
        return None, None
    report = local / total
    backward = report * dist_world_size() if _sync_mode(gradient_sync) else report
    return report, backward


def is_node_regression_mse(task: StarryTask) -> bool:
    return (
        str(getattr(task, "name", "")).lower() in {"node_prediction", "node_regression"}
        and str(getattr(task, "loss", "")).lower() == "mse"
        and str(getattr(task, "target_owner", "")).lower() == "node_master"
    )


def dist_world_size() -> int:
    return int(dist.get_world_size()) if dist.is_available() and dist.is_initialized() else 1


def dist_sum_scalar(value: Tensor | float) -> Tensor | float:
    out = dist_sum(value)
    return out.reshape(()) if isinstance(out, Tensor) else out


def dist_sum(value: Tensor | float) -> Tensor | float:
    if dist_world_size() <= 1:
        return value
    if isinstance(value, Tensor):
        out = value.detach().clone()
    else:
        device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}" if torch.cuda.is_available() else "cpu")
        out = torch.tensor(float(value), device=device)
    dist.all_reduce(out)
    return out


def scalar_float(value: Tensor | float) -> float:
    return float(value.detach().cpu()) if isinstance(value, Tensor) else float(value)


def _sync_mode(mode: str | None) -> bool:
    return str(mode or "").strip().lower() in {"all_reduce", "ddp", "mean"}


def _distributed_sync(mode: str | None) -> bool:
    return _sync_mode(mode) and dist_world_size() > 1


def _cpu_generator(generator: torch.Generator | None) -> bool:
    return generator is not None and str(getattr(generator, "device", "cpu")) == "cpu"


def snapshot_chunk_count(store: StoreBundle) -> int:
    view = getattr(store.graph, "snapshot_csc_view", None)
    if not isinstance(view, Mapping):
        return 0
    count = 0
    for row in view.get("slices", ()) or ():
        chunk = row.get("node_chunk") if isinstance(row, Mapping) else None
        if isinstance(chunk, Tensor) and chunk.numel():
            count = max(count, int(chunk.max() - chunk.min() + 1))
    return count


def _local_node_owner_count(store: StoreBundle) -> int:
    index = store.graph.partition.get("node_dist_index") if store.graph.prepare is not None else None
    if isinstance(index, Tensor) and index.numel():
        return int(((index.long() >> 48) == store.graph.rank).sum())
    for value in (
        getattr(store.labels, "node_label_ids", None),
        getattr(store.labels, "node_label", None),
        getattr(store.features, "node_ids", None),
    ):
        if isinstance(value, Tensor) and value.numel():
            return int(value.shape[1] if value.dim() >= 3 else value.shape[0])
    return 0


__all__ = ["EpochResult"]
