from __future__ import annotations

from typing import Literal, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from starrygl.batch import Batch
from starrygl.model import ModelOutput

from .base import StarryTask
from .target import TaskTarget


class NodePredictionTask(StarryTask):
    target_owner = "node_master"
    output_owner = "node_master"
    name = "node_prediction"

    def __init__(
        self,
        *,
        name: Literal["node_prediction", "node_classification", "node_regression"] = "node_prediction",
        loss: str = "cross_entropy",
        train_loss_mode: Literal["last_only", "window_mean"] = "last_only",
    ) -> None:
        if name not in {"node_prediction", "node_classification", "node_regression"}:
            raise ValueError(f"unsupported node task: {name!r}")
        self.name = name
        self.loss = loss
        if train_loss_mode not in {"last_only", "window_mean"}:
            raise ValueError("train_loss_mode must be last_only or window_mean")
        self.train_loss_mode = train_loss_mode

    def compute_loss(self, output: ModelOutput, batch: Batch) -> Tensor:
        windows = _window_node_supervision(output, batch, predictions=self.loss == "mse")
        if windows:
            losses = []
            for value, label in windows:
                if int(value.shape[0]) == 0:
                    losses.append(value.sum() * 0.0)
                elif self.loss == "mse":
                    value, label = _align_regression_target(value, label)
                    losses.append(F.mse_loss(value, label))
                else:
                    target = label.argmax(dim=-1) if label.dim() == value.dim() else label.long()
                    losses.append(F.cross_entropy(value, target))
            return torch.stack(losses).mean()
        label = _label(batch)
        if self.loss == "mse":
            pred = _required(output.predictions if output.predictions is not None else output.logits, "predictions")
            pred, label = _select_node_supervision(pred, label, batch)
            if int(pred.shape[0]) == 0:
                return pred.sum() * 0.0
            pred, label = _align_regression_target(pred, label)
            return F.mse_loss(pred, label)
        logits = _required(output.logits, "logits")
        logits, label = _select_node_supervision(logits, label, batch)
        if int(logits.shape[0]) == 0:
            return logits.sum() * 0.0
        target = label.argmax(dim=-1) if label.dim() == logits.dim() else label.long()
        return F.cross_entropy(logits, target)

    def compute_metrics(self, output: ModelOutput, batch: Batch) -> Mapping[str, Tensor]:
        windows = _window_node_supervision(output, batch, predictions=self.loss == "mse")
        if windows:
            if self.loss == "mse":
                losses = [F.mse_loss(*_align_regression_target(value, label)) if value.shape[0]
                          else value.sum() * 0.0 for value, label in windows]
                return {"mse": torch.stack(losses).mean()}
            confusion = windows[0][0].new_zeros((int(windows[0][0].shape[-1]),) * 2)
            for logits, label in windows:
                truth = label.argmax(dim=-1) if label.dim() == logits.dim() else label.long()
                confusion = confusion + _confusion_matrix(logits.argmax(dim=-1), truth, int(logits.shape[-1]))
            return _classification_metrics(confusion)
        label = _label(batch)
        if self.loss == "mse":
            pred = _required(output.predictions if output.predictions is not None else output.logits, "predictions")
            pred, label = _select_node_supervision(pred, label, batch)
            if int(pred.shape[0]) == 0:
                return {}
            pred, label = _align_regression_target(pred, label)
            return {"mse": F.mse_loss(pred, label)}
        logits = _required(output.logits, "logits")
        logits, label = _select_node_supervision(logits, label, batch)
        if int(logits.shape[0]) == 0:
            return {}
        pred = logits.argmax(dim=-1)
        truth = label.argmax(dim=-1) if label.dim() == logits.dim() else label.long()
        return _classification_metrics(_confusion_matrix(pred, truth, int(logits.shape[-1])))


class EdgePredictionTask(StarryTask):
    target_owner = "edge_master"
    output_owner = "edge_master"
    name = "edge_prediction"

    def __init__(self, *, loss: str = "bce") -> None:
        self.loss = str(loss)

    def compute_loss(self, output: ModelOutput, batch: Batch) -> Tensor:
        if "precomputed_loss" in output.aux:
            return output.aux["precomputed_loss"]
        if "pos_score" in output.aux and "neg_score" in output.aux:
            pos = output.aux["pos_score"]
            neg = output.aux["neg_score"]
            if self.loss in {"softmax", "cross_entropy", "ranking"}:
                logits = _ranking_logits(pos, neg)
                return F.cross_entropy(logits, torch.zeros(int(logits.shape[0]), dtype=torch.long, device=logits.device))
            pos_loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
            neg_loss = F.binary_cross_entropy_with_logits(
                neg,
                torch.zeros_like(neg),
                weight=_neg_weight(batch, neg),
            )
            return pos_loss + neg_loss
        logits = _required(output.logits, "logits")
        label = _label(batch).to(dtype=logits.dtype, device=logits.device)
        weight = _maybe_tensor(batch, "weight", device=logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, label, reduction="none")
        return (loss * weight).mean() if weight is not None else loss.mean()

    def compute_metrics(self, output: ModelOutput, batch: Batch) -> Mapping[str, Tensor]:
        if "precomputed_metrics" in output.aux:
            return output.aux["precomputed_metrics"]
        if "pos_score" in output.aux and "neg_score" in output.aux:
            pos = output.aux["pos_score"]
            neg = output.aux["neg_score"]
            metrics = _binary_rank_metrics(pos, neg)
            metrics["margin"] = pos.mean() - neg.mean()
            return metrics
        logits = _required(output.logits, "logits")
        label = _label(batch).to(dtype=logits.dtype, device=logits.device)
        pred = (torch.sigmoid(logits) >= 0.5).to(dtype=label.dtype)
        metrics = _binary_label_metrics(logits, label)
        metrics["accuracy"] = (pred == label).float().mean()
        return metrics


def _task_target(batch: Batch) -> TaskTarget | None:
    value = batch.targets.get("task") if isinstance(batch.targets, Mapping) else None
    return value if isinstance(value, TaskTarget) else None


def _label(batch: Batch) -> Tensor:
    target = _task_target(batch)
    if target is not None and target.label is not None:
        return target.label
    return batch.targets["label"]


def _neg_weight(batch: Batch, neg: Tensor) -> Tensor | None:
    target = _task_target(batch)
    if target is not None and target.neg_loss_weight is not None:
        return target.neg_loss_weight.to(device=neg.device, dtype=neg.dtype)
    weight = batch.targets.get("neg_weight") if isinstance(batch.targets, Mapping) else None
    return None if weight is None else weight.to(device=neg.device, dtype=neg.dtype)


def _maybe_tensor(batch: Batch, key: str, *, device: torch.device) -> Tensor | None:
    value = batch.targets.get(key) if isinstance(batch.targets, Mapping) else None
    return None if value is None else value.to(device=device)


def _select_roots(value: Tensor, batch: Batch) -> Tensor:
    target = _task_target(batch)
    if target is not None and target.target_route is not None and target.target_route.target_rows is not None:
        return value.index_select(0, target.target_route.target_rows.to(device=value.device).long())
    root_lids = batch.targets.get("root_lids") if isinstance(batch.targets, Mapping) else None
    if root_lids is None:
        return value
    return value.index_select(0, root_lids.to(device=value.device).long())


def _select_node_supervision(value: Tensor, label: Tensor, batch: Batch) -> tuple[Tensor, Tensor]:
    target = _task_target(batch)
    return _select_target_supervision(value, label, target, batch=batch)


def _select_target_supervision(
    value: Tensor,
    label: Tensor,
    target: TaskTarget | None,
    *,
    batch: Batch | None = None,
) -> tuple[Tensor, Tensor]:
    rows = None
    row_bound = None
    if target is not None and target.target_route is not None:
        rows = target.target_route.target_rows
        row_bound = target.target_route.target_row_bound if rows is not None else None
    if rows is None and batch is not None and isinstance(batch.targets, Mapping):
        rows = batch.targets.get("root_lids")
    if rows is None:
        return value, label
    rows = rows.to(device=value.device).long()
    if row_bound == int(value.shape[0]):
        return value.index_select(0, rows), label
    valid = (rows >= 0) & (rows < int(value.shape[0]))
    if not bool(torch.all(valid).item()):
        keep = valid.nonzero(as_tuple=True)[0]
        rows = rows.index_select(0, keep)
        label = label.index_select(0, keep.to(device=label.device))
    return value.index_select(0, rows), label


def _window_node_supervision(
    output: ModelOutput,
    batch: Batch,
    *,
    predictions: bool,
) -> tuple[tuple[Tensor, Tensor], ...]:
    values = output.aux.get("window_predictions" if predictions else "window_logits")
    if values is None and predictions:
        values = output.aux.get("window_logits")
    targets = batch.targets.get("window_tasks") if isinstance(batch.targets, Mapping) else None
    if targets is None:
        return ()
    if not isinstance(values, (tuple, list)) or not isinstance(targets, (tuple, list)):
        raise ValueError("window_mean requires model window predictions/logits and window task targets")
    if len(values) != len(targets):
        raise ValueError("snapshot window predictions and targets must have the same length")
    pairs = []
    for value, target in zip(values, targets):
        if not isinstance(value, Tensor) or not isinstance(target, TaskTarget) or target.label is None:
            raise ValueError("window_mean requires a tensor output and labeled TaskTarget for every snapshot")
        pairs.append(_select_target_supervision(value, target.label, target))
    return tuple(pairs)


def _required(value: Tensor | None, name: str) -> Tensor:
    if value is None:
        raise KeyError(name)
    return value


def _align_regression_target(pred: Tensor, label: Tensor) -> tuple[Tensor, Tensor]:
    label = label.to(device=pred.device, dtype=pred.dtype)
    if pred.shape == label.shape:
        return pred, label
    if int(pred.numel()) != int(label.numel()):
        raise ValueError(
            f"regression prediction and label shapes differ: {tuple(pred.shape)} != {tuple(label.shape)}"
        )
    return pred, label.reshape_as(pred)


def _binary_rank_metrics(pos: Tensor, neg: Tensor) -> dict[str, Tensor]:
    scores = torch.cat((pos, neg), dim=0)
    labels = torch.cat((torch.ones_like(pos), torch.zeros_like(neg)), dim=0)
    return _binary_label_metrics(scores, labels)


def _ranking_logits(pos: Tensor, neg: Tensor) -> Tensor:
    pos = pos.reshape(-1)
    neg = neg.reshape(-1)
    if int(pos.numel()) == 0:
        return pos.new_empty((0, 1 + int(neg.numel())))
    neg_samples = max(1, int(neg.numel()) // int(pos.numel()))
    neg = neg[: int(pos.numel()) * neg_samples].reshape(int(pos.numel()), neg_samples)
    return torch.cat((pos.reshape(-1, 1), neg), dim=1)


def _confusion_matrix(pred: Tensor, truth: Tensor, num_classes: int) -> Tensor:
    index = truth.reshape(-1).long() * int(num_classes) + pred.reshape(-1).long()
    return torch.bincount(index, minlength=int(num_classes) ** 2).reshape(num_classes, num_classes).float()


def _classification_metrics(confusion: Tensor) -> dict[str, Tensor]:
    true_positive = confusion.diagonal()
    count = confusion.sum()
    accuracy = true_positive.sum() / count.clamp_min(1.0)
    denominator = confusion.sum(dim=0) + confusion.sum(dim=1)
    valid = denominator > 0
    macro = (2 * true_positive[valid] / denominator[valid]).mean() if bool(valid.any().item()) else count * 0.0
    return {
        "accuracy": accuracy,
        "f1_micro": accuracy,
        "f1_macro": macro,
        "num_examples": count,
        "_classification_confusion": confusion,
    }


def _binary_label_metrics(scores: Tensor, labels: Tensor) -> dict[str, Tensor]:
    labels = (labels > 0.5).to(dtype=torch.float32, device=scores.device)
    scores = scores.to(dtype=torch.float32)
    return _torch_binary_label_metrics(scores=scores, labels=labels)


def _torch_binary_label_metrics(*, scores: Tensor, labels: Tensor) -> dict[str, Tensor]:
    labels = labels.reshape(-1).to(device=scores.device, dtype=torch.float32)
    scores = scores.reshape(-1).to(device=labels.device, dtype=torch.float32)
    pos = labels.sum()
    total = scores.new_tensor(int(scores.numel()), dtype=torch.float32)
    neg = total - pos
    if bool((pos <= 0).item() or (neg <= 0).item()):
        nan = scores.new_tensor(float("nan"))
        return {"ap": nan, "auc": nan}
    desc = torch.argsort(scores, descending=True)
    scores_desc = scores.index_select(0, desc)
    truth_desc = labels.index_select(0, desc)
    counts_desc = torch.unique_consecutive(scores_desc, return_counts=True)[1].long()
    group_end_desc = torch.cumsum(counts_desc, dim=0) - 1
    tps = torch.cumsum(truth_desc, dim=0).index_select(0, group_end_desc)
    total_seen = (group_end_desc + 1).to(dtype=torch.float32)
    precision = tps / total_seen
    recall = tps / pos
    recall_prev = torch.cat((recall.new_zeros((1,)), recall[:-1]), dim=0)
    ap = ((recall - recall_prev) * precision).sum()

    asc = torch.argsort(scores, descending=False)
    scores_asc = scores.index_select(0, asc)
    truth_asc = labels.index_select(0, asc)
    counts_asc = torch.unique_consecutive(scores_asc, return_counts=True)[1].long()
    group_end_asc = torch.cumsum(counts_asc, dim=0)
    group_start_asc = group_end_asc - counts_asc + 1
    avg_rank = (group_start_asc.to(dtype=torch.float32) + group_end_asc.to(dtype=torch.float32)) * 0.5
    group_pos = torch.cumsum(truth_asc, dim=0).index_select(0, group_end_asc - 1)
    group_pos_prev = torch.cat((group_pos.new_zeros((1,)), group_pos[:-1]), dim=0)
    pos_rank_sum = ((group_pos - group_pos_prev) * avg_rank).sum()
    auc = (pos_rank_sum - pos * (pos + 1.0) * 0.5) / (pos * neg)
    return {"ap": ap, "auc": auc}


__all__ = ["EdgePredictionTask", "NodePredictionTask"]
