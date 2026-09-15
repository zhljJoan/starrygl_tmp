import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

import starrygl as sg


def _batch(targets):
    graph = sg.graph_block_from_coo(
        src=torch.tensor([0]),
        dst=torch.tensor([1]),
        edge_ids=torch.tensor([0]),
        num_nodes=2,
        format="csr",
    )
    return sg.Batch(mode="snapshot", graph=graph, targets=targets)


def test_node_prediction_task_uses_node_master_owner_and_cross_entropy() -> None:
    task = sg.NodePredictionTask()
    output = sg.ModelOutput(
        embeddings=torch.empty(2, 2),
        logits=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
    )
    batch = _batch({"label": torch.tensor([0, 1])})

    loss = task.compute_loss(output, batch)
    metrics = task.compute_metrics(output, batch)

    assert task.target_owner == "node_master"
    assert task.output_owner == "node_master"
    assert torch.allclose(loss, F.cross_entropy(output.logits, batch.targets["label"]))
    assert torch.equal(metrics["accuracy"], torch.tensor(1.0))


def test_node_prediction_task_supports_mse() -> None:
    task = sg.NodePredictionTask(loss="mse")
    output = sg.ModelOutput(
        embeddings=torch.tensor([[1.0], [3.0]]),
        predictions=torch.tensor([[1.0], [3.0]]),
    )
    batch = _batch({"label": torch.tensor([[2.0], [1.0]])})

    loss = task.compute_loss(output, batch)

    assert torch.allclose(loss, torch.tensor(2.5))


def test_node_regression_reshapes_scalar_labels_without_broadcasting() -> None:
    task = sg.NodePredictionTask(name="node_regression", loss="mse")
    output = sg.ModelOutput(
        embeddings=torch.tensor([[1.0], [3.0]]),
        predictions=torch.tensor([[1.0], [3.0]]),
    )
    batch = _batch({"label": torch.tensor([2.0, 1.0])})

    loss = task.compute_loss(output, batch)
    metrics = task.compute_metrics(output, batch)

    assert torch.allclose(loss, torch.tensor(2.5))
    assert torch.allclose(metrics["mse"], torch.tensor(2.5))


def test_snapshot_node_regression_uses_flare_window_mean() -> None:
    task = sg.NodePredictionTask(name="node_regression", loss="mse")
    first = sg.TaskTarget(
        target_kind="node",
        target_ids=torch.tensor([0, 1]),
        label=torch.tensor([2.0, 1.0]),
    )
    second = sg.TaskTarget(
        target_kind="node",
        target_ids=torch.tensor([0]),
        label=torch.tensor([2.0]),
    )
    output = sg.ModelOutput(
        logits=torch.tensor([[0.0]]),
        aux={"window_logits": (torch.tensor([[1.0], [3.0]]), torch.tensor([[0.0]]))},
    )
    batch = _batch({"task": second, "window_tasks": (first, second)})

    loss = task.compute_loss(output, batch)
    metrics = task.compute_metrics(output, batch)

    assert torch.allclose(loss, torch.tensor(3.25))
    assert torch.allclose(metrics["mse"], torch.tensor(3.25))


def test_node_prediction_task_expands_duplicate_root_labels_via_root_lids() -> None:
    task = sg.NodePredictionTask()
    output = sg.ModelOutput(
        embeddings=torch.empty(2, 2),
        logits=torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
    )
    batch = _batch(
        {
            "label": torch.tensor([0, 1, 0]),
            "root_lids": torch.tensor([0, 1, 0]),
        }
    )

    loss = task.compute_loss(output, batch)
    metrics = task.compute_metrics(output, batch)
    expanded = output.logits.index_select(0, batch.targets["root_lids"])

    assert torch.allclose(loss, F.cross_entropy(expanded, batch.targets["label"]))
    assert torch.equal(metrics["accuracy"], torch.tensor(1.0))


def test_node_prediction_task_accepts_one_hot_labels() -> None:
    task = sg.NodePredictionTask()
    output = sg.ModelOutput(
        embeddings=torch.empty(2, 2),
        logits=torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
    )
    label = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    batch = _batch({"label": label})

    loss = task.compute_loss(output, batch)
    metrics = task.compute_metrics(output, batch)

    assert torch.allclose(loss, F.cross_entropy(output.logits, label))
    assert torch.equal(metrics["accuracy"], torch.tensor(1.0))


def test_node_classification_aggregates_macro_f1_from_global_confusion() -> None:
    from starrygl.runtime.epoch import accumulate, result

    task = sg.NodePredictionTask(name="node_classification")
    batches = (
        (
            sg.ModelOutput(logits=torch.tensor([[3.0, 0.0], [3.0, 0.0]])),
            _batch({"label": torch.tensor([0, 0])}),
        ),
        (
            sg.ModelOutput(logits=torch.tensor([[3.0, 0.0], [0.0, 3.0]])),
            _batch({"label": torch.tensor([1, 1])}),
        ),
    )
    sums = {}
    for output, batch in batches:
        accumulate(sums, task.compute_metrics(output, batch))
    epoch = result(total_loss=torch.tensor(0.0), total_steps=2, metric_sums=sums)

    assert abs(epoch.metrics["accuracy"] - 0.75) < 1e-6
    assert abs(epoch.metrics["f1_micro"] - 0.75) < 1e-6
    assert abs(epoch.metrics["f1_macro"] - (11.0 / 15.0)) < 1e-6


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two torchrun ranks")
def test_epoch_metrics_aggregate_when_one_rank_has_no_targets() -> None:
    from starrygl.runtime.epoch import accumulate, result

    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("gloo")
    try:
        sums = {}
        if dist.get_rank() == 0:
            task = sg.NodePredictionTask(name="node_classification")
            output = sg.ModelOutput(logits=torch.tensor([[3.0, 0.0], [0.0, 3.0]]))
            accumulate(sums, task.compute_metrics(output, _batch({"label": torch.tensor([0, 1])})))
        epoch = result(
            total_loss=torch.tensor(float(dist.get_rank() == 0)),
            total_steps=int(dist.get_rank() == 0),
            metric_sums=sums,
        )

        assert epoch.loss == 1.0
        assert epoch.steps == 1
        assert epoch.metrics["accuracy"] == 1.0
        assert epoch.metrics["f1_macro"] == 1.0

        edge = result(
            total_loss=torch.tensor(0.0),
            total_steps=1,
            metric_sums={"ap": torch.tensor(float(dist.get_rank() + 1))},
        )
        assert edge.metrics["ap"] == 1.5
    finally:
        if created_group and dist.is_initialized():
            dist.destroy_process_group()


def test_node_prediction_task_reads_task_target_label() -> None:
    task = sg.NodePredictionTask()
    output = sg.ModelOutput(
        embeddings=torch.empty(2, 2),
        logits=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
    )
    target = sg.TaskTarget(
        target_kind="node",
        target_ids=torch.tensor([0, 1]),
        label=torch.tensor([0, 1]),
    )
    batch = _batch({"task": target})

    loss = task.compute_loss(output, batch)

    assert torch.allclose(loss, F.cross_entropy(output.logits, target.label))


def test_edge_prediction_task_consumes_negative_loss_weight() -> None:
    task = sg.EdgePredictionTask()
    output = sg.ModelOutput(
        embeddings=torch.empty(1, 1),
        aux={
            "pos_score": torch.tensor([2.0]),
            "neg_score": torch.tensor([0.0, 0.0]),
        },
    )
    batch = _batch({"neg_weight": torch.tensor([1.0, 3.0])})

    loss = task.compute_loss(output, batch)
    expected_pos = F.binary_cross_entropy_with_logits(
        torch.tensor([2.0]),
        torch.ones(1),
    )
    expected_neg = F.binary_cross_entropy_with_logits(
        torch.tensor([0.0, 0.0]),
        torch.zeros(2),
        weight=torch.tensor([1.0, 3.0]),
    )
    expected = expected_pos + expected_neg

    assert task.target_owner == "edge_master"
    assert task.output_owner == "edge_master"
    assert torch.allclose(loss, expected)


def test_edge_prediction_task_reads_task_target_negative_weight() -> None:
    task = sg.EdgePredictionTask()
    output = sg.ModelOutput(
        embeddings=torch.empty(1, 1),
        aux={
            "pos_score": torch.tensor([2.0]),
            "neg_score": torch.tensor([0.0, 0.0]),
        },
    )
    target = sg.TaskTarget(
        target_kind="edge",
        target_ids=torch.tensor([0]),
        neg_loss_weight=torch.tensor([2.0, 4.0]),
    )
    batch = _batch({"task": target})

    loss = task.compute_loss(output, batch)
    expected = F.binary_cross_entropy_with_logits(torch.tensor([2.0]), torch.ones(1)) + F.binary_cross_entropy_with_logits(
        torch.tensor([0.0, 0.0]),
        torch.zeros(2),
        weight=target.neg_loss_weight,
    )

    assert torch.allclose(loss, expected)


def test_edge_prediction_task_supports_memshare_softmax_loss() -> None:
    task = sg.EdgePredictionTask(loss="softmax")
    output = sg.ModelOutput(
        aux={
            "pos_score": torch.tensor([2.0, 1.0]),
            "neg_score": torch.tensor([0.5, -1.0]),
        }
    )
    batch = _batch({})

    loss = task.compute_loss(output, batch)
    expected = F.cross_entropy(torch.tensor([[2.0, 0.5], [1.0, -1.0]]), torch.zeros(2, dtype=torch.long))

    assert torch.allclose(loss, expected)


def test_softmax_loss_keeps_target_major_negative_order() -> None:
    from starrygl.task.prediction import _ranking_logits

    logits = _ranking_logits(torch.tensor([10.0, 20.0]), torch.tensor([1.0, 2.0, 3.0, 4.0]))

    assert torch.equal(logits, torch.tensor([[10.0, 1.0, 2.0], [20.0, 3.0, 4.0]]))


def test_materialize_negative_samples_uses_local_dst_pool_and_weight() -> None:
    target = sg.TaskTarget(
        target_kind="edge",
        target_ids=torch.tensor([0, 1]),
        pos_src=torch.tensor([10, 11]),
        pos_dst=torch.tensor([2, 3]),
        negative_pool=sg.NegativeSamplePool(
            mode="dst",
            local_node_ids=torch.tensor([4, 5]),
            local_loss_weight=2.5,
        ),
    )
    generator = torch.Generator().manual_seed(1)

    out = sg.materialize_negative_samples(target, num_negatives=2, generator=generator)

    assert out.neg_src is None
    assert out.neg_dst is not None
    assert torch.isin(out.neg_dst, torch.tensor([4, 5])).all()
    assert out.neg_loss_weight is not None
    assert torch.equal(out.neg_loss_weight, torch.full((4,), 2.5))


def test_materialize_negative_samples_uses_global_dst_pool_when_local_disabled() -> None:
    target = sg.TaskTarget(
        target_kind="edge",
        target_ids=torch.tensor([0]),
        pos_src=torch.tensor([7]),
        pos_dst=torch.tensor([2]),
        negative_pool=sg.NegativeSamplePool(
            mode="dst",
            local_node_ids=torch.tensor([4]),
            global_node_ids=torch.tensor([8, 9]),
            local_prob=0.0,
            global_prob=1.0,
            global_loss_weight=3.0,
        ),
    )

    out = sg.materialize_negative_samples(target, num_negatives=3, generator=torch.Generator().manual_seed(2))

    assert out.neg_dst is not None
    assert torch.isin(out.neg_dst, torch.tensor([8, 9])).all()
    assert out.neg_loss_weight is not None
    assert torch.equal(out.neg_loss_weight, torch.full((3,), 3.0))


def test_materialize_negative_samples_supports_src_dst_mode() -> None:
    target = sg.TaskTarget(
        target_kind="edge",
        target_ids=torch.tensor([0, 1]),
        pos_src=torch.tensor([10, 11]),
        pos_dst=torch.tensor([2, 3]),
        negative_pool=sg.NegativeSamplePool(
            mode="src_dst",
            local_src_ids=torch.tensor([6, 7]),
            local_dst_ids=torch.tensor([8, 9]),
        ),
    )

    out = sg.materialize_negative_samples(target, num_negatives=2, generator=torch.Generator().manual_seed(4))

    assert out.neg_src is not None and out.neg_dst is not None
    assert out.neg_src.shape == out.neg_dst.shape == (4,)
    assert torch.isin(out.neg_src, torch.tensor([6, 7])).all()
    assert torch.isin(out.neg_dst, torch.tensor([8, 9])).all()


def test_materialize_negative_samples_mixes_local_and_global_weights() -> None:
    target = sg.TaskTarget(
        target_kind="edge",
        target_ids=torch.tensor([0, 1]),
        pos_src=torch.tensor([1, 2]),
        pos_dst=torch.tensor([3, 4]),
        negative_pool=sg.NegativeSamplePool(
            mode="dst",
            local_node_ids=torch.tensor([5]),
            global_node_ids=torch.tensor([9]),
            local_prob=0.5,
            global_prob=0.5,
            local_loss_weight=1.0,
            global_loss_weight=4.0,
        ),
    )

    out = sg.materialize_negative_samples(target, num_negatives=4, generator=torch.Generator().manual_seed(4))

    assert out.neg_dst is not None
    assert out.neg_loss_weight is not None
    assert torch.isin(out.neg_dst, torch.tensor([5, 9])).all()
    assert torch.equal(out.neg_loss_weight[out.neg_dst == 5], torch.ones_like(out.neg_loss_weight[out.neg_dst == 5]))
    assert torch.equal(out.neg_loss_weight[out.neg_dst == 9], torch.full_like(out.neg_loss_weight[out.neg_dst == 9], 4.0))


def test_materialize_negative_samples_weights_by_sampling_branch() -> None:
    target = sg.TaskTarget(
        target_kind="edge",
        target_ids=torch.tensor([0, 1]),
        pos_src=torch.tensor([10, 11]),
        pos_dst=torch.tensor([2, 3]),
        negative_pool=sg.NegativeSamplePool(
            mode="dst",
            local_dst_ids=torch.tensor([5, 6]),
            global_dst_ids=torch.tensor([5, 6, 7, 8]),
            local_prob=0.0,
            global_prob=1.0,
            local_loss_weight=2.0,
            global_loss_weight=5.0,
        ),
    )

    out = sg.materialize_negative_samples(target, num_negatives=20, generator=torch.Generator().manual_seed(3))

    assert out.neg_dst is not None
    assert out.neg_loss_weight is not None
    assert bool(torch.isin(out.neg_dst, torch.tensor([5, 6])).any().item())
    assert torch.equal(out.neg_loss_weight, torch.full_like(out.neg_loss_weight, 5.0))


def test_edge_prediction_task_reports_ap_auc_for_pos_neg_scores() -> None:
    task = sg.EdgePredictionTask()
    output = sg.ModelOutput(
        embeddings=torch.empty(1, 1),
        aux={
            "pos_score": torch.tensor([3.0, 2.0]),
            "neg_score": torch.tensor([1.0, 0.0]),
        },
    )
    batch = _batch({})

    metrics = task.compute_metrics(output, batch)

    assert torch.allclose(metrics["margin"], torch.tensor(2.0))
    assert torch.allclose(metrics["ap"], torch.tensor(1.0))
    assert torch.allclose(metrics["auc"], torch.tensor(1.0))


def test_edge_prediction_task_matches_memshare_sklearn_ap_auc() -> None:
    task = sg.EdgePredictionTask()
    pos = torch.tensor([0.2, 0.7, 0.5])
    neg = torch.tensor([0.6, 0.1, 0.4])
    output = sg.ModelOutput(
        embeddings=torch.empty(1, 1),
        aux={
            "pos_score": pos,
            "neg_score": neg,
        },
    )
    y_pred = torch.cat((pos.view(-1), neg.view(-1))).numpy()
    y_true = torch.cat((torch.ones(pos.numel()), torch.zeros(neg.numel()))).numpy()

    metrics = task.compute_metrics(output, _batch({}))

    assert torch.allclose(metrics["ap"], torch.tensor(average_precision_score(y_true, y_pred), dtype=metrics["ap"].dtype))
    assert torch.allclose(metrics["auc"], torch.tensor(roc_auc_score(y_true, y_pred), dtype=metrics["auc"].dtype))


def test_edge_prediction_torch_rank_metrics_match_sklearn_with_ties() -> None:
    from starrygl.task.prediction import _torch_binary_label_metrics

    scores = torch.tensor([0.2, 0.7, 0.5, 0.6, 0.1, 0.4, 0.4, 0.6])
    labels = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    metrics = _torch_binary_label_metrics(scores=scores, labels=labels)

    assert torch.allclose(metrics["ap"], torch.tensor(average_precision_score(labels.numpy(), scores.numpy()), dtype=metrics["ap"].dtype))
    assert torch.allclose(metrics["auc"], torch.tensor(roc_auc_score(labels.numpy(), scores.numpy()), dtype=metrics["auc"].dtype))


def test_edge_prediction_torch_rank_metrics_match_sklearn_without_ties() -> None:
    from starrygl.task.prediction import _torch_binary_label_metrics

    scores = torch.tensor([0.2, 0.7, 0.5, 0.6, 0.1, 0.3])
    labels = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    metrics = _torch_binary_label_metrics(scores=scores, labels=labels)

    assert torch.allclose(metrics["ap"], torch.tensor(average_precision_score(labels.numpy(), scores.numpy()), dtype=metrics["ap"].dtype))
    assert torch.allclose(metrics["auc"], torch.tensor(roc_auc_score(labels.numpy(), scores.numpy()), dtype=metrics["auc"].dtype))


def test_edge_prediction_task_supports_binary_logits() -> None:
    task = sg.EdgePredictionTask()
    output = sg.ModelOutput(
        embeddings=torch.empty(3),
        logits=torch.tensor([2.0, -2.0, 0.0]),
    )
    batch = _batch(
        {
            "label": torch.tensor([1.0, 0.0, 1.0]),
            "weight": torch.tensor([1.0, 1.0, 2.0]),
        }
    )

    loss = task.compute_loss(output, batch)
    metrics = task.compute_metrics(output, batch)
    expected = (
        F.binary_cross_entropy_with_logits(
            output.logits,
            batch.targets["label"],
            reduction="none",
        )
        * batch.targets["weight"]
    ).mean()

    assert torch.allclose(loss, expected)
    assert torch.allclose(metrics["accuracy"], torch.tensor(1.0))
    assert torch.allclose(metrics["ap"], torch.tensor(1.0))
    assert torch.allclose(metrics["auc"], torch.tensor(1.0))

