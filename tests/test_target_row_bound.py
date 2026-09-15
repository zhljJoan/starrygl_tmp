"""Accessor range proofs remove only redundant task-side row validation."""
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

import starrygl as sg
from starrygl.prepare.snapshot_csc import build_snapshot_csc_views
from starrygl.runtime.dataloader.pipeline import _move_value
from starrygl.runtime.snapshot.materialize import access_snapshot_window
from starrygl.store import FeatureManager, GraphStore, LabelStore, StoreBundle
from starrygl.task.prediction import _select_target_supervision


def _graph():
    return sg.graph_block_from_coo(src=torch.tensor([0]), dst=torch.tensor([1]),
                                  edge_ids=torch.tensor([0]), num_nodes=4, format="coo")


@pytest.mark.parametrize("loss_name", ["mse", "cross_entropy"])
def test_bound_preserves_duplicate_loss_metrics_and_gradients_without_item(loss_name):
    rows = torch.tensor([2, 0, 2])
    label = torch.tensor([1., 4., 3.]) if loss_name == "mse" else torch.tensor([0, 1, 1])
    target = sg.TaskTarget(target_kind="node", target_ids=rows, label=label,
                           target_route=sg.TargetRoute(target_rows=rows, target_row_bound=4))
    generic = replace(target, target_route=replace(target.target_route, target_row_bound=None))
    task = sg.NodePredictionTask(loss=loss_name)
    width = 1 if loss_name == "mse" else 2
    original = torch.arange(4 * width, dtype=torch.float64).reshape(4, width) / 5
    expected_value = original.clone().requires_grad_()
    actual_value = original.clone().requires_grad_()

    def evaluate(value, item):
        batch = sg.Batch(mode="snapshot", graph=_graph(), targets={"task": item})
        output = sg.ModelOutput(logits=value)
        return task.compute_loss(output, batch), task.compute_metrics(output, batch)

    expected_loss, expected_metrics = evaluate(expected_value, generic)
    expected_loss.backward()
    # Classification metric reporting has its own unrelated scalar checks.
    with patch.object(torch.Tensor, "item", side_effect=AssertionError("revalidated proven rows")):
        batch = sg.Batch(mode="snapshot", graph=_graph(), targets={"task": target})
        actual_loss = task.compute_loss(sg.ModelOutput(logits=actual_value), batch)
        actual_loss.backward()
        actual_metrics = task.compute_metrics(sg.ModelOutput(logits=actual_value), batch) if loss_name == "mse" else None
    if actual_metrics is None:
        actual_metrics = task.compute_metrics(sg.ModelOutput(logits=actual_value), batch)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(actual_value.grad, expected_value.grad, rtol=0, atol=0)
    for name in expected_metrics:
        torch.testing.assert_close(actual_metrics[name], expected_metrics[name], rtol=0, atol=0)


@pytest.mark.parametrize("source", ["custom", "mismatched_bound", "root_lids", "empty_route_root_lids"])
def test_unknown_mismatched_and_root_routes_keep_original_invalid_filter(source):
    rows = torch.tensor([-1, 2, 0, 2, 4])
    labels = torch.tensor([-9., 0., 1., 3., -9.])
    route = sg.TargetRoute(target_rows=rows)
    if source == "mismatched_bound":
        rows, labels = rows[1:], labels[1:]
        route = sg.TargetRoute(target_rows=rows, target_row_bound=5)
    if source == "empty_route_root_lids":
        route = sg.TargetRoute(target_row_bound=4)
    target = None if source == "root_lids" else sg.TaskTarget(
        target_kind="node", target_ids=rows, label=labels, target_route=route)
    batch = sg.Batch(mode="event", graph=_graph(), targets={"root_lids": rows})
    value = torch.arange(4.).reshape(4, 1)
    selected, actual_labels = _select_target_supervision(value, labels, target, batch=batch)
    torch.testing.assert_close(selected, value[torch.tensor([2, 0, 2])])
    torch.testing.assert_close(actual_labels, torch.tensor([0., 1., 3.]))


def _owner_store():
    src, dst = torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 0])
    ptr = torch.arange(3)[:, None] * 4 + torch.tensor([0, 4])
    views = build_snapshot_csc_views(
        src=src.repeat(3), dst=dst.repeat(3), ts=torch.arange(3).repeat_interleave(4).float(),
        edge_ids=torch.arange(12), edge_dist_index=torch.arange(12),
        node_master=torch.zeros(4, dtype=torch.long), hot_node_ids=torch.empty(0, dtype=torch.long),
        node_is_hot=torch.zeros(4, dtype=torch.bool), node_to_chunk=torch.arange(4),
        time_ptr_2=ptr, num_nodes=4, world_size=1)
    return StoreBundle(
        graph=GraphStore(num_nodes=4, prepare={"meta": {"world_size": 1},
            "partition": {"node_dist_index": torch.arange(4)}, "snapshot_csc_views": views}),
        features=FeatureManager(), labels=LabelStore(task_kind="node",
            task_ptr=torch.tensor([0, 5, 10, 15]), task_payload={
                "node_ids": torch.tensor([3, 1, 3, 0, 2]).repeat(3),
                "label": torch.arange(15).float() / 4}))


def test_accessor_proves_only_filtered_full_partial_and_empty_owner_routes():
    store = _owner_store()
    args = dict(split="train", window_id=2, input_window=range(3), chunk_limits=(2, -1, 0),
                sampling_policy="full", sampler_options={"_snapshot_train_loss_mode": "window_mean"})
    _, targets, blocks, _, _ = access_snapshot_window(store, store.graph.snapshot_csc_view["slices"], **args)
    windows = targets["window_tasks"]
    assert [item.target_ids.tolist() for item in windows] == [[1, 0], [3, 1, 3, 0, 2], []]
    for item, (block,) in zip(windows, blocks):
        assert item.target_route.target_row_bound == block.num_dst
        torch.testing.assert_close(block.dst_nodes[item.target_route.target_rows], item.target_ids)
        assert _move_value(item, torch.device("cpu")).target_route.target_row_bound == block.num_dst
    task = sg.NodePredictionTask(loss="mse", train_loss_mode="window_mean")
    reference_windows = tuple(replace(item, target_route=replace(item.target_route, target_row_bound=None))
                              for item in windows)
    values = tuple(torch.arange(block.num_dst, dtype=torch.float64).reshape(-1, 1).requires_grad_()
                   for (block,) in blocks)
    references = tuple(value.detach().clone().requires_grad_() for value in values)
    expected_batch = sg.Batch(mode="snapshot", blocks=blocks,
        targets={"window_tasks": reference_windows, "task": reference_windows[-1]})
    expected = task.compute_loss(sg.ModelOutput(aux={"window_logits": references}), expected_batch)
    expected.backward()
    actual_batch = sg.Batch(mode="snapshot", blocks=blocks, targets=targets)
    output = sg.ModelOutput(aux={"window_logits": values})
    with patch.object(torch.Tensor, "item", side_effect=AssertionError("revalidated proven owner rows")):
        actual = task.compute_loss(output, actual_batch)
        metrics = task.compute_metrics(output, actual_batch)
        actual.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(metrics["mse"], expected, rtol=0, atol=0)
    for value, reference in zip(values, references):
        torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
    _, generic, _, _, _ = access_snapshot_window(store, store.graph.snapshot_csc_view["slices"],
        **{**args, "chunk_limits": (-1,) * 3, "sampler_options": {}})
    assert generic["task"].target_route.target_row_bound is None
