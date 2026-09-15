"""One dst join preserves actual partial-window supervision and its gradients."""
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

import starrygl as sg
from starrygl.runtime.snapshot.materialize import access_snapshot_window
from starrygl.store import LabelStore
from test_snapshot_window_supervision import _store


def _access(*, limits, all_empty=False, window_mean=True):
    store = _store()
    if all_empty:
        labels = LabelStore(task_kind="node", task_ptr=torch.zeros(4, dtype=torch.long),
            task_payload={"node_ids": torch.empty(0, dtype=torch.long), "label": torch.empty(0),
                          "cutoff_ts": torch.empty(0)})
    else:
        labels = LabelStore(task_kind="node", task_ptr=torch.tensor([0, 3, 7, 7]),
            task_payload={"node_ids": torch.tensor([0, 0, 3, 2, 1, 1, 3]),
                          "label": torch.tensor([10., 11., 12., 20., 21., 22., 23.]),
                          "cutoff_ts": torch.tensor([1., 1., 1., 2., 2., 2., 2.])})
    store = replace(store, labels=labels)
    return access_snapshot_window(store, store.graph.snapshot_csc_view["slices"], split="train",
        window_id=2, input_window=range(3), chunk_limits=limits, sampling_policy="full",
        sampler_options={"_snapshot_train_loss_mode": "window_mean" if window_mean else "last_only"})


def test_partial_accessor_preserves_duplicate_owner_targets_loss_and_gradient_without_membership_pass():
    with patch.object(torch, "isin", wraps=torch.isin) as membership:
        _, targets, blocks, _, _ = _access(limits=(1, 2, -1))
    windows = targets["window_tasks"]
    assert [target.target_ids.tolist() for target in windows] == [[0, 0], [1, 1], []]
    assert [target.target_route.target_rows.tolist() for target in windows] == [[0, 0], [1, 1], []]
    assert [target.label.tolist() for target in windows] == [[10., 11.], [21., 22.], []]
    assert [target.target_ts.tolist() for target in windows] == [[1., 1.], [2., 2.], []]
    for target, (block,) in zip(windows, blocks):
        torch.testing.assert_close(target.node_ids, target.target_ids)
        torch.testing.assert_close(block.dst_nodes[target.target_route.target_rows], target.target_ids)
    actual_values = tuple(torch.arange(block[0].num_dst, dtype=torch.float32).reshape(-1, 1).requires_grad_()
                          for block in blocks)
    expected_values = tuple(value.detach().clone().requires_grad_() for value in actual_values)
    batch = sg.Batch(mode="snapshot", blocks=blocks, targets=targets)
    output = sg.ModelOutput(logits=actual_values[-1], aux={"window_logits": actual_values})
    task = sg.NodePredictionTask(name="node_regression", loss="mse", train_loss_mode="window_mean")
    actual = task.compute_loss(output, batch)
    expected = ((expected_values[0][0] - torch.tensor([10., 11.])) ** 2).mean()
    expected = (expected + ((expected_values[1][1] - torch.tensor([21., 22.])) ** 2).mean()
                + expected_values[2].sum() * 0) / 3
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(task.compute_metrics(output, batch)["mse"], expected, rtol=0, atol=0)
    actual.backward(); expected.backward()
    for value, reference in zip(actual_values, expected_values):
        torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
    assert membership.call_count == 0, "window target projection must reuse the row join for membership"


@pytest.mark.parametrize("all_empty,limits", [(True, (1, 2, -1)), (False, (0, 0, 0))])
def test_empty_query_or_empty_dst_has_empty_targets_and_finite_window_loss(all_empty, limits):
    _, targets, blocks, _, _ = _access(limits=limits, all_empty=all_empty)
    for target in targets["window_tasks"]:
        assert target.target_ids.numel() == target.target_route.target_rows.numel() == 0
        assert target.label.numel() == target.target_ts.numel() == 0
    values = tuple(torch.zeros(block[0].num_dst, 1, requires_grad=True) for block in blocks)
    batch = sg.Batch(mode="snapshot", blocks=blocks, targets=targets)
    output = sg.ModelOutput(logits=values[-1], aux={"window_logits": values})
    task = sg.NodePredictionTask(name="node_regression", loss="mse")
    loss = task.compute_loss(output, batch)
    assert loss == task.compute_metrics(output, batch)["mse"] == 0
    loss.backward()
    assert all(value.grad is not None for value in values)


def test_last_only_keeps_existing_target_route():
    _, targets, _, _, _ = _access(limits=(1, 2, -1), window_mean=False)
    assert "window_tasks" not in targets
    assert targets["task"].target_ids.numel() == 0
    assert targets["task"].target_route.target_rows.numel() == 0
