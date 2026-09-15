"""Flare-style per-window supervision remains on the common runtime path."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import starrygl as sg
from starrygl.prepare.snapshot_csc import build_snapshot_csc_views
from starrygl.runtime import loop
from starrygl.runtime.builders import build_task_from_config
from starrygl.runtime.epoch import empty_supervision
from starrygl.runtime.snapshot.materialize import access_snapshot_window
from starrygl.store import FeatureManager, GraphStore, LabelStore, StoreBundle


def _store(*, all_empty=False):
    src, dst = torch.tensor([0, 1, 2, 3, 0]), torch.tensor([1, 2, 3, 0, 3])
    ptr = torch.arange(3)[:, None] * 5 + torch.tensor([0, 5])
    views = build_snapshot_csc_views(src=src.repeat(3), dst=dst.repeat(3), ts=torch.arange(3).repeat_interleave(5).float(),
        edge_ids=torch.arange(15), edge_dist_index=torch.arange(15), node_master=torch.zeros(4, dtype=torch.long),
        hot_node_ids=torch.empty(0, dtype=torch.long), node_is_hot=torch.zeros(4, dtype=torch.bool),
        node_to_chunk=torch.arange(4), time_ptr_2=ptr, num_nodes=4, world_size=1)
    count = 0 if all_empty else 8
    return StoreBundle(graph=GraphStore(num_nodes=4, prepare={
            "meta": {"world_size": 1}, "partition": {"node_dist_index": torch.arange(4)},
            "time_ptr_2": ptr, "snapshot_csc_views": views}),
        features=FeatureManager(node_features={"x": torch.arange(12).reshape(3, 4, 1).float() / 10}),
        labels=LabelStore(task_kind="node", task_ptr=torch.zeros(4, dtype=torch.long) if all_empty else torch.tensor([0, 4, 8, 8]),
            task_payload={"node_ids": torch.arange(4).repeat(2)[:count], "label": torch.arange(count).float() / 8}))


@pytest.mark.parametrize("mode", ["last_only", "window_mean"])
def test_public_task_option_defaults_and_lowers(mode):
    trainer = sg.compile(data_source={"source": "snapshots"}, backbone={"name": "tgcn"},
        task_segment={"name": "node_regression", "train_loss_mode": mode})
    task = build_task_from_config(trainer.task)
    assert task.train_loss_mode == mode
    assert build_task_from_config({"name": "node_regression"}).train_loss_mode == "last_only"
    assert sg.NodePredictionTask().train_loss_mode == "last_only"


@pytest.mark.parametrize("config", [
    {"name": "node_regression", "train_loss_mode": "unknown"},
    {"name": "edge_prediction", "train_loss_mode": "window_mean"},
])
def test_public_rejects_unsupported_loss_declaration(config):
    with pytest.raises(ValueError, match="train_loss_mode|snapshot node"):
        sg.compile(data_source={"source": "snapshots"}, backbone={"name": "tgcn"}, task_segment=config)


@pytest.mark.parametrize("mode,sampling", [("event", "full"), ("snapshot", "neighbor")])
def test_runtime_rejects_unsupported_window_targets_before_loader(mode, sampling):
    model = sg.TGCNModel(1, 2, 1)
    task = sg.NodePredictionTask(name="node_regression", loss="mse", train_loss_mode="window_mean")
    with patch.object(loop, "DataLoader", side_effect=AssertionError("unsupported path reached loader")):
        with pytest.raises(ValueError, match="window_mean requires"):
            loop.run_epoch(store=SimpleNamespace(), model=model, task=task, mode=mode,
                training=True, window_policy="full_snapshot", sampling_policy=sampling)


def test_accessor_produces_owner_targets_per_actual_partial_snapshot():
    store = _store()
    _, targets, blocks, _, _ = access_snapshot_window(store, store.graph.snapshot_csc_view["slices"],
        split="train", window_id=2, input_window=range(3), chunk_limits=(1, 2, -1),
        sampling_policy="full", sampler_options={"_snapshot_train_loss_mode": "window_mean"})
    windows = targets["window_tasks"]
    assert [item.target_ids.tolist() for item in windows] == [[0], [0, 1], []]
    assert targets["task"] is windows[-1]
    for target, (block,) in zip(windows, blocks):
        assert torch.all(torch.isin(target.target_ids, block.dst_nodes))
        torch.testing.assert_close(block.dst_nodes[target.target_route.target_rows], target.target_ids)
    torch.testing.assert_close(windows[0].label, torch.tensor([0.]))
    torch.testing.assert_close(windows[1].label, torch.tensor([0.5, 0.625]))
    assert not empty_supervision(sg.Batch(mode="snapshot", blocks=blocks, targets=targets))
    _, latest_only, _, _, _ = access_snapshot_window(store, store.graph.snapshot_csc_view["slices"],
        split="train", window_id=2, input_window=range(3), chunk_limits=(-1,) * 3, sampling_policy="full")
    assert "window_tasks" not in latest_only


def test_empty_windows_have_zero_loss_metrics_and_correct_gradients():
    task = sg.NodePredictionTask(name="node_regression", loss="mse", train_loss_mode="window_mean")
    target = sg.TaskTarget(target_kind="node", target_ids=torch.tensor([0]), label=torch.tensor([2.]))
    empty = sg.TaskTarget(target_kind="node", target_ids=torch.empty(0, dtype=torch.long), label=torch.empty(0))
    first, last = torch.tensor([[4.]], requires_grad=True), torch.empty(0, 1, requires_grad=True)
    graph = sg.graph_block_from_coo(src=torch.tensor([0]), dst=torch.tensor([0]),
        edge_ids=torch.tensor([0]), num_nodes=1, format="coo")
    batch = sg.Batch(mode="snapshot", graph=graph, targets={"task": empty, "window_tasks": (target, empty)})
    output = sg.ModelOutput(logits=last, aux={"window_logits": (first, last)})
    loss = task.compute_loss(output, batch)
    torch.testing.assert_close(loss, torch.tensor(2.))
    torch.testing.assert_close(task.compute_metrics(output, batch)["mse"], loss)
    loss.backward()
    torch.testing.assert_close(first.grad, torch.tensor([[2.]]))
    assert last.grad is not None and not empty_supervision(batch)
    batch.targets = {"task": empty, "window_tasks": (empty, empty)}
    output.aux["window_logits"] = (last, last)
    assert empty_supervision(batch)
    assert task.compute_loss(output, batch) == 0
    assert task.compute_metrics(output, batch)["mse"] == 0
    with pytest.raises(ValueError, match="window_mean requires model"):
        task.compute_loss(sg.ModelOutput(logits=last), batch)


@pytest.mark.parametrize("window_policy", ["full_snapshot", "chunk_decay"])
@pytest.mark.parametrize("model_cls", [sg.TGCNModel, sg.MPNNLSTMModel, sg.EvolveGCNModel,
                                       sg.GConvGRUModel, sg.DCRNNModel])
def test_train_pad_eval_carry_keeps_coupled_state_semantics(model_cls, window_policy):
    model = model_cls(1, 2, 1)
    coupled = model_cls in (sg.GConvGRUModel, sg.DCRNNModel)
    assert loop._batch_local_snapshot_state(model, mode="snapshot", training=True,
        window_policy=window_policy) == (not coupled)
    assert not loop._batch_local_snapshot_state(model, mode="snapshot", training=False,
        window_policy=window_policy)
    assert not loop._batch_local_snapshot_state(model, mode="event", training=True,
        window_policy=window_policy)


@pytest.mark.parametrize("loss_mode,expected_steps", [("last_only", 2), ("window_mean", 3)])
def test_common_run_epoch_optimizes_older_supervision_and_eval_is_last_only(loss_mode, expected_steps):
    torch.manual_seed(37)
    model = sg.TGCNModel(1, 2, 1)
    task = sg.NodePredictionTask(name="node_regression", loss="mse", train_loss_mode=loss_mode)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    store, observed = _store(), []
    options = dict(store=store, model=model, task=task, mode="snapshot", window_policy="full_snapshot",
        sampling_policy="full", num_full_snapshots=2, device="cpu", sampler_options={"access_pipeline": False})
    with patch.object(loop, "encode_model", wraps=loop.encode_model) as encode:
        result = loop.run_epoch(**options, training=True, optimizer=optimizer,
            batch_callback=lambda batch: observed.append(batch))
        assert all(call.kwargs["persist_state"] is False for call in encode.call_args_list)
    assert result.steps == expected_steps
    assert optimizer.state[model.output.weight]["step"] == expected_steps
    assert torch.isfinite(torch.tensor(result.loss))
    if loss_mode == "window_mean":
        assert observed[-1].targets["task"].target_ids.numel() == 0
        assert observed[-1].targets["window_tasks"][0].target_ids.numel() == 4
    evaluated = []
    loop.run_epoch(**options, training=False, batch_callback=lambda batch: evaluated.append(batch))
    assert all("window_tasks" not in batch.targets for batch in evaluated)
    model = sg.TGCNModel(1, 2, 1)
    empty_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    result = loop.run_epoch(**{**options, "store": _store(all_empty=True), "model": model},
        training=True, optimizer=empty_optimizer)
    assert result.steps == 0 and not empty_optimizer.state
