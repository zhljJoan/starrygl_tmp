"""Prepared labels follow accessor placement without changing owner targets."""
import os

import pytest
import torch
import starrygl as sg

from starrygl.batch import EventRows
from starrygl.runtime.event.materialize import _event_sampling_roots
from starrygl.runtime.loop import run_epoch
from starrygl.store import LabelStore
from starrygl.task import build_window_task_target
from test_snapshot_device_materialization import _store


@pytest.mark.parametrize("window,expected", [(0, [2, 0, 2]), (1, []), (2, [1, 3])])
def test_task_slice_preserves_duplicates_empty_windows_and_scalar_cutoff(window, expected):
    labels = LabelStore(task_kind="node", task_ptr=torch.tensor([0, 3, 3, 5]),
        task_payload={"node_ids": torch.tensor([2, 0, 2, 1, 3]), "label": torch.arange(5).float()})
    target = build_window_task_target(labels, window, target_ts=torch.tensor(7.))
    assert target.target_ids.tolist() == expected
    assert target.node_ids is target.target_ids
    assert target.target_ts.device == target.node_ids.device
    torch.testing.assert_close(target.target_ts, torch.full((len(expected),), 7.))
    assert labels.task_ptr.device.type == "cpu"
    assert target.label.untyped_storage().data_ptr() == labels.task_payload["label"].untyped_storage().data_ptr()


def _evaluate(store, model, *, device, placement):
    outputs = []
    result = run_epoch(store=store, model=model.to(device),
        task=sg.NodePredictionTask(name="node_regression", loss="mse"),
        mode="snapshot", training=False, window_policy="full_snapshot", sampling_policy="full",
        num_full_snapshots=1, num_layers=2, device=device,
        sampler_options={"snapshot_materialize_on_device": placement, "snapshot_dgl_gcn": True,
                         "access_pipeline": False},
        output_callback=lambda output: outputs.append(output.logits.detach().cpu().clone()))
    return result, outputs


def test_cpu_epoch_placement_keeps_payload_storage_and_evaluation_values():
    torch.manual_seed(17)
    store, model = _store(), sg.TGCNModel(1, 3, 1)
    original = dict(store.labels.task_payload)
    before = _evaluate(store, model, device="cpu", placement=False)
    for placement in (True, False):
        after = _evaluate(store, model, device="cpu", placement=placement)
        assert before[0].loss == after[0].loss and before[0].steps == after[0].steps
        for left, right in zip(before[1], after[1]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert all(store.labels.task_payload[name] is value for name, value in original.items())
        assert store.labels.task_ptr.device.type == "cpu"


@pytest.mark.skipif(os.environ.get("STARRYGL_TEST_CUDA_MATERIALIZE") != "1", reason="explicit CUDA opt-in required")
def test_epoch_rebinds_payload_on_device_and_placement_switches():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(17)
    store, model = _store(), sg.TGCNModel(1, 3, 1)
    original = {name: value.clone() for name, value in store.labels.task_payload.items()}
    before = _evaluate(store, model, device="cpu", placement=False)
    previous_device, previous_payload = "cpu", dict(store.labels.task_payload)
    for device, placement in (("cuda:0", True), ("cuda:0", True), ("cuda:0", False),
                              ("cuda:0", True), ("cpu", False), ("cpu", True)):
        after = _evaluate(store, model, device=device, placement=placement)
        expected_device = device if placement else "cpu"
        assert all(value.device == torch.device(expected_device) for value in store.labels.task_payload.values())
        if expected_device == previous_device:
            assert all(store.labels.task_payload[name] is value for name, value in previous_payload.items())
        previous_device, previous_payload = expected_device, dict(store.labels.task_payload)
        assert store.labels.task_ptr.device.type == "cpu"
        for name, value in store.labels.task_payload.items():
            torch.testing.assert_close(value.cpu(), original[name], rtol=0, atol=0)
        target = build_window_task_target(store.labels, 0, target_ts=torch.tensor(7.))
        assert target.target_ts.device == target.node_ids.device
        torch.testing.assert_close(target.target_ts.cpu(), torch.full((target.node_ids.numel(),), 7.))
        if expected_device == "cpu":
            events = EventRows(src=torch.tensor([0]), dst=torch.tensor([1]),
                               edge_ids=torch.tensor([0]), ts=torch.tensor([1.]))
            roots = _event_sampling_roots(target, events)
            assert roots.node_ids.device.type == roots.ts.device.type == "cpu"
        assert after[0].steps == before[0].steps
        assert after[0].loss == pytest.approx(before[0].loss, rel=1e-5, abs=1e-6)
        for left, right in zip(before[1], after[1]):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
