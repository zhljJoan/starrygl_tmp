"""Globally empty windows keep backward/state communication but skip Adam."""
import copy
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

import starrygl as sg
from starrygl.runtime import loop
from starrygl.runtime.comm import CommScheduler
from starrygl.runtime.epoch import step_optimizer
from starrygl.store import LabelStore
from starrygl.task import StarryTask


def _target(active):
    ids = torch.tensor([0], dtype=torch.long) if active else torch.empty(0, dtype=torch.long)
    return sg.TaskTarget(target_kind="node", target_ids=ids)


class _BackwardReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, synchronized, events):
        ctx.synchronized, ctx.events = synchronized, events
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        ctx.events.append("backward")
        gradient = gradient.clone()
        if ctx.synchronized:
            dist.all_reduce(gradient)
            gradient /= dist.get_world_size()
        return gradient, None, None


class _Model(sg.StarryModel):
    def __init__(self, synchronized, events):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.unused = torch.nn.Parameter(torch.tensor(3.0))
        self.synchronized, self.events = synchronized, events

    def encode(self, batch):
        self.events.append("encode")
        values = _BackwardReduce.apply(self.weight, self.synchronized, self.events)
        return sg.ModelOutput(embeddings=values.reshape(1, 1))

    def state_update(self, batch, output):
        self.events.append("state_update")
        return sg.StateDelta(torch.tensor([0]), output.embeddings.detach())


class _Task(StarryTask):
    name = "edge_prediction"

    def supervision(self, batch):
        # Prepared counts alone would incorrectly classify this as nonempty.
        if batch.targets["window"] == 1:
            batch.targets["task"] = _target(False)
        return batch

    def compute_loss(self, output, batch):
        return output.embeddings.sum()


def _snapshot(optimizer, model):
    state = copy.deepcopy(optimizer.state_dict())
    for values in state["state"].values():
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                values[key] = value.detach().cpu().clone()
    return [p.detach().cpu().clone() for p in model.parameters()], state


def _assert_snapshot_equal(left, right):
    for before, after in zip(left[0], right[0]):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert left[1]["param_groups"] == right[1]["param_groups"]
    assert left[1]["state"].keys() == right[1]["state"].keys()
    for key in left[1]["state"]:
        for name in left[1]["state"][key]:
            torch.testing.assert_close(left[1]["state"][key][name], right[1]["state"][key][name], atol=0, rtol=0)


@pytest.mark.parametrize("sync_mode", ["all_reduce", None])
def test_loop_preserves_empty_backward_and_commits_without_empty_adam_step(monkeypatch, sync_mode):
    cuda = os.environ.get("STARRYGL_TEST_NCCL") == "1"
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    created = int(os.environ.get("WORLD_SIZE", "1")) == 2 and not dist.is_initialized()
    if created:
        dist.init_process_group("nccl" if cuda else "gloo")
    try:
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        synchronized = world > 1 and sync_mode is not None
        events, committed, names = [], [], []
        model = _Model(synchronized, events).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        graph = sg.graph_block_from_coo(src=torch.tensor([0]), dst=torch.tensor([0]),
                                        edge_ids=torch.tensor([0]), num_nodes=1, format="csc")
        batches = [sg.Batch(mode="snapshot", graph=graph,
                            state={"keep_state_execution": torch.zeros(1)},
                            targets={"task": _target(True), "window": step}) for step in range(3)]
        monkeypatch.setattr(loop, "DataLoader", lambda *args, **kwargs: batches)
        monkeypatch.setattr(loop, "poll_state_update", lambda *args: None)
        monkeypatch.setattr(loop, "finish_state_update", lambda *args, **kwargs: None)
        monkeypatch.setattr(loop, "submit_hydrate_state", lambda *args, **kwargs: ())
        monkeypatch.setattr(loop, "launch_state_update", lambda *args: committed.append(_snapshot(optimizer, model)))

        def callback(batch):
            # Final effective supervision can also change after task.supervision.
            if batch.targets["window"] == 2 and rank == 1:
                batch.targets["task"] = _target(False)

        comm = CommScheduler()
        original_reduce = comm.all_reduce

        def record_reduce(value, **kwargs):
            names.append(kwargs["name"])
            return original_reduce(value, **kwargs)

        monkeypatch.setattr(comm, "all_reduce", record_reduce)
        with patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reductions:
            result = loop.run_epoch(
                store=SimpleNamespace(graph=SimpleNamespace(rank=rank), labels=LabelStore()), model=model,
                task=_Task(), mode="snapshot", training=True, optimizer=optimizer,
                window_policy="full_snapshot", sampling_policy="full", device=device,
                state_manager=SimpleNamespace(comm=comm), comm=comm,
                gradient_sync=sync_mode, compute_metrics=False, batch_callback=callback,
            )
        assert result.steps == 2
        assert events == ["encode", "backward", "state_update"] * 3
        assert len(committed) == 3
        _assert_snapshot_equal(committed[0], committed[1])
        final_active = synchronized or rank == 0
        expected_steps = 2 if final_active else 1
        assert optimizer.state[model.weight]["step"].item() == expected_steps
        assert (not torch.equal(committed[1][0][0], committed[2][0][0])) == final_active

        reference = _Model(False, []).to(device)
        reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.001)
        gradients = [1.0] + ([1.0 / world if synchronized else 1.0] if final_active else [])
        for gradient in gradients:
            reference_optimizer.zero_grad(set_to_none=True)
            reference.weight.grad = torch.tensor(gradient, device=device)
            if synchronized:
                reference.unused.grad = torch.zeros_like(reference.unused)
            reference_optimizer.step()
        _assert_snapshot_equal(_snapshot(optimizer, model), _snapshot(reference_optimizer, reference))

        if synchronized:
            # Per window: existing autograd reduce, two existing gradient reduces,
            # then the single new flag reduce. No old collective is skipped.
            assert [call.args[0].dtype for call in reductions.call_args_list] == [
                torch.float32, torch.float32, torch.float32, torch.int64,
            ] * 3
            assert names == ["optimizer_supervision"] * 3
            gathered = [None] * world
            dist.all_gather_object(gathered, _snapshot(optimizer, model))
            _assert_snapshot_equal(gathered[0], gathered[1])
        else:
            assert reductions.call_count == 0 and not names
    finally:
        if created:
            dist.destroy_process_group()


def test_zero_gradient_adam_would_move_but_explicit_empty_step_does_not():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model(torch.ones(1, 1)).sum().backward()
    step_optimizer(model, optimizer, None, CommScheduler(), has_supervision=True)
    before = _snapshot(optimizer, model)
    optimizer.zero_grad(set_to_none=True)
    (model(torch.ones(1, 1)).sum() * 0).backward()
    step_optimizer(model, optimizer, None, CommScheduler(), has_supervision=False)
    _assert_snapshot_equal(before, _snapshot(optimizer, model))
    optimizer.step()
    assert not torch.equal(before[0][0], model.weight)
    assert optimizer.state[model.weight]["step"].item() == 2


@pytest.mark.parametrize(
    ("task", "mode", "window_policy", "sampling_policy"),
    [
        (sg.NodePredictionTask(name="node_regression", loss="mse"),
         "snapshot", "full_snapshot", "full"),
        (sg.EdgePredictionTask(loss="bce"),
         "event", "event_window", "neighbor"),
    ],
)
def test_prepared_activity_uses_one_cached_collective(
    monkeypatch, task, mode, window_policy, sampling_policy,
):
    store = SimpleNamespace(
        graph=SimpleNamespace(runtime_cache={}),
        labels=LabelStore(
            task_kind="node",
            task_ptr=torch.tensor([0, 0, 2, 2]),
            task_payload={"node_ids": torch.tensor([1, 2])},
        ),
    )
    batches = SimpleNamespace(
        window_ids=range(3), skip=0, maximum=0, split="train",
    )
    comm = CommScheduler()
    calls = []

    def reduce_remote_activity(active, **kwargs):
        calls.append(kwargs["name"])
        active[2] = 1
        return active

    monkeypatch.setattr(loop, "dist_world_size", lambda: 2)
    monkeypatch.setattr(comm, "all_reduce", reduce_remote_activity)
    args = dict(
        task=task, batches=batches, mode=mode,
        window_policy=window_policy, sampling_policy=sampling_policy,
        batch_callback=None, comm=comm, device="cpu",
    )
    assert loop._prepared_supervision_schedule(store, **args) == (False, True, True)
    assert loop._prepared_supervision_schedule(store, **args) == (False, True, True)
    assert calls == ["prepared_supervision"]
    assert loop._prepared_supervision_schedule(
        store, **{**args, "batch_callback": lambda batch: batch}
    ) is None
