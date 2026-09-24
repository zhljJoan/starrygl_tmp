from __future__ import annotations

import os
from threading import Event, get_ident
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from starrygl.batch import Batch
from starrygl.runtime.comm import Route
from starrygl.runtime.exchange import PendingNodeFeatureFetch
from starrygl.model import ModelOutput, StarryModel
import starrygl.runtime.dataloader.loader as runtime_batches
import starrygl.runtime.dataloader.pipeline as runtime_pipeline
import starrygl.runtime.loop as runtime_loop
from starrygl.store import LabelStore
from starrygl.view import GraphBlock


class _RecordingModel(StarryModel):
    def __init__(self, events: list[str], *, record_lifecycle: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.events = events
        self.record_lifecycle = record_lifecycle
        self.step = -1
        if record_lifecycle:
            self.weight.register_hook(self._record_backward)

    def encode(self, batch: Batch) -> ModelOutput:
        self.step += 1
        self.events.append(f"encode:{self.step}")
        return ModelOutput(embeddings=self.weight.reshape(1, 1))

    def state_update(self, batch: Batch, output: ModelOutput):
        del output
        if self.record_lifecycle:
            self.events.append(f"state_update:{self.step}")
        return None

    def _record_backward(self, grad: torch.Tensor) -> torch.Tensor:
        self.events.append(f"backward:{self.step}")
        return grad


class _Task:
    name = "edge_prediction"

    def supervision(self, batch: Batch) -> Batch:
        batch.targets = {"task": object()}
        return batch

    def compute_loss(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        del batch
        return output.embeddings.sum()

    def compute_metrics(self, output: ModelOutput, batch: Batch):
        del output, batch
        return {}


def _batch(step: int) -> Batch:
    empty = torch.empty(0, dtype=torch.long)
    block = GraphBlock(
        src_nodes=torch.tensor([0]),
        dst_nodes=torch.tensor([0]),
        edge_ids=empty,
        format="coo",
        row=empty,
        col=empty,
        num_src=1,
        num_dst=1,
    )
    batch = Batch(mode="event", graph=block)
    return batch


def _accessed(step: int):
    block = _batch(step).graph
    assert block is not None
    return (
        step,
        {},
        ((block,),),
        (block.src_nodes,),
        (block.edge_ids,),
    )


def _pipeline_store(rank: int = 0):
    ptr = torch.zeros((3, 2), dtype=torch.long)
    return SimpleNamespace(
        graph=SimpleNamespace(
            rank=rank,
            time_ptr_2=ptr,
            split_time_ptr_2={"train": ptr},
            event_view={},
        ),
        features=SimpleNamespace(node_features={}, edge_features={}),
        labels=LabelStore(),
    )


def _run_epoch(
    monkeypatch,
    *,
    events: list[str],
    max_batches: int = 0,
    access_pipeline: bool | None = None,
    bounded_stale: bool = False,
    record_materialize: bool = False,
    record_lifecycle: bool = False,
    materialize_threads: list[int] | None = None,
    wait_policy: str = "block",
    state_commit_wait_interval: int | None = None,
) -> None:
    accessed = tuple(_accessed(step) for step in range(3))
    store = _pipeline_store()

    monkeypatch.setattr(
        runtime_batches,
        "_bind_source",
        lambda *args, **kwargs: (
            range(len(accessed)),
            lambda *, window_id, **unused: accessed[int(window_id)],
        ),
    )
    real_materialize = runtime_batches.materialize_accessed_window

    def materialize(item, **kwargs):
        if record_materialize:
            events.append(f"materialize:{item[0]}")
        if materialize_threads is not None:
            materialize_threads.append(get_ident())
        return real_materialize(item, **kwargs)

    monkeypatch.setattr(runtime_batches, "materialize_accessed_window", materialize)

    feature_step = -1

    def launch_features(batch, *args, **kwargs):
        nonlocal feature_step
        del args, kwargs
        feature_step += 1
        events.append(f"feature:{feature_step}")
        return batch, (), None

    state_step = -1

    def submit_state(batch, state_manager, **_):
        nonlocal state_step
        del batch, state_manager
        state_step += 1
        events.append(f"state:{state_step}")
        return []

    monkeypatch.setattr(
        runtime_pipeline,
        "launch_batch_features",
        launch_features,
    )
    monkeypatch.setattr(runtime_loop, "submit_hydrate_state", submit_state)

    options = {
        "defer_state_hydrate": True,
        "max_batches_per_epoch": max_batches,
    }
    if access_pipeline is not None:
        options["access_pipeline"] = access_pipeline
    if access_pipeline is False:
        options["defer_feature_launch"] = True
    if state_commit_wait_interval is not None:
        options["state_commit_wait_interval"] = state_commit_wait_interval

    runtime_loop.run_epoch(
        store=store,
        model=_RecordingModel(events, record_lifecycle=record_lifecycle),
        task=_Task(),
        optimizer=None,
        mode="event",
        training=True,
        window_policy="event_window",
        sampling_policy="full",
        state_manager=(
            SimpleNamespace(freshness_policy="bounded_stale")
            if bounded_stale
            else object()
        ),
        sampler_options=options,
        compute_metrics=False,
        wait_policy=wait_policy,
    )


def test_double_buffer_launches_one_feature_window_ahead_without_state_lookahead(
    monkeypatch,
) -> None:
    events: list[str] = []

    _run_epoch(monkeypatch, events=events)

    assert events == [
        "feature:0",
        "feature:1",
        "state:0",
        "encode:0",
        "feature:2",
        "state:1",
        "encode:1",
        "state:2",
        "encode:2",
    ]


def test_request_queue_never_runs_beyond_stage_c_lookahead(monkeypatch) -> None:
    third_started = Event()
    fourth_started = Event()
    accessed = tuple(_accessed(step) for step in range(4))
    store = _pipeline_store()

    def access(*, window_id, **unused):
        del unused
        if window_id == 2:
            third_started.set()
        if window_id == 3:
            fourth_started.set()
        return accessed[int(window_id)]

    monkeypatch.setattr(
        runtime_batches,
        "_bind_source",
        lambda *args, **kwargs: (range(4), access),
    )
    loader = runtime_batches.DataLoader(
        store,
        mode="event",
        split="train",
        window_policy="event_window",
        sampling_policy="full",
        chunk_decay=None,
        num_full_snapshots=1,
        num_layers=1,
        fanouts=None,
        sampler_options={},
        num_negatives=0,
        generator=None,
        comm=runtime_batches.CommScheduler(),
        device=None,
        prefetch_state=None,
    )
    values = iter(loader)

    assert isinstance(next(values), Batch)
    assert third_started.wait(1.0)
    assert not fourth_started.wait(0.05)
    assert isinstance(next(values), Batch)
    assert fourth_started.wait(1.0)
    values.close()


def test_double_buffer_does_not_launch_beyond_max_batches(monkeypatch) -> None:
    events: list[str] = []

    _run_epoch(monkeypatch, events=events, max_batches=1)

    assert events == ["feature:0", "state:0", "encode:0"]


def test_access_pipeline_false_disables_default_double_buffer(monkeypatch) -> None:
    events: list[str] = []

    _run_epoch(monkeypatch, events=events, access_pipeline=False)

    assert events == [
        "feature:0",
        "state:0",
        "encode:0",
        "feature:1",
        "state:1",
        "encode:1",
        "feature:2",
        "state:2",
        "encode:2",
    ]


def test_materialize_and_dependency_launch_stay_one_window_ahead(monkeypatch) -> None:
    events: list[str] = []

    _run_epoch(monkeypatch, events=events, record_materialize=True)

    assert events == [
        "materialize:0",
        "feature:0",
        "materialize:1",
        "feature:1",
        "state:0",
        "encode:0",
        "materialize:2",
        "feature:2",
        "state:1",
        "encode:1",
        "state:2",
        "encode:2",
    ]


def test_materialize_runs_in_stage_b_worker(monkeypatch) -> None:
    events: list[str] = []
    materialize_threads: list[int] = []

    _run_epoch(monkeypatch, events=events, materialize_threads=materialize_threads)

    assert materialize_threads
    assert set(materialize_threads) == {materialize_threads[0]}
    assert materialize_threads[0] != get_ident()


def test_ready_pipeline_does_not_claim_reschedule_semantics(monkeypatch) -> None:
    with pytest.raises(NotImplementedError, match="wait_policy='block'"):
        _run_epoch(monkeypatch, events=[], wait_policy="reschedule")


def test_ready_batch_waits_on_compute_stream_without_host_sync(monkeypatch) -> None:
    waited: list[object] = []
    recorded: list[tuple[Batch, object]] = []
    stream = SimpleNamespace(wait_event=waited.append)
    event = SimpleNamespace(synchronize=lambda: pytest.fail("host synchronize called"))
    loader = object.__new__(runtime_batches.DataLoader)
    loader.device = torch.device("cuda:0")
    batch = _batch(0)

    monkeypatch.setattr(torch.cuda, "current_stream", lambda **_: stream)
    monkeypatch.setattr(
        runtime_batches,
        "record_batch_stream",
        lambda value, current: recorded.append((value, current)),
    )

    assert loader._wait_ready((batch, event)) is batch
    assert waited == [event]
    assert recorded == [(batch, stream)]


def test_bounded_stale_state_launches_with_stage_b(monkeypatch) -> None:
    events: list[str] = []

    _run_epoch(monkeypatch, events=events, bounded_stale=True)

    assert events == [
        "feature:0",
        "state:0",
        "feature:1",
        "state:1",
        "encode:0",
        "feature:2",
        "state:2",
        "encode:1",
        "encode:2",
    ]


def test_stage_a_waits_exact_state_but_not_stage_b_prefetched_state(monkeypatch) -> None:
    exact_events: list[str] = []
    bounded_events: list[str] = []

    def record_finish(events):
        def finish(_manager, *, final=False):
            events.append("finish:final" if final else "finish:owner")

        return finish

    monkeypatch.setattr(runtime_loop, "finish_state_update", record_finish(exact_events))
    _run_epoch(
        monkeypatch,
        events=exact_events,
        state_commit_wait_interval=0,
    )

    monkeypatch.setattr(runtime_loop, "finish_state_update", record_finish(bounded_events))
    _run_epoch(monkeypatch, events=bounded_events, bounded_stale=True)

    assert exact_events == [
        "feature:0",
        "feature:1",
        "finish:owner",
        "state:0",
        "encode:0",
        "feature:2",
        "finish:owner",
        "state:1",
        "encode:1",
        "finish:owner",
        "state:2",
        "encode:2",
        "finish:final",
    ]
    assert bounded_events == [
        "feature:0",
        "state:0",
        "feature:1",
        "state:1",
        "encode:0",
        "feature:2",
        "state:2",
        "encode:1",
        "encode:2",
        "finish:final",
    ]


def test_state_update_runs_after_backward(monkeypatch) -> None:
    events: list[str] = []

    _run_epoch(monkeypatch, events=events, max_batches=1, record_lifecycle=True)

    assert events == [
        "feature:0",
        "state:0",
        "encode:0",
        "backward:0",
        "state_update:0",
    ]


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="requires torchrun with at least two ranks",
)
def test_distributed_empty_owner_runs_zero_gradient_step(monkeypatch) -> None:
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        batch = _batch(0)
        batch.targets = {
            "task": SimpleNamespace(
                target_ids=torch.tensor([0]) if rank == 0 else torch.empty(0, dtype=torch.long)
            )
        }
        monkeypatch.setattr(runtime_loop, "DataLoader", lambda *args, **kwargs: (batch,))
        model = _RecordingModel([])
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        task = _Task()
        task.supervision = lambda value: value
        store = _pipeline_store(rank)

        result = runtime_loop.run_epoch(
            store=store,
            model=model,
            task=task,
            optimizer=optimizer,
            mode="event",
            training=True,
            window_policy="event_window",
            sampling_policy="full",
            sampler_options={},
            compute_metrics=False,
            gradient_sync="all_reduce",
        )

        assert model.weight.item() == 0.5
        assert result.steps == 1
    finally:
        if created_group and dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="requires torchrun with at least two ranks",
)
def test_distributed_double_buffer_preserves_collective_order(monkeypatch) -> None:
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert world_size == 2
        events: list[str] = []
        accessed = tuple(_accessed(step) for step in range(3))
        store = _pipeline_store(rank)
        monkeypatch.setattr(
            runtime_batches,
            "_bind_source",
            lambda *args, **kwargs: (
                range(len(accessed)),
                lambda *, window_id, **unused: accessed[int(window_id)],
            ),
        )

        feature_step = -1

        def launch_features(batch, store, *, comm, sampler_options, **_):
            nonlocal feature_step
            del store, sampler_options
            feature_step += 1
            step = feature_step
            events.append(f"feature:{step}")
            payload = torch.tensor(
                [[step * 100 + rank * 10 + dst] for dst in range(world_size)],
                dtype=torch.float32,
            )
            route = Route(
                send_sizes=(1,) * world_size,
                recv_sizes=(1,) * world_size,
            )
            handle = comm.launch_push(route, payload, name=f"double_buffer:{step}")
            pending = PendingNodeFeatureFetch(
                keys=("node",),
                out={"node": torch.zeros(world_size, 1)},
                remote=True,
                missing_pos=torch.arange(world_size),
                order=torch.arange(world_size),
                response_handles={"node": handle},
                scheduler=comm,
            )
            return batch, ({"pending": pending},), None

        class _DistributedModel(_RecordingModel):
            def encode(self, batch: Batch) -> ModelOutput:
                step = self.step + 1
                expected = torch.tensor(
                    [[step * 100 + src * 10 + rank] for src in range(world_size)],
                    dtype=torch.float32,
                )
                assert torch.equal(batch.features["node"], expected)
                return super().encode(batch)

        monkeypatch.setattr(
            runtime_pipeline,
            "launch_batch_features",
            launch_features,
        )

        result = runtime_loop.run_epoch(
            store=store,
            model=_DistributedModel(events),
            task=_Task(),
            optimizer=None,
            mode="event",
            training=True,
            window_policy="event_window",
            sampling_policy="full",
            sampler_options={},
            compute_metrics=False,
        )

        assert result.steps == 3
        assert events == [
            "feature:0",
            "feature:1",
            "encode:0",
            "feature:2",
            "encode:1",
            "encode:2",
        ]
        dist.barrier()
    finally:
        if created_group and dist.is_initialized():
            dist.destroy_process_group()
