import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import starrygl as sg

from starrygl.runtime import memory as runtime_memory
from starrygl.runtime import state as runtime_state
from starrygl.runtime import train as runtime_train
from starrygl.runtime.comm import CommScheduler
from starrygl.runtime.loop import EpochResult
from starrygl.model import StateDelta
from starrygl.store import FeatureManager, GraphStore, LabelStore, MailboxManager, StateManager, StoreBundle


class _Manager:
    kind = "node_memory"

    def __init__(self) -> None:
        self.values = torch.zeros(3, 2)
        self.timestamps = torch.zeros(3)
        self.async_owner_collective = False
        self.commits = []
        self.calls = []

    def commit(self, delta: StateDelta) -> None:
        self.commits.append(delta)

    def handle_owner_async(self) -> None:
        self.calls.append("owner")

    def launch_shared(self) -> None:
        self.calls.append("launch_shared")

    def handle_last_async(self) -> None:
        self.calls.append("last")

    def finish_shared_async(self) -> None:
        self.calls.append("finish_shared")

    def finish_shared_ready(self) -> None:
        self.calls.append("poll_shared")


def test_launch_state_update_dispatches_by_kind() -> None:
    memory = _Manager()
    mailbox = _Manager()
    mailbox.kind = "mailbox"
    delta = StateDelta(
        kind="mailbox",
        node_ids=torch.tensor([1]),
        values=torch.ones(1, 2),
    )

    runtime_state.launch_state_update({"node_memory": memory, "mailbox": mailbox}, delta)

    assert memory.commits == []
    assert mailbox.commits == [delta]
    assert mailbox.calls == ["launch_shared"]


def test_launch_state_update_submits_empty_collective_delta(monkeypatch) -> None:
    manager = _Manager()
    manager.async_owner_collective = True
    monkeypatch.setattr(runtime_state.dist, "is_available", lambda: True)
    monkeypatch.setattr(runtime_state.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runtime_state.dist, "get_world_size", lambda: 2)

    runtime_state.launch_state_update(manager, None)

    assert len(manager.commits) == 1
    delta = manager.commits[0]
    assert delta.kind == "node_memory"
    assert delta.node_ids.numel() == 0
    assert delta.values.shape == (0, 2)
    assert delta.timestamps is not None
    assert delta.timestamps.numel() == 0
    assert manager.calls == ["launch_shared"]


def test_state_update_helpers_call_manager_boundaries() -> None:
    first = _Manager()
    second = _Manager()
    managers = {"a": first, "b": second}

    runtime_state.finish_state_update(managers)
    runtime_state.poll_state_update(managers)
    runtime_state.finish_state_update(managers, final=True)

    assert first.calls == [
        "owner",
        "poll_shared",
        "last",
    ]
    assert second.calls == first.calls


def test_evaluate_and_predict_advance_temporal_state_by_default(monkeypatch) -> None:
    calls = []

    def run_epoch(**kwargs):
        calls.append(kwargs)
        return EpochResult(loss=0.0, steps=0)

    monkeypatch.setattr(runtime_train, "run_epoch", run_epoch)
    model = torch.nn.Linear(1, 1)
    trainer = sg.compile(
        data_source={"temporal_representation": "event_stream"},
        backbone=model,
        task_segment={"name": "node_regression"},
        runtime={"device": "cpu"},
    )
    store = StoreBundle(
        graph=GraphStore(num_nodes=1),
        features=FeatureManager(node_features={"x": torch.zeros(1, 1)}),
        labels=LabelStore(node_label=torch.zeros(1)),
    )
    task = object()

    trainer.evaluate(store=store, model=model, task=task)
    trainer.predict(store=store, model=model, task=task)

    assert calls[0]["commit_state"] is True
    assert calls[1]["commit_state"] is True


def test_fit_forwards_batch_trace_callback(monkeypatch) -> None:
    calls = []

    def run_epoch(**kwargs):
        calls.append(kwargs)
        return EpochResult(loss=0.0, steps=0)

    monkeypatch.setattr(runtime_train, "run_epoch", run_epoch)
    trainer = sg.compile(
        data_source={"temporal_representation": "event_stream"},
        backbone=torch.nn.Linear(1, 1),
        task_segment={"name": "node_regression"},
        runtime={"device": "cpu"},
    )
    store = StoreBundle(
        graph=GraphStore(num_nodes=1),
        features=FeatureManager(node_features={"x": torch.zeros(1, 1)}),
        labels=LabelStore(node_label=torch.zeros(1)),
    )
    callback = object()

    trainer.fit(store=store, model=torch.nn.Linear(1, 1), task=object(),
                optimizer=object(), epochs=1, batch_callback=callback)

    assert calls[0]["batch_callback"] is callback


def test_decoupled_snapshot_eval_and_predict_replay_exact_history(monkeypatch) -> None:
    calls = []

    def run_epoch(**kwargs):
        calls.append(kwargs)
        if kwargs.get("output_callback") is not None:
            kwargs["output_callback"](sg.ModelOutput())
        return EpochResult(loss=0.0, steps=0)

    class Model(torch.nn.Module):
        runtime_cell = SimpleNamespace(reads_neighbor_state=False)

    monkeypatch.setattr(runtime_train, "run_epoch", run_epoch)
    trainer = sg.compile(
        data_source={"temporal_representation": "snapshot_sequence"},
        backbone={"name": "tgcn", "in_dim": 1, "hidden_dim": 1, "out_dim": 1},
        task_segment={"name": "node_regression", "loss": "mse"},
        runtime={"sampling": {"mode": "full", "window": {"policy": "chunk_decay"}}},
    )
    ptr = torch.tensor([[0, 1]], dtype=torch.long)
    store = StoreBundle(
        graph=GraphStore(
            num_nodes=1,
            prepare={
                "partition": {},
                "time_ptr_2": ptr,
                "split_time_ptr_2": {
                    "train": ptr,
                    "val": torch.empty((0, 2), dtype=torch.long),
                    "test": ptr,
                },
            },
        ),
        features=FeatureManager(node_features={"x": torch.zeros(1, 1)}),
        labels=LabelStore(node_label=torch.zeros(1)),
    )
    task = SimpleNamespace(name="node_regression", target_owner="node_master")
    state = _Manager()

    trainer.evaluate(store=store, model=Model(), task=task, state_manager=state)
    outputs = trainer.predict(store=store, model=Model(), task=task, state_manager=state, split="test")

    assert [call["split"] for call in calls] == ["train", "test"] * 2
    assert all(call["window_policy"] == "full_snapshot" for call in calls)
    assert all(call["num_full_snapshots"] == 1 for call in calls)
    assert len(outputs) == 1


def test_shared_update_launch_does_not_wait_or_relaunch_owner(monkeypatch) -> None:
    events = []
    owner = StateManager(
        values=torch.zeros(2, 2),
        timestamps=torch.zeros(2),
        row_map=torch.arange(2),
        kind="node_memory",
    )
    shared = StateManager(
        values=torch.zeros(2, 2),
        timestamps=torch.zeros(2),
        row_map=torch.arange(2),
        kind="node_memory",
    )
    owner.submit_commit_async = lambda delta: events.append("owner_launch") or delta
    owner.finish_commit_async = lambda pending: events.append("owner_finish")
    monkeypatch.setattr(runtime_memory, "distributed", lambda: True)
    manager = runtime_memory.AsyncMemoryCommitter(
        owner,
        shared_manager=shared,
        background_owner=True,
    )
    delta = StateDelta(
        kind="node_memory",
        node_ids=torch.tensor([1]),
        values=torch.ones(1, 2),
        timestamps=torch.ones(1),
    )

    manager.commit(delta)
    manager.launch_shared()
    manager.launch_shared()

    assert events == ["owner_launch"]
    assert len(manager.pending_shared_sync) == 1

    manager.handle_last_async()

    assert events == ["owner_launch", "owner_finish"]
    assert manager.pending == []


def test_async_committer_refreshes_and_reads_shared_hot_state() -> None:
    row_map = torch.arange(3)
    owner = StateManager(values=torch.zeros(3, 2), timestamps=torch.zeros(3), row_map=row_map, kind="node_memory")
    mailbox = MailboxManager(values=torch.zeros(3, 1, 2), timestamps=torch.zeros(3, 1), row_map=row_map)
    shared = StateManager(values=torch.zeros(3, 2), timestamps=torch.zeros(3), row_map=row_map, kind="node_memory")
    shared_mailbox = MailboxManager(values=torch.zeros(3, 1, 2), timestamps=torch.zeros(3, 1), row_map=row_map)
    manager = runtime_memory.AsyncMemoryCommitter(
        owner,
        mailbox_manager=mailbox,
        shared_manager=shared,
        shared_mailbox_manager=shared_mailbox,
    )
    delta = StateDelta(
        kind="node_memory",
        node_ids=torch.tensor([1]),
        values=torch.tensor([[2.0, 3.0]]),
        timestamps=torch.tensor([4.0]),
        metadata={
            "mailbox_nodes": torch.tensor([1]),
            "mailbox_messages": torch.tensor([[5.0, 6.0]]),
            "mailbox_timestamps": torch.tensor([4.0]),
        },
    )

    manager.commit(delta)
    manager.launch_shared()
    manager.finish_shared_async()
    state, messages = manager.materialize_local_with_mailbox_if_all_present(torch.tensor([1]))

    assert state.values.tolist() == [[2.0, 3.0]]
    assert messages.values.reshape(1, -1).tolist() == [[5.0, 6.0]]
    assert manager.pending == []


def test_shared_refresh_keeps_all_hot_updates_when_compensation_metadata_is_partial() -> None:
    owner = StateManager(values=torch.zeros(2, 2), row_map=torch.arange(2), kind="node_memory")
    shared = StateManager(values=torch.zeros(2, 2), row_map=torch.arange(2), kind="node_memory")
    manager = runtime_memory.AsyncMemoryCommitter(owner, shared_manager=shared)
    delta = StateDelta(
        kind="node_memory",
        node_ids=torch.tensor([0, 1]),
        values=torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
        timestamps=torch.tensor([1.0, 1.0]),
        metadata={"state_compensation_rows": torch.tensor([0])},
    )

    handle = manager.submit_commit(delta)

    assert handle.shared_delta is not None
    assert handle.shared_delta.node_ids.tolist() == [0, 1]


def test_bounded_state_read_marks_only_rows_loaded_from_shared_hot() -> None:
    owner = StateManager(
        values=torch.tensor([[1.0, 1.0]]),
        row_map=torch.tensor([0, -1]),
        kind="neighbor_recurrent",
    )
    shared = StateManager(
        values=torch.tensor([[2.0, 2.0]]),
        row_map=torch.tensor([-1, 0]),
        kind="neighbor_recurrent",
    )
    manager = runtime_memory.AsyncMemoryCommitter(
        owner,
        shared_manager=shared,
        bounded_stale_reads=True,
    )

    read = manager.materialize(torch.tensor([0, 1]))

    assert torch.equal(read.values, torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
    assert torch.equal(read.metadata["shared_mask"], torch.tensor([False, True]))
    assert torch.equal(read.metadata["shared_rows"], torch.tensor([-1, 0]))


def test_trainer_builds_plan_owned_temporal_state_once() -> None:
    trainer = sg.compile(
        data_source={"source": "events"},
        backbone={"name": "tgn", "in_dim": 2, "hidden_dim": 4, "out_dim": 4, "edge_dim": 2},
        task_segment=sg.EdgePrediction(),
    )
    store = StoreBundle(
        graph=GraphStore(
            num_nodes=4,
            prepare={"partition": {"hot_node_ids": torch.tensor([1, 3])}},
        ),
        features=FeatureManager(
            node_features={"x": torch.zeros(4, 2)},
            node_ids=torch.arange(4),
            node_row_map=torch.arange(4),
        ),
        labels=LabelStore(),
    )
    scheduler = CommScheduler()

    state = trainer._state_manager(
        None,
        model=trainer.model,
        store=store,
        device="cpu",
        comm=scheduler,
    )
    again = trainer._state_manager(
        None,
        model=trainer.model,
        store=store,
        device="cpu",
        comm=scheduler,
    )

    manager = state["node_memory"]
    assert again is state
    assert isinstance(manager, runtime_memory.AsyncMemoryCommitter)
    assert manager.bounded_stale_reads is True
    assert manager.memory_manager.values.shape == (4, 4)
    assert manager.mailbox_manager.values.shape == (4, 1, 10)
    assert manager.shared_manager.values.shape == (2, 4)
    assert manager.change_filter.min_cosine_distance == 0.3
    assert manager.change_filter.max_skip == 1
    assert manager.freshness_policy == "bounded_stale"
    assert manager.max_staleness == 1


def test_state_owner_rows_do_not_reuse_feature_replicas() -> None:
    trainer = sg.compile(
        data_source={"source": "events"},
        backbone={"name": "tgn", "in_dim": 2, "hidden_dim": 4, "out_dim": 4, "edge_dim": 2},
        task_segment=sg.EdgePrediction(),
    )
    node_dist_index = torch.tensor([1, 1 << 48], dtype=torch.long)
    store = StoreBundle(
        graph=GraphStore(
            num_nodes=2,
            rank=0,
            prepare={
                "meta": {"world_size": 2},
                "partition": {
                    "node_dist_index": node_dist_index,
                    "hot_node_ids": torch.tensor([1]),
                },
            },
        ),
        features=FeatureManager(
            node_features={"x": torch.zeros(2, 2)},
            node_ids=torch.arange(2),
            node_row_map=torch.arange(2),
        ),
        labels=LabelStore(),
    )

    state = trainer._state_manager(
        None,
        model=trainer.model,
        store=store,
        device="cpu",
        comm=CommScheduler(),
    )["node_memory"]
    state.shared_manager.values[0] = 9.0

    assert state.memory_manager.row_map.tolist() == [1, -1]
    assert state.materialize(torch.tensor([1])).values.tolist() == [[9.0] * 4]


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two torchrun ranks")
def test_distributed_combined_memory_mailbox_fetch() -> None:
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        owner = torch.tensor([0, 1, 0, 1])
        location = torch.tensor([0, 0, 1, 1])
        dist_index = (owner << 48) | location
        owned = (owner == rank).nonzero(as_tuple=True)[0]
        row_map = torch.full((4,), -1, dtype=torch.long)
        row_map[owned] = torch.arange(2)
        memory = StateManager(
            values=(owned + 10).float().reshape(-1, 1),
            timestamps=torch.zeros(2),
            row_map=row_map,
            node_dist_index=dist_index,
            kind="node_memory",
            comm=CommScheduler(),
        )
        mailbox = MailboxManager(
            values=(owned + 20).float().reshape(-1, 1, 1),
            timestamps=torch.zeros(2, 1),
            row_map=row_map,
            node_dist_index=dist_index,
            comm=memory.comm,
        )
        manager = runtime_memory.AsyncMemoryCommitter(memory, mailbox_manager=mailbox)
        query = torch.tensor([rank, 1 - rank, rank + 2, 3 - rank])

        pending = manager.submit_materialize_with_mailbox_async(query)
        state, messages = manager.finish_materialize_with_mailbox_async(pending)

        assert torch.equal(state.values.reshape(-1), (query + 10).float())
        assert torch.equal(messages.values.reshape(-1), (query + 20).float())
    finally:
        if created_group:
            dist.destroy_process_group()


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two torchrun ranks")
def test_bounded_memory_mailbox_pair_uses_shared_hot_without_owner_fetch() -> None:
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        owners = torch.tensor([0, 1, 0, 1])
        locations = torch.tensor([0, 0, 1, 1])
        dist_index = (owners << 48) | locations
        owned = (owners == rank).nonzero(as_tuple=True)[0]
        row_map = torch.full((4,), -1, dtype=torch.long)
        row_map[owned] = torch.arange(2)
        comm = CommScheduler()
        memory = StateManager(
            values=(owned + 10).float().reshape(-1, 1),
            timestamps=torch.ones(2), row_map=row_map,
            node_dist_index=dist_index, kind="node_memory", comm=comm,
        )
        mailbox = MailboxManager(
            values=(owned + 20).float().reshape(-1, 1, 1),
            timestamps=torch.ones(2, 1), row_map=row_map,
            node_dist_index=dist_index, comm=comm,
        )
        shared_memory = StateManager(
            values=(torch.arange(4) + 10).float().reshape(-1, 1),
            timestamps=torch.ones(4), row_map=torch.arange(4), kind="node_memory",
        )
        shared_mailbox = MailboxManager(
            values=(torch.arange(4) + 20).float().reshape(-1, 1, 1),
            timestamps=torch.ones(4, 1), row_map=torch.arange(4),
        )
        manager = runtime_memory.AsyncMemoryCommitter(
            memory, mailbox_manager=mailbox, shared_manager=shared_memory,
            shared_mailbox_manager=shared_mailbox, bounded_stale_reads=True,
        )
        query = torch.arange(4)

        pending = manager.submit_materialize_with_mailbox_async(query)
        state, messages = manager.finish_materialize_with_mailbox_async(pending)

        assert state.values.reshape(-1).tolist() == [10.0, 11.0, 12.0, 13.0]
        assert messages.values.reshape(-1).tolist() == [20.0, 21.0, 22.0, 23.0]
        assert torch.equal(state.metadata["shared_mask"], owners != rank)
    finally:
        if created_group:
            dist.destroy_process_group()


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two torchrun ranks")
def test_distributed_state_and_mailbox_fetch_keep_empty_collectives() -> None:
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        row_map = torch.tensor([0]) if rank == 0 else torch.tensor([-1])
        rows = 1 if rank == 0 else 0
        comm = CommScheduler()
        state = StateManager(
            values=torch.full((rows, 1), 7.0),
            row_map=row_map,
            node_dist_index=torch.tensor([0]),
            comm=comm,
        )
        mailbox = MailboxManager(
            values=torch.full((rows, 1, 1), 8.0),
            row_map=row_map,
            node_dist_index=torch.tensor([0]),
            comm=comm,
        )

        state_read = state.finish_materialize_async(state.submit_materialize_async(torch.tensor([0])))
        mailbox_read = mailbox.finish_materialize_async(mailbox.submit_materialize_async(torch.tensor([0])))

        assert state_read.values.item() == 7.0
        assert mailbox_read.values.item() == 8.0
    finally:
        if created_group:
            dist.destroy_process_group()


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two torchrun ranks")
def test_distributed_recurrent_commit_keeps_empty_collectives() -> None:
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        manager = StateManager(
            values=torch.zeros((1 if rank == 0 else 0, 1)),
            row_map=torch.tensor([0]) if rank == 0 else torch.tensor([-1]),
            node_dist_index=torch.tensor([0]),
            kind="node_recurrent",
            comm=CommScheduler(),
        )
        delta = (
            StateDelta(
                node_ids=torch.tensor([0]),
                values=torch.tensor([[11.0]]),
                kind="node_recurrent",
            )
            if rank == 0
            else None
        )

        runtime_state.launch_state_update(manager, delta)

        if rank == 0:
            assert manager.values.item() == 11.0
    finally:
        if created_group:
            dist.destroy_process_group()


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two torchrun ranks")
def test_distributed_owner_and_shared_updates_launch_together() -> None:
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        owners = torch.tensor([0, 1, 0, 1])
        locations = torch.tensor([0, 0, 1, 1])
        dist_index = (owners << 48) | locations
        owned = (owners == rank).nonzero(as_tuple=True)[0]
        row_map = torch.full((4,), -1, dtype=torch.long)
        row_map[owned] = torch.arange(2)
        comm = CommScheduler()
        owner = StateManager(
            values=torch.zeros(2, 1),
            timestamps=torch.zeros(2),
            row_map=row_map,
            node_dist_index=dist_index,
            kind="node_memory",
            comm=comm,
        )
        shared = StateManager(
            values=torch.zeros(4, 1),
            timestamps=torch.zeros(4),
            row_map=torch.arange(4),
            kind="node_memory",
            comm=comm,
        )
        manager = runtime_memory.AsyncMemoryCommitter(
            owner,
            shared_manager=shared,
            background_owner=True,
        )
        node_id = torch.tensor([1 - rank])
        delta = StateDelta(
            kind="node_memory",
            node_ids=node_id,
            values=torch.tensor([[float(rank + 1)]]),
            timestamps=torch.tensor([float(rank + 1)]),
        )

        runtime_state.launch_state_update(manager, delta)
        runtime_state.finish_state_update(manager)
        runtime_state.finish_state_update(manager, final=True)

        expected_owner = torch.tensor([2.0 if rank == 0 else 1.0])
        assert torch.equal(owner.values[0], expected_owner)
        assert torch.equal(shared.values[:2, 0], torch.tensor([2.0, 1.0]))
        assert manager.pending == []
        assert manager.pending_shared_sync == []
    finally:
        if created_group:
            dist.destroy_process_group()
