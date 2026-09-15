from types import SimpleNamespace

import pytest
import torch

import starrygl.runtime.dataloader.loader as runtime_loader
import starrygl.runtime.snapshot.cache as snapshot_cache
import starrygl.runtime.snapshot.materialize as snapshot_materialize
from starrygl.batch import EventRows
from starrygl.prepare.task import build_task_shards
from starrygl.runtime.comm import CommScheduler
from starrygl.runtime.event.materialize import _event_sampling_roots, event_window_ids
from starrygl.runtime.dataloader.materialize import materialize_accessed_window
from starrygl.store.graph import split_window_range
from starrygl.runtime.snapshot.materialize import access_snapshot_window
from starrygl.store import LabelStore
from starrygl.task.negative import snapshot_negative_dst_pool
from starrygl.task.target import (
    build_task_target,
    build_window_task_target,
    with_negative_samples,
)
from starrygl.task import NegativeSamplePool, TargetRoute
from starrygl.view import GraphBlock


def test_event_context_keeps_state_update_fields_out_of_task_target() -> None:
    target = build_task_target(
        target_kind="edge",
        target_ids=torch.tensor([10, 11]),
        target_ts=torch.tensor([0.5, 1.5]),
        label=None,
        pos_src=torch.tensor([0, 1]),
        pos_dst=torch.tensor([1, 2]),
        neg_src=None,
        neg_dst=torch.tensor([3, 4]),
        neg_loss_weight=None,
        negative_pool=None,
        target_route=None,
        edge_ids=torch.tensor([10, 11]),
    )
    events = EventRows(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([1, 2]),
        edge_ids=torch.tensor([10, 11]),
        ts=torch.tensor([0.5, 1.5]),
        state_write_mask=torch.tensor([1, 2], dtype=torch.uint8),
    )

    assert torch.equal(target.neg_dst, torch.tensor([3, 4]))
    assert torch.equal(events.state_write_mask, torch.tensor([1, 2], dtype=torch.uint8))


def test_node_target_uses_node_master_route_rows() -> None:
    target = build_task_target(
        target_kind="node",
        target_ids=torch.tensor([2, 5]),
        target_ts=torch.tensor([7.0, 7.0]),
        label=torch.tensor([1, 0]),
        negative_pool=NegativeSamplePool(mode="dst", local_node_ids=torch.tensor([2, 5])),
        target_route=TargetRoute(target_rows=torch.tensor([0, 1])),
    )

    assert target.target_kind == "node"
    assert torch.equal(target.node_ids, torch.tensor([2, 5]))
    assert target.target_route is not None
    assert torch.equal(target.target_route.target_rows, torch.tensor([0, 1]))


def test_snapshot_edge_target_materializes_negatives_and_preserves_route() -> None:
    generator = torch.Generator().manual_seed(1)
    route = TargetRoute(pos_src_rows=torch.tensor([0, 1]), pos_dst_rows=torch.tensor([1, 2]))
    target = build_task_target(
        target_kind="edge",
        target_ids=torch.tensor([20, 21]),
        target_ts=torch.tensor([1.0, 2.0]),
        label=torch.tensor([1.0, 1.0]),
        pos_src=torch.tensor([0, 1]),
        pos_dst=torch.tensor([1, 2]),
        negative_pool=NegativeSamplePool(mode="dst", local_dst_ids=torch.tensor([3, 4, 5])),
        num_negatives=2,
        generator=generator,
        target_route=route,
    )

    assert target.target_kind == "edge"
    assert target.neg_src is None
    assert target.neg_dst is not None
    assert target.neg_dst.shape == (4,)
    assert target.negative_pool is None
    assert target.target_route is route


def test_snapshot_negative_dst_pool_normalizes_train_mix() -> None:
    pool = snapshot_negative_dst_pool(
        local_dst_ids=torch.tensor([1, 2]),
        global_dst_ids=torch.tensor([3, 4]),
        split="train",
        options={"negative_local_prob": 2.0, "negative_global_prob": 1.0},
    )

    assert pool.mode == "dst"
    assert abs(pool.local_prob - (2.0 / 3.0)) < 1e-6
    assert abs(pool.global_prob - (1.0 / 3.0)) < 1e-6


def test_sampling_root_targets_feed_negative_materialization() -> None:
    root = build_task_target(
        target_kind="edge",
        target_ids=torch.tensor([7, 8]),
        target_ts=torch.tensor([1.0, 2.0]),
        pos_src=torch.tensor([0, 1]),
        pos_dst=torch.tensor([1, 2]),
        negative_pool=NegativeSamplePool(mode="dst", local_dst_ids=torch.tensor([3, 4])),
    )

    sampled = with_negative_samples(
        root,
        num_negatives=1,
        generator=torch.Generator().manual_seed(0),
    )

    assert sampled.target_kind == "edge"
    assert sampled.neg_dst is not None
    assert sampled.neg_dst.shape == (2,)
    assert sampled.negative_pool is root.negative_pool


def test_node_sampling_roots_keep_label_and_event_timestamps_separate() -> None:
    root = build_task_target(
        target_kind="node",
        target_ids=torch.tensor([2]),
        target_ts=torch.tensor([5.0]),
        label=torch.tensor([1]),
    )
    events = EventRows(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([1, 2]),
        edge_ids=torch.tensor([10, 11]),
        ts=torch.tensor([1.0, 2.0]),
    )
    roots = _event_sampling_roots(root, events)

    assert root.pos_src is None
    assert root.edge_ids is None
    assert torch.equal(roots.node_ids, torch.tensor([2, 0, 1, 1, 2]))
    assert torch.equal(roots.ts, torch.tensor([5.0, 1.0, 2.0, 1.0, 2.0]))


def test_event_node_targets_follow_global_event_windows_once() -> None:
    ptr = torch.tensor([[0, 2], [2, 4], [4, 6]])
    prepared = SimpleNamespace(
        meta={"world_size": 1},
        partition={
            "node_dist_index": torch.arange(7),
            "edge_dist_index": torch.arange(6),
        },
        time_ptr_2=ptr,
        split_time_ptr_2={"train": ptr},
    )
    shard = build_task_shards(
        prepared=prepared,
        task="node_classification",
        temporal="event",
        src=torch.arange(6),
        dst=torch.arange(1, 7),
        ts=torch.arange(1, 7, dtype=torch.float32),
        edge_ids=torch.arange(6),
        node_label=torch.tensor([0, 1, 0, 1]),
        node_label_nodes=torch.tensor([0, 1, 2, 3]),
        node_label_ts=torch.tensor([0.5, 3.0, 3.0, 9.0]),
        node_label_split=torch.zeros(4, dtype=torch.uint8),
        node_label_temporal=False,
        node_label_horizon=0,
        edge_label=None,
    )[0]
    labels = LabelStore.from_shard(shard)

    assert [labels.task_slice(index)["node_ids"].tolist() for index in range(3)] == [[0], [1, 2], [3]]


def test_snapshot_task_table_encodes_horizon_without_crossing_split() -> None:
    ptr = torch.tensor([[0, 1], [1, 2], [2, 3]])
    prepared = SimpleNamespace(
        meta={"world_size": 1},
        partition={
            "node_dist_index": torch.arange(3),
            "edge_dist_index": torch.arange(3),
        },
        time_ptr_2=ptr,
        split_time_ptr_2={
            "train": ptr[:2],
            "val": ptr[2:],
            "test": torch.empty((0, 2), dtype=torch.long),
        },
    )
    common = dict(
        prepared=prepared,
        temporal="snapshot",
        src=torch.tensor([0, 1, 2]),
        dst=torch.tensor([1, 2, 0]),
        ts=torch.tensor([1.0, 2.0, 3.0]),
        edge_ids=torch.tensor([10, 11, 12]),
        node_label_nodes=None,
        node_label_ts=None,
        node_label_split=None,
        edge_label=None,
    )
    edge = build_task_shards(
        task="edge_prediction",
        node_label=None,
        node_label_temporal=False,
        node_label_horizon=0,
        **common,
    )[0]
    node = build_task_shards(
        task="node_regression",
        node_label=torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]]),
        node_label_temporal=True,
        node_label_horizon=1,
        **common,
    )[0]

    assert edge["task_ptr"].tolist() == [0, 1, 1, 1]
    assert edge["task_payload"]["edge_ids"].tolist() == [11]
    assert node["task_ptr"].tolist() == [0, 3, 3, 3]
    assert node["task_payload"]["label"].tolist() == [10.0, 11.0, 12.0]


def test_event_split_uses_global_time_ptr_rows() -> None:
    ptr = torch.tensor([[0, 1], [1, 3], [3, 4], [4, 6]])
    split_ptrs = {"train": ptr[:1], "val": ptr[1:3], "test": ptr[3:]}
    view = {"time_ptr_2": ptr}
    store = SimpleNamespace(
        graph=SimpleNamespace(
            event_view=view,
            split_time_ptr_2=split_ptrs,
            time_ptr_2=ptr,
            rank=0,
        )
    )

    window_ids = split_window_range(store.graph.split_time_ptr_2, "val")

    assert list(window_ids) == [1, 2]
    assert [view["time_ptr_2"][index].tolist() for index in window_ids] == [[1, 3], [3, 4]]


def test_snapshot_neighbor_sampling_uses_task_target_roots() -> None:
    input_row = _snapshot_row(0, edge_ids=torch.tensor([0, 1]))
    target_row = _snapshot_row(1, edge_ids=torch.tensor([10, 11]))
    store = _snapshot_store(input_row, target_row, task_kind="edge")

    class Sampler:
        calls = []

        def sample_blocks(self, root_nodes, root_ts):
            self.calls.append((root_nodes.clone(), root_ts.clone()))
            return (_root_block(root_nodes),)

    sampler = Sampler()
    accessed = access_snapshot_window(
        store,
        store.graph.snapshot_csc_view["slices"],
        split="train",
        window_id=0,
        input_window=range(1),
        chunk_limits=(-1,),
        sampling_policy="neighbor",
        native_sampler=sampler,
        sampler_options={"negative_local_prob": 1.0},
        num_negatives=1,
        generator=torch.Generator().manual_seed(0),
    )

    assert accessed is not None
    target = accessed[1]["task"]
    assert target.target_kind == "edge"
    assert torch.equal(target.pos_src, torch.tensor([0, 1]))
    assert torch.equal(target.pos_dst, torch.tensor([1, 2]))
    assert target.neg_dst is not None
    assert torch.equal(sampler.calls[0][0], torch.cat((target.pos_src, target.pos_dst, target.neg_dst)))
    assert torch.equal(sampler.calls[0][1], torch.full_like(sampler.calls[0][1], 2))


def test_snapshot_edge_target_uses_edge_master_task_surface() -> None:
    input_row = _snapshot_row(0, edge_ids=torch.tensor([0, 1]))
    target_row = _snapshot_row(1, edge_ids=torch.tensor([10, 11]))
    store = _snapshot_store(input_row, target_row, task_kind="edge")
    store.labels = LabelStore(
        task_kind="edge",
        task_ptr=torch.tensor([0, 1, 1]),
        task_payload={
            "src": torch.tensor([2]),
            "dst": torch.tensor([0]),
            "edge_ids": torch.tensor([20]),
            "edge_rows": torch.tensor([1]),
        },
    )
    target = build_window_task_target(store.labels, 0)

    assert target.target_ids.tolist() == [20]
    assert target.pos_src.tolist() == [2]
    assert target.pos_dst.tolist() == [0]


def test_snapshot_target_rejects_missing_task_table() -> None:
    row = _snapshot_row(0, edge_ids=torch.tensor([0, 1]))
    store = _snapshot_store(row)
    with pytest.raises(ValueError, match="re-run prepare"):
        build_window_task_target(LabelStore(), 0)


def test_snapshot_materialization_reuses_prebuilt_target(monkeypatch) -> None:
    row = _snapshot_row(0, edge_ids=torch.tensor([0, 1]))
    store = _snapshot_store(row)
    target = build_window_task_target(store.labels, 0)
    block = _root_block(row["dst_nodes"])
    batch = materialize_accessed_window(
        (0, {"task": target}, ((block,),), (block.src_nodes,), (block.edge_ids,)),
        mode="snapshot",
        num_layers=1,
    )

    assert batch.targets["task"].target_kind == "node"
    assert batch.blocks == ((block,),)


def test_full_snapshot_builds_task_target_before_materialize() -> None:
    row = _snapshot_row(0, edge_ids=torch.tensor([0, 1]))
    store = _snapshot_store(row)

    accessed = access_snapshot_window(
        store,
        store.graph.snapshot_csc_view["slices"],
        split="train",
        window_id=0,
        input_window=range(1),
        chunk_limits=(-1,),
        sampling_policy="full",
    )

    assert accessed is not None
    assert set(accessed[1]) == {"task"}
    assert accessed[1]["task"].target_kind == "node"
    assert len(accessed[2]) == 1


def test_snapshot_cache_uses_the_same_sample_and_materialize_path(monkeypatch) -> None:
    row = _snapshot_row(0, edge_ids=torch.tensor([0, 1]))
    store = _snapshot_store(row)
    target = build_window_task_target(store.labels, 0)
    sampled = []
    caches = []
    block = _root_block(row["dst_nodes"])
    accessed = (0, {"task": target}, ((block,),), (block.src_nodes,), (block.edge_ids,))

    def fake_access(*args, **kwargs):
        sampled.append(
            (kwargs["window_id"], list(kwargs["input_window"]), tuple(kwargs["chunk_limits"]))
        )
        caches.append((kwargs["entry_cache"], kwargs["blob_cache"]))
        return accessed

    monkeypatch.setattr(snapshot_materialize, "access_snapshot_window", fake_access)

    for enabled in (False, True):
        list(
            runtime_loader.DataLoader(
                store,
                mode="snapshot",
                split="train",
                window_policy="chunk_decay",
                sampling_policy="full",
                chunk_decay=None,
                num_full_snapshots=1,
                num_layers=1,
                fanouts=None,
                sampler_options={"rolling_snapshot_cache": enabled},
                num_negatives=0,
                generator=None,
                comm=CommScheduler(),
                device=None,
                prefetch_state=None,
                enabled=False,
            )
        )

    assert sampled == [(0, [0], (-1,)), (0, [0], (-1,))]
    assert caches[0] == (None, None)
    assert isinstance(caches[1][0], dict)
    assert isinstance(caches[1][1], dict)


def test_accessed_window_short_tuple_materializes_directly() -> None:
    block = _root_block(torch.tensor([0]))
    batch = materialize_accessed_window(
        (0, {"task": object()}, ((block,),), (block.src_nodes,), (block.edge_ids,)),
        mode="event",
        num_layers=1,
    )

    assert batch.blocks == ((block,),)
    assert not hasattr(batch, "meta")


def test_event_drop_last_only_removes_an_incomplete_tail() -> None:
    def ids(ptr: torch.Tensor) -> list[int]:
        store = SimpleNamespace(
            graph=SimpleNamespace(
                time_ptr_2=ptr,
                split_time_ptr_2={"train": ptr},
                prepare={"meta": {"time_split": "batch", "target_batch_size": 2}},
            )
        )
        return list(
            event_window_ids(store, "train", drop_last=True)
        )

    assert ids(torch.tensor([[0, 2], [2, 4]])) == [0, 1]
    assert ids(torch.tensor([[0, 2], [2, 3]])) == [0]


def test_snapshot_entry_cache_reuses_full_materialization(monkeypatch) -> None:
    row = _snapshot_row(0, edge_ids=torch.tensor([0, 1]))
    store = _snapshot_store(row)
    created = []

    def fake_materialize(_store, materialize_row, **kwargs):
        entry = snapshot_cache._SnapshotEntry(
            row=materialize_row,
            graph=_root_block(materialize_row["dst_nodes"]),
            features={},
            chunk_limit=kwargs["chunk_limit"],
        )
        created.append(entry)
        return entry

    monkeypatch.setattr(snapshot_cache, "_materialize_snapshot_entry", fake_materialize)
    entry_cache = {}
    blob_cache = {}
    args = dict(
        store=store,
        rows=((row, -1),),
        chunk_order=None,
        comm=None,
        options={},
        entry_cache=entry_cache,
        blob_cache=blob_cache,
    )

    first = snapshot_cache._materialize_snapshot_entries(**args)
    second = snapshot_cache._materialize_snapshot_entries(**args)

    assert len(created) == 1
    assert first[0] is second[0]


def _snapshot_row(snapshot_id: int, *, edge_ids: torch.Tensor) -> dict:
    return {
        "snapshot_id": snapshot_id,
        "src_nodes": torch.tensor([0, 1, 2]),
        "dst_nodes": torch.tensor([1, 2]),
        "edge_ids": edge_ids,
        "indptr": torch.tensor([0, 1, 2]),
        "indices": torch.tensor([0, 1]),
        "ts": torch.tensor([float(snapshot_id + 1)] * 2),
    }


def _snapshot_store(*rows: dict, task_kind: str = "node") -> SimpleNamespace:
    counts = torch.tensor([int(row["edge_ids"].numel()) for row in rows], dtype=torch.long)
    ends = counts.cumsum(0)
    ptr = torch.stack((torch.cat((torch.zeros(1, dtype=torch.long), ends[:-1])), ends), dim=1)
    if task_kind == "edge":
        targets = list(rows[1:]) + [None]
        task_counts = torch.tensor(
            [0 if row is None else int(row["edge_ids"].numel()) for row in targets]
        )
        task_ptr = torch.cat((torch.zeros(1, dtype=torch.long), task_counts.cumsum(0)))
        edge_rows = [row for row in targets if row is not None]
        payload = {
            "src": torch.cat([row["src_nodes"].index_select(0, row["indices"]) for row in edge_rows]),
            "dst": torch.cat([
                row["dst_nodes"].index_select(
                    0,
                    torch.repeat_interleave(
                        torch.arange(int(row["dst_nodes"].numel())),
                        row["indptr"][1:] - row["indptr"][:-1],
                    ),
                )
                for row in edge_rows
            ]),
            "edge_ids": torch.cat([row["edge_ids"] for row in edge_rows]),
            "edge_rows": torch.cat([row["edge_ids"] for row in edge_rows]),
        }
    else:
        task_ptr = torch.cat((torch.zeros(1, dtype=torch.long), torch.full((len(rows),), 2).cumsum(0)))
        payload = {"node_ids": torch.cat([row["dst_nodes"] for row in rows])}
    return SimpleNamespace(
        graph=SimpleNamespace(
            rank=0,
            num_nodes=3,
            prepare=None,
            partition={},
            runtime_cache={},
            time_ptr_2=ptr,
            split_time_ptr_2={"train": ptr},
            snapshot_csc_view={"slices": list(rows)},
        ),
        features=SimpleNamespace(node_ids=None, node_features={}, edge_features={}),
        labels=LabelStore(task_kind=task_kind, task_ptr=task_ptr, task_payload=payload),
    )


def _root_block(nodes: torch.Tensor) -> GraphBlock:
    empty = torch.empty(0, dtype=torch.long)
    return GraphBlock(
        src_nodes=nodes.long(),
        dst_nodes=nodes.long(),
        edge_ids=empty,
        format="csc",
        indptr=torch.zeros(int(nodes.numel()) + 1, dtype=torch.long),
        indices=empty,
        num_src=int(nodes.numel()),
        num_dst=int(nodes.numel()),
    )
