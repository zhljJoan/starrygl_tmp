import pytest
import torch

import starrygl as sg
from starrygl.runtime.snapshot.features import _read_snapshot_features
from starrygl.runtime.snapshot.rows import _slice_csc_prefix
from starrygl.task import attach_target_route, build_window_task_target


ALL_VIEWS = sg.ViewPlan(
    kind="test",
    required_layouts=("event_view", "temporal_csr", "snapshot_csc"),
)


def _prepare_graph_data(data, *, config, node_master, edge_master, hot_node_ids):
    graph = sg.load_graph_data(data)
    partition_plan = sg.partition_graph_data(
        graph,
        config=config,
        node_master=node_master,
        edge_master=edge_master,
        hot_node_ids=hot_node_ids,
    )
    return sg.materialize_graph_data(
        graph,
        config=config,
        partition_plan=partition_plan,
        view_plan=ALL_VIEWS,
    )


def test_graph_store_reads_edges_by_id() -> None:
    store = sg.GraphStore.from_mapping(
        {
            "num_nodes": 4,
            "src": torch.tensor([0, 1, 2]),
            "dst": torch.tensor([1, 2, 3]),
        }
    )

    src, dst = store.edges(torch.tensor([2, 0]))

    assert store.num_nodes == 4
    assert store.num_edges == 3
    assert torch.equal(src, torch.tensor([2, 0]))
    assert torch.equal(dst, torch.tensor([3, 1]))


def test_graph_data_uses_canonical_fields_and_sorts_edge_rows() -> None:
    graph = sg.load_graph_data(
        {
            "src": torch.tensor([2, 0]),
            "dst": torch.tensor([0, 1]),
            "ts": torch.tensor([2.0, 1.0]),
            "edge_feat": torch.tensor([[20.0], [10.0]]),
            "split_labels": torch.tensor([2, 0]),
        }
    )

    assert graph.src.tolist() == [0, 2]
    assert graph.edge_feat.tolist() == [[10.0], [20.0]]
    assert graph.split_labels.tolist() == [0, 2]

    try:
        sg.load_graph_data({"u": [0], "i": [1]})
    except ValueError as exc:
        assert "src" in str(exc)
    else:
        raise AssertionError("legacy field aliases must not enter canonical prepare")


def test_graph_data_normalizes_timestamped_node_labels() -> None:
    graph = sg.load_graph_data(
        {
            "src": torch.tensor([0, 1, 2, 3]),
            "dst": torch.tensor([1, 2, 3, 0]),
            "ts": torch.tensor([10.0, 20.0, 30.0, 40.0]),
            "split_labels": torch.tensor([0, 0, 2, 2]),
            "node_label_dict": {
                9: {0: torch.tensor([1.0, 0.0])},
                29: {2: torch.tensor([0.0, 1.0])},
            },
        }
    )

    assert graph.node_label_nodes.tolist() == [0, 2]
    assert graph.node_label_ts.tolist() == [9.0, 29.0]
    assert graph.node_label_split.tolist() == [0, 2]
    assert graph.node_label.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_graph_data_recognizes_snapshot_scalar_labels_and_temporal_node_axis() -> None:
    graph = sg.load_graph_data(
        {
            "src": torch.tensor([0, 1, 2]),
            "dst": torch.tensor([1, 2, 0]),
            "time_ptr_2": torch.tensor([[0, 1], [1, 2], [2, 3]]),
            "node_feat": torch.zeros(3, 4, 2),
            "node_label": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        }
    )

    assert graph.num_nodes == 4
    assert graph.node_label_temporal


def test_prepare_artifacts_uses_compiled_view_and_new_store_contract(tmp_path, monkeypatch) -> None:
    trainer = sg.compile(
        data_source={
            "src": torch.tensor([0, 1]),
            "dst": torch.tensor([1, 2]),
            "ts": torch.tensor([1.0, 2.0]),
            "num_nodes": 3,
            "node_feat": torch.arange(6, dtype=torch.float32).reshape(3, 2),
            "temporal_representation": "snapshot_sequence",
        },
        backbone={
            "name": "tgcn",
            "temporal_representation": "snapshot_sequence",
            "spatial_aggregation": "full_neighbor",
        },
        task_segment={"name": "node_regression"},
        artifact_root=tmp_path,
    )

    root = trainer.prepare_artifacts(world_size=1)

    assert trainer.plan.partition_plan is not None
    assert trainer.plan.view.required_layouts == ("snapshot_csc",)
    assert (root / "prepare.pt").exists()
    assert (root / "graph_000.pt").exists()
    assert not (root / "graph.pt").exists()
    store = sg.load_starrygl_store(root)
    assert store.graph.snapshot_csc_view
    first = trainer._store(None, artifact_root=None, rank=0, map_location="cpu", mmap=False)
    second = trainer._store(None, artifact_root=None, rank=0, map_location="cpu", mmap=False)
    assert first is second

    monkeypatch.setattr(trainer, "prepare", lambda **_: (_ for _ in ()).throw(AssertionError("unexpected rebuild")))
    assert trainer.prepare_artifacts(world_size=1) == root


def test_prepare_signature_ignores_training_options_and_rejects_changed_data(tmp_path) -> None:
    data = {
        "src": torch.tensor([0, 1]),
        "dst": torch.tensor([1, 2]),
        "ts": torch.tensor([1.0, 2.0]),
        "num_nodes": 3,
        "node_feat": torch.zeros(3, 2),
        "temporal_representation": "snapshot_sequence",
    }
    backbone = {
        "name": "tgcn",
        "temporal_representation": "snapshot_sequence",
        "spatial_aggregation": "full_neighbor",
    }
    trainer = sg.compile(
        data_source=data,
        backbone=backbone,
        task_segment={"name": "node_regression"},
        artifact_root=tmp_path,
    )
    trainer.prepare_artifacts()

    training_change = sg.compile(
        data_source=data,
        backbone=backbone,
        task_segment={"name": "node_regression"},
        artifact_root=tmp_path,
        runtime={"train": {"lr": 0.2}},
    )
    changed_data = dict(data, ts=torch.tensor([1.0, 3.0]))
    stale = sg.compile(
        data_source=changed_data,
        backbone=backbone,
        task_segment={"name": "node_regression"},
        artifact_root=tmp_path,
    )
    changed_task = sg.compile(
        data_source=data,
        backbone=backbone,
        task_segment={"name": "edge_prediction"},
        artifact_root=tmp_path,
    )

    assert training_change._artifacts_ready(tmp_path)
    assert not stale._artifacts_ready(tmp_path)
    assert not changed_task._artifacts_ready(tmp_path)
    with pytest.raises(ValueError, match="do not match"):
        stale._store(None, artifact_root=None, rank=0, map_location="cpu", mmap=False)


def test_snapshot_csc_layout_packs_only_temporal_node_features(tmp_path) -> None:
    temporal_x = torch.tensor(
        [
            [[1.0], [2.0], [3.0]],
            [[10.0], [20.0], [30.0]],
        ]
    )
    trainer = sg.compile(
        data_source={
            "src": torch.tensor([0, 1]),
            "dst": torch.tensor([1, 2]),
            "ts": torch.tensor([1.0, 2.0]),
            "time_ptr_2": torch.tensor([[0, 1], [1, 2]]),
            "node_feat": temporal_x,
            "edge_feat": torch.tensor([[4.0], [5.0]]),
            "temporal_representation": "snapshot_sequence",
        },
        backbone={
            "name": "tgcn",
            "temporal_representation": "snapshot_sequence",
            "spatial_aggregation": "full_neighbor",
        },
        task_segment={"name": "node_regression"},
        artifact_root=tmp_path,
        runtime={"preprocess": {"feature_layout": "snapshot_csc"}},
    )

    root = trainer.prepare_artifacts()
    graph_shard = torch.load(root / "graph_000.pt", weights_only=False)
    feature_shard = torch.load(root / "feature_000.pt", weights_only=False)
    packed_x = graph_shard["snapshot_csc_view"]["node_data"]["x"]

    assert packed_x["ptr"].tolist() == [0, 3, 6]
    assert torch.equal(packed_x["data"], temporal_x.flatten(0, 1))
    assert "node_feat" not in feature_shard
    assert feature_shard["edge_feat"].tolist() == [[4.0], [5.0]]

    store = sg.load_starrygl_store(root)
    rows = store.graph.snapshot_csc_view["slices"]
    assert bool(getattr(rows, "_by_snapshot_id", False))
    assert "target_edge_rows" not in rows[0]
    assert store.labels.task_ptr.tolist() == [0, 3, 6]
    assert store.labels.task_slice(1)["node_ids"].tolist() == [0, 1, 2]
    features, remote, pending = _read_snapshot_features(store, rows[1])
    assert torch.equal(features["x"], temporal_x[1])
    assert not remote and pending is None
    prefix_features, _, _ = _read_snapshot_features(store, _slice_csc_prefix(rows[1], 2))
    assert torch.equal(prefix_features["x"], temporal_x[1, :2])
    assert "route" in rows[0]
    assert not store.features.node_features

    static_root = tmp_path / "static"
    prepared = trainer.prepare()
    static_features = sg.build_static_feature_shards(
        prepared=prepared,
        node_feat=temporal_x[0],
    )
    sg.write_prepare_artifacts(
        static_root,
        prepared=prepared,
        feature_shards=static_features,
        feature_layout="snapshot_csc",
    )
    static_graph = torch.load(static_root / "graph_000.pt", weights_only=False)
    static_feature = torch.load(static_root / "feature_000.pt", weights_only=False)
    assert "node_data" not in static_graph["snapshot_csc_view"]
    assert torch.equal(static_feature["node_feat"], temporal_x[0])


def test_feature_manager_reads_node_and_edge_features() -> None:
    manager = sg.FeatureManager(
        node_features={"x": torch.arange(12, dtype=torch.float32).reshape(4, 3)},
        edge_features={"w": torch.arange(3, dtype=torch.float32).reshape(3, 1)},
    )

    node = manager.read_nodes(torch.tensor([3, 1]), names=("x",))
    edge = manager.read_edges(torch.tensor([2, 0]), names=("w",))

    assert torch.equal(node["x"], torch.tensor([[9.0, 10.0, 11.0], [3.0, 4.0, 5.0]]))
    assert torch.equal(edge["w"], torch.tensor([[2.0], [0.0]]))


def test_static_edge_feature_replica_is_locally_readable() -> None:
    prepared = _prepare_graph_data(
        {"src": torch.tensor([0, 1]), "dst": torch.tensor([1, 2]), "num_nodes": 3},
        config=sg.PrepareConfig(world_size=2, chunks_per_rank=1),
        node_master=torch.tensor([0, 0, 1]),
        edge_master=torch.tensor([0, 1]),
        hot_node_ids=torch.empty(0, dtype=torch.long),
    )
    shards = sg.build_static_feature_shards(
        prepared=prepared,
        edge_feat=torch.tensor([[3.0], [4.0]]),
        replicate_edge_features=True,
    )

    for shard in shards:
        manager = sg.FeatureManager.from_shard(shard)
        assert manager.edge_features_replicated
        assert torch.equal(manager.read_edges(torch.tensor([1, 0]))["edge"], torch.tensor([[4.0], [3.0]]))


def test_feature_manager_row_map_reads_logical_node_and_edge_ids() -> None:
    manager = sg.FeatureManager(
        node_features={"x": torch.tensor([[10.0], [20.0]], dtype=torch.float32)},
        edge_features={"w": torch.tensor([[30.0], [40.0]], dtype=torch.float32)},
        node_row_map=torch.tensor([-1, 1, -1, 0], dtype=torch.long),
        edge_row_map=torch.tensor([1, -1, -1, 0], dtype=torch.long),
    )

    node = manager.read_nodes(torch.tensor([3, 1]), names=("x",))
    edge = manager.read_edges(torch.tensor([3, 0]), names=("w",))

    assert torch.equal(node["x"], torch.tensor([[10.0], [20.0]]))
    assert torch.equal(edge["w"], torch.tensor([[30.0], [40.0]]))


def test_snapshot_csc_separates_compute_and_edge_target_owners() -> None:
    prepared = _prepare_graph_data(
        {
            "src": torch.tensor([0, 1]),
            "dst": torch.tensor([1, 2]),
            "ts": torch.tensor([1.0, 2.0]),
            "edge_ids": torch.tensor([10, 11]),
            "num_nodes": 3,
        },
        config=sg.PrepareConfig(
            world_size=2,
            chunks_per_rank=1,
            time_ptr_2=torch.tensor([[0, 1], [1, 2]]),
        ),
        node_master=torch.tensor([0, 0, 1]),
        edge_master=torch.tensor([1, 0]),
        hot_node_ids=torch.empty(0, dtype=torch.long),
    )

    rank0 = prepared.snapshot_csc_views[0]["slices"][0]
    rank1 = prepared.snapshot_csc_views[1]["slices"][1]

    assert rank0["edge_ids"].tolist() == [10]
    assert rank1["edge_ids"].tolist() == [11]
    assert "target_edge_rows" not in rank0

    task_shards = sg.build_label_shards(
        prepared=prepared,
        task="edge_prediction",
        temporal="snapshot",
        src=torch.tensor([0, 1]),
        dst=torch.tensor([1, 2]),
        ts=torch.tensor([1.0, 2.0]),
        edge_ids=torch.tensor([10, 11]),
    )

    store = sg.StoreBundle(
        graph=sg.GraphStore.from_prepare(prepared.as_dict(), rank=0),
        features=sg.FeatureManager(),
        labels=sg.LabelStore.from_shard(task_shards[0]),
    )
    target = build_window_task_target(store.labels, 0)
    graph = sg.GraphBlock(
        src_nodes=rank0["src_nodes"],
        dst_nodes=rank0["dst_nodes"],
        edge_ids=rank0["edge_ids"],
        format="csc",
        indptr=rank0["indptr"],
        indices=rank0["indices"],
        cache={"node_dist_index": prepared.partition["node_dist_index"]},
    )
    target = attach_target_route(graph, target, collect_remote_endpoints=True)

    assert target.target_ids.tolist() == [11]
    assert target.target_route.endpoint_collect.remote_endpoint_nodes.tolist() == [2]


def test_build_static_feature_shards_include_hot_and_owned_features_by_default() -> None:
    prepared = _prepare_graph_data(
        {"src": torch.tensor([2, 1]), "dst": torch.tensor([0, 2]), "num_nodes": 3},
        config=sg.PrepareConfig(world_size=2, chunks_per_rank=1, time_ptr_2=torch.tensor([[0, 2]])),
        node_master=torch.tensor([0, 0, 1]),
        edge_master=torch.tensor([0, 1]),
        hot_node_ids=torch.tensor([2]),
    )
    node_feat = torch.tensor([[0.0], [10.0], [20.0]])
    edge_feat = torch.tensor([[100.0], [200.0]])

    shards = sg.build_static_feature_shards(
        prepared=prepared,
        node_feat=node_feat,
        edge_feat=edge_feat,
    )

    rank0, rank1 = shards
    assert rank0["node_ids"].tolist() == [2, 0, 1]
    assert rank0["node_row_map"].tolist() == [1, 2, 0]
    assert rank0["node_feat"].tolist() == [[20.0], [0.0], [10.0]]
    assert rank0["edge_ids"].tolist() == [0]
    assert rank0["edge_row_map"].tolist() == [0, -1]
    assert rank0["edge_feat"].tolist() == [[100.0]]
    assert rank1["node_ids"].tolist() == [2]
    assert rank1["node_row_map"].tolist() == [-1, -1, 0]
    assert rank1["node_feat"].tolist() == [[20.0]]
    assert rank1["edge_ids"].tolist() == [1]
    assert rank1["edge_row_map"].tolist() == [-1, 0]
    assert rank1["edge_feat"].tolist() == [[200.0]]


def test_build_static_feature_shards_can_include_static_one_hop_nodes() -> None:
    prepared = _prepare_graph_data(
        {"src": torch.tensor([2, 1]), "dst": torch.tensor([0, 2]), "num_nodes": 3},
        config=sg.PrepareConfig(world_size=2, chunks_per_rank=1, time_ptr_2=torch.tensor([[0, 2]])),
        node_master=torch.tensor([0, 0, 1]),
        edge_master=torch.tensor([0, 1]),
        hot_node_ids=torch.tensor([2]),
    )

    shards = sg.build_static_feature_shards(
        prepared=prepared,
        include_static_one_hop=True,
    )

    assert shards[0]["node_ids"].tolist() == [2, 0, 1]
    assert shards[0]["node_row_map"].tolist() == [1, 2, 0]
    assert shards[1]["node_ids"].tolist() == [2, 1]
    assert shards[1]["node_row_map"].tolist() == [-1, 1, 0]


def test_temporal_feature_and_scalar_label_shards_share_logical_node_rows() -> None:
    prepared = _prepare_graph_data(
        {
            "src": torch.tensor([0, 1, 2]),
            "dst": torch.tensor([1, 2, 3]),
            "num_nodes": 4,
        },
        config=sg.PrepareConfig(
            world_size=2,
            chunks_per_rank=1,
            time_ptr_2=torch.tensor([[0, 1], [1, 2], [2, 3]]),
        ),
        node_master=torch.tensor([0, 1, 0, 1]),
        edge_master=torch.tensor([0, 1, 0]),
        hot_node_ids=torch.tensor([3]),
    )
    node_feat = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
    node_label = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    features = sg.build_static_feature_shards(prepared=prepared, node_feat=node_feat)
    labels = sg.build_label_shards(
        prepared=prepared,
        node_label=node_label,
        node_label_temporal=True,
    )

    rank0_features = sg.FeatureManager.from_shard(features[0])
    rank0_labels = sg.LabelStore.from_shard(labels[0])
    assert torch.equal(
        rank0_features.read_nodes_at(torch.tensor([3, 2]), 1)["x"],
        node_feat[1, torch.tensor([3, 2])],
    )
    assert torch.equal(
        rank0_labels.read_node_rows_at(torch.tensor([1, 2]), 1),
        node_label[1, torch.tensor([0, 2])],
    )


def test_build_label_shards_write_labels_by_owner_loc() -> None:
    prepared = _prepare_graph_data(
        {"src": torch.tensor([0, 1]), "dst": torch.tensor([1, 2]), "num_nodes": 3},
        config=sg.PrepareConfig(world_size=2, chunks_per_rank=1, time_ptr_2=torch.tensor([[0, 2]])),
        node_master=torch.tensor([0, 0, 1]),
        edge_master=torch.tensor([0, 1]),
        hot_node_ids=torch.tensor([2]),
    )
    node_label = torch.tensor([[0.0], [10.0], [20.0]])
    edge_label = torch.tensor([[100.0], [200.0]])

    shards = sg.build_label_shards(
        prepared=prepared,
        node_label=node_label,
        edge_label=edge_label,
    )

    rank0, rank1 = shards
    assert rank0["node_label_ids"].tolist() == [0, 1]
    assert rank0["node_label"].tolist() == [[0.0], [0.0], [10.0]]
    assert rank0["edge_label"].tolist() == [[100.0]]
    assert rank1["node_label_ids"].tolist() == [2]
    assert rank1["node_label"].tolist() == [[20.0]]
    assert rank1["edge_label"].tolist() == [[200.0]]


def test_load_starrygl_store_reads_rank_shards(tmp_path) -> None:
    prepared = _prepare_graph_data(
        {
            "src": torch.tensor([0, 1]),
            "dst": torch.tensor([1, 2]),
            "ts": torch.tensor([1.0, 2.0]),
            "num_nodes": 3,
        },
        config=sg.PrepareConfig(world_size=2, chunks_per_rank=1, time_ptr_2=torch.tensor([[0, 2]])),
        node_master=torch.tensor([0, 0, 1]),
        edge_master=torch.tensor([0, 1]),
        hot_node_ids=torch.tensor([2]),
    )
    feature_shards = sg.build_static_feature_shards(
        prepared=prepared,
        node_feat=torch.tensor([[0.0], [10.0], [20.0]]),
        edge_feat=torch.tensor([[100.0], [200.0]]),
    )
    label_shards = sg.build_label_shards(
        prepared=prepared,
        node_label=torch.tensor([[0.0], [1.0], [2.0]]),
        edge_label=torch.tensor([[3.0], [4.0]]),
    )
    sg.write_prepare_artifacts(
        tmp_path,
        prepared=prepared,
        feature_shards=feature_shards,
        label_shards=label_shards,
    )

    bundle = sg.load_starrygl_store(tmp_path, rank=1)

    assert bundle.graph.rank == 1
    assert bundle.graph.world_size == 2
    assert bundle.graph.time_ptr_2.tolist() == [[0, 2]]
    assert bundle.graph.split_time_ptr_2["train"].tolist() == [[0, 2]]
    assert bundle.graph.split_masks["train"].tolist() == [True, True]
    assert bundle.graph.event_view["edge_ids"].tolist() == [1]
    assert bundle.graph.event_view["split_window_end_ts"]["train"].tolist() == [2.0]
    assert bundle.graph.snapshot_csc_view["rank"] == 1
    assert bundle.features.node_ids.tolist() == [2]
    assert bundle.features.read_nodes(torch.tensor([2]))["x"].tolist() == [[20.0]]
    assert bundle.features.edge_ids.tolist() == [1]
    assert bundle.features.read_edges(torch.tensor([1]))["edge"].tolist() == [[200.0]]
    assert bundle.labels.read_node_rows(torch.tensor([0])).tolist() == [[2.0]]
    assert bundle.labels.read_edge_rows(torch.tensor([0])).tolist() == [[4.0]]


def test_state_manager_reads_and_commits_state_delta() -> None:
    manager = sg.StateManager(values=torch.zeros(4, 2))

    manager.commit(
        sg.StateDelta(
            node_ids=torch.tensor([1, 3]),
            values=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            timestamps=torch.tensor([10.0, 30.0]),
        )
    )
    read = manager.read(torch.tensor([3, 1]))

    assert torch.equal(read.values, torch.tensor([[3.0, 4.0], [1.0, 2.0]]))
    assert read.timestamps is not None
    assert torch.equal(read.timestamps, torch.tensor([30.0, 10.0]))


def test_state_manager_row_map_reads_and_commits_logical_node_ids() -> None:
    manager = sg.StateManager(
        values=torch.zeros(2, 1),
        timestamps=torch.zeros(2),
        row_map=torch.tensor([-1, 1, -1, 0], dtype=torch.long),
    )

    manager.commit(
        sg.StateDelta(
            node_ids=torch.tensor([1, 3]),
            values=torch.tensor([[2.0], [4.0]]),
            timestamps=torch.tensor([5.0, 7.0]),
        )
    )
    read = manager.read(torch.tensor([3, 1]))

    assert torch.equal(read.values, torch.tensor([[4.0], [2.0]]))
    assert read.timestamps is not None
    assert torch.equal(read.timestamps, torch.tensor([7.0, 5.0]))


def test_mailbox_manager_row_map_reads_and_appends_logical_node_ids() -> None:
    manager = sg.MailboxManager(
        values=torch.zeros(2, 1, 2),
        row_map=torch.tensor([-1, 1, -1, 0], dtype=torch.long),
    )

    manager.append(
        torch.tensor([1, 3]),
        torch.tensor([[2.0, 2.5], [4.0, 4.5]]),
        torch.tensor([5.0, 7.0]),
    )
    read = manager.read(torch.tensor([3, 1]))

    assert torch.equal(read.values[:, 0], torch.tensor([[4.0, 4.5], [2.0, 2.5]]))
    assert torch.equal(read.timestamps[:, 0], torch.tensor([7.0, 5.0]))


def test_state_manager_reset_clears_values_and_timestamps() -> None:
    manager = sg.StateManager(values=torch.ones(3, 2))
    manager.commit(
        sg.StateDelta(
            node_ids=torch.tensor([0, 2]),
            values=torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
            timestamps=torch.tensor([1.0, 2.0]),
        )
    )

    manager.reset()

    assert torch.equal(manager.values, torch.zeros(3, 2))
    assert manager.timestamps is not None
    assert torch.equal(manager.timestamps, torch.zeros(3))
