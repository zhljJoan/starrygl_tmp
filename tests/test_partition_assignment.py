import torch

import starrygl as sg
import starrygl.runtime.trainer as trainer_module
from starrygl.partition import PartitionConfig, PartitionPlan, partition_graph


def test_partition_graph_preserves_explicit_node_and_edge_masters() -> None:
    result = partition_graph(
        src=torch.tensor([0, 1, 2, 3]),
        dst=torch.tensor([1, 2, 3, 0]),
        ts=torch.arange(4, dtype=torch.float32),
        num_nodes=4,
        config=PartitionConfig(num_parts=2, chunks_per_rank=1, backend="round_robin"),
        node_master=torch.tensor([0, 0, 1, 1]),
        edge_master=torch.tensor([1, 0, 1, 0]),
        hot_node_ids=torch.tensor([1, 3]),
    )

    assert isinstance(result, PartitionPlan)
    assert result.node_master.tolist() == [0, 0, 1, 1]
    assert result.edge_master.tolist() == [1, 0, 1, 0]
    assert result.shared_nodes.tolist() == [1, 3]
    assert result.chunk_table.chunk_ids.tolist() == [0, 0, 1, 1]
    assert result.chunk_table.values["edge_chunk"].tolist() == [0, 1, 1, 0]


def test_partition_graph_derives_edge_master_from_destination_owner() -> None:
    dst = torch.tensor([1, 2, 3, 0])
    result = partition_graph(
        src=torch.tensor([0, 1, 2, 3]),
        dst=dst,
        ts=None,
        num_nodes=4,
        config=PartitionConfig(num_parts=2, chunks_per_rank=1, backend="round_robin"),
    )

    assert result.node_master.tolist() == [0, 1, 0, 1]
    assert torch.equal(result.edge_master, result.node_master.index_select(0, dst))


def test_trainer_prepare_uses_canonical_partition_config(monkeypatch) -> None:
    captured = {}
    original = trainer_module.partition_graph_data

    def capture_partition_graph_data(graph, *, config, **kwargs):
        captured["config"] = config
        return original(graph, config=config, **kwargs)

    monkeypatch.setattr(trainer_module, "partition_graph_data", capture_partition_graph_data)
    trainer = sg.compile(
        data_source={"source": "unused"},
        backbone={"name": "tgn"},
        task_segment=sg.EdgePrediction(),
        runtime={
            "preprocess": {
                "num_parts": 3,
                "chunks_per_rank": 4,
                "hot_node_ratio": 0.25,
                "partition_backend": "round_robin",
            }
        },
    )
    assert trainer.plan.partition_plan is None

    trainer.prepare(
        graph={
            "src": torch.tensor([0, 1]),
            "dst": torch.tensor([1, 2]),
            "ts": torch.tensor([0.0, 1.0]),
            "num_nodes": 3,
        }
    )

    config = captured["config"]
    assert config.world_size == 3
    assert config.chunks_per_rank == 4
    assert config.speed_partition_topk_ratio == 0.25
    assert trainer.plan.partition_plan is not None
    assert trainer.plan.partition_plan.metadata["num_parts"] == 3
    assert trainer.plan.partition_plan.metadata["chunks_per_rank"] == 4


def test_materialize_graph_views_consumes_existing_partition_plan() -> None:
    src = torch.tensor([0, 1])
    dst = torch.tensor([1, 2])
    plan = partition_graph(
        src=src,
        dst=dst,
        ts=torch.tensor([0.0, 1.0]),
        num_nodes=3,
        config=PartitionConfig(num_parts=2, backend="round_robin"),
    )

    prepared = sg.materialize_graph_views(
        src=src,
        dst=dst,
        ts=torch.tensor([0.0, 1.0]),
        num_nodes=3,
        config=sg.PrepareConfig(
            world_size=2,
            time_ptr_2=torch.tensor([[0, 2]]),
        ),
        partition_plan=plan,
        view_plan=sg.ViewPlan(kind="none", required_layouts=()),
    )

    assert prepared.partition["node_to_chunk"].tolist() == [0, 1, 0]
    assert prepared.partition["edge_chunk"].tolist() == [1, 0]
    assert prepared.event_views == []
    assert prepared.temporal_csr_view == {}
    assert prepared.snapshot_csc_views == []
    assert prepared.split_masks["train"].tolist() == [True, True]
    assert prepared.split_masks["val"].tolist() == [False, False]
    assert prepared.split_masks["test"].tolist() == [False, False]


def test_event_negative_pool_follows_node_replicas_not_edge_owners() -> None:
    src = torch.tensor([0, 1, 2, 3])
    dst = torch.tensor([0, 1, 2, 3])
    plan = partition_graph(
        src=src,
        dst=dst,
        ts=torch.arange(4, dtype=torch.float32),
        num_nodes=4,
        config=PartitionConfig(num_parts=2, backend="round_robin"),
        node_master=torch.tensor([0, 0, 1, 1]),
        edge_master=torch.tensor([1, 1, 0, 0]),
        hot_node_ids=torch.tensor([3]),
    )

    prepared = sg.materialize_graph_views(
        src=src,
        dst=dst,
        ts=torch.arange(4, dtype=torch.float32),
        config=sg.PrepareConfig(
            world_size=2,
            time_ptr_2=torch.tensor([[0, 4]]),
            include_state_write_routes=False,
        ),
        partition_plan=plan,
        view_plan=sg.ViewPlan(kind="event", required_layouts=("event_view",)),
    )

    assert prepared.event_views[0]["dst_pool"].tolist() == [0, 1, 3]
    assert prepared.event_views[1]["dst_pool"].tolist() == [2, 3]


def test_materialize_graph_views_preserves_split_label_masks() -> None:
    src = torch.tensor([0, 1, 2])
    dst = torch.tensor([1, 2, 0])
    partition = partition_graph(
        src=src,
        dst=dst,
        ts=torch.arange(3, dtype=torch.float32),
        num_nodes=3,
        config=PartitionConfig(num_parts=1, backend="round_robin"),
    )
    prepared = sg.materialize_graph_views(
        src=src,
        dst=dst,
        ts=torch.arange(3, dtype=torch.float32),
        split_labels=torch.tensor([0, 1, 2]),
        config=sg.PrepareConfig(world_size=1, time_ptr_2=torch.tensor([[0, 1], [1, 2], [2, 3]])),
        partition_plan=partition,
        view_plan=sg.ViewPlan(kind="none", required_layouts=()),
    )

    assert prepared.split_masks["train"].tolist() == [True, False, False]
    assert prepared.split_masks["val"].tolist() == [False, True, False]
    assert prepared.split_masks["test"].tolist() == [False, False, True]


def test_snapshot_csc_gcn_norm_preserves_edge_weights() -> None:
    src = torch.tensor([0, 2])
    dst = torch.tensor([1, 1])
    partition = partition_graph(
        src=src,
        dst=dst,
        ts=torch.arange(2, dtype=torch.float32),
        num_nodes=3,
        config=PartitionConfig(num_parts=1, backend="round_robin"),
    )

    prepared = sg.materialize_graph_views(
        src=src,
        dst=dst,
        ts=torch.arange(2, dtype=torch.float32),
        edge_weight=torch.tensor([2.0, 1.0]),
        config=sg.PrepareConfig(world_size=1, time_ptr_2=torch.tensor([[0, 2]])),
        partition_plan=partition,
        view_plan=sg.ViewPlan(kind="snapshot", required_layouts=("snapshot_csc",)),
    )
    row = prepared.snapshot_csc_views[0]["slices"][0]

    assert torch.allclose(
        row["edge_gcn_norm"],
        torch.tensor([2.0 / (12.0**0.5), 1.0 / (8.0**0.5)]),
    )


def test_compile_lowers_view_requirements_from_semantics() -> None:
    event_sampled = sg.compile(
        data_source={"source": "events", "temporal_representation": "event_stream"},
        backbone={"name": "tgn", "spatial_aggregation": "sampled_neighbor"},
        task_segment=sg.EdgePrediction(),
    ).plan.view
    snapshot_default = sg.compile(
        data_source={"source": "snapshots", "temporal_representation": "snapshot_sequence"},
        backbone={"name": "gconv_gru", "spatial_aggregation": "sampled_neighbor"},
        task_segment={"name": "node_prediction"},
    ).plan.view

    assert event_sampled.required_layouts == ("event_view", "temporal_csr")
    assert event_sampled.temporal_csr_bidirectional is True
    assert snapshot_default.required_layouts == ("snapshot_csc", "snapshot_hot_compute")
