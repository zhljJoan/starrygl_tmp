import torch
import pytest

import starrygl as sg


def test_core_skeleton_compile_exposes_observable_plan() -> None:
    trainer = sg.compile(
        data_source={"source": "wiki"},
        backbone={"name": "tgn"},
        task_segment=sg.EdgePrediction(),
    )

    assert isinstance(trainer.plan, sg.ExecutionPlan)
    assert trainer.plan.view.kind == "temporal_sampling"
    assert trainer.plan.view.required_layouts == ("event_view", "temporal_csr")
    assert trainer.plan.owner_policy == "edge_owner"
    assert trainer.plan.comm_mode == "collective_epoch"
    assert "execution_spine='temporal_sampling'" in trainer.plan.explain()
    assert trainer.plan.await_dependencies


def test_snapshot_full_graph_defaults_to_csc_layout() -> None:
    trainer = sg.compile(
        data_source={"source": "snapshots", "temporal_representation": "snapshot_sequence"},
        backbone={"name": "tgcn", "spatial_aggregation": "full_neighbor"},
        task_segment=sg.NodeRegression(),
    )

    assert trainer.plan.storage_view == "snapshot_block_view"
    assert trainer.plan.view.required_layouts == ("snapshot_csc",)


def test_event_defaults_to_temporal_sampling_layout() -> None:
    trainer = sg.compile(
        data_source={"source": "events", "temporal_representation": "event_stream"},
        backbone={"name": "stateless_gnn", "spatial_aggregation": "full_neighbor"},
        task_segment=sg.NodeClassification(),
    )

    assert trainer.plan.storage_view == "temporal_sampling_view"
    assert trainer.plan.view.required_layouts == ("event_view", "temporal_csr")


def test_core_skeleton_batch_and_graph_block_contract() -> None:
    block = sg.GraphBlock(
        src_nodes=torch.tensor([0, 1]),
        dst_nodes=torch.tensor([1]),
        edge_ids=torch.tensor([10]),
        format="csc",
        indptr=torch.tensor([0, 1]),
        indices=torch.tensor([0]),
    )
    batch = sg.Batch(mode="sampled", blocks=((block,),), features={"x": torch.ones(2, 4)})

    assert block.num_edges == 1
    assert batch.blocks == ((block,),)
    assert not hasattr(batch, "mfgs")
    assert list(batch.layer_blocks()) == [block]


def test_batch_supports_sampled_and_full_graph_inputs() -> None:
    mfg0 = sg.GraphBlock(
        src_nodes=torch.tensor([0, 1]),
        dst_nodes=torch.tensor([1]),
        edge_ids=torch.tensor([10]),
        format="csc",
    )
    mfg1 = sg.GraphBlock(
        src_nodes=torch.tensor([1, 2]),
        dst_nodes=torch.tensor([2]),
        edge_ids=torch.tensor([11]),
        format="csc",
    )
    full_graph = sg.GraphBlock(
        src_nodes=torch.tensor([0, 1, 2]),
        dst_nodes=torch.tensor([0, 1, 2]),
        edge_ids=torch.tensor([20, 21]),
        format="csc",
    )

    sampled_element = sg.Batch(mode="sampled", blocks=((mfg0, mfg1),), num_layers=2)
    full_graph_element = sg.Batch(mode="snapshot", graph=full_graph, num_layers=2)
    assert list(sampled_element.layer_blocks()) == [mfg0, mfg1]
    assert list(full_graph_element.layer_blocks()) == [full_graph, full_graph]


def test_core_skeleton_model_contract_consumes_batch() -> None:
    class ToyModel(sg.StarryModel):
        def encode(self, batch: sg.Batch) -> sg.ModelOutput:
            return sg.ModelOutput(embeddings=batch.features["x"] + 1)

    block = sg.GraphBlock(
        src_nodes=torch.tensor([0]),
        dst_nodes=torch.tensor([0]),
        edge_ids=torch.tensor([0]),
        format="csr",
    )
    batch = sg.Batch(mode="snapshot", graph=block, features={"x": torch.ones(1, 2)})
    output = ToyModel().encode(batch)

    assert torch.equal(output.embeddings, torch.full((1, 2), 2.0))
    assert output.commit_embeddings is output.embeddings


def test_from_config_uses_top_level_artifact_root_not_runtime() -> None:
    trainer = sg.from_config(
        {
            "data": {"source": "wiki"},
            "backbone": {"name": "tgn"},
            "task": {"name": "edge_prediction"},
            "runtime": {"device": "cuda", "profile": False},
            "artifact_root": "logs/artifacts/wiki",
        }
    )

    assert str(trainer.artifact_root) == "logs/artifacts/wiki"
    assert trainer.runtime_config["device"] == "cuda"
    assert trainer.runtime_config["profile"] is False
    assert trainer.to_config()["artifact_root"] == "logs/artifacts/wiki"
    assert "artifact_root" not in trainer.to_config()["runtime"]


def test_from_config_rejects_runtime_artifact_root() -> None:
    with pytest.raises(ValueError, match="artifact_root belongs at config top level"):
        sg.from_config(
            {
                "data": {"source": "events"},
                "backbone": {"name": "stateless_gnn"},
                "task": {"name": "node_classification"},
                "runtime": {"artifact_root": "logs/artifacts/wiki"},
            }
        )
