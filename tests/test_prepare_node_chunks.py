from types import SimpleNamespace

import pytest
import torch

import starrygl as sg
import starrygl.partition.build as partition_module
from starrygl.partition import PartitionConfig, PartitionPlan, partition_graph


def graph_data():
    nodes = torch.arange(12)
    return {
        "temporal_representation": "snapshot_sequence",
        "num_nodes": 12,
        "src": nodes.repeat(3),
        "dst": ((nodes + 1) % 12).repeat(3),
        "ts": torch.arange(3, dtype=torch.float32).repeat_interleave(12),
        "time_ptr_2": torch.tensor([[0, 12], [12, 24], [24, 36]]),
        "node_feat": torch.arange(24, dtype=torch.float32).reshape(12, 2),
        "node_label": torch.arange(12, dtype=torch.float32),
        "edge_weight": torch.linspace(0.5, 2.0, 36),
    }


def assignments():
    owners = torch.arange(12) % 4
    chunks = owners * 4 + torch.tensor([3, 1, 0, 2, 0, 3, 2, 0, 1, 0, 3, 1])
    return owners, chunks


def trainer_for(chunks=None):
    owners, _ = assignments()
    preprocess = {
        "num_parts": 4, "chunks_per_rank": 4, "partition_backend": "round_robin",
        "node_master_source": owners,
    }
    if chunks is not None:
        preprocess["node_to_chunk_source"] = chunks
    return sg.compile(
        data_source=graph_data(),
        backbone={"name": "tgcn", "spatial_aggregation": "full_neighbor"},
        task_segment={"name": "node_regression", "loss": "mse"},
        runtime={"preprocess": preprocess, "temporal_state": {"consistency": "exact"}},
    )


def legacy_signature(trainer):
    # Capture the previous, three-input signature contract independently.
    from starrygl.prepare import PREPARE_FORMAT
    from starrygl.store.artifact import artifact_fingerprint
    config = trainer._prepare_config(world_size=4)
    return artifact_fingerprint({
        "format": PREPARE_FORMAT,
        "data": trainer._prepare_source(None),
        "prepare": {k: v for k, v in vars(config).items() if k != "profile_prepare"},
        "view": trainer.plan.view.as_dict(),
        "task": trainer.task.get("name"),
        "temporal": trainer.spec.temporal,
        "partition_inputs": (assignments()[0], None, None),
        "feature_layout": "separate",
        "include_static_one_hop": trainer._include_static_one_hop_default(has_snapshot_view=True),
    })


def test_supplied_chunks_reach_four_rank_prepare_and_csc(monkeypatch):
    def unexpected_repartition(**kwargs):
        raise AssertionError("supplied chunks must bypass within-owner partitioning")
    monkeypatch.setattr(partition_module, "_assign_chunks_within_master", unexpected_repartition)
    owners, chunks = assignments()
    trainer = trainer_for()
    prepared = trainer.prepare(node_to_chunk=chunks)
    assert torch.equal(trainer.plan.partition_plan.chunk_table.chunk_ids, chunks)
    assert torch.equal(prepared.partition["node_to_chunk"], chunks)
    assert torch.equal(PartitionPlan.from_artifact(prepared.partition).node_master, owners)
    assert torch.equal(prepared.partition["edge_chunk"], chunks[graph_data()["dst"]])
    assert len(prepared.snapshot_csc_views) == 4
    for rank, view in enumerate(prepared.snapshot_csc_views):
        assert len(view["slices"]) == 3
        for row in view["slices"]:
            assert torch.equal(row["node_chunk"], chunks[row["src_nodes"]])
            assert torch.all(owners[row["dst_nodes"]] == rank)


def test_imported_chunks_preserve_owner_edges_and_normalization():
    owners, chunks = assignments()
    original = trainer_for().prepare()
    imported = trainer_for(chunks).prepare()
    for key in ("node_dist_index", "edge_dist_index", "hot_node_ids"):
        torch.testing.assert_close(original.partition[key], imported.partition[key], rtol=0, atol=0)
    for before, after in zip(original.snapshot_csc_views, imported.snapshot_csc_views):
        for left, right in zip(before["slices"], after["slices"]):
            assert torch.equal(left["dst_nodes"], right["dst_nodes"])
            assert torch.equal(left["self_gcn_norm"], right["self_gcn_norm"])
            lp, rp = left["edge_ids"].argsort(), right["edge_ids"].argsort()
            assert torch.equal(left["edge_ids"][lp], right["edge_ids"][rp])
            torch.testing.assert_close(left["edge_gcn_norm"][lp], right["edge_gcn_norm"][rp], rtol=0, atol=0)


def test_source_roundtrip_strict_fingerprint_and_direct_override(tmp_path):
    _, chunks = assignments()
    source = tmp_path / "chunks.pt"
    torch.save(chunks, source)
    trainer = trainer_for(str(source))
    trainer = sg.from_config(trainer.to_config())
    assert trainer.preprocess_config["node_to_chunk_source"] == str(source)
    root = tmp_path / "prepared"
    imported = trainer.prepare(save=True, artifact_root=root)
    assert trainer._artifacts_ready(root, world_size=4)
    for rank in range(4):
        store = trainer._store(None, artifact_root=root, rank=rank, map_location="cpu", mmap=True)
        assert torch.equal(store.graph.partition["node_to_chunk"], chunks)
    changed = chunks.clone()
    changed[0] = 2
    torch.save(changed, source)
    assert not trainer._artifacts_ready(root, world_size=4)
    with pytest.raises(ValueError, match="do not match"):
        trainer._validate_artifact_store(store)
    with pytest.raises(ValueError, match="do not match"):
        trainer_for()._validate_artifact_store(store)
    overridden = trainer.prepare(node_to_chunk=chunks, save=True, artifact_root=tmp_path / "override")
    assert overridden.meta["artifact_signature"] == imported.meta["artifact_signature"]


def test_no_input_keeps_legacy_signature_and_loads_old_contract(tmp_path):
    trainer = trainer_for()
    signature = legacy_signature(trainer)
    prepared = trainer.prepare(save=True, artifact_root=tmp_path)
    assert prepared.meta["artifact_signature"] == signature
    assert trainer._artifacts_ready(tmp_path, world_size=4)
    for rank in range(4):
        trainer._store(None, artifact_root=tmp_path, rank=rank, map_location="cpu", mmap=True)
    legacy_store = SimpleNamespace(graph=SimpleNamespace(prepare={"meta": {"artifact_signature": signature}}))
    trainer._validate_artifact_store(legacy_store)


@pytest.mark.parametrize("bad, message", [
    (torch.zeros(11, dtype=torch.long), "one integer per node"),
    (torch.zeros((3, 4), dtype=torch.long), "one integer per node"),
    (torch.zeros(12), "integer"),
    (torch.zeros(12, dtype=torch.bool), "integer"),
    (torch.zeros(12, dtype=torch.complex64), "integer"),
    (torch.full((12,), -1), "range"),
    (torch.full((12,), 16), "range"),
    (torch.zeros(12, dtype=torch.long), "owner"),
])
def test_rejects_invalid_chunk_assignments(bad, message):
    data = graph_data()
    with pytest.raises(ValueError, match=message):
        partition_graph(src=data["src"], dst=data["dst"], ts=data["ts"], num_nodes=12,
                        config=PartitionConfig(num_parts=4, chunks_per_rank=4, backend="round_robin"),
                        node_master=assignments()[0], node_to_chunk=bad)


def test_source_does_not_silently_truncate_floating_ids(tmp_path):
    source = tmp_path / "bad.pt"
    torch.save(assignments()[1].float(), source)
    with pytest.raises(ValueError, match="integer"):
        trainer_for(str(source)).prepare()


def test_accepts_integer_ids_and_empty_owner_buckets():
    owners = torch.tensor([0, 0, 2, 2])
    chunks = torch.tensor([3, 0, 11, 8], dtype=torch.int32)
    result = partition_graph(src=torch.arange(4), dst=torch.arange(4), ts=None, num_nodes=4,
                             config=PartitionConfig(num_parts=4, chunks_per_rank=4, backend="round_robin"),
                             node_master=owners, node_to_chunk=chunks)
    assert torch.equal(result.chunk_table.chunk_ids, chunks.long())
