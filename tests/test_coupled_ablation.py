import torch
import pytest
import json
import os
import torch.distributed as dist

import starrygl as sg
from starrygl.runtime.builders import build_model_from_config


def graph():
    return sg.graph_block_from_coo(src=torch.tensor([0, 1]), dst=torch.tensor([1, 2]),
                                   edge_ids=torch.arange(2), num_nodes=3, format="coo")


def test_dcrnn_diffusion_and_gradient_match_dense_random_walk():
    block = graph()
    conv = sg.DiffusionGraphConv(2, 2)
    x = torch.randn(3, 2, requires_grad=True)
    adjacency = torch.eye(3)
    adjacency[0, 1] = adjacency[1, 2] = 1
    forward = (adjacency / adjacency.sum(1, keepdim=True)).T
    backward = (adjacency.T / adjacency.sum(0).unsqueeze(1)).T
    expected = x @ (conv.weight[0, 0] + conv.weight[1, 0])
    expected = expected + forward @ x @ conv.weight[0, 1] + backward @ x @ conv.weight[1, 1] + conv.bias
    actual = conv(block, x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x, retain_graph=True)[0],
                               torch.autograd.grad(expected.sum(), x)[0])


@pytest.mark.parametrize("name", ["dcrnn", "gconv_gru"])
def test_models_lower_to_coupled_state_and_use_neighbors(name):
    torch.manual_seed(7)
    trainer = sg.compile(data_source={"source": "snapshots"},
                         backbone={"name": name, "in_dim": 2, "hidden_dim": 3, "out_dim": 1},
                         task_segment={"name": "node_regression"})
    assert trainer.plan.coupling == "coupled"
    assert trainer.plan.state_dependencies[0].kind == "neighbor_recurrent"
    model = build_model_from_config(trainer.model_config)
    block = graph()
    x = torch.ones(3, 3)
    previous = torch.zeros(3, 3)
    first, _ = model.cell.materialize((block,), {"x": x, "h_prev": previous})
    previous[0] = 2
    second, _ = model.cell.materialize((block,), {"x": x, "h_prev": previous})
    local = {"h_prev": torch.zeros(3, 3)}
    assert not torch.allclose(model.cell.local_forward(block, first, local)[1],
                              model.cell.local_forward(block, second, local)[1])


def test_ablation_replays_unlabeled_split_boundary_before_validation(tmp_path):
    from starrygl.cli.coupled_ablation import main

    data = tmp_path / "data"
    data.mkdir()
    torch.manual_seed(9)
    features = torch.rand(12, 3, 2)
    labels = torch.cat((features[1:, :, 0].log1p(), torch.full((1, 3), float("nan"))))
    torch.save(dict(src=torch.tensor([0, 1, 2]).repeat(12), dst=torch.tensor([1, 2, 0]).repeat(12),
                    ts=torch.arange(12).repeat_interleave(3).float(), edge_ids=torch.arange(36),
                    num_nodes=3, time_ptr_2=torch.arange(12).unsqueeze(1)*3+torch.tensor([0, 3]),
                    node_label_horizon=1), data / "graph.pt")
    torch.save(features, data / "node_feat.pt")
    torch.save(labels, data / "node_label.pt")
    output = tmp_path / "run"
    args = ["--data", str(data), "--artifact-root", str(tmp_path / "prepared"),
            "--output", str(output), "--model", "dcrnn", "--policy", "exact", "--world-size", "1",
            "--epochs", "1", "--eval-every", "1", "--device", "cpu",
            "--num-full-snapshots", "3", "--access-pipeline"]
    main(args + ["--prepare-only"])
    main(args)
    result = json.loads((output / "result.json").read_text())
    assert result["best_epoch"] == 1 and result["test_steps"] > 0
    assert result["test_mse_exact"] >= 0
    assert result["test_state_reads"]["owner_stale_rows"] == 0
    assert json.loads((output / "manifest.json").read_text())["history_window_size"] == 3


def test_hydration_restores_permuted_unique_node_order():
    from starrygl.runtime.state.access import _store_hydrated_read
    from starrygl.store.state import StateRead

    state = {}
    _store_hydrated_read(state, kind="neighbor_recurrent",
                        read=StateRead(torch.arange(3), torch.tensor([[10.], [20.], [30.]]), torch.arange(3).float()),
                        nodes=torch.tensor([2, 0, 1]), compact_nodes=torch.arange(3), inverse=torch.tensor([2, 0, 1]))
    torch.testing.assert_close(state["neighbor_recurrent"], torch.tensor([[30.], [10.], [20.]]))
    torch.testing.assert_close(state["neighbor_recurrent_ts"], torch.tensor([2., 0., 1.]))


def test_history_audit_distinguishes_hot_cold_and_nonzero_stale_increments():
    from starrygl.cli.coupled_ablation import StateReadAudit
    from starrygl.runtime.comm import CommScheduler

    block = graph()
    block.cache["snapshot_id"] = 2
    batch = sg.Batch(mode="snapshot", graph=block, state={
        "neighbor_recurrent_ts": torch.tensor([2., 1., 0.]),
        "neighbor_recurrent_shared_mask": torch.tensor([False, True, True]),
        "neighbor_recurrent_cold_src_rows": (torch.tensor([1, 2]),),
        "neighbor_recurrent_snapshots": (torch.tensor([[1., 0., 2., 2.],
                                                       [1., 2., 1., 1.],
                                                       [0., 3., 0., 0.]]),),
    })
    audit = StateReadAudit(3, "cpu", CommScheduler(), hot_nodes=torch.tensor([1]), num_nodes=3)
    audit(batch)
    result = audit.report()
    assert result["history_remote_hot_rows"] == result["history_remote_cold_rows"] == 1
    assert result["history_hot_mean_lag"] == 1 and result["history_cold_mean_lag"] == 2
    assert result["stale_remote_nonzero_increment_rows"] == 2
    assert result["future_rows"] == result["owner_stale_rows"] == 0
    audit.reset()
    assert audit.report()["history_rows"] == 0


def test_slot_audit_reads_boundary_packets_without_shared_hot_plane():
    from starrygl.cli.coupled_ablation import StateReadAudit
    from starrygl.runtime.comm import CommScheduler

    block = graph()
    block.cache["snapshot_id"] = 2
    packet = torch.tensor([[1., 1., 1., 2.], [1., 1., 1., 1.], [0., 0., 0., 0.]])
    batch = sg.Batch(mode="snapshot", graph=block, state={
        "neighbor_recurrent_ts": packet[:, -1],
        "neighbor_recurrent_shared_mask": torch.tensor([False, True, True]),
        "neighbor_recurrent_snapshots": (packet,),
        "neighbor_recurrent_cold_src_rows": (torch.tensor([1, 2]),),
    })
    audit = StateReadAudit(4, "cpu", CommScheduler(), num_nodes=3, window_size=3)
    audit(batch)
    report = audit.report()
    assert report["cold_history_inputs"][2]["mean_lag"] == 1.5
    assert report["cold_history_inputs"][2]["nonzero_extrapolation_rows"] == 1
    assert report["cold_history_inputs"][0]["rows"] == 0
    assert "hot_shared_predictions" not in report
    assert report["shared_lag_histogram"] == {"1": 1, "2": 1}
    assert report["shared_initial_rows"] == 1 and report["future_rows"] == 0
    audit.reset()
    assert audit.report()["cold_history_inputs"][2]["rows"] == 0


def test_unlabeled_coupled_state_commits_owned_rows_and_snapshot_version():
    model = sg.GConvGRUModel(1, 1, 1)
    block = graph()
    block.cache["snapshot_id"] = 6
    batch = sg.Batch(mode="snapshot", graph=block,
                     state={"neighbor_recurrent_node_ids": torch.tensor([2, 0, 1])})
    delta = model.state_update(batch, sg.ModelOutput(state_embeddings=torch.tensor([[30.], [10.], [20.]])))
    torch.testing.assert_close(delta.values, torch.tensor([[10.], [20.], [30.]]))
    torch.testing.assert_close(delta.timestamps, torch.full((3,), 7.))


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two torchrun ranks")
@pytest.mark.parametrize("hot_compute", [False, True])
def test_distributed_dcrnn_output_and_gradient_match_complete_snapshot(hot_compute):
    from copy import deepcopy
    from starrygl.prepare.snapshot_csc import build_snapshot_csc_views
    from starrygl.runtime.dataloader.blocks import snapshot_row_to_graph_block
    from starrygl.runtime.snapshot.layerwise import materialize_coupled_cell
    from starrygl.runtime.comm import CommScheduler

    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        src, dst = torch.tensor([0, 0, 1, 2, 3]), torch.tensor([1, 3, 2, 0, 2])
        views = build_snapshot_csc_views(src=src, dst=dst, ts=torch.zeros(5), edge_ids=torch.arange(5),
            edge_dist_index=torch.arange(5), node_master=torch.tensor([0, 1, 0, 1]),
            hot_node_ids=torch.tensor([0, 1]), node_is_hot=torch.tensor([True, True, False, False]),
            node_to_chunk=torch.zeros(4, dtype=torch.long), time_ptr_2=torch.tensor([[0, 5]]),
            num_nodes=4, world_size=2, diffusion=True, hot_compute=hot_compute)
        block = snapshot_row_to_graph_block(views[rank]["slices"][0], attach_reverse=False)
        from starrygl.store.feature import _slim_snapshot_view
        from starrygl.store.artifact import _restore_snapshot_view
        restored = _restore_snapshot_view(_slim_snapshot_view(views[rank]))["slices"][0]
        for name, value in block.cache["diffusion"].items():
            torch.testing.assert_close(restored["diffusion"][name], value)
        block.cache["comm"] = CommScheduler()
        full = sg.graph_block_from_coo(src=src, dst=dst, edge_ids=torch.arange(5), num_nodes=4, format="coo")
        torch.manual_seed(31)
        cell = sg.DCRNNCell(3)
        reference = deepcopy(cell)
        x, previous = torch.randn(4, 3), torch.randn(4, 3)
        states, _ = materialize_coupled_cell(cell, (block,), x[block.src_nodes], previous[block.src_nodes])
        actual = cell.local_forward(block, states, {"h_prev": previous[block.dst_nodes]})
        full_states, _ = reference.materialize((full,), {"x": x, "h_prev": previous})
        expected = reference.local_forward(full, full_states, {"h_prev": previous})
        torch.testing.assert_close(actual, expected[block.dst_nodes])
        owned = torch.tensor([0, 1, 0, 1])[block.dst_nodes] == rank
        actual[owned].square().sum().backward()
        expected.square().sum().backward()
        for parameter, ref in zip(cell.parameters(), reference.parameters()):
            dist.all_reduce(parameter.grad)
            torch.testing.assert_close(parameter.grad, ref.grad, atol=1e-6, rtol=1e-5)
    finally:
        dist.destroy_process_group()


def test_temporal_feature_reads_use_shard_node_mapping():
    from starrygl.store import FeatureManager, GraphStore, LabelStore, StoreBundle
    from starrygl.runtime.snapshot.features import _read_features

    features = FeatureManager(node_features={"x": torch.tensor([[[20.], [30.], [10.]]])},
                              node_row_map=torch.tensor([2, 0, 1]))
    store = StoreBundle(graph=GraphStore(num_nodes=3), features=features, labels=LabelStore())
    actual, _, _ = _read_features(store, src_nodes=torch.tensor([0, 2]), edge_ids=None,
        snapshot_id=0, embedded=None, src_feature_row=torch.tensor([0, 1]), comm=None, defer_finish=False)
    torch.testing.assert_close(actual["x"], torch.tensor([[10.], [30.]]))
