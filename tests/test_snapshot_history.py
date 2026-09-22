import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import starrygl as sg
from starrygl.runtime.comm import CommScheduler
from starrygl.runtime.dataloader.blocks import snapshot_row_to_graph_block, move_graph_block
from starrygl.runtime.memory import AsyncMemoryCommitter, SharedStateRefreshFilter
from starrygl.runtime.memory.snapshot import bind_snapshot_history, hydrate_snapshot_history
from starrygl.runtime.state import finish_state_update, launch_state_update
from starrygl.store import StateManager
from starrygl.store.snapshot_history import SnapshotHistory
from starrygl.prepare.snapshot_csc import build_snapshot_csc_views


def test_history_is_causal_and_recomputed_time_does_not_inflate_increment():
    nodes = torch.tensor([2, 0])
    history = SnapshotHistory(nodes, 3, 5, 1)
    history.update(nodes, 1, torch.tensor([[2.], [4.]]))
    history.update(nodes, 2, torch.tensor([[4.], [8.]]))
    history.update(nodes, 2, torch.tensor([[6.], [12.]]))
    history.update(nodes, 4, torch.tensor([[10.], [20.]]))
    torch.testing.assert_close(history.read(nodes, 1), torch.tensor([[2., 2., 1., 1.], [4., 4., 1., 1.]]))
    torch.testing.assert_close(history.read(nodes, 3), torch.tensor([[6., 3., 2., 2.], [12., 6., 2., 2.]]))
    torch.testing.assert_close(history.read(nodes, 4)[:, 1], torch.tensor([8/3, 16/3]))
    history.reset()
    assert history.read(nodes, 4).count_nonzero() == 0
    assert not history.valid[1:].any()


def test_history_predicts_any_cache_channel_from_cumulative_increment():
    nodes = torch.tensor([0, 1])
    history = SnapshotHistory(nodes, 2, 4, 1)
    history.update(nodes, 1, torch.tensor([[2.], [3.]]))
    history.update(nodes, 2, torch.tensor([[4.], [6.]]))
    torch.testing.assert_close(history.predict(nodes, 4), torch.tensor([[8.], [12.]]))


def test_coupled_scan_overlays_current_local_candidate_input_on_stale_remote_prediction():
    from starrygl.runtime.snapshot.coupled import run_coupled_window_scan

    class Cell:
        reads_neighbor_state = True
        state_key = "neighbor_recurrent"
        cache_channels = {"candidate_input": "increment"}

        def materialize_cached(self, block, x, previous_src, previous_dst, cached):
            del x, previous_src
            candidate_input = previous_dst
            candidate_src = cached["candidate_input"].index_copy(0, torch.tensor([0]), candidate_input)
            self.observed = candidate_src
            return {"candidate": candidate_src[:1], "state_like": candidate_src[:1],
                    "candidate_input": candidate_input}, block

        def local_forward(self, block, src, dst):
            return src["candidate"]

    block = sg.GraphBlock(src_nodes=torch.tensor([0, 1]), dst_nodes=torch.tensor([0]),
                          edge_ids=torch.tensor([0]), format="coo", row=torch.tensor([1]),
                          col=torch.tensor([0]), num_src=2, num_dst=1)
    block.cache["snapshot_id"] = 2
    state_packet = torch.tensor([[2., 0., 1., 2.], [4., 0., 1., 2.]])
    candidate_packet = torch.tensor([[1., 1., 1., 1.], [3., 2., 1., 1.]])
    batch = sg.Batch(mode="snapshot", graph=block, blocks=((block,),),
                     features={"x": (torch.ones(2, 1),)}, state={
                         "neighbor_recurrent_snapshots": (state_packet,),
                         "neighbor_recurrent_window_state": (state_packet[:, :1],),
                         "snapshot_cache_channels": {"candidate_input": (candidate_packet,)},
                     })
    cell = Cell()
    result = run_coupled_window_scan(batch, input_project=lambda value: value, cell=cell)
    torch.testing.assert_close(cell.observed, torch.tensor([[2.], [7.]]))
    assert result.cache_history["candidate_input"][0][1] == 3
    torch.testing.assert_close(result.cache_history["candidate_input"][0][2], torch.tensor([[2.]]))


def test_window_slots_wrap_without_resetting_cumulative_increment():
    nodes = torch.arange(2)
    history = SnapshotHistory(nodes, 2, 3, 1)
    allocation = history.packets.data_ptr()
    increment = torch.tensor([[2.], [4.]])
    for end in range(1, 13):
        start = max(0, end - 3)
        history.advance(start)
        for version in range(start + 1, end + 1):
            history.update(nodes, version, version * increment)
        actual = history.read(nodes, end)
        torch.testing.assert_close(actual[:, :1], end * increment)
        torch.testing.assert_close(actual[:, 1:2], increment)
        torch.testing.assert_close(actual[:, -2], torch.full((2,), float(end)))
        assert torch.all(history.read(nodes, start)[:, -1] == start)
        assert history.packets.shape == (4, 2, 4)
        assert history.packets.data_ptr() == allocation
    with pytest.raises(ValueError, match="reset"):
        history.advance(1)


def test_window_predecessor_preserves_filtered_seed_and_rejects_late_overwrite():
    nodes = torch.tensor([0])
    history = SnapshotHistory(nodes, 1, 2, 1)
    history.update(nodes, 1, torch.tensor([[2.]]))
    old = history.read(nodes, 1)
    history.advance(2)
    history.update(nodes, 3, torch.tensor([[6.]]))
    history.advance(3)
    history.update(nodes, 4, torch.tensor([[8.]]))
    retained = history.packets.clone()
    history.install(nodes, old)  # version 1 and 3 reuse the same output slot
    history.install(nodes, torch.tensor([[999., 999., 999., 7.]]))  # outside active window
    torch.testing.assert_close(history.packets, retained)
    torch.testing.assert_close(old, torch.tensor([[2., 2., 1., 1.]]))
    history.advance(10)  # no refresh for several windows
    assert history.read(nodes, 10)[0, -1] == 4
    history.install(nodes, torch.tensor([[16., 2., 8., 8.]]))  # useful delayed predecessor
    history.update(nodes, 11, torch.tensor([[22.]]))
    torch.testing.assert_close(history.read(nodes, 11), torch.tensor([[22., 2., 9., 11.]]))
    assert history.read(nodes, 3).count_nonzero() == 0  # never return a future seed


def test_each_snapshot_compensates_missing_rows_and_carries_local_live_state():
    torch.manual_seed(11)
    model = sg.GConvGRUModel(1, 2, 1, gamma_boundary_init=0.0)
    nodes = torch.arange(2)
    history = SnapshotHistory(nodes, 2, 5, 2)
    history.update(nodes, 1, torch.tensor([[1., 2.], [3., 4.]]))
    history.update(nodes, 4, torch.full((2, 2), 999.))
    blocks = []
    for t in (2, 3):
        block = sg.GraphBlock(src_nodes=nodes, dst_nodes=nodes[:1], edge_ids=torch.tensor([0]),
                             format="coo", row=torch.tensor([1]), col=torch.tensor([0]), num_src=2, num_dst=1)
        block.cache["snapshot_id"] = t
        blocks.append((block,))
    batch = sg.Batch(mode="snapshot", graph=blocks[-1][-1], blocks=tuple(blocks),
                     features={"x": (torch.ones(2, 1), torch.ones(2, 1))})
    shared = SnapshotHistory(nodes[:1], 2, 5, 2)
    shared.update(nodes[:1], 1, torch.tensor([[1., 2.]]))
    runtime = SimpleNamespace(snapshot_history=history, snapshot_shared_history=shared,
                              memory_manager=SimpleNamespace(row_map=torch.tensor([0, -1])))
    hydrate_snapshot_history(batch, runtime)
    observed = []
    materialize = model.cell.materialize

    def record(blocks, src):
        observed.append(src["h_prev"].clone())
        return materialize(blocks, src)

    model.cell.materialize = record
    output = model.encode(batch)
    output.embeddings.square().sum().backward()
    assert model.gamma_boundary.grad is not None and model.gamma_boundary.grad.abs() > 1e-8
    first_output = output.aux["snapshot_states"][0][2]
    torch.testing.assert_close(observed[1][0], first_output[0])
    expected_remote = torch.tensor([3., 4.]) * 2  # cache + sigmoid(0) * age(2) * CMA
    torch.testing.assert_close(observed[1][1], expected_remote)
    assert batch.state["neighbor_recurrent_snapshots"][0][:, -1].max() == 1
    assert history.values[3].count_nonzero() == 0  # encode cannot mutate caches


def test_filter_compares_with_synchronized_shared_reference():
    from starrygl.model import StateDelta
    from starrygl.runtime.memory.shared import shared_delta

    shared = StateManager(values=torch.tensor([[0., 1.]]), row_map=torch.tensor([0]))
    change_filter = SharedStateRefreshFilter(dim=2, num_rows=1, min_cosine_distance=0.3, max_skip=1)
    change_filter.historical.copy_(torch.tensor([[1., 0.]]))
    runtime = SimpleNamespace(shared_manager=shared, change_filter=change_filter, use_shared_filter=True, snapshot_history=None)
    delta = StateDelta(node_ids=torch.tensor([0]), values=torch.tensor([[1., 0.]]), kind="neighbor_recurrent")
    assert shared_delta(runtime, delta).node_ids.numel() == 1
    shared.values.copy_(delta.values)
    change_filter.historical.copy_(torch.tensor([[0., 1.]]))
    assert shared_delta(runtime, delta).node_ids.numel() == 0


def test_overlap_reuses_shared_slots_but_filtered_nodes_can_have_older_holes():
    nodes = torch.arange(2)
    local = SnapshotHistory(nodes, 2, 3, 2)
    shared = SnapshotHistory(nodes, 2, 3, 2)
    value = torch.tensor([[1., 2.], [2., 4.]])
    for version in (1, 2, 3):
        local.update(nodes, version, value * version)
        sent = nodes if version == 1 else nodes[:1]  # node 1 is filtered after its first update
        shared.update(sent, version, value[sent] * version)
    blocks = [sg.graph_block_from_coo(src=nodes, dst=nodes.flip(0), edge_ids=nodes,
              num_nodes=2, format="coo") for _ in range(3)]
    for block, sid in zip(blocks, (1, 2, 3)):
        block.cache["snapshot_id"] = sid
    batch = sg.Batch(mode="snapshot", blocks=tuple((b,) for b in blocks))
    runtime = SimpleNamespace(snapshot_history=local, snapshot_shared_history=shared,
                              memory_manager=SimpleNamespace(row_map=torch.tensor([0, -1])))
    hydrate_snapshot_history(batch, runtime)
    ages = torch.stack([sid + 1 - packet[:, -1] for sid, packet in
                        zip((1, 2, 3), batch.state["neighbor_recurrent_shared_snapshots"])])
    torch.testing.assert_close(ages[:, 0], torch.tensor([0., 0., 1.]))
    torch.testing.assert_close(ages[:, 1], torch.tensor([1., 2., 3.]))
    assert torch.all(batch.state["neighbor_recurrent_snapshots"][-1][:, -1] == 3)


def _views(hot_compute, diffusion=False):
    src, dst = torch.tensor([0, 1, 2, 3, 0]), torch.tensor([1, 2, 3, 0, 3])
    return build_snapshot_csc_views(
        src=src.repeat(4), dst=dst.repeat(4), ts=torch.arange(4).repeat_interleave(5).float(),
        edge_ids=torch.arange(20), edge_dist_index=torch.arange(20),
        node_master=torch.tensor([0, 1, 0, 1]), hot_node_ids=torch.tensor([0]),
        node_is_hot=torch.tensor([True, False, False, False]), node_to_chunk=torch.zeros(4, dtype=torch.long),
        time_ptr_2=torch.arange(4)[:, None] * 5 + torch.tensor([0, 5]),
        num_nodes=4, world_size=2, hot_compute=hot_compute, diffusion=diffusion,
    )


@pytest.mark.parametrize("model_name", ["gconv_gru", "dcrnn"])
@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two torchrun ranks")
def test_exact_sliding_outputs_and_gradients_match_full_graph(model_name):
    from copy import deepcopy

    cuda = os.environ.get("STARRYGL_TEST_NCCL") == "1"
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if cuda else "gloo")
    try:
        rank, comm = dist.get_rank(), CommScheduler()
        masters = torch.tensor([0, 1, 0, 1], device=device)
        owned = (masters == rank).nonzero(as_tuple=True)[0]
        row_map = torch.full((4,), -1, device=device, dtype=torch.long)
        row_map[owned] = torch.arange(2, device=device)
        manager = StateManager(values=torch.zeros(2, 2, device=device), row_map=row_map,
            node_dist_index=(masters << 48) | torch.tensor([0, 0, 1, 1], device=device),
            kind="neighbor_recurrent", comm=comm)
        view = _views(False, model_name == "dcrnn")[rank]
        store = SimpleNamespace(graph=SimpleNamespace(snapshot_csc_view=view, num_nodes=4))
        bind_snapshot_history(manager, store, owned[:0], window_size=3)
        torch.manual_seed(47)
        cls = sg.DCRNNModel if model_name == "dcrnn" else sg.GConvGRUModel
        model = cls(1, 2, 1).to(device)
        reference = deepcopy(model)
        full = sg.graph_block_from_coo(src=torch.tensor([0, 1, 2, 3, 0], device=device),
            dst=torch.tensor([1, 2, 3, 0, 3], device=device), edge_ids=torch.arange(5, device=device),
            num_nodes=4, format="coo")
        history = [torch.zeros(4, 2, device=device)]
        for t in range(4):
            begin = max(0, t - 2)
            blocks = []
            for sid in range(begin, t + 1):
                block = move_graph_block(snapshot_row_to_graph_block(view["slices"][sid], attach_reverse=False), device)
                block.cache["comm"] = comm
                blocks.append((block,))
            batch = sg.Batch(mode="snapshot", graph=blocks[-1][-1], blocks=tuple(blocks),
                features={"x": tuple((b[0].src_nodes.float() + sid + 1)[:, None]
                    for sid, b in zip(range(begin, t + 1), blocks))})
            hydrate_snapshot_history(batch, manager)
            first_packet = batch.state["neighbor_recurrent_snapshots"][0]
            local = row_map[blocks[0][0].src_nodes] >= 0
            assert torch.all(first_packet[local, -1] == begin)
            model.zero_grad()
            actual = model.encode(batch)
            state = history[begin].detach()
            reference.zero_grad()
            for sid in range(begin, t + 1):
                x = reference.input((torch.arange(4, device=device).float() + sid + 1)[:, None])
                from starrygl.runtime.snapshot.layerwise import materialize_coupled_cell
                src, block = materialize_coupled_cell(reference.cell, (full,), x, state)
                state = reference.cell.local_forward(block, src, {"h_prev": state})
                if len(history) <= sid + 1:
                    history.append(state.detach())
                else:
                    history[sid + 1] = state.detach()
            torch.testing.assert_close(actual.embeddings, state[blocks[-1][0].dst_nodes])
            actual.embeddings.square().sum().backward()
            state.square().sum().backward()
            for param, expected in zip(model.parameters(), reference.parameters()):
                if expected.grad is not None:
                    dist.all_reduce(param.grad)
                    torch.testing.assert_close(param.grad, expected.grad, atol=1e-5, rtol=1e-5)
            manager.commit(model.state_update(batch, actual))
        manager.reset()
        assert manager.snapshot_history.packets.count_nonzero() == 0
    finally:
        dist.destroy_process_group()


def test_hot_compute_keeps_feature_rows_and_static_routes_consistent():
    views = _views(True)
    for rank, view in enumerate(views):
        row = view["slices"][0]
        assert 0 in row["dst_nodes"]
        assert row["src_nodes"].unique().numel() == row["src_nodes"].numel()
        feature_ids = torch.cat((torch.tensor([0]), row["node_feature_ids"]))
        torch.testing.assert_close(feature_ids[row["src_feature_row"]], row["src_nodes"])
        assert torch.all(row["route"]["send_index"] < row["dst_nodes"].numel())
    trainer = sg.compile(data_source={"source": "snapshots"}, backbone={"name": "gconv_gru"},
                         task_segment={"name": "node_regression"},
                         runtime={"temporal_state": {"consistency": "bounded_stale"}})
    assert not trainer.plan.view.requires("snapshot_hot_compute")
    assert "sliding_window_slots" in trainer.plan.explain()


def test_history_capacity_uses_resolved_sliding_window_arguments():
    from starrygl.store import GraphStore, StoreBundle, FeatureManager, LabelStore

    trainer = sg.compile(data_source={"source": "snapshots"},
        backbone={"name": "gconv_gru", "in_dim": 1, "hidden_dim": 2, "out_dim": 1},
        task_segment={"name": "node_regression"},
        runtime={"sampling": {"window": {"policy": "full_snapshot", "num_full_snapshots": 2}}})
    packed = torch.tensor([0, 1 << 48, 1, (1 << 48) | 1])
    store = StoreBundle(graph=GraphStore(num_nodes=4, prepare={
        "meta": {"world_size": 2, "view_plan": trainer.plan.view.as_dict()},
        "partition": {"node_dist_index": packed, "hot_node_ids": torch.tensor([0])},
        "snapshot_csc_views": _views(True),
    }), features=FeatureManager(), labels=LabelStore())
    common = dict(model=sg.GConvGRUModel(1, 2, 1), store=store, device="cpu", comm=CommScheduler())
    default = trainer._state_manager(None, **common)["neighbor_recurrent"].snapshot_history
    assert default.window_size == 2 and default.packets.shape[0] == 3
    overridden = trainer._state_manager(None, **common, window_policy="chunk_decay",
        num_full_snapshots=3, chunk_decay=[1, -1, 0])["neighbor_recurrent"].snapshot_history
    assert overridden.window_size == 5 and overridden.packets.shape[0] == 6


@pytest.mark.parametrize("model_name", ["gconv_gru", "dcrnn"])
@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two torchrun ranks")
def test_distributed_compile_prepare_and_sliding_train_inference(model_name):
    cuda = os.environ.get("STARRYGL_TEST_NCCL") == "1"
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if cuda else "gloo")
    temporary = TemporaryDirectory(prefix="starrygl_history_") if dist.get_rank() == 0 else None
    try:
        paths = [None if temporary is None else temporary.name]
        dist.broadcast_object_list(paths, src=0)
        root = Path(paths[0])
        if dist.get_rank() == 0:
            data = root / "data"
            data.mkdir()
            torch.manual_seed(19)
            features = torch.rand(8, 4, 1)
            torch.save(dict(src=torch.tensor([0, 1, 2, 3, 0]).repeat(8),
                dst=torch.tensor([1, 2, 3, 0, 3]).repeat(8), ts=torch.arange(8).repeat_interleave(5).float(),
                edge_ids=torch.arange(40), num_nodes=4, time_ptr_2=torch.arange(8)[:, None]*5+torch.tensor([0, 5]),
                node_label_horizon=1), data / "graph.pt")
            torch.save(features, data / "node_feat.pt")
            torch.save(torch.cat((features[1:, :, 0], torch.full((1, 4), float("nan")))), data / "node_label.pt")
        dist.barrier()
        trainer = sg.compile(data_source={"source": str(root / "data"), "temporal_representation": "snapshot_sequence"},
            backbone={"name": model_name, "in_dim": 1, "hidden_dim": 2, "out_dim": 1},
            task_segment={"name": "node_regression", "loss": "mse"},
            runtime={"temporal_state": {"consistency": "bounded_stale", "max_staleness": 2,
                     "boundary_prediction": {"learnable": True},
                     "filter": {"min_cosine_distance": 0.3, "max_skip": 2}},
                     "sampling": {"mode": "full", "window": {"policy": "full_snapshot", "num_full_snapshots": 2}},
                     "preprocess": {"num_parts": 2, "chunks_per_rank": 1, "hot_node_ratio": 0.25,
                                    "split_ratios": [0.5, 0.25, 0.25], "include_static_one_hop": True}})
        trainer.artifact_root = root / "prepared"
        if dist.get_rank() == 0:
            exact_runtime = {**trainer.runtime_config, "preprocess": trainer.preprocess_config, "train": trainer.train_config}
            exact_runtime["temporal_state"] = {**exact_runtime["temporal_state"], "consistency": "exact", "max_staleness": 0}
            exact = sg.compile(data_source=trainer.graph, backbone=trainer.model_config, task_segment={"name": "node_regression", "loss": "mse"}, runtime=exact_runtime)
            exact.artifact_root = trainer.artifact_root
            assert exact.plan.view == trainer.plan.view
            exact.prepare_artifacts(world_size=2)
        dist.barrier()
        result = trainer.fit(epochs=1, rank=dist.get_rank(), device=device,
                             sampler_options={"access_pipeline": True, "snapshot_reverse_direction": False})[0]
        assert result.steps == 3 and torch.isfinite(torch.tensor(result.loss))
        runtime = trainer._runtime_state_manager["neighbor_recurrent"]
        assert runtime.snapshot_version == 4  # unlabeled split boundary still commits
        assert runtime.snapshot_history.window_size == 2
        assert runtime.snapshot_history.packets.shape[0] == 3
        assert runtime.snapshot_history.valid[1:].any()
        assert runtime.snapshot_history.packets[:, :, -2].max() >= 2
        if model_name == "dcrnn":
            candidate = runtime.snapshot_cache_channels["candidate_input"]
            assert candidate.valid[1:].any()
            assert candidate.packets[:, :, -2].max() >= 2
        assert not runtime.pending_snapshot_pushes
        validation = trainer.evaluate(rank=dist.get_rank(), device=device,
                                      sampler_options={"access_pipeline": True, "snapshot_reverse_direction": False})
        assert validation.steps == 1 and torch.isfinite(torch.tensor(validation.loss))
        assert runtime.snapshot_version == 6 and not runtime.pending_snapshot_pushes
        outputs = trainer.predict(rank=dist.get_rank(), device=device,
                                  sampler_options={"access_pipeline": True, "snapshot_reverse_direction": False})
        assert len(outputs) == 2 and runtime.snapshot_version == 8
        assert not runtime.pending_snapshot_pushes
        dist.barrier()
    finally:
        dist.destroy_process_group()
        if temporary is not None:
            temporary.cleanup()
