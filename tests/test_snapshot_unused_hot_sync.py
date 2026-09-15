"""Owner-only snapshot boundary publication; historical hot tests live in V5."""
import os
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

import starrygl as sg
from starrygl.prepare.snapshot_csc import build_snapshot_csc_views
from starrygl.runtime.builders import build_model_from_config
from starrygl.runtime.comm import CommScheduler
from starrygl.runtime.dataloader.blocks import move_graph_block, snapshot_row_to_graph_block
from starrygl.runtime.epoch import sync_gradients
from starrygl.runtime.memory.snapshot import hydrate_snapshot_history
from starrygl.runtime.state import finish_state_update, launch_state_update
from starrygl.runtime.state.build import build_state_managers
from starrygl.store import FeatureManager, GraphStore, LabelStore, StoreBundle


def _store(model_cls, rank=0, *, snapshot=True):
    src, dst = torch.tensor([0, 1, 2, 3, 0]), torch.tensor([1, 2, 3, 0, 3])
    masters, hot = torch.tensor([0, 1, 0, 1]), torch.tensor([0])
    views = build_snapshot_csc_views(
        src=src.repeat(7), dst=dst.repeat(7), ts=None,
        edge_ids=torch.arange(35), edge_dist_index=torch.arange(35),
        node_master=masters, hot_node_ids=hot, node_is_hot=torch.arange(4) == 0,
        node_to_chunk=torch.zeros(4, dtype=torch.long),
        time_ptr_2=torch.arange(7)[:, None] * 5 + torch.tensor([0, 5]),
        num_nodes=4, world_size=2, hot_compute=False, diffusion=model_cls is sg.DCRNNModel,
    )
    # Global hot row 0 remains reserved on rank 1: two owned nodes, three storage rows.
    graph = GraphStore(num_nodes=4, rank=rank, prepare={
        "meta": {"world_size": 2, "view_plan": {"required_layouts": ["snapshot_csc"] if snapshot else ["temporal_csr"]}},
        "partition": {"node_dist_index": (masters << 48) | torch.tensor([0, 1, 1, 2]), "hot_node_ids": hot},
        "snapshot_csc_views": views,
    })
    return StoreBundle(graph=graph, features=FeatureManager(), labels=LabelStore())


@pytest.mark.parametrize("model_name", ["gconv_gru", "dcrnn"])
@pytest.mark.parametrize("mode", ["cache", "mean", "learnable"])
def test_public_boundary_prediction_preserves_owner_layout_and_parameter(model_name, mode):
    options = dict(data_source={"source": "snapshots"},
        backbone={"name": model_name, "in_dim": 1, "hidden_dim": 2, "out_dim": 1,
                  "state_extrapolation": mode != "cache"}, task_segment={"name": "node_regression"})
    exact = sg.compile(**options)
    bounded = sg.compile(**options, runtime={"temporal_state": {"consistency": "bounded_stale",
        "max_staleness": 2, "boundary_prediction": {"learnable": mode == "learnable", "gamma_init": 0.7}}})
    assert exact.plan.view == bounded.plan.view
    assert bounded.plan.cache_policy == "local"
    assert not bounded.plan.view.requires("snapshot_hot_compute")
    assert bounded.plan.view.requires("snapshot_diffusion") == (model_name == "dcrnn")
    state = bounded.runtime_config["temporal_state"]
    assert state["boundary_prediction"]["gamma_init"] == 0.7
    model = build_model_from_config(bounded.model_config, temporal_state=state)
    assert not hasattr(model, "gamma") and not hasattr(model, "increment")
    assert (model.gamma_boundary is not None) == (mode == "learnable")
    if model.gamma_boundary is not None:
        torch.testing.assert_close(model.gamma_boundary.detach(), torch.tensor(0.7))
    assert model.state_extrapolation == (mode != "cache")


@pytest.mark.parametrize("model_cls", [sg.GConvGRUModel, sg.DCRNNModel])
@pytest.mark.parametrize("snapshot", [True, False])
def test_snapshot_owner_only_preserves_other_state_construction(model_cls, snapshot):
    runtime = build_state_managers(model=model_cls(1, 2, 1), store=_store(model_cls, snapshot=snapshot),
        kinds=("neighbor_recurrent",), temporal_state={"consistency": "bounded_stale",
        "filter": {"enabled": True}}, device=torch.device("cpu"), comm=CommScheduler(), window_size=3)["neighbor_recurrent"]
    assert (runtime.shared_manager is None) == snapshot
    if snapshot:
        assert runtime.snapshot_shared_history.node_ids.numel() == 0
        assert runtime.snapshot_owned_count == 2
        assert runtime.skip_remote_non_hot_owner_commit


@pytest.mark.parametrize("model_cls", [sg.GConvGRUModel, sg.DCRNNModel])
@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two torchrun ranks")
def test_owner_filtered_history_has_bounded_age_compensation_and_owner_only_commit(model_cls):
    cuda = os.environ.get("STARRYGL_TEST_NCCL") == "1"
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if cuda else "gloo")
    try:
        rank, comm = dist.get_rank(), CommScheduler()
        comm.launch_all_gather = Mock(side_effect=AssertionError("snapshot must not synchronize hot replicas"))
        comm.launch_push = Mock(wraps=comm.launch_push)
        torch.manual_seed(7)
        model = model_cls(1, 2, 1, gamma_boundary_init=0.0).to(device)
        store = _store(model_cls, rank)
        runtime = build_state_managers(model=model, store=store, kinds=("neighbor_recurrent",),
            temporal_state={"consistency": "bounded_stale", "max_staleness": 2,
            "filter": {"enabled": True, "min_cosine_distance": 0.3, "max_skip": 20}},
            device=device, comm=comm, window_size=3)["neighbor_recurrent"]
        assert runtime.shared_manager is None and runtime.change_filter.max_skip == 2
        owner = runtime.memory_manager
        owned = (owner.row_map >= 0).nonzero(as_tuple=True)[0]
        if rank == 1:
            assert owner.values.shape[0] == 3 and runtime.snapshot_owned_count == 2
        owner.submit_materialize_async = Mock(side_effect=AssertionError("no per-snapshot owner fetch"))
        ages, gamma_grad = [], 0.0
        for t in range(7):
            blocks = []
            for sid in range(max(0, t - 2), t + 1):
                block = move_graph_block(snapshot_row_to_graph_block(store.graph.snapshot_csc_view["slices"][sid], attach_reverse=False), device)
                assert torch.all(owner.row_map[block.dst_nodes] >= 0)
                block.cache["comm"] = comm
                blocks.append((block,))
            batch = sg.Batch(mode="snapshot", graph=blocks[-1][-1], blocks=tuple(blocks),
                features={"x": tuple((b[0].src_nodes.float() + 1)[:, None] for b in blocks)})
            hydrate_snapshot_history(batch, runtime)  # awaits prior batch push once
            packet = batch.state["neighbor_recurrent_snapshots"][-1]
            cold = batch.state["neighbor_recurrent_cold_src_rows"][-1]
            age = t - packet[cold, -1]
            ages.append(int(age.max()))
            assert torch.all((age >= 0) & (age <= 2))
            prepared, _ = model.runtime_prepare_scan(batch)
            expected = packet[cold, :2] + 0.5 * age[:, None] * packet[cold, 2:4]
            torch.testing.assert_close(prepared.state["neighbor_recurrent_window_state"][-1][cold], expected)
            if t >= 2 and ages[-1]:
                assert torch.any(expected != packet[cold, :2])
            model.zero_grad()
            output = model.encode(batch)
            output.embeddings.square().sum().backward()
            sync_gradients(model, "all_reduce")
            if model.gamma_boundary.grad is not None:
                gamma_grad += float(model.gamma_boundary.grad.abs())
                gathered = [torch.zeros_like(model.gamma_boundary.grad) for _ in range(2)]
                dist.all_gather(gathered, model.gamma_boundary.grad)
                torch.testing.assert_close(gathered[0], gathered[1])
            delta = model.state_update(batch, output)
            assert torch.all(owner.row_map[delta.node_ids] >= 0)
            # Controlled same-direction owner outputs force cosine filtering without delayed packets.
            states = tuple((nodes, version, torch.full_like(value, float(version)))
                           for nodes, version, value in delta.metadata["snapshot_states"])
            delta = sg.StateDelta(node_ids=delta.node_ids, values=torch.full_like(delta.values, float(t + 1)),
                timestamps=delta.timestamps, kind=delta.kind, metadata={"snapshot_states": states})
            launch_state_update(runtime, delta)
            finish_state_update(runtime)  # owner commits are never filtered
            torch.testing.assert_close(owner.values[owner.row_map[owned]], torch.full((2, 2), float(t + 1), device=device))
            assert len(runtime.pending_snapshot_pushes) == 1
        assert ages == [0, 0, 1, 2, 0, 1, 2]
        assert gamma_grad > 0
        finish_state_update(runtime, final=True)
        assert torch.all(runtime.snapshot_history.read(runtime.snapshot_recv_nodes, 7)[:, -1] == 7)
        assert comm.launch_all_gather.call_count == 0
        pushes = [call for call in comm.launch_push.call_args_list if call.kwargs.get("name", "").startswith("snapshot_boundary:")]
        assert len(pushes) == 14
        assert sum(call.args[1].shape[0] == 0 for call in pushes) == 8
        assert runtime.snapshot_history.packets.shape[0] == 4
        runtime.reset()
        assert not runtime.pending_snapshot_pushes
        assert runtime.snapshot_history.packets.count_nonzero() == 0
        assert runtime.change_filter.count.count_nonzero() == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("model_cls", [sg.GConvGRUModel, sg.DCRNNModel])
@pytest.mark.parametrize("mode,beta", [("cache", 0.), ("mean", 1.), ("learnable", 0.5)])
def test_boundary_prediction_uses_only_remote_rows_and_has_exact_ablation_math(model_cls, mode, beta):
    model = model_cls(1, 2, 1, state_extrapolation=mode != "cache",
                      gamma_boundary_init=0.0 if mode == "learnable" else None)
    block = sg.GraphBlock(src_nodes=torch.arange(2), dst_nodes=torch.tensor([0]),
        edge_ids=torch.tensor([0]), format="coo", row=torch.tensor([1]), col=torch.tensor([0]), num_src=2, num_dst=1)
    block.cache["snapshot_id"] = 3
    # Owner row deliberately has an old cache too; it must never be compensated.
    packet = torch.tensor([[10., 20., 8., 9., 1., 1.], [2., 4., 1., 2., 1., 1.]])
    batch = sg.Batch(mode="snapshot", blocks=((block,),), state={
        "neighbor_recurrent_snapshots": (packet,), "neighbor_recurrent_cold_src_rows": (torch.tensor([1]),)})
    prepared, _ = model.runtime_prepare_scan(batch)
    actual = prepared.state["neighbor_recurrent_window_state"][0]
    torch.testing.assert_close(actual[0], packet[0, :2])
    torch.testing.assert_close(actual[1], packet[1, :2] + beta * 2 * packet[1, 2:4])
    assert not hasattr(model, "runtime_smooth_state")
    if mode == "learnable":
        actual.sum().backward()
        torch.testing.assert_close(model.gamma_boundary.grad, torch.tensor(1.5))
