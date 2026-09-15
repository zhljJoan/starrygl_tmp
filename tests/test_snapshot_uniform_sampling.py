"""Independent snapshot sampling; no claim to implement R-GraphSAGE math."""
import copy
import pytest
import torch

import starrygl as sg
from starrygl.native.sampling import NativeSamplingUnavailable
from starrygl.model.graph_conv import MeanGraphConv
from starrygl.prepare.snapshot_csc import build_snapshot_csc_views
from starrygl.prepare.temporal_csr import build_temporal_csr_view
from starrygl.runtime.comm import CommScheduler
from starrygl.runtime.dataloader.loader import DataLoader
from starrygl.runtime.loop import run_epoch
from starrygl.runtime.sample import build_native_sampler, _snapshot_sample_times, _sampler_topology
from starrygl.store import FeatureManager, GraphStore, LabelStore, StoreBundle
from starrygl.task import NodePredictionTask


def _store():
    src, dst = torch.tensor([0, 1, 2, 3, 0, 1]), torch.tensor([1, 2, 1, 2, 3, 0])
    ids = torch.tensor([50, 10, 40, 20, 70, 60])
    ts = torch.tensor([.2, .9, 1.1, 1.8, 4.2, 4.8])
    ptr = torch.tensor([[0, 2], [2, 2], [2, 4], [4, 6]])
    temporal = build_temporal_csr_view(src=src, dst=dst, ts=ts, edge_ids=ids,
        node_dist_index=torch.arange(4), edge_dist_index=torch.arange(6),
        node_to_chunk=torch.zeros(4, dtype=torch.long), edge_chunk=torch.zeros(6, dtype=torch.long),
        time_ptr_2=ptr, num_nodes=4, bidirectional=True, shared=False,
        node_is_hot=torch.zeros(4, dtype=torch.bool))
    # Keep reversed duplicates but deliberately remove physical/EID ordering.
    order = torch.tensor([5, 0, 10, 3, 8, 7, 4, 1, 11, 2, 9, 6])
    for key in ("src", "dst", "ts", "edge_ids", "edge_feature_ids", "edge_dist_index"):
        temporal[key] = temporal[key][order]
    views = build_snapshot_csc_views(src=src, dst=dst, ts=ts, edge_ids=ids,
        edge_dist_index=torch.arange(6), node_master=torch.zeros(4, dtype=torch.long),
        hot_node_ids=torch.empty(0, dtype=torch.long), node_is_hot=torch.zeros(4, dtype=torch.bool),
        node_to_chunk=torch.zeros(4, dtype=torch.long), time_ptr_2=ptr, num_nodes=4, world_size=1)
    targets = torch.tensor([0, 1, 2, 3, 1])
    return StoreBundle(
        graph=GraphStore(num_nodes=4, prepare={"meta": {"world_size": 1},
            "partition": {"node_dist_index": torch.arange(4)}, "time_ptr_2": ptr,
            "temporal_csr_view": temporal, "snapshot_csc_views": views}),
        features=FeatureManager(node_features={"x": torch.arange(32).reshape(4, 4, 2).float() / 30}),
        labels=LabelStore(task_kind="node", task_ptr=torch.arange(5) * 5,
            task_payload={"node_ids": targets.repeat(4),
                          "label": torch.arange(20).reshape(20, 1).float() / 20}))


def _sampler(store, **options):
    try:
        return build_native_sampler(store, store.graph.temporal_csr_view, fanouts=(8,),
            num_layers=1, options={"policy": "snapshot_uniform", "workers": 1, **options}, snapshot=True)
    except NativeSamplingUnavailable:
        pytest.skip("native sampler is not built")


def test_snapshot_topology_uses_physical_rows_and_existing_cache():
    store = _store()
    original = store.graph.temporal_csr_view["ts"].clone()
    options = {"policy": "snapshot_uniform"}
    topology = _sampler_topology(store, store.graph.temporal_csr_view, options)
    expected = torch.tensor([0, 0, 2, 2, 3, 3])[topology["edge_feature_ids"]]
    torch.testing.assert_close(topology["ts"], expected)
    assert _sampler_topology(store, store.graph.temporal_csr_view, options) is topology
    torch.testing.assert_close(store.graph.temporal_csr_view["ts"], original)
    assert len(topology["edge_ids"]) == 12  # Reverse copies are not duplicated again.


def test_native_snapshot_membership_empty_and_nonidentity_eids():
    store = _store()
    sampler = _sampler(store)
    assert _sampler(store) is sampler
    for sid, allowed in enumerate(({10, 50}, set(), {20, 40}, {60, 70})):
        blocks = sampler.sample_blocks(torch.tensor([0, 1, 2, 3, 1]), torch.full((5,), sid))
        assert len(blocks) == 1
        for block in blocks:
            assert set(block.edge_ids.tolist()) == allowed
            assert set(block.srcdata["ts"].tolist()) == {sid}
            assert set(block.dstdata["ts"].tolist()) == {sid}
            expected_rows = {0, 1} if sid == 0 else {2, 3} if sid == 2 else {4, 5} if sid == 3 else set()
            assert set(block.cache["edge_feature_ids"].tolist()) == expected_rows


@pytest.mark.parametrize("rows,windows", [([5], [[0, 2]]), ([-1], [[0, 2]]),
    ([2], [[0, 1], [3, 4]]), ([0], [[0, 2], [1, 3]]), ([0], [[2, 1]])])
def test_snapshot_topology_rejects_uncovered_or_overlapping_ranges(rows, windows):
    with pytest.raises(ValueError, match="snapshot_uniform"):
        _snapshot_sample_times({"edge_feature_ids": torch.tensor(rows)}, torch.tensor(windows))


def test_snapshot_policy_rejects_event_at_construction():
    store = _store()
    with pytest.raises(ValueError, match="requires Snapshot"):
        build_native_sampler(store, store.graph.temporal_csr_view, fanouts=(2,), num_layers=1,
                             options={"policy": "snapshot_uniform"})


def test_snapshot_policy_rejects_unavailable_multilayer_root_chain():
    store = _store()
    with pytest.raises(ValueError, match="requires num_layers=1"):
        build_native_sampler(store, store.graph.temporal_csr_view, fanouts=(2, 2), num_layers=2,
                             options={"policy": "snapshot_uniform"}, snapshot=True)


def test_sampled_mean_operator_matches_snapshot_matrix_and_gradients():
    sampler = _sampler(_store())
    block = sampler.sample_blocks(torch.arange(4), torch.full((4,), 2))[0]
    torch.manual_seed(3)
    conv = MeanGraphConv(2, 3)
    reference = copy.deepcopy(conv)
    x = torch.randn(4, 2, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    actual = conv(block, x[block.src_nodes])
    # Snapshot 2 is only 2--1 and 3--2; node 0 is isolated.
    mean = xr.new_tensor([[0, 0, 0, 0], [0, 0, 1, 0],
                         [0, .5, 0, .5], [0, 0, 1, 0]])
    expected = (mean @ reference.linear(xr) + reference.self_linear(xr))[block.dst_nodes]
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, xr.grad)
    for p, q in zip(conv.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_common_loader_keeps_duplicate_targets_and_existing_model_trains():
    store = _store()
    _sampler(store)  # Skip only when the real native prerequisite is absent.
    options = {"policy": "snapshot_uniform", "workers": 1, "access_pipeline": False}
    loader = DataLoader(store, mode="snapshot", split="train", window_policy="full_snapshot",
        sampling_policy="neighbor", chunk_decay=None, num_full_snapshots=2, num_layers=1,
        fanouts=(8,), sampler_options=options, num_negatives=0, generator=None,
        comm=CommScheduler(), device="cpu", prefetch_state=None, enabled=False)
    batch = list(loader)[3]
    target, graph = batch.targets["task"], batch.blocks[-1][-1]
    torch.testing.assert_close(graph.dst_nodes[target.target_route.target_rows], target.target_ids)
    assert target.target_ids.tolist() == [0, 1, 2, 3, 1]
    assert set(graph.dstdata["ts"].tolist()) == {3}
    assert set(target.target_ts.tolist()) == {5}
    assert target.label.shape == (5, 1)
    assert len(batch.blocks) == 2
    torch.manual_seed(19)
    model = sg.TGCNModel(2, 3, 1, num_layers=1)
    before = model.output.weight.detach().clone()
    result = run_epoch(store=store, model=model, task=NodePredictionTask(name="node_regression", loss="mse"),
        mode="snapshot", training=True, window_policy="full_snapshot", sampling_policy="neighbor",
        optimizer=torch.optim.Adam(model.parameters(), lr=.001), num_full_snapshots=2,
        num_layers=1, fanouts=(8,), sampler_options=options, device="cpu")
    assert result.steps == 4
    assert torch.isfinite(torch.tensor(result.loss))
    assert not torch.equal(before, model.output.weight)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
