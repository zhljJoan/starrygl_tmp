"""Full and partial snapshots share an epoch permutation without changing math."""
import pytest
import torch
import starrygl as sg

from starrygl.runtime.snapshot.cache import _materialize_snapshot_entries
from test_flare_snapshot_hotpath import _chunk_loader
from test_snapshot_device_materialization import _store


@pytest.mark.parametrize("model_cls", [sg.TGCNModel, sg.MPNNLSTMModel, sg.EvolveGCNModel])
def test_full_order_preserves_two_epoch_outputs_gradients_and_owner_targets(model_cls):
    records = []
    for ordered in (False, True):
        torch.manual_seed(19)
        model = model_cls(1, 3, 1)
        task = sg.NodePredictionTask(name="node_regression", loss="mse", train_loss_mode="window_mean")
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        loader = _chunk_loader(model, reuse=True, decay=(1,), full=1)
        loader.options.update(snapshot_materialize_on_device=ordered, _materialize_device="cpu",
                              _snapshot_train_loss_mode="window_mean")
        values, epochs = [], []
        for priority in (torch.tensor([3, 1, 2, 0]), torch.tensor([1, 3, 0, 2])):
            loader.options["chunk_order"] = priority
            expected = torch.argsort(priority[torch.arange(8) // 2], stable=True)
            layouts = []
            for batch in loader:
                for feature in batch.features["x"]:
                    feature.requires_grad_()
                optimizer.zero_grad()
                output = model.encode(batch)
                loss = task.compute_loss(output, batch)
                loss.backward()
                values.append(loss.detach())
                for slot, (block,) in enumerate(batch.blocks):
                    torch.testing.assert_close(batch.features["x"][slot],
                        loader.store.features.node_features["x"][block.cache["snapshot_id"], block.src_nodes])
                    values.append(output.aux["window_logits"][slot].detach()[block.dst_nodes.argsort()])
                    grad = batch.features["x"][slot].grad
                    values.append(None if grad is None else grad[block.src_nodes.argsort()])
                    if not block.cache["chunk_limited"]:
                        assert block.dst_nodes.tolist() == (expected if ordered else torch.arange(8)).tolist()
                        assert bool(block.cache.get("chunk_prefix_ordered", False)) is ordered
                        layouts.append(block.dst_nodes.tolist())
                target = batch.targets["task"]
                torch.testing.assert_close(batch.blocks[-1][-1].dst_nodes[target.target_route.target_rows], target.target_ids)
                values.extend(p.grad.clone() for p in model.parameters() if p.grad is not None)
                optimizer.step()
                assert len(loader._blob_cache) <= 2
                assert len({sid for sid, _ in loader._entry_cache}) <= 2
                assert not loader.store.graph.runtime_cache.get("snapshot_full_entries")
            epochs.append(layouts)
        if ordered:
            assert epochs[0] != epochs[1]
        records.append(values)
    assert len(records[0]) == len(records[1])
    for before, after in zip(*records):
        if before is None:
            assert after is None
        else:
            torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("rolling", [False, True])
@pytest.mark.parametrize("defer_launch", [False, True])
def test_full_order_remaps_route_and_features_without_mutating_prepared_rows(rolling, defer_launch):
    store = _store(rank=0, world=2)
    store.features.edge_features["w"] = torch.arange(48).float().unsqueeze(1)
    row = store.graph.snapshot_csc_view["slices"][0]
    original = row["dst_nodes"].clone()
    args = dict(store=store, rows=((row, -1), (row, 1)), chunk_order=torch.tensor([1, 0]), comm=None,
        options={"snapshot_materialize_on_device": True, "_materialize_device": "cpu",
                 "_snapshot_train_loss_mode": "window_mean",
                 "snapshot_reverse_direction": False, "defer_feature_launch": defer_launch},
        entry_cache={} if rolling else None, blob_cache={} if rolling else None)
    full, partial = _materialize_snapshot_entries(**args)
    assert full.graph.dst_nodes.tolist() == [2, 3, 0, 1]
    assert partial.graph.dst_nodes.tolist() == [2, 3]
    assert full.graph.cache["chunk_prefix_ordered"] is True
    assert not full.graph.cache["chunk_limited"] and partial.graph.cache["chunk_limited"]
    assert partial.graph.route is None
    torch.testing.assert_close(full.graph.dst_nodes[full.graph.route["send_index"]],
                               row["dst_nodes"][row["route"]["send_index"]])
    torch.testing.assert_close(full.graph.route["recv_src_row"], row["route"]["recv_src_row"])
    torch.testing.assert_close(full.row["src_feature_row"],
                               row["src_feature_row"][[2, 3, 0, 1, *range(4, row["src_nodes"].numel())]])
    torch.testing.assert_close(row["dst_nodes"], original)
    if not defer_launch:
        torch.testing.assert_close(full.features["w"], store.features.edge_features["w"][full.graph.edge_ids])
    assert not store.graph.runtime_cache.get("snapshot_full_entries")


@pytest.mark.parametrize("loss_mode", [None, "last_only"])
def test_default_and_evaluation_keep_full_owner_order(loss_mode):
    loader = _chunk_loader(sg.TGCNModel(1, 3, 1), reuse=True,
                           order=torch.tensor([3, 1, 2, 0]), decay=(1,), full=1)
    loader.options.update(snapshot_materialize_on_device=True, _materialize_device="cpu",
                          _snapshot_train_loss_mode=loss_mode)
    for batch in loader:
        full = batch.blocks[-1][-1]
        torch.testing.assert_close(full.dst_nodes, torch.arange(8))
        assert not full.cache.get("chunk_prefix_ordered", False)
