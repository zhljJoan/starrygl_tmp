"""Only exact built-in snapshot model types may omit unconsumed edge fields."""
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

import starrygl as sg
from starrygl.runtime.comm import CommScheduler
from starrygl.runtime.dataloader import features as feature_launch
from starrygl.runtime.dataloader.pipeline import finish_pending_feature_fetches
from starrygl.runtime.epoch import with_model_snapshot_options
from starrygl.store import FeatureManager, GraphStore, LabelStore, StoreBundle
from test_snapshot_window_graph_reuse import _loader


def _case(mode="snapshot"):
    store = StoreBundle(
        graph=GraphStore(num_nodes=4),
        features=FeatureManager(
            node_features={"x": torch.arange(4).float().reshape(4, 1) / 3},
            edge_features={"edge": torch.arange(5).float().reshape(5, 1),
                           "w": torch.tensor([1., 2., 4., 3., 7.]).reshape(5, 1),
                           "custom": torch.ones(5, 2)},
        ), labels=LabelStore(),
    )
    block = sg.graph_block_from_coo(
        src=torch.tensor([0, 1, 2, 3, 0]), dst=torch.tensor([1, 2, 3, 0, 3]),
        edge_ids=torch.arange(5), num_nodes=4, format="coo",
    )
    block.cache["snapshot_id"] = 0
    targets = {} if mode == "snapshot" else {"events": sg.EventRows(
        src=torch.tensor([0]), dst=torch.tensor([1]), edge_ids=torch.tensor([0]))}
    return store, sg.Batch(mode=mode, graph=block, blocks=((block,),), targets=targets)


def _finish(store, batch, options):
    batch, nodes, edges = feature_launch.launch_batch_features(
        batch, store, comm=CommScheduler(), sampler_options=options,
    )
    return finish_pending_feature_fetches(batch, nodes, edges)


@pytest.mark.parametrize("model_cls", [sg.GConvGRUModel, sg.DCRNNModel,
                                       sg.TGCNModel, sg.MPNNLSTMModel, sg.EvolveGCNModel])
def test_model_loader_never_launches_unconsumed_edge_feature_fetch(model_cls):
    loader = _loader(model_cls(1, 2, 1), rolling=True)
    original = loader.store.features.edge_features["edge"]
    with patch.object(feature_launch, "launch_edge_feature_fetch",
                      wraps=feature_launch.launch_edge_feature_fetch) as fetch:
        batches = list(loader)
    assert fetch.call_count == 0
    assert len(batches) == 5
    assert loader.store.features.edge_features == {"edge": original}
    assert all("edge" not in batch.features for batch in batches)
    assert all("edge_feat" not in block.edata for batch in batches
               for blocks in batch.iter_blocks() for block in blocks)


class CustomGConvGRU(sg.GConvGRUModel):
    pass


@pytest.mark.parametrize("model_cls,mode,expected", [
    (sg.GConvGRUModel, "snapshot", ("w",)),
    (sg.DCRNNModel, "snapshot", ("w",)),
    (sg.GConvGRUModel, "event", ("edge", "w", "custom")),
    (CustomGConvGRU, "snapshot", ("edge", "w", "custom")),
    (sg.TGCNModel, "snapshot", ("w",)),
    (sg.MPNNLSTMModel, "snapshot", ("w",)),
    (sg.EvolveGCNModel, "snapshot", ("w",)),
    (None, "snapshot", ("edge", "w", "custom")),
])
def test_projection_keeps_weights_and_preserves_other_callers(model_cls, mode, expected):
    store, batch = _case(mode)
    before = dict(store.features.edge_features)
    options = {} if model_cls is None else with_model_snapshot_options({}, model_cls(1, 2, 1))
    with patch.object(feature_launch, "launch_edge_feature_fetch",
                      wraps=feature_launch.launch_edge_feature_fetch) as fetch:
        result = _finish(store, batch, options)
    assert fetch.call_args.kwargs["names"] == expected
    assert tuple(store.features.edge_features) == tuple(before)
    assert all(store.features.edge_features[name] is value for name, value in before.items())
    torch.testing.assert_close(result.graph.edata["w"], before["w"], rtol=0, atol=0)
    assert ("edge_feat" in result.graph.edata) == ("edge" in expected)
    assert ("custom" in result.graph.edata) == ("custom" in expected)
    if mode == "event":
        torch.testing.assert_close(result.features["pos_edge_feat"][0], before["edge"][:1])


def test_direct_block_feature_launch_keeps_all_fields():
    store, batch = _case()
    with patch.object(feature_launch, "launch_edge_feature_fetch",
                      wraps=feature_launch.launch_edge_feature_fetch) as fetch:
        pending = feature_launch.launch_block_edge_features(
            store, batch.blocks, comm=CommScheduler())
    assert pending is not None
    assert fetch.call_args.kwargs["names"] == ("edge", "w", "custom")


def test_weight_field_still_launches_with_zero_local_edges():
    store, _ = _case()
    store.features.edge_features = {"edge": torch.empty(0, 1), "w": torch.empty(0, 1)}
    empty = torch.empty(0, dtype=torch.long)
    block = sg.graph_block_from_coo(src=empty, dst=empty, edge_ids=empty, num_nodes=4, format="coo")
    block.cache["snapshot_id"] = 0
    batch = sg.Batch(mode="snapshot", graph=block, blocks=((block,),))
    options = with_model_snapshot_options({}, sg.GConvGRUModel(1, 2, 1))
    with patch.object(feature_launch, "launch_edge_feature_fetch",
                      wraps=feature_launch.launch_edge_feature_fetch) as fetch:
        _, _, pending = feature_launch.launch_batch_features(batch, store, comm=CommScheduler(),
                                                             sampler_options=options)
    assert pending is not None and fetch.call_count == 1
    assert fetch.call_args.kwargs["names"] == ("w",)
    assert fetch.call_args.args[1].numel() == 0


@pytest.mark.parametrize("model_cls", [sg.GConvGRUModel, sg.DCRNNModel])
def test_weighted_cell_outputs_and_all_gradients_match_all_features(model_cls):
    torch.manual_seed(47)
    initial = model_cls(1, 3, 1)
    results = []
    for selected in (False, True):
        model = deepcopy(initial)
        store, batch = _case()
        options = with_model_snapshot_options({}, model) if selected else {}
        batch = _finish(store, batch, options)
        batch.features["x"][0].requires_grad_()
        hidden = torch.arange(12).float().reshape(4, 3).div(7).requires_grad_()
        batch.state = {"neighbor_recurrent": hidden, "neighbor_recurrent_node_ids": torch.arange(4)}
        output = model.encode(batch)
        output.embeddings.square().sum().backward()
        grads = {name: None if param.grad is None else param.grad.clone()
                 for name, param in model.named_parameters()}
        results.append((output.embeddings.detach().clone(), grads,
                        hidden.grad.clone(), batch.features["x"][0].grad.clone()))
    for before, after in zip(*results):
        if isinstance(before, dict):
            for name in before:
                if before[name] is None:
                    assert after[name] is None
                else:
                    torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
        else:
            torch.testing.assert_close(before, after, rtol=0, atol=0)
