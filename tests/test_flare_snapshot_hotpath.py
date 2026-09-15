"""Snapshot topology reuse and node-identity recurrence."""
from unittest.mock import patch
import dgl
import pytest
import torch
import starrygl as sg
from starrygl.prepare.snapshot_csc import build_snapshot_csc_views
from starrygl.runtime.comm import CommScheduler
from starrygl.runtime.dataloader.loader import DataLoader
from starrygl.runtime.epoch import with_model_snapshot_options
from starrygl.runtime.snapshot import cache
from starrygl.runtime.snapshot.scan import run_decoupled_window_dag_scan
from starrygl.store import FeatureManager, GraphStore, LabelStore, StoreBundle
from test_snapshot_window_graph_reuse import _assert_no_live_tensors

MODELS = (sg.TGCNModel, sg.MPNNLSTMModel, sg.EvolveGCNModel)

def _chunk_loader(model, *, reuse=False, pipeline=False, order=None, decay=(2,1,1), full=2):
    n,t=8,5
    src=torch.arange(n).repeat_interleave(2)
    dst=(src+torch.tensor([1,2]).repeat(n))%n
    e=src.numel()
    ptr=torch.arange(t)[:,None]*e+torch.tensor([0,e])
    views=build_snapshot_csc_views(src=src.repeat(t),dst=dst.repeat(t),ts=torch.arange(t).repeat_interleave(e).float(),
        edge_ids=torch.arange(t*e),edge_dist_index=torch.arange(t*e),
        node_master=torch.zeros(n,dtype=torch.long),hot_node_ids=torch.empty(0,dtype=torch.long),
        node_is_hot=torch.zeros(n,dtype=torch.bool),node_to_chunk=torch.arange(n)//2,
        time_ptr_2=ptr,num_nodes=n,world_size=1,diffusion=False)
    store=StoreBundle(graph=GraphStore(num_nodes=n,prepare={'meta':{'world_size':1},
        'partition':{'node_dist_index':torch.arange(n)},'time_ptr_2':ptr,'snapshot_csc_views':views}),
        features=FeatureManager(node_features={'x':torch.arange(t*n).reshape(t,n,1).float()/10},edge_features={}),
        labels=LabelStore(task_kind='node',task_ptr=torch.arange(t+1)*n,
                         task_payload={'node_ids':torch.arange(n).repeat(t),'label':torch.arange(t*n).reshape(-1,1).float()/20}))
    options={'rolling_snapshot_cache':True,'snapshot_dgl_gcn':True,
             'snapshot_reverse_direction':False,'_snapshot_edge_feature_names':('w',),
             'chunk_order':torch.arange(4) if order is None else order}
    options=with_model_snapshot_options(options,model)
    options['_reuse_static_snapshot_graph'] = bool(reuse and options['_reuse_static_snapshot_graph'])
    out=DataLoader(store,mode='snapshot',split='train',window_policy='chunk_decay',sampling_policy='full',
        chunk_decay=decay,num_full_snapshots=full,num_layers=1,fanouts=None,sampler_options=options,
        num_negatives=0,generator=None,comm=CommScheduler(),device='cpu',prefetch_state=None,enabled=pipeline)
    return out


class _SumCell:
    state_key = "node_recurrent"
    num_gcn_layers = 1

    def materialize(self, blocks, src):
        return self.finalize_gcn(src["x"]), blocks[-1]

    def compute_gcn_layer(self, layer, blocks, value):
        return value

    def finalize_gcn(self, value):
        return {"x": value, "state_like": value}

    def local_forward(self, block, src, dst):
        return src["x"] + dst["h_prev"]


@pytest.mark.parametrize("sequential", [False, True])
def test_chunk_to_full_recurrence_follows_node_ids_and_gradients(monkeypatch, sequential):
    monkeypatch.setenv("STARRYGL_DISABLE_LAYERWISE_DAG", str(int(sequential)))
    batch = list(_chunk_loader(sg.TGCNModel(1, 2, 1), pipeline=False,
        order=torch.tensor([3, 1, 2, 0]), decay=(1,), full=1))[1]
    assert batch.blocks[0][0].dst_nodes.tolist() == [6, 7]
    assert batch.blocks[1][0].dst_nodes.tolist() == list(range(8))
    batch.features = {"x": (torch.ones(2, 1, requires_grad=True),
                             torch.ones(8, 1, requires_grad=True))}
    output = run_decoupled_window_dag_scan(batch, input_project=lambda x: x, cell=_SumCell())
    torch.testing.assert_close(output.embeddings.flatten(), torch.tensor([1., 1., 1., 1., 1., 1., 2., 2.]))
    (output.embeddings.flatten() * torch.arange(1, 9)).sum().backward()
    torch.testing.assert_close(batch.features["x"][0].grad.flatten(), torch.tensor([7., 8.]))


@pytest.mark.parametrize("model_cls", MODELS)
def test_chunk_static_cache_preserves_two_epoch_training(model_cls):
    records, counts, layouts = [], [], []
    for reuse in (False, True):
        torch.manual_seed(19)
        model = model_cls(1, 2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        load = _chunk_loader(model, reuse=reuse, pipeline=True)
        record, layout = [], []
        with patch.object(cache, "_snapshot_graph_block", wraps=cache._snapshot_graph_block) as builds, \
             patch.object(dgl, "graph", wraps=dgl.graph) as dgl_builds:
            for epoch in range(2):
                load.options["chunk_order"] = torch.arange(4) if not epoch else torch.tensor([3, 1, 2, 0])
                for batch in load:
                    for value in batch.features["x"]:
                        value.requires_grad_()
                    optimizer.zero_grad()
                    output = model.encode(batch)
                    (output.logits - batch.targets["task"].label).square().mean().backward()
                    record.append(output.logits.detach().clone())
                    record.extend(p.grad.clone() for p in model.parameters() if p.grad is not None)
                    record.extend(value.grad.clone() for value in batch.features["x"] if value.grad is not None)
                    optimizer.step()
                    layout.append([(b[0].cache["snapshot_id"], b[0].dst_nodes.tolist()) for b in batch.blocks])
                    if reuse:
                        assert load._graph_cache is not None and len(load._graph_cache) <= 5
                        for source, static in load._graph_cache.values():
                            _assert_no_live_tensors(static.cache)
                            assert "edge_feat" not in source.edata
                        for entry in load._entry_cache.values():
                            assert not entry.features and entry.pending_node_features is None
            counts.append((builds.call_count, dgl_builds.call_count))
        records.append(record)
        layouts.append(layout)
    assert layouts[0] == layouts[1]
    assert len(records[0]) == len(records[1])
    for before, after in zip(*records):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert counts == [(30, 30), (20, 20)]


def test_new_graph_reuse_is_dgl_only_and_does_not_apply_to_custom_models():
    class CustomTGCN(sg.TGCNModel):
        pass
    for model_cls in MODELS:
        model = model_cls(1, 2, 1)
        assert with_model_snapshot_options({}, model)["_snapshot_edge_feature_names"] == ("w",)
        assert not with_model_snapshot_options({}, model)["_reuse_static_snapshot_graph"]
        assert with_model_snapshot_options({"snapshot_dgl_gcn": True}, model)["_reuse_static_snapshot_graph"]
        assert not with_model_snapshot_options({"snapshot_dgl_gcn": True, "snapshot_sparse_gcn": True}, model)["_reuse_static_snapshot_graph"]
    options = with_model_snapshot_options({"snapshot_dgl_gcn": True}, CustomTGCN(1, 2, 1))
    assert not options["_reuse_static_snapshot_graph"] and options["_snapshot_edge_feature_names"] is None


@pytest.mark.parametrize("model_cls", MODELS)
def test_weighted_projection_preserves_outputs_and_gradients(model_cls):
    from copy import deepcopy
    from test_model_edge_features import _case, _finish

    torch.manual_seed(47)
    initial = model_cls(1, 3, 1)
    results = []
    for projected in (False, True):
        model = deepcopy(initial)
        store, batch = _case()
        batch = _finish(store, batch, with_model_snapshot_options({}, model) if projected else {})
        value = batch.features["x"][0].requires_grad_()
        output = model.encode(batch)
        output.embeddings.square().sum().backward()
        results.append((output.embeddings.detach(), value.grad,
                        {name: p.grad for name, p in model.named_parameters()}))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
    for name, before in results[0][2].items():
        after = results[1][2][name]
        if before is None:
            assert after is None
        else:
            torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize("model_cls", MODELS)
@pytest.mark.parametrize("pipeline", [False, True])
def test_full_graph_reuse_preserves_five_step_training(model_cls, pipeline):
    from test_snapshot_window_graph_reuse import _loader

    records, counts = [], []
    for reuse in (False, True):
        torch.manual_seed(19)
        model = model_cls(1, 2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        load = _loader(model, rolling=reuse, pipeline=pipeline)
        record = []
        with patch.object(cache, "_snapshot_graph_block", wraps=cache._snapshot_graph_block) as builds, \
             patch.object(dgl, "graph", wraps=dgl.graph) as dgl_builds:
            for batch in load:
                for value in batch.features["x"]:
                    value.requires_grad_()
                optimizer.zero_grad()
                output = model.encode(batch)
                (output.logits - batch.targets["task"].label).square().mean().backward()
                record.append(output.logits.detach().clone())
                record.extend(p.grad.clone() for p in model.parameters() if p.grad is not None)
                record.extend(value.grad.clone() for value in batch.features["x"] if value.grad is not None)
                optimizer.step()
                if reuse:
                    for _, static in load._graph_cache.values():
                        _assert_no_live_tensors(static.cache)
            counts.append((builds.call_count, dgl_builds.call_count))
        records.append(record)
    assert len(records[0]) == len(records[1])
    for before, after in zip(*records):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert counts == [(12, 12), (5, 5)]


@pytest.mark.parametrize("ordered_full", [False, True])
def test_manual_snapshot_rows_are_matched_without_prepared_layout_flags(ordered_full):
    from starrygl.runtime.state.recurrent import previous_state

    first = sg.graph_block_from_coo(src=torch.arange(3), dst=torch.arange(3),
                                    edge_ids=torch.arange(3), num_nodes=3, format="coo")
    second = sg.graph_block_from_coo(src=torch.arange(3), dst=torch.arange(3),
                                     edge_ids=torch.arange(3), num_nodes=3, format="coo")
    second.dst_nodes = torch.tensor([2, 0, 1])
    if ordered_full:
        first.cache["chunk_limited"] = second.cache["chunk_limited"] = False
        second.cache["chunk_prefix_ordered"] = True
    carried = torch.tensor([[1.], [2.], [3.]], requires_grad=True)
    actual = previous_state(batch_state=None, current=torch.zeros(3, 1), carried=carried,
                            persist_state=False, previous_block=first, current_block=second)
    torch.testing.assert_close(actual.flatten(), torch.tensor([3., 1., 2.]))
    actual[0].backward()
    torch.testing.assert_close(carried.grad.flatten(), torch.tensor([0., 0., 1.]))


def test_non_nested_partial_layouts_follow_node_ids():
    from starrygl.runtime.state.recurrent import previous_state

    first = sg.graph_block_from_coo(src=torch.arange(2), dst=torch.arange(2),
                                    edge_ids=torch.arange(2), num_nodes=2, format="coo")
    second = sg.graph_block_from_coo(src=torch.arange(4), dst=torch.arange(4),
                                     edge_ids=torch.arange(4), num_nodes=4, format="coo")
    first.dst_nodes, second.dst_nodes = torch.tensor([6, 7]), torch.tensor([2, 3, 6, 7])
    first.cache["chunk_limited"] = second.cache["chunk_limited"] = True
    result = previous_state(batch_state=None, current=torch.zeros(4, 1), carried=torch.ones(2, 1),
                            persist_state=False, previous_block=first, current_block=second)
    torch.testing.assert_close(result.flatten(), torch.tensor([0., 0., 1., 1.]))
