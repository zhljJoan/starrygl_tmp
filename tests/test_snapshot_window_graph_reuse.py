"""Bounded layout reuse must preserve fresh batch inputs and training math."""
from collections.abc import Mapping
from unittest.mock import patch

import dgl
import pytest
import torch

import starrygl as sg
from starrygl.prepare.snapshot_csc import build_snapshot_csc_views
from starrygl.runtime.comm import CommScheduler
from starrygl.runtime.dataloader.loader import DataLoader
from starrygl.runtime.dataloader.pipeline import move_batch
from starrygl.runtime.epoch import with_model_snapshot_options
from starrygl.runtime.snapshot import cache as snapshot_cache
from starrygl.store import FeatureManager, GraphStore, LabelStore, StoreBundle


class _CountRows(list):
    reads = 0

    def __getitem__(self, index):
        self.reads += 1
        return super().__getitem__(index)


def _loader(model, *, rolling, pipeline=False, device="cpu", **overrides):
    src, dst = torch.tensor([0, 1, 2, 3, 0]), torch.tensor([1, 2, 3, 0, 3])
    ptr = torch.arange(5)[:, None] * 5 + torch.tensor([0, 5])
    views = build_snapshot_csc_views(
        src=src.repeat(5), dst=dst.repeat(5), ts=None,
        edge_ids=torch.arange(25), edge_dist_index=torch.arange(25),
        node_master=torch.zeros(4, dtype=torch.long),
        hot_node_ids=torch.empty(0, dtype=torch.long),
        node_is_hot=torch.zeros(4, dtype=torch.bool),
        node_to_chunk=torch.zeros(4, dtype=torch.long), time_ptr_2=ptr,
        num_nodes=4, world_size=1, diffusion=type(model) is sg.DCRNNModel,
    )
    store = StoreBundle(
        graph=GraphStore(num_nodes=4, prepare={
            "meta": {"world_size": 1}, "partition": {"node_dist_index": torch.arange(4)},
            "time_ptr_2": ptr, "snapshot_csc_views": views,
        }),
        features=FeatureManager(
            node_features={"x": torch.arange(20).reshape(5, 4, 1).float() / 10},
            edge_features={"edge": torch.arange(25).reshape(25, 1).float()},
        ),
        labels=LabelStore(task_kind="node", task_ptr=torch.arange(6) * 4,
                          task_payload={"node_ids": torch.arange(4).repeat(5),
                                        "label": torch.arange(20).reshape(20, 1).float() / 20}),
    )
    options = with_model_snapshot_options(
        {"rolling_snapshot_cache": rolling, "snapshot_dgl_gcn": True}, model,
    )
    options.update(overrides)
    store.graph.snapshot_csc_view["slices"] = _CountRows(store.graph.snapshot_csc_view["slices"])
    return DataLoader(store, mode="snapshot", split="train", window_policy="full_snapshot",
                      sampling_policy="full", chunk_decay=None, num_full_snapshots=3,
                      num_layers=1, fanouts=None, sampler_options=options, num_negatives=0,
                      generator=None, comm=CommScheduler(), device=device,
                      prefetch_state=None, enabled=pipeline)


def _assert_no_live_tensors(value):
    if isinstance(value, torch.Tensor):
        assert not value.requires_grad and value.grad_fn is None
    elif isinstance(value, Mapping):
        for child in value.values():
            _assert_no_live_tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _assert_no_live_tensors(child)
    elif isinstance(value, dgl.DGLGraph):
        assert not value.ndata and not value.edata


@pytest.mark.parametrize("model_cls", [sg.GConvGRUModel, sg.DCRNNModel])
@pytest.mark.parametrize("pipeline", [False, True])
def test_reuse_preserves_outputs_gradients_and_fresh_batch_inputs(model_cls, pipeline):
    records, counts = [], []
    for rolling in (False, True):
        torch.manual_seed(19)
        model = model_cls(1, 2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loader = _loader(model, rolling=rolling, pipeline=pipeline)
        record, observed = [], {}
        with patch.object(snapshot_cache, "_snapshot_graph_block", wraps=snapshot_cache._snapshot_graph_block) as builds, \
             patch.object(dgl, "graph", wraps=dgl.graph) as dgl_builds:
            for step, batch in enumerate(loader):
                assert batch.graph is batch.blocks[-1][-1]
                assert batch.targets["task"].label[0].item() == pytest.approx(step / 5)
                state = torch.full((4, 2), step / 10, requires_grad=True)
                batch.state = {"neighbor_recurrent": state, "neighbor_recurrent_node_ids": torch.arange(4)}
                for (block,), features in zip(batch.blocks, batch.features["x"]):
                    sid = block.cache["snapshot_id"]
                    assert block.cache["comm"] is loader.comm
                    features.requires_grad_()
                    if sid in observed:
                        previous, previous_features = observed[sid]
                        assert block is not previous  # batch-owned mutable dictionaries
                        assert (block.cache is previous.cache) == rolling
                        if rolling:
                            assert block.indices is previous.indices
                        assert block.edata is not previous.edata
                        assert features.data_ptr() != previous_features.data_ptr()
                    observed[sid] = block, features
                optimizer.zero_grad()
                output = model.encode(batch)
                loss = (output.logits - batch.targets["task"].label).square().mean()
                loss.backward()
                record.append(output.logits.detach().clone())
                record.extend(p.grad.clone() for p in model.parameters() if p.grad is not None)
                assert state.grad is not None
                record.append(state.grad.clone())
                optimizer.step()
                if rolling:
                    assert len(loader._graph_cache) <= 3
                    for source, static in loader._graph_cache.values():
                        assert "edge_feat" not in source.edata and "edge_feat" not in static.edata
                        _assert_no_live_tensors(static.cache)
            counts.append((builds.call_count, dgl_builds.call_count,
                           loader.store.graph.snapshot_csc_view["slices"].reads))
        if rolling:
            assert {source.cache["snapshot_id"] for source, _ in loader._graph_cache.values()} == {2, 3, 4}
            assert len(loader._entry_cache) == 3
            for entry in loader._entry_cache.values():
                assert not entry.features and entry.pending_node_features is None
                _assert_no_live_tensors(entry.row)
            assert not loader.store.graph.runtime_cache.get("snapshot_full_entries")
        records.append(record)
    assert len(records[0]) == len(records[1])
    for before, after in zip(*records):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert [count[0] for count in counts] == [12, 5]
    assert [count[2] for count in counts] == [12, 5]
    factor = 2 if model_cls is sg.DCRNNModel else 1
    assert [count[1] for count in counts] == [12 * factor, 5 * factor]


def test_reuse_bounds_scope_and_restarts_cleanly():
    loader = _loader(sg.GConvGRUModel(1, 2, 1), rolling=True, device=None)
    for _ in range(2):
        iterator = iter(loader)
        first, second = next(iterator), next(iterator)
        assert len(loader._graph_cache) == 2
        first.graph.edata["batch_only"] = torch.ones(1, requires_grad=True) * 2
        assert "batch_only" not in second.blocks[0][0].edata
        assert "batch_only" not in loader._graph_cache[id(loader._entry_cache[(0, -1)].graph)][1].edata
        iterator.close()
    assert _loader(sg.GConvGRUModel(1, 2, 1), rolling=False)._graph_cache is None
    for limit in (0, 1):
        assert _loader(sg.GConvGRUModel(1, 2, 1), rolling=True, full_snapshot_chunk_limit=limit)._graph_cache is None
    class CustomModel(sg.GConvGRUModel):
        pass
    assert _loader(CustomModel(1, 2, 1), rolling=True)._graph_cache is None


def test_device_memo_reuses_only_static_graph_storage_without_a_gpu():
    loader = _loader(sg.GConvGRUModel(1, 2, 1), rolling=True)
    batch = next(iter(loader))
    source, memo = batch.graph, {}
    moved = []
    for step in range(2):
        current = sg.Batch(mode="snapshot", graph=source, blocks=((source,),),
                           features={"x": torch.full((4, 1), float(step))},
                           state={"hidden": torch.zeros(4, 2, requires_grad=True)},
                           targets={"ids": torch.tensor([step])})
        moved.append(move_batch(current, "meta", graph_cache=memo))
    before, after = moved
    assert before.graph is not after.graph
    assert before.graph.indices is after.graph.indices
    assert before.graph.cache is after.graph.cache
    assert before.features["x"] is not after.features["x"]
    assert before.state["hidden"] is not after.state["hidden"]
    assert before.targets["ids"] is not after.targets["ids"]
    assert memo[id(source)][0] is source and len(memo) == 1
