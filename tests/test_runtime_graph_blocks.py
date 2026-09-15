from types import SimpleNamespace

import torch
import pytest

from starrygl.native import sampling as native_sampling
from starrygl.native.sampling import NativeSamplingUnavailable, NativeTemporalSampler
from starrygl.native.sampling_output import graph_block_from_native_mfg
from starrygl.runtime.dataloader.blocks import (
    event_rows_to_graph_block,
    root_nodes_to_graph_block,
    snapshot_row_to_graph_block,
)
from starrygl.runtime.sample.blocks import normalize_mfg_blocks, sampled_edge_ids, sampled_feature_graph
from starrygl.view import GraphBlock


def test_native_sampler_treats_zero_workers_as_synchronous(monkeypatch) -> None:
    captured = {}

    class NativeModule:
        @staticmethod
        def get_neighbors(*args):
            return args

        @staticmethod
        def ParallelSampler(*args):
            captured["workers"] = args[3]
            return SimpleNamespace()

    monkeypatch.setattr(native_sampling, "_load_native_sampler_module", lambda: NativeModule)

    NativeTemporalSampler.from_graph(
        {"src": torch.tensor([0]), "dst": torch.tensor([1]), "num_nodes": 2},
        fanouts=[1],
        workers=0,
    )

    assert captured["workers"] == 1


def test_native_compaction_preserves_node_timestamp_instances() -> None:
    try:
        sampler = NativeTemporalSampler.from_graph(
            {
                "src": torch.tensor([0, 0]),
                "dst": torch.tensor([1, 2]),
                "ts": torch.tensor([1, 2]),
                "num_nodes": 3,
            },
            fanouts=[2],
            workers=1,
            policy="recent",
            add_reverse_edges=True,
        )
    except NativeSamplingUnavailable:
        pytest.skip("native sampler is not built")

    block = sampler.sample_blocks(
        torch.tensor([0, 0, 0]),
        torch.tensor([2, 3, 2]),
    )[0]

    assert torch.equal(block.dst_nodes, torch.tensor([0, 0]))
    assert torch.equal(block.dstdata["ts"], torch.tensor([2, 3]))


def test_native_snapshot_stream_samples_prior_time_slices() -> None:
    try:
        sampler = NativeTemporalSampler.from_graph(
            {
                "src": torch.tensor([0, 2]),
                "dst": torch.tensor([1, 1]),
                "edge_ids": torch.tensor([10, 20]),
                "edge_feature_ids": torch.tensor([100, 200]),
                "ts": torch.tensor([0, 1]),
                "num_nodes": 3,
            },
            fanouts=[4],
            workers=1,
            policy="uniform",
            graph_name="test_native_snapshot_stream_prior",
        )
    except NativeSamplingUnavailable:
        pytest.skip("native sampler is not built")

    first = sampler.sample_blocks(torch.tensor([1]), torch.tensor([1]))[0]
    second = sampler.sample_blocks(torch.tensor([1]), torch.tensor([2]))[0]

    assert torch.equal(first.edge_ids, torch.tensor([10]))
    assert torch.equal(torch.sort(second.edge_ids).values, torch.tensor([10, 20]))


def test_root_nodes_to_graph_block_deduplicates_and_keeps_latest_ts() -> None:
    block = root_nodes_to_graph_block(
        torch.tensor([3, 1, 3, 2]),
        torch.tensor([0.1, 0.2, 0.9, 0.4]),
    )

    assert torch.equal(block.src_nodes, torch.tensor([1, 2, 3]))
    assert torch.equal(block.dst_nodes, torch.tensor([1, 2, 3]))
    assert torch.allclose(block.srcdata["ts"], torch.tensor([0.2, 0.4, 0.9]))
    assert block.exec_mode == "LOCAL_EVENT"


def test_event_rows_to_graph_block_adds_extra_nodes_and_event_ts() -> None:
    view = {
        "src": torch.tensor([0, 1, 2]),
        "dst": torch.tensor([1, 2, 3]),
        "edge_ids": torch.tensor([10, 11, 12]),
        "ts": torch.tensor([1.0, 2.0, 3.0]),
    }

    block = event_rows_to_graph_block(view, torch.tensor([1, 2]), extra_nodes=torch.tensor([5]))

    assert torch.equal(block.src_nodes, torch.tensor([1, 2, 3, 5]))
    assert torch.equal(block.edge_ids, torch.tensor([11, 12]))
    assert torch.equal(block.row, torch.tensor([0, 1]))
    assert torch.equal(block.col, torch.tensor([1, 2]))
    assert torch.allclose(block.edata["ts"], torch.tensor([2.0, 3.0]))


def test_snapshot_row_to_graph_block_attaches_cache_and_reverse_buffers() -> None:
    row = {
        "src_nodes": torch.tensor([0, 1, 2]),
        "dst_nodes": torch.tensor([0, 1]),
        "edge_ids": torch.tensor([20, 21]),
        "indptr": torch.tensor([0, 1, 2]),
        "indices": torch.tensor([1, 2]),
        "ts": torch.tensor([4.0, 5.0]),
        "src_feature_row": torch.tensor([10, 11, 12]),
        "dst_feature_row": torch.tensor([10, 11]),
        "route": {"send_sizes": [0, 1]},
    }

    block = snapshot_row_to_graph_block(row, precompute_edge_rows=True)

    assert block.format == "csc"
    assert block.exec_mode == "DISTRIBUTED_FULL"
    assert torch.equal(block.row, torch.tensor([1, 2]))
    assert torch.equal(block.col, torch.tensor([0, 1]))
    assert torch.equal(block.cache["src_state_rows"], torch.tensor([10, 11, 12]))
    assert torch.equal(block.cache["dst_state_rows"], torch.tensor([10, 11]))
    assert torch.equal(block.cache["reverse_row"], torch.tensor([0]))
    assert torch.equal(block.cache["reverse_col"], torch.tensor([1]))


@pytest.mark.parametrize("prepared", (True, False))
def test_embedding_route_reuses_bound_without_reducing_each_exchange(monkeypatch, prepared) -> None:
    from starrygl.runtime.snapshot.layerwise import ReadyLayerAwaitable, materialize_embedding_src_async

    nodes = torch.tensor([0, 1])
    empty = torch.empty(0, dtype=torch.long)
    route = {"send_sizes": (0, 1), "recv_sizes": (0, 1),
             "send_index": torch.tensor([1]), "recv_index": torch.tensor([2])}
    if prepared:
        block = snapshot_row_to_graph_block({
            "src_nodes": torch.tensor([0, 1, 2]), "dst_nodes": nodes,
            "edge_ids": empty, "indices": empty, "indptr": torch.zeros(3, dtype=torch.long),
            "route": route,
        }, attach_reverse=False)
    else:
        block = GraphBlock(src_nodes=nodes, dst_nodes=nodes, edge_ids=empty, format="coo", route=route)
    calls = []
    tensor_max = torch.Tensor.max

    def count_max(tensor, *args, **kwargs):
        calls.append(1)
        return tensor_max(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "max", count_max)
    comm = SimpleNamespace(launch_autograd_pull=lambda route, owned, **kwargs: ReadyLayerAwaitable(owned))
    owned = torch.randn(2, 3, requires_grad=True)
    for name in ("snapshot_exact_previous_state", "snapshot_reset_gate", "h:gcn_layer_1"):
        result = materialize_embedding_src_async(block, owned, comm=comm, name=name).wait()
        assert result is owned
    assert len(calls) == (0 if prepared else 1)
    with pytest.raises(RuntimeError, match="max_send_index=1, owned_rows=1"):
        materialize_embedding_src_async(block, owned[:1], comm=comm)
    assert len(calls) == (0 if prepared else 1)


def test_normalize_mfg_blocks_orders_chain_and_drops_trailing_empty() -> None:
    first = GraphBlock(
        src_nodes=torch.tensor([0, 1, 2]),
        dst_nodes=torch.tensor([1, 2]),
        edge_ids=torch.tensor([0, 1]),
        format="csc",
    )
    second = GraphBlock(
        src_nodes=torch.tensor([1, 2]),
        dst_nodes=torch.tensor([2]),
        edge_ids=torch.tensor([2]),
        format="csc",
    )
    empty_tail = GraphBlock(
        src_nodes=torch.tensor([], dtype=torch.long),
        dst_nodes=torch.tensor([], dtype=torch.long),
        edge_ids=torch.tensor([], dtype=torch.long),
        format="csc",
    )

    assert normalize_mfg_blocks((second, first)) == (first, second)
    assert normalize_mfg_blocks((first, second, empty_tail)) == (first, second)
    first.dstdata["ts"] = torch.tensor([1., 2.])
    second.srcdata["ts"] = torch.tensor([3., 4.])
    assert normalize_mfg_blocks((second, first)) == (second, first)


def test_sampled_feature_graph_uses_all_mfg_source_nodes_and_unique_edges() -> None:
    first = GraphBlock(
        src_nodes=torch.tensor([0, 1, 2]),
        dst_nodes=torch.tensor([1, 2]),
        edge_ids=torch.tensor([3, 1]),
        format="csc",
    )
    second = GraphBlock(
        src_nodes=torch.tensor([1, 2, 4]),
        dst_nodes=torch.tensor([2, 4]),
        edge_ids=torch.tensor([3, 2]),
        format="csc",
    )

    feature_graph = sampled_feature_graph((first, second))

    assert torch.equal(feature_graph.src_nodes, torch.tensor([0, 1, 2, 4]))
    assert feature_graph.exec_mode == "LOCAL_SAMPLE_FEATURE_TABLE"
    assert torch.equal(sampled_edge_ids((first, second)), torch.tensor([1, 2, 3]))


def test_native_block_keeps_external_edge_id_separate_from_feature_row() -> None:
    mfg = SimpleNamespace(
        dst_lids=torch.tensor([0]),
        src_lids=torch.tensor([0, 1]),
        csc_indptr=torch.tensor([0, 1]),
        csc_indices=torch.tensor([1]),
        edge_lids=torch.tensor([0]),
    )

    block = graph_block_from_native_mfg(
        mfg,
        node_gids=torch.tensor([2, 1]),
        edge_gids=torch.tensor([1]),
        edge_id_map=torch.tensor([10, 11]),
        edge_feature_id_map=torch.tensor([5, 7]),
        deduplicate_edges=False,
    )

    assert torch.equal(block.edge_ids, torch.tensor([11]))
    assert torch.equal(block.cache["edge_feature_ids"], torch.tensor([7]))
    assert torch.equal(sampled_edge_ids((block,)), torch.tensor([7]))
