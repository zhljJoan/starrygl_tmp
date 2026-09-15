from dataclasses import replace
import os

import pytest
import torch
import torch.distributed as dist

import starrygl as sg
from starrygl.runtime.comm import CommScheduler, Route
from starrygl.runtime.epoch import sync_gradients
from starrygl.runtime.snapshot.layerwise import (
    _run_flare_style_coroutines,
    materialize_embedding_src_async,
)
from starrygl.runtime.snapshot.scan import (
    _needs_layer_exchange,
    _run_sequential_window_scan,
    encode_model,
    materialize_coupled_neighbor_state,
    run_coupled_window_scan,
    run_decoupled_window_dag_scan,
)
from starrygl.runtime.endpoint import (
    materialize_endpoint_embeddings,
    materialize_endpoint_output,
)
from starrygl.model.apan import _apan_mailbox_update_values
from starrygl.model.tgcn import _linear_cat
from starrygl.model.layers import (
    EdgePredictor,
    IdentityNormLayer,
    JODIETimeEmbedding,
    StateIncrementEstimator,
    TGNMemoryUpdater,
    TemporalTransformerAttentionLayer,
    TimeEncode,
)


def _event_batch() -> sg.Batch:
    graph = sg.graph_block_from_coo(
        src=torch.tensor([0, 1, 2]),
        dst=torch.tensor([1, 2, 0]),
        edge_ids=torch.tensor([0, 1, 2]),
        num_nodes=3,
        format="coo",
    )
    target = sg.TaskTarget(
        target_kind="edge",
        target_ids=torch.tensor([0, 1]),
        target_ts=torch.tensor([0.0, 1.0]),
        pos_src=torch.tensor([0, 1]),
        pos_dst=torch.tensor([1, 2]),
        neg_src=torch.tensor([0, 1]),
        neg_dst=torch.tensor([2, 0]),
        edge_ids=torch.tensor([0, 1]),
    )
    events = sg.EventRows(
        src=target.pos_src,
        dst=target.pos_dst,
        edge_ids=target.edge_ids,
        ts=target.target_ts,
    )
    return sg.Batch(
        mode="event",
        graph=graph,
        blocks=((graph,),),
        features={"x": (torch.ones(3, 3),)},
        state={"node_memory": torch.zeros(3, 5)},
        targets={"task": target, "events": events},
    )


def _snapshot_batch() -> sg.Batch:
    graph = sg.graph_block_from_coo(
        src=torch.tensor([0, 1, 2]),
        dst=torch.tensor([1, 2, 0]),
        edge_ids=torch.tensor([0, 1, 2]),
        num_nodes=3,
        format="csc",
    )
    target = sg.TaskTarget(
        target_kind="node",
        target_ids=torch.tensor([0, 1, 2]),
        target_ts=torch.tensor([2.0, 2.0, 2.0]),
        label=torch.tensor([0, 1, 0]),
        node_ids=torch.tensor([0, 1, 2]),
    )
    return sg.Batch(
        mode="snapshot",
        graph=graph,
        blocks=((graph,),),
        features={"x": (torch.ones(3, 3),)},
        state={"node_recurrent": torch.zeros(3, 5)},
        targets={"task": target},
    )


def _two_window_two_layer_batch(mode: str = "snapshot") -> sg.Batch:
    graph = sg.graph_block_from_coo(
        src=torch.tensor([0, 1, 2]),
        dst=torch.tensor([1, 2, 0]),
        edge_ids=torch.tensor([0, 1, 2]),
        num_nodes=3,
        format="coo",
    )
    edge_target = sg.TaskTarget(
        target_kind="edge",
        target_ids=torch.tensor([0, 1]),
        target_ts=torch.tensor([1.0, 2.0]),
        pos_src=torch.tensor([0, 1]),
        pos_dst=torch.tensor([1, 2]),
        neg_src=torch.tensor([0, 1]),
        neg_dst=torch.tensor([2, 0]),
        edge_ids=torch.tensor([0, 1]),
    )
    node_target = sg.TaskTarget(
        target_kind="node",
        target_ids=torch.tensor([0, 1, 2]),
        target_ts=torch.tensor([2.0, 2.0, 2.0]),
        label=torch.tensor([0, 1, 0]),
        node_ids=torch.tensor([0, 1, 2]),
    )
    return sg.Batch(
        mode=mode,
        graph=graph,
        blocks=((graph, graph), (graph, graph)),
        features={"x": (torch.ones(3, 3), torch.full((3, 3), 2.0))},
        state={"node_memory": torch.zeros(3, 5), "node_recurrent": torch.zeros(3, 5)},
        targets={"task": edge_target if mode == "event" else node_target},
        num_layers=2,
    )


def _backward(output: sg.ModelOutput) -> None:
    loss = output.embeddings.sum()
    if output.logits is not None:
        loss = loss + output.logits.sum()
    if "pos_score" in output.aux:
        loss = loss + output.aux["pos_score"].sum()
    if "neg_score" in output.aux:
        loss = loss + output.aux["neg_score"].sum()
    loss.backward()


def test_temporal_layers_are_reusable_model_components() -> None:
    assert TimeEncode(4)(torch.tensor([0.0, 1.0])).shape == (2, 4)
    assert IdentityNormLayer(5)(torch.ones(2, 5)).shape == (2, 5)
    assert JODIETimeEmbedding(5)(torch.ones(2, 5), torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])).shape == (2, 5)
    assert StateIncrementEstimator(num_rows=3, dim=5).estimate(torch.tensor([0, 2])).shape == (2, 5)
    pos, neg = EdgePredictor(5)(torch.ones(2, 5), torch.ones(2, 5), h_neg_dst=torch.ones(4, 5), neg_samples=2)
    assert pos.shape == (2,)
    assert neg is not None
    assert neg.shape == (4,)
    updater = TGNMemoryUpdater(
        memory_dim=5,
        message_dim=10,
        time_dim=4,
        node_dim=5,
        combine_node_feature=True,
        memory_update="gru",
        state_compensation=False,
        compensation_num_rows=1,
        gamma_init=0.5,
    )
    assert updater is not None
    layer = TemporalTransformerAttentionLayer(
        node_dim=5,
        edge_dim=2,
        time_dim=4,
        num_heads=1,
        out_dim=5,
        dropout=0.0,
        att_dropout=0.0,
    )
    assert layer is not None


def test_transformer_memory_matches_memshare_update_order() -> None:
    torch.manual_seed(3)
    updater = TGNMemoryUpdater(
        memory_dim=4,
        message_dim=6,
        time_dim=2,
        node_dim=4,
        combine_node_feature=False,
        memory_update="transformer",
        state_compensation=False,
        compensation_num_rows=1,
        gamma_init=0.5,
        mailbox_size=2,
        transformer_heads=2,
    ).eval()
    memory = torch.randn(2, 4)
    mem_input = torch.randn(2, 6)
    memory_ts = torch.tensor([0.0, 1.0])
    node_ts = torch.tensor([4.0, 7.0])
    mailbox_ts = torch.tensor([[1.0, 3.0], [2.0, 5.0]])

    updated, _, _ = updater(
        node_feat=memory,
        memory=memory,
        memory_ts=memory_ts,
        node_ts=node_ts,
        mem_input=mem_input,
        historical_memory=None,
        shared_mask=None,
        shared_rows=None,
        mailbox_ts=mailbox_ts,
    )
    mails = torch.cat(
        (mem_input.reshape(2, 2, 3), updater.time_enc(node_ts[:, None] - mailbox_ts).reshape(2, 2, 2)),
        dim=2,
    )
    query = updater.w_q(memory).reshape(2, 2, 2)
    key = updater.w_k(mails).reshape(2, 2, 2, 2)
    value = updater.w_v(mails).reshape(2, 2, 2, 2)
    attention = torch.softmax(updater.att_act((query[:, None] * key).sum(dim=3)), dim=1)
    context = (attention[..., None] * value).sum(dim=1).reshape(2, 4)
    expected = torch.relu(updater.mlp(updater.layer_norm(memory + context)))

    assert torch.allclose(updated, expected)


def test_state_compensation_normalizes_transition_before_blending() -> None:
    updater = TGNMemoryUpdater(
        memory_dim=2,
        message_dim=2,
        time_dim=0,
        node_dim=2,
        combine_node_feature=False,
        memory_update="rnn",
        state_compensation=True,
        compensation_num_rows=2,
        gamma_init=0.0,
    )
    with torch.no_grad():
        for parameter in updater.updater.parameters():
            parameter.zero_()
        updater.increment.count.fill_(1)
        updater.increment.increment.copy_(torch.tensor([[1.0, 2.0], [3.0, 3.0]]))
    historical = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
    updated, _, aux = updater(
        node_feat=torch.zeros(2, 2),
        memory=torch.zeros(2, 2),
        memory_ts=torch.zeros(2),
        node_ts=torch.ones(2),
        mem_input=torch.zeros(2, 2),
        historical_memory=historical,
        shared_mask=torch.ones(2, dtype=torch.bool),
        shared_rows=torch.arange(2),
    )
    transition = historical + torch.tensor([[1.0, 2.0], [3.0, 3.0]])
    expected_prediction = 2 * (transition - transition.min()) / (transition.max() - transition.min()) - 1

    assert torch.allclose(aux["state_compensation_prediction"], expected_prediction)
    assert torch.allclose(updated, expected_prediction * 0.5)


def test_edge_predictor_triplet_matches_tgl_negative_order() -> None:
    predictor = EdgePredictor(2)
    with torch.no_grad():
        predictor.src_fc.weight.copy_(torch.eye(2))
        predictor.src_fc.bias.zero_()
        predictor.dst_fc.weight.copy_(torch.eye(2))
        predictor.dst_fc.bias.zero_()
        predictor.out_fc.weight.copy_(torch.tensor([[1.0, 10.0]]))
        predictor.out_fc.bias.zero_()
    h_pos_src = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    h_pos_dst = torch.zeros(2, 2)
    h_neg_dst = torch.zeros(4, 2)

    _, neg = predictor(h_pos_src, h_pos_dst, h_neg_dst=h_neg_dst, neg_samples=2)

    assert neg is not None
    assert torch.equal(neg.detach(), torch.tensor([1.0, 2.0, 1.0, 2.0]))


def test_tgcn_linear_cat_matches_explicit_cat() -> None:
    torch.manual_seed(3)
    linear = torch.nn.Linear(8, 5)
    left = torch.randn(7, 3, requires_grad=True)
    right = torch.randn(7, 5, requires_grad=True)
    expected = linear(torch.cat((left, right), dim=-1))
    actual = _linear_cat(linear, left, right)

    assert torch.allclose(actual, expected)
    actual.sum().backward(retain_graph=True)
    left_grad = left.grad.clone()
    right_grad = right.grad.clone()
    left.grad.zero_()
    right.grad.zero_()
    expected.sum().backward()
    assert torch.allclose(left.grad, left_grad)
    assert torch.allclose(right.grad, right_grad)


def test_tgcn_linear_cat_strided_inputs_and_parameter_gradients() -> None:
    torch.manual_seed(31)
    linear = torch.nn.Linear(16, 8, dtype=torch.float64)
    gate_storage = torch.randn(5, 24, dtype=torch.float64, requires_grad=True)
    state_storage = torch.randn(5, 16, dtype=torch.float64, requires_grad=True)
    left, right = gate_storage[:, 8:16], state_storage[:, ::2]
    assert not left.is_contiguous() and not right.is_contiguous()
    actual = _linear_cat(linear, left, right)
    expected = (left @ linear.weight[:, :8].T
                + right @ linear.weight[:, 8:].T + linear.bias)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    weights = torch.randn_like(actual)
    inputs = (gate_storage, state_storage, linear.weight, linear.bias)
    actual_grad = torch.autograd.grad((actual.tanh() * weights).sum(), inputs, retain_graph=True)
    expected_grad = torch.autograd.grad((expected.tanh() * weights).sum(), inputs)
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_value, expected_value, rtol=1e-11, atol=1e-12)
    assert torch.count_nonzero(actual_grad[0][:, :8]) == 0
    assert torch.count_nonzero(actual_grad[0][:, 16:]) == 0
    assert torch.count_nonzero(actual_grad[1][:, 1::2]) == 0


def test_edge_endpoint_collect_route_overrides_block_rows() -> None:
    graph = sg.graph_block_from_coo(
        src=torch.tensor([0]),
        dst=torch.tensor([0]),
        edge_ids=torch.tensor([0]),
        num_nodes=1,
        format="coo",
    )
    target = sg.TaskTarget(
        target_kind="edge",
        target_ids=torch.tensor([10, 11]),
        pos_src=torch.tensor([100, 101]),
        pos_dst=torch.tensor([200, 201]),
        neg_dst=torch.tensor([300, 301]),
        target_route=sg.TargetRoute(
            endpoint_collect=sg.EndpointCollectRoute(
                num_endpoints=6,
                groups={"pos_src": (0, 2), "pos_dst": (2, 4), "neg_dst": (4, 6)},
                local_endpoint_rows=torch.arange(6),
                local_embedding_rows=torch.tensor([1, 0, 3, 2, 5, 4]),
            )
        ),
    )
    batch = sg.Batch(
        mode="snapshot",
        graph=graph,
        blocks=((graph,),),
        features={"x": (torch.ones(1, 2),)},
        targets={"task": target},
    )
    embeddings = torch.tensor(
        [
            [10.0, 0.0],
            [11.0, 0.0],
            [20.0, 0.0],
            [21.0, 0.0],
            [30.0, 0.0],
            [31.0, 0.0],
        ]
    )

    endpoints = materialize_endpoint_embeddings(batch, embeddings)

    assert endpoints is not None
    assert torch.equal(endpoints["pos_src"], torch.tensor([[11.0, 0.0], [10.0, 0.0]]))
    assert torch.equal(endpoints["pos_dst"], torch.tensor([[21.0, 0.0], [20.0, 0.0]]))
    assert torch.equal(endpoints["neg_dst"], torch.tensor([[31.0, 0.0], [30.0, 0.0]]))

    output = materialize_endpoint_output(
        type("Model", (), {"edge_score": staticmethod(lambda left, right: (left + right).sum(dim=1))})(),
        batch,
        sg.ModelOutput(embeddings=embeddings),
    )
    assert torch.equal(output.aux["pos_score"], torch.tensor([32.0, 30.0]))
    assert torch.equal(output.aux["neg_score"], torch.tensor([42.0, 40.0]))


def test_edge_endpoint_collect_empty_target_is_materializable() -> None:
    graph = sg.graph_block_from_coo(
        src=torch.tensor([0]),
        dst=torch.tensor([0]),
        edge_ids=torch.tensor([0]),
        num_nodes=1,
        format="coo",
    )
    target = sg.TaskTarget(
        target_kind="edge",
        target_ids=torch.empty(0, dtype=torch.long),
        pos_src=torch.empty(0, dtype=torch.long),
        pos_dst=torch.empty(0, dtype=torch.long),
        target_route=sg.TargetRoute(
            endpoint_collect=sg.EndpointCollectRoute(
                num_endpoints=0,
                groups={"pos_src": (0, 0), "pos_dst": (0, 0)},
                local_endpoint_rows=torch.empty(0, dtype=torch.long),
                local_embedding_rows=torch.empty(0, dtype=torch.long),
                remote_endpoint_nodes=torch.empty(0, dtype=torch.long),
                remote_endpoint_rows=torch.empty(0, dtype=torch.long),
                node_dist_index=torch.zeros(1, dtype=torch.long),
            )
        ),
    )
    batch = sg.Batch(
        mode="snapshot",
        graph=graph,
        blocks=((graph,),),
        features={"x": (torch.ones(1, 2),)},
        targets={"task": target},
    )

    endpoints = materialize_endpoint_embeddings(batch, torch.ones(1, 2))

    assert endpoints is not None
    assert endpoints["pos_src"].shape == (0, 2)
    assert endpoints["pos_dst"].shape == (0, 2)


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two torchrun ranks")
@pytest.mark.parametrize("route_kind", ("dynamic", "prepared"))
def test_endpoint_collect_returns_snapshot_outputs_to_edge_owner(route_kind: str) -> None:
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        empty = torch.empty(0, dtype=torch.long)
        graph = sg.GraphBlock(
            src_nodes=torch.tensor([rank]),
            dst_nodes=torch.tensor([rank]),
            edge_ids=empty,
            format="coo",
            row=empty,
            col=empty,
            cache={"comm": CommScheduler()},
        )
        node_dist_index = torch.tensor([0, 1 << 48], dtype=torch.long)
        if route_kind == "dynamic":
            owner_route = {
                "remote_endpoint_nodes": torch.tensor([1]),
                "remote_endpoint_rows": torch.tensor([1]),
                "node_dist_index": node_dist_index,
            }
            worker_route = {
                "remote_endpoint_nodes": empty,
                "remote_endpoint_rows": empty,
                "node_dist_index": node_dist_index,
            }
        else:
            owner_route = {
                "route": Route(send_sizes=(0, 0), recv_sizes=(0, 1), send_index=empty),
                "recv_endpoint_rows": torch.tensor([1]),
            }
            worker_route = {
                "route": Route(send_sizes=(1, 0), recv_sizes=(0, 0), send_index=torch.tensor([0])),
                "recv_endpoint_rows": empty,
            }
        if rank == 0:
            target = sg.TaskTarget(
                target_kind="edge",
                target_ids=torch.tensor([0]),
                pos_src=torch.tensor([0]),
                pos_dst=torch.tensor([1]),
                target_route=sg.TargetRoute(
                    endpoint_collect=sg.EndpointCollectRoute(
                        num_endpoints=2,
                        groups={"pos_src": (0, 1), "pos_dst": (1, 2)},
                        local_endpoint_rows=torch.tensor([0]),
                        local_embedding_rows=torch.tensor([0]),
                        **owner_route,
                    )
                ),
            )
        else:
            target = sg.TaskTarget(
                target_kind="edge",
                target_ids=empty,
                pos_src=empty,
                pos_dst=empty,
                target_route=sg.TargetRoute(
                    endpoint_collect=sg.EndpointCollectRoute(
                        num_endpoints=0,
                        groups={"pos_src": (0, 0), "pos_dst": (0, 0)},
                        **worker_route,
                    )
                ),
            )
        batch = sg.Batch(mode="snapshot", graph=graph, targets={"task": target})

        embeddings = torch.tensor([[10.0 + 10.0 * rank]], requires_grad=True)
        endpoints = materialize_endpoint_embeddings(batch, embeddings)

        if rank == 0:
            assert endpoints["pos_src"].tolist() == [[10.0]]
            assert endpoints["pos_dst"].tolist() == [[20.0]]
            loss = endpoints["pos_src"].sum() + endpoints["pos_dst"].sum()
        else:
            assert endpoints["pos_src"].shape == (0, 1)
            loss = endpoints["pos_src"].sum()
        loss.backward()
        assert embeddings.grad.tolist() == [[1.0]]
    finally:
        if created_group and dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two torchrun ranks")
def test_layerwise_exchange_returns_remote_gradients_to_embedding_owner() -> None:
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        peer = 1 - rank
        sizes = (0, 1) if rank == 0 else (1, 0)
        block = sg.GraphBlock(
            src_nodes=torch.tensor([rank, peer]),
            dst_nodes=torch.tensor([rank]),
            edge_ids=torch.empty(0, dtype=torch.long),
            format="coo",
            num_src=2,
            num_dst=1,
            route=Route(
                send_sizes=sizes,
                recv_sizes=sizes,
                send_index=torch.tensor([0]),
                recv_index=torch.tensor([1]),
                output_len=2,
            ),
        )
        owned = torch.tensor([[float(rank + 1)]], requires_grad=True)

        source = materialize_embedding_src_async(block, owned, comm=CommScheduler()).wait()

        assert source.tolist() == [[float(rank + 1)], [float(peer + 1)]]
        source.sum().backward()
        assert owned.grad.tolist() == [[2.0]]
    finally:
        if created_group and dist.is_initialized():
            dist.destroy_process_group()


def test_layerwise_schedule_keeps_empty_rank_in_collective_epoch() -> None:
    empty = torch.empty(0, dtype=torch.long)
    block = sg.GraphBlock(
        src_nodes=empty,
        dst_nodes=empty,
        edge_ids=empty,
        format="coo",
        route=Route(send_sizes=(0, 0), recv_sizes=(0, 0), send_index=empty),
    )

    assert _needs_layer_exchange(block)


def test_layerwise_scheduler_launches_all_routes_before_first_wait() -> None:
    events: list[str] = []

    class Handle:
        async def async_wait(self, stream=None):
            del stream
            import asyncio

            await asyncio.sleep(0.0)
            events.append("wait")
            return torch.tensor([1.0])

    async def window(window_id: int):
        events.append(f"launch:{window_id}")
        return await Handle().async_wait()

    result = _run_flare_style_coroutines([window(0), window(1), window(2)])

    assert events[:3] == ["launch:0", "launch:1", "launch:2"]
    assert len(result) == 3


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two torchrun ranks")
def test_gradient_sync_materializes_missing_rank_gradients() -> None:
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("gloo")
    try:
        model = torch.nn.Linear(1, 1, bias=False)
        if dist.get_rank() == 0:
            model.weight.sum().backward()

        sync_gradients(model, "all_reduce")

        assert model.weight.grad.tolist() == [[0.5]]
    finally:
        if created_group and dist.is_initialized():
            dist.destroy_process_group()


def test_tgat_event_batch_forward_backward() -> None:
    model = sg.TGATModel(in_dim=3, hidden_dim=5, out_dim=2, num_layers=1)
    batch = _event_batch()

    output = model.encode(batch)
    _backward(output)

    assert output.embeddings.shape == (3, 5)
    assert output.logits is None
    assert output.aux["pos_score"].shape == (2,)
    assert output.aux["neg_score"].shape == (2,)
    assert model.state_update(batch, output) is None


def test_tgat_compact_frontier_uses_second_sampled_block_and_second_layer() -> None:
    class CountingLayer(torch.nn.Module):
        def __init__(self, layer_id: int) -> None:
            super().__init__()
            self.layer_id = int(layer_id)
            self.calls = []

        def forward(self, block, h, edge_feat, edge_dt):
            self.calls.append(
                {
                    "edge_ids": block.edge_ids.detach().clone(),
                    "src_nodes": block.src_nodes.detach().clone(),
                    "edge_feat_shape": tuple(edge_feat.shape),
                    "edge_dt_shape": tuple(edge_dt.shape),
                }
            )
            return h.new_full((int(block.dst_nodes.numel()), int(h.shape[1])), float(self.layer_id + 1))

    block0 = sg.GraphBlock(
        src_nodes=torch.tensor([0, 1, 2]),
        dst_nodes=torch.tensor([0, 1]),
        edge_ids=torch.tensor([10, 11]),
        format="coo",
        row=torch.tensor([1, 2]),
        col=torch.tensor([0, 1]),
        num_src=3,
        num_dst=2,
        edata={"edge_feat": torch.ones(2, 4), "dt": torch.ones(2)},
    )
    block1 = sg.GraphBlock(
        src_nodes=torch.tensor([1, 2, 3]),
        dst_nodes=torch.tensor([2, 3]),
        edge_ids=torch.tensor([20, 21]),
        format="coo",
        row=torch.tensor([0, 1]),
        col=torch.tensor([0, 1]),
        num_src=3,
        num_dst=2,
        edata={"edge_feat": torch.ones(2, 4) * 2.0, "dt": torch.ones(2)},
    )
    model = sg.TGATModel(in_dim=3, hidden_dim=4, out_dim=2, edge_dim=4, num_layers=2)
    model.layers = torch.nn.ModuleList([CountingLayer(0), CountingLayer(1)])
    batch = sg.Batch(
        mode="event",
        graph=block0,
        blocks=((block0, block1),),
        features={"x": (torch.arange(12, dtype=torch.float32).reshape(4, 3),),
                  "node_ids": (torch.arange(4),)},
        targets={},
    )

    output = model.encode(batch)

    assert len(model.layers[0].calls) == 2
    assert torch.equal(model.layers[0].calls[0]["edge_ids"], torch.tensor([10, 11]))
    assert torch.equal(model.layers[0].calls[1]["edge_ids"], torch.tensor([20, 21]))
    assert len(model.layers[1].calls) == 1
    assert torch.equal(model.layers[1].calls[0]["edge_ids"], torch.tensor([10, 11]))
    assert model.layers[1].calls[0]["edge_feat_shape"] == (2, 4)
    assert output.embeddings.shape == (2, 3)
    assert torch.all(output.embeddings == 2.0)


def test_tgat_frontier_merge_preserves_query_cutoffs_and_gradients() -> None:
    from starrygl.model.tgat import _merge_frontier_embeddings
    from starrygl.runtime.sample.blocks import native_target_block

    root = sg.GraphBlock(
        src_nodes=torch.tensor([1, 1, 2]), dst_nodes=torch.tensor([1, 1]),
        edge_ids=torch.empty(0, dtype=torch.long), format="coo",
        srcdata={"ts": torch.tensor([5., 8., 6.])},
        dstdata={"ts": torch.tensor([5., 8.])})
    child = sg.GraphBlock(
        src_nodes=torch.tensor([1, 2, 3]), dst_nodes=torch.tensor([1, 2]),
        edge_ids=torch.empty(0, dtype=torch.long), format="coo",
        dstdata={"ts": torch.tensor([6., 6.])})
    root_h = torch.tensor([[10.], [20.]], requires_grad=True)
    child_h = torch.tensor([[99.], [30.]], requires_grad=True)
    merged = _merge_frontier_embeddings(root, (root, root_h), (child, child_h))
    torch.testing.assert_close(merged, torch.tensor([[10.], [20.], [30.]]))
    merged.sum().backward()
    torch.testing.assert_close(root_h.grad, torch.ones_like(root_h))
    torch.testing.assert_close(child_h.grad, torch.tensor([[0.], [1.]]))
    assert native_target_block((root, child)) is root


def test_tgat_node_batch_skips_edge_scores() -> None:
    model = sg.TGATModel(in_dim=3, hidden_dim=5, out_dim=2, num_layers=1)
    output = model.encode(_snapshot_batch())

    assert output.logits is not None
    assert output.logits.shape == (3, 2)
    assert "pos_score" not in output.aux
    assert "neg_score" not in output.aux


def test_tgn_event_batch_forward_backward_and_state_delta() -> None:
    model = sg.TGNModel(in_dim=3, hidden_dim=5, out_dim=2)
    batch = _event_batch()

    output = model.encode(batch)
    _backward(output)
    delta = model.state_update(batch, output)

    assert output.embeddings.shape == (3, 5)
    assert output.logits is None
    assert output.aux["pos_score"].shape == (2,)
    assert output.aux["neg_score"].shape == (2,)
    assert delta is not None
    assert delta.kind == "node_memory"
    assert torch.equal(delta.node_ids, torch.tensor([0, 1, 2]))
    assert delta.values.shape == (3, 5)


def test_tgn_state_update_uses_prepared_state_write_mask() -> None:
    model = sg.TGNModel(in_dim=3, hidden_dim=5, out_dim=2)
    batch = _event_batch()
    batch.targets["events"] = replace(
        batch.targets["events"],
        state_write_mask=torch.tensor([1, 2], dtype=torch.uint8),
    )

    output = model.encode(batch)
    delta = model.state_update(batch, output)

    assert delta is not None
    assert delta.kind == "node_memory"
    assert torch.equal(delta.node_ids.cpu(), torch.tensor([0, 2]))
    assert delta.timestamps is not None
    assert torch.equal(delta.timestamps.cpu(), torch.tensor([0.0, 1.0]))


def test_tgn_state_update_emits_mailbox_payload() -> None:
    model = sg.TGNModel(in_dim=3, hidden_dim=5, out_dim=2)
    batch = _event_batch()

    output = model.encode(batch)
    delta = model.state_update(batch, output)

    assert delta is not None
    assert torch.equal(delta.metadata["mailbox_nodes"].cpu(), torch.tensor([0, 1, 1, 2]))
    assert delta.metadata["mailbox_messages"].shape == (4, 10)
    assert torch.equal(delta.metadata["mailbox_timestamps"].cpu(), torch.tensor([0.0, 1.0, 0.0, 1.0]))


def test_tgn_mailbox_uses_explicit_positive_edge_features() -> None:
    model = sg.TGNModel(in_dim=3, hidden_dim=5, out_dim=2, edge_dim=1)
    batch = _event_batch()
    batch.features = {
        "x": batch.features["x"],
        "pos_edge_feat": (torch.tensor([[9.0], [8.0]]),),
    }

    output = model.encode(batch)
    delta = model.state_update(batch, output)

    assert delta is not None
    assert delta.metadata["mailbox_messages"].shape == (4, 11)
    assert torch.equal(delta.metadata["mailbox_messages"][:, -1].detach().cpu(), torch.tensor([9.0, 8.0, 9.0, 8.0]))


def test_apan_mailbox_keeps_self_messages_and_delivers_them_to_neighbors() -> None:
    batch = _event_batch()
    graph = batch.graph
    assert graph is not None
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])

    nodes, messages, timestamps = _apan_mailbox_update_values(batch, graph, embeddings, 0)

    assert torch.equal(nodes, torch.tensor([0, 1, 2]))
    assert torch.equal(timestamps, torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(
        messages,
        torch.tensor(
            [
                [2.0, 2.0, 0.0, 1.0],
                [0.0, 1.0, 2.0, 2.0],
                [0.0, 1.0, 2.0, 2.0],
            ]
        ),
    )

    empty_graph = sg.graph_block_from_coo(
        src=torch.empty(0, dtype=torch.long),
        dst=torch.empty(0, dtype=torch.long),
        edge_ids=torch.empty(0, dtype=torch.long),
        num_nodes=3,
        format="coo",
    )
    self_only = _apan_mailbox_update_values(batch, empty_graph, embeddings, 0)
    assert self_only is not None
    assert torch.equal(self_only[0], torch.tensor([0, 1, 2]))


def test_tgn_optional_node_head_is_explicit() -> None:
    model = sg.TGNModel(in_dim=3, hidden_dim=5, out_dim=2, node_output_dim=4)
    batch = _snapshot_batch()
    batch.state = {"node_memory": torch.zeros(3, 5)}
    output = model.encode(batch)

    assert output.logits is not None
    assert output.logits.shape == (3, 4)
    assert "pos_score" not in output.aux
    assert "neg_score" not in output.aux


def test_tgn_uses_time_edge_features_and_shared_filter() -> None:
    batch = _event_batch()
    graph = batch.graph
    assert graph is not None
    graph.edata["f"] = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    graph.edata["delta_t"] = torch.tensor([0.5, 1.0, 1.5])
    graph.srcdata["ts"] = torch.tensor([10.0, 20.0, 30.0])
    batch.state = {
        "node_memory": torch.zeros(3, 5),
        "node_memory_ts": torch.tensor([0.0, 1.0, 2.0]),
        "mailbox_ts": torch.tensor([[4.0], [5.0], [6.0]]),
        "node_memory_historical": torch.full((3, 5), 0.25),
        "node_memory_shared_mask": torch.tensor([True, False, True]),
        "node_memory_shared_rows": torch.tensor([0, 2]),
    }
    model = sg.TGNModel(
        in_dim=3,
        hidden_dim=5,
        out_dim=2,
        edge_dim=2,
        time_dim=4,
        state_compensation=True,
        compensation_num_rows=3,
    )

    output = model.encode(batch)
    _backward(output)
    delta = model.state_update(batch, output)

    assert output.embeddings.shape == (3, 5)
    assert output.aux["memory_node_ts"].tolist() == [10.0, 20.0, 30.0]
    assert "state_compensation_change" in output.aux
    assert output.aux["state_compensation_rows"].tolist() == [0, 2]
    assert delta is not None
    assert delta.timestamps is not None
    assert "state_compensation_rows" not in delta.metadata


def test_models_execute_window_layer_mfgs() -> None:
    tgn = sg.TGNModel(in_dim=3, hidden_dim=5, out_dim=2, num_layers=2)
    tgn_out = tgn.encode(_two_window_two_layer_batch("event"))
    assert tgn_out.embeddings.shape == (3, 5)
    assert tgn_out.logits is None
    assert tgn_out.aux["pos_score"].shape == (2,)

    tgcn = sg.TGCNModel(in_dim=3, hidden_dim=5, out_dim=2, num_layers=2)
    tgcn_out = tgcn.encode(_two_window_two_layer_batch("snapshot"))
    assert tgcn_out.embeddings.shape == (3, 5)
    assert tgcn_out.logits is not None

    gconv_gru = sg.GConvGRUModel(in_dim=3, hidden_dim=5, out_dim=2)
    gconv_gru_out = gconv_gru.encode(_two_window_two_layer_batch("snapshot"))
    assert gconv_gru_out.embeddings.shape == (3, 5)
    assert gconv_gru_out.logits is not None

    mpnn = sg.MPNNLSTMModel(in_dim=3, hidden_dim=5, out_dim=2, num_layers=2)
    mpnn_out = mpnn.encode(_two_window_two_layer_batch("snapshot"))
    assert mpnn_out.embeddings.shape == (3, 5)
    assert mpnn_out.logits is not None


def test_gcnconv_dgl_kernel_matches_torch_path() -> None:
    torch.manual_seed(17)
    block = sg.GraphBlock(
        src_nodes=torch.tensor([0, 1, 2, 3]),
        dst_nodes=torch.tensor([0, 1, 2]),
        edge_ids=torch.tensor([0, 1, 2, 3]),
        format="csc",
        indptr=torch.tensor([0, 2, 3, 4]),
        indices=torch.tensor([0, 3, 1, 2]),
        num_src=4,
        num_dst=3,
        edata={
            "gcn_norm": torch.tensor([0.5, 0.25, 1.0, 0.75]),
            "self_gcn_norm": torch.tensor([1.0, 0.8, 0.6]),
        },
    )
    dgl_block = sg.GraphBlock(
        src_nodes=block.src_nodes,
        dst_nodes=block.dst_nodes,
        edge_ids=block.edge_ids,
        format=block.format,
        indptr=block.indptr,
        indices=block.indices,
        num_src=block.num_src,
        num_dst=block.num_dst,
        edata={name: value.clone() for name, value in block.edata.items()},
    )
    dgl_block.cache["use_dgl_gcn"] = True
    base = sg.GCNConv(4, 3, bias=True)
    dgl_conv = sg.GCNConv(4, 3, bias=True)
    dgl_conv.load_state_dict(base.state_dict())
    x0 = torch.randn(4, 4, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)

    out0 = base(block, x0)
    out1 = dgl_conv(dgl_block, x1)
    assert torch.allclose(out0, out1, atol=1e-5)

    out0.sum().backward()
    out1.sum().backward()
    assert torch.allclose(x0.grad, x1.grad, atol=1e-5)
    assert torch.allclose(base.weight.grad, dgl_conv.weight.grad, atol=1e-5)


def test_tgcn_snapshot_batch_forward_backward_and_state_delta() -> None:
    model = sg.TGCNModel(in_dim=3, hidden_dim=5, out_dim=2)
    batch = _snapshot_batch()

    output = model.encode(batch)
    _backward(output)
    delta = model.state_update(batch, output)

    assert output.embeddings.shape == (3, 5)
    assert output.logits is not None
    assert output.logits.shape == (3, 2)
    assert delta is None
    assert isinstance(model.input, torch.nn.Identity)
    assert model.gcn.convs[0].in_dim == 3
    assert model.gcn.convs[-1].out_dim == 15
    assert model.gcn.convs[0].add_self_loops is True
    assert model.update_gate.in_features == 10


def test_tgcn_persistent_state_is_explicit() -> None:
    model = sg.TGCNModel(in_dim=3, hidden_dim=5, out_dim=2, persist_state=True)
    batch = _snapshot_batch()

    output = model.encode(batch)
    delta = model.state_update(batch, output)

    assert delta is not None
    assert delta.kind == "node_recurrent"
    assert torch.equal(delta.node_ids, torch.tensor([0, 1, 2]))


def test_batch_local_window_scan_runs_tgcn_cell() -> None:
    batch = _two_window_two_layer_batch("snapshot")
    input_layer = torch.nn.Linear(3, 5)
    cells = (
        sg.TGCNLocalCell(hidden_dim=5, num_layers=2),
        sg.MPNNLSTMLocalCell(hidden_dim=5, num_layers=2),
    )

    for cell in cells:
        scan = _run_sequential_window_scan(batch, input_project=input_layer, cell=cell)

        assert scan.embeddings.shape == (3, 5)
        assert scan.final_block is batch.blocks[-1][-1]
        assert cell.reads_neighbor_state is False
        if isinstance(cell, sg.MPNNLSTMLocalCell):
            assert scan.state_embeddings.shape == (3, 20)
        else:
            assert scan.state_embeddings.shape == (3, 5)


def test_decoupled_window_dag_scan_matches_sync_scan_for_local_cells() -> None:
    torch.manual_seed(7)
    batch = _two_window_two_layer_batch("snapshot")
    input_layer = torch.nn.Linear(3, 5)
    cells = (
        sg.TGCNLocalCell(hidden_dim=5, num_layers=2),
        sg.MPNNLSTMLocalCell(hidden_dim=5, num_layers=2),
    )

    for cell in cells:
        sync = _run_sequential_window_scan(batch, input_project=input_layer, cell=cell)
        dag = run_decoupled_window_dag_scan(batch, input_project=input_layer, cell=cell)

        assert torch.allclose(dag.embeddings, sync.embeddings)
        assert torch.allclose(dag.state_embeddings, sync.state_embeddings)
        assert dag.final_block is sync.final_block


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two torchrun ranks")
@pytest.mark.parametrize("cell_cls", [sg.TGCNLocalCell, sg.MPNNLSTMLocalCell])
def test_sequential_scan_preserves_distributed_layers_and_gradients(monkeypatch, cell_cls):
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        sizes = (0, 1) if rank == 0 else (1, 0)
        block = sg.GraphBlock(
            src_nodes=torch.tensor([rank, 1 - rank]), dst_nodes=torch.tensor([rank]),
            edge_ids=torch.tensor([rank]), format="coo", row=torch.tensor([1]),
            col=torch.tensor([0]), num_src=2, num_dst=1,
            route=Route(send_sizes=sizes, recv_sizes=sizes, send_index=torch.tensor([0]),
                        recv_index=torch.tensor([1]), output_len=2))
        records = []
        for disabled in ("0", "1"):
            monkeypatch.setenv("STARRYGL_DISABLE_LAYERWISE_DAG", disabled)
            torch.manual_seed(41)
            cell = cell_cls(hidden_dim=2, in_dim=1, num_layers=2)
            features = tuple(torch.tensor([[rank + 1.], [2. - rank]], requires_grad=True)
                             for _ in range(2))
            batch = sg.Batch(mode="snapshot", blocks=((block,), (block,)), features={"x": features})
            scan = run_decoupled_window_dag_scan(batch, input_project=torch.nn.Identity(),
                                                cell=cell, comm=CommScheduler())
            scan.embeddings.square().sum().backward()
            records.append([scan.embeddings.detach(), scan.state_embeddings.detach(),
                            *(x.grad for x in features),
                            *(p.grad for p in cell.parameters())])
        for layerwise, sequential in zip(*records):
            torch.testing.assert_close(layerwise, sequential, rtol=1e-5, atol=1e-7)
    finally:
        dist.destroy_process_group()


def test_batch_local_window_scan_pads_carried_state_across_chunk_windows() -> None:
    class _AddOneCell:
        reads_neighbor_state = False
        state_key = "node_recurrent"

        def materialize(self, blocks, src):
            block = blocks[-1]
            rows = int(block.num_dst)
            return {"state_like": src["x"].new_zeros((rows, 1))}, block

        def local_forward(self, block, src, dst):
            del block, src
            return dst["h_prev"] + 1.0

    first = sg.GraphBlock(
        src_nodes=torch.tensor([0, 1]),
        dst_nodes=torch.tensor([0, 1]),
        edge_ids=torch.empty(0, dtype=torch.long),
        format="csc",
        indptr=torch.zeros(3, dtype=torch.long),
        indices=torch.empty(0, dtype=torch.long),
        num_src=2,
        num_dst=2,
    )
    second = sg.GraphBlock(
        src_nodes=torch.tensor([0, 1, 2]),
        dst_nodes=torch.tensor([0, 1, 2]),
        edge_ids=torch.empty(0, dtype=torch.long),
        format="csc",
        indptr=torch.zeros(4, dtype=torch.long),
        indices=torch.empty(0, dtype=torch.long),
        num_src=3,
        num_dst=3,
    )
    batch = sg.Batch(
        mode="snapshot",
        blocks=((first,), (second,)),
        features={"x": (torch.zeros(2, 1), torch.zeros(3, 1))},
        state={},
        targets={},
    )

    scan = _run_sequential_window_scan(batch, input_project=torch.nn.Identity(), cell=_AddOneCell())

    assert torch.equal(scan.state_embeddings, torch.tensor([[2.0], [2.0], [1.0]]))
    assert scan.final_block is second


def test_coupled_window_scan_runs_gconv_gru_cell() -> None:
    batch = _two_window_two_layer_batch("snapshot")
    for window in batch.blocks:
        for block in window:
            block.cache["src_state_rows"] = torch.arange(int(block.num_src), dtype=torch.long)
            block.cache["dst_state_rows"] = torch.arange(int(block.num_dst), dtype=torch.long)
    input_layer = torch.nn.Linear(3, 5)
    cell = sg.GConvGRUCell(hidden_dim=5)

    scan = run_coupled_window_scan(
        batch,
        input_project=input_layer,
        cell=cell,
        state_materializer=materialize_coupled_neighbor_state,
    )

    assert cell.reads_neighbor_state is True
    assert scan.embeddings.shape == (3, 5)
    assert scan.state_embeddings.shape == (3, 5)
    assert scan.final_block is batch.blocks[-1][-1]


def test_gconv_gru_cell_output_depends_on_neighbor_previous_state() -> None:
    torch.manual_seed(7)
    block = sg.GraphBlock(
        src_nodes=torch.tensor([0, 1, 2]),
        dst_nodes=torch.tensor([0, 1]),
        edge_ids=torch.tensor([0]),
        format="coo",
        row=torch.tensor([2]),
        col=torch.tensor([0]),
        num_src=3,
        num_dst=2,
    )
    cell = sg.GConvGRUCell(hidden_dim=4)
    x = torch.zeros(3, 4)
    h0 = torch.zeros(3, 4)
    h1 = h0.clone()
    h1[2] = torch.tensor([3.0, -2.0, 1.0, 0.5])

    src0, _ = cell.materialize((block,), {"x": x, "h_prev": h0})
    src1, _ = cell.materialize((block,), {"x": x, "h_prev": h1})
    dst = {"h_prev": torch.zeros(2, 4)}
    out0 = cell.local_forward(block, src0, dst)
    out1 = cell.local_forward(block, src1, dst)

    assert not torch.allclose(out0[0], out1[0])


def test_snapshot_graph_block_carries_state_row_maps() -> None:
    from starrygl.runtime.snapshot import snapshot_row_to_graph_block

    block = snapshot_row_to_graph_block(
        {
            "src_nodes": torch.tensor([0, 1, 2]),
            "dst_nodes": torch.tensor([0, 1]),
            "edge_ids": torch.tensor([0, 1]),
            "indptr": torch.tensor([0, 1, 2]),
            "indices": torch.tensor([2, 0]),
            "route": {"send_sizes": (0,), "recv_sizes": (0,)},
        }
    )

    assert torch.equal(block.cache["src_state_rows"], torch.tensor([0, 1, 2]))
    assert torch.equal(block.cache["dst_state_rows"], torch.tensor([0, 1]))


def test_snapshot_graph_block_uses_source_node_rows_for_chunk_state() -> None:
    from starrygl.runtime.snapshot import snapshot_row_to_graph_block

    block = snapshot_row_to_graph_block(
        {
            "src_nodes": torch.tensor([5, 9, 12]),
            "dst_nodes": torch.tensor([5, 9]),
            "edge_ids": torch.tensor([0]),
            "indptr": torch.tensor([0, 1, 1]),
            "indices": torch.tensor([2]),
            "chunk_limited": True,
        }
    )

    assert torch.equal(block.cache["src_state_rows"], torch.tensor([5, 9, 12]))
    assert torch.equal(block.cache["dst_state_rows"], torch.tensor([5, 9]))


def test_snapshot_graph_block_carries_reverse_direction() -> None:
    from starrygl.runtime.snapshot import snapshot_row_to_graph_block

    block = snapshot_row_to_graph_block(
        {
            "src_nodes": torch.tensor([0, 1, 2]),
            "dst_nodes": torch.tensor([0, 1]),
            "edge_ids": torch.tensor([0, 1]),
            "indptr": torch.tensor([0, 1, 2]),
            "indices": torch.tensor([2, 0]),
        }
    )

    assert torch.equal(block.cache["reverse_row"], torch.tensor([1]))
    assert torch.equal(block.cache["reverse_col"], torch.tensor([0]))


def test_coupled_window_scan_materializes_src_and_dst_state() -> None:
    class ToyCoupledCell(torch.nn.Module):
        reads_neighbor_state = True
        state_key = "neighbor_recurrent"

        def __init__(self) -> None:
            super().__init__()
            self.src_history = []
            self.dst_history = []

        def materialize(self, blocks, src):
            block = blocks[-1]
            return {
                "x": src["x"],
                "h_prev": src["h_prev"],
                "state_like": src["h_prev"][: block.num_dst],
            }, block

        def local_forward(self, block, src, dst):
            self.src_history.append(src["h_prev"].detach().clone())
            self.dst_history.append(dst["h_prev"].detach().clone())
            num_dst = int(dst["h_prev"].shape[0])
            return src["x"][:num_dst] + src["h_prev"][:num_dst] + dst["h_prev"]

    graph = sg.graph_block_from_coo(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([0, 1]),
        edge_ids=torch.tensor([0, 1]),
        num_nodes=3,
        format="coo",
    )
    graph.cache["src_state_rows"] = torch.tensor([2, 1, 0])
    graph.cache["dst_state_rows"] = torch.tensor([0, 2])
    graph.src_nodes = torch.tensor([2, 1, 0])
    graph.dst_nodes = torch.tensor([0, 2])
    graph.num_src = 3
    graph.num_dst = 2
    batch = sg.Batch(
        mode="snapshot",
        graph=graph,
        blocks=((graph,), (graph,)),
        features={"x": (torch.ones(3, 2), torch.full((3, 2), 2.0))},
        state={"neighbor_recurrent": torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])},
        targets={},
    )
    cell = ToyCoupledCell()

    scan = run_coupled_window_scan(
        batch,
        input_project=torch.nn.Identity(),
        cell=cell,
        persist_state=True,
    )

    assert cell.reads_neighbor_state is True
    assert torch.equal(cell.src_history[0], torch.tensor([[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]]))
    assert torch.equal(cell.dst_history[0], torch.tensor([[1.0, 1.0], [3.0, 3.0]]))
    assert torch.equal(cell.src_history[1], torch.tensor([[6.0, 6.0], [2.0, 2.0], [5.0, 5.0]]))
    assert torch.equal(scan.state_embeddings, torch.tensor([[13.0, 13.0], [2.0, 2.0], [10.0, 10.0]]))


def test_coupled_window_scan_carries_local_updates_across_hydrated_snapshots() -> None:
    class ToyCoupledCell(torch.nn.Module):
        reads_neighbor_state = True
        state_key = "neighbor_recurrent"

        def __init__(self) -> None:
            super().__init__()
            self.src_history = []
            self.dst_history = []

        def materialize(self, blocks, src):
            return {"x": src["x"], "h_prev": src["h_prev"], "state_like": src["h_prev"]}, blocks[-1]

        def local_forward(self, block, src, dst):
            self.src_history.append(src["h_prev"].detach().clone())
            self.dst_history.append(dst["h_prev"].detach().clone())
            return dst["h_prev"] + 10.0

    block0 = sg.GraphBlock(
        src_nodes=torch.tensor([2, 1, 0]),
        dst_nodes=torch.tensor([0, 2]),
        edge_ids=torch.tensor([0]),
        format="coo",
        row=torch.tensor([0]),
        col=torch.tensor([0]),
        num_src=3,
        num_dst=2,
    )
    block1 = sg.GraphBlock(
        src_nodes=torch.tensor([4, 2, 3]),
        dst_nodes=torch.tensor([3, 4]),
        edge_ids=torch.tensor([1]),
        format="coo",
        row=torch.tensor([0]),
        col=torch.tensor([0]),
        num_src=3,
        num_dst=2,
    )
    block2 = sg.GraphBlock(
        src_nodes=torch.tensor([2, 4, 1]),
        dst_nodes=torch.tensor([1, 2]),
        edge_ids=torch.tensor([2]),
        format="coo",
        row=torch.tensor([0]),
        col=torch.tensor([0]),
        num_src=3,
        num_dst=2,
    )
    batch = sg.Batch(
        mode="snapshot",
        graph=block2,
        blocks=((block0,), (block1,), (block2,)),
        features={"x": (torch.zeros(3, 1), torch.zeros(3, 1), torch.zeros(3, 1))},
        state={
            "neighbor_recurrent": torch.tensor([[20.0], [10.0], [30.0], [40.0], [20.0], [10.0]]),
            "neighbor_recurrent_node_ids": torch.tensor([2, 1, 3, 4, 2, 1]),
        },
        targets={},
    )
    cell = ToyCoupledCell()

    scan = run_coupled_window_scan(batch, input_project=torch.nn.Identity(), cell=cell, persist_state=True)

    assert torch.equal(cell.src_history[0], torch.tensor([[20.0], [10.0], [0.0]]))
    assert torch.equal(cell.dst_history[0], torch.tensor([[0.0], [20.0]]))
    assert torch.equal(cell.src_history[1], torch.tensor([[40.0], [30.0], [30.0]]))
    assert torch.equal(cell.dst_history[1], torch.tensor([[30.0], [40.0]]))
    assert torch.equal(cell.src_history[2], torch.tensor([[30.0], [50.0], [10.0]]))
    assert torch.equal(cell.dst_history[2], torch.tensor([[10.0], [30.0]]))
    assert torch.equal(scan.embeddings, torch.tensor([[20.0], [40.0]]))
    assert torch.equal(scan.state_embeddings, torch.tensor([[40.0], [20.0], [40.0], [50.0], [40.0], [20.0]]))


def test_coupled_state_materialization_accepts_compact_initial_state() -> None:
    block = sg.GraphBlock(
        src_nodes=torch.tensor([5, 9, 12]),
        dst_nodes=torch.tensor([5, 9]),
        edge_ids=torch.tensor([0]),
        format="coo",
        row=torch.tensor([2]),
        col=torch.tensor([0]),
        num_src=3,
        num_dst=2,
    )
    block.cache["src_state_rows"] = torch.tensor([5, 9, 12])
    block.cache["dst_state_rows"] = torch.tensor([5, 9])
    compact = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    state = materialize_coupled_neighbor_state(
        batch=None,
        blocks=(block,),
        block=block,
        owned_state=compact,
        dst_owned_state=compact,
        state_key="neighbor_recurrent",
        state_reader=None,
    )

    assert torch.equal(state.src_state, compact)
    assert torch.equal(state.dst_state, compact[:2])
    assert torch.equal(state.dst_rows, torch.tensor([5, 9]))


def test_coupled_state_materialization_zero_fills_missing_partial_global_rows() -> None:
    block = sg.GraphBlock(
        src_nodes=torch.tensor([0, 3, 5]),
        dst_nodes=torch.tensor([0, 5]),
        edge_ids=torch.tensor([0]),
        format="coo",
        row=torch.tensor([1]),
        col=torch.tensor([0]),
        num_src=3,
        num_dst=2,
    )
    block.cache["src_state_rows"] = torch.tensor([0, 3, 5])
    block.cache["dst_state_rows"] = torch.tensor([0, 5])
    partial = torch.ones(4, 2)

    state = materialize_coupled_neighbor_state(
        batch=None,
        blocks=(block,),
        block=block,
        owned_state=partial,
        dst_owned_state=partial,
        state_key="neighbor_recurrent",
        state_reader=None,
    )

    assert torch.equal(state.src_state, torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]))
    assert torch.equal(state.dst_state, torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
    assert torch.equal(state.dst_rows, torch.tensor([0, 5]))


def test_coupled_window_scan_can_delay_neighbor_state_only() -> None:
    class ToyDelayedCoupledCell(torch.nn.Module):
        reads_neighbor_state = True
        state_key = "neighbor_recurrent"

        def __init__(self) -> None:
            super().__init__()
            self.src_history = []
            self.dst_history = []

        def materialize(self, blocks, src):
            return {"x": src["x"], "h_prev": src["h_prev"], "state_like": src["h_prev"]}, blocks[-1]

        def local_forward(self, block, src, dst):
            self.src_history.append(src["h_prev"].detach().clone())
            self.dst_history.append(dst["h_prev"].detach().clone())
            num_dst = int(dst["h_prev"].shape[0])
            return dst["h_prev"] + src["h_prev"][:num_dst] + src["x"][:num_dst]

    graph = sg.graph_block_from_coo(
        src=torch.tensor([2, 1]),
        dst=torch.tensor([0, 2]),
        edge_ids=torch.tensor([0, 1]),
        num_nodes=3,
        format="coo",
    )
    graph.cache["src_state_rows"] = torch.tensor([2, 1, 0])
    graph.cache["dst_state_rows"] = torch.tensor([0, 2])
    graph.src_nodes = torch.tensor([2, 1, 0])
    graph.dst_nodes = torch.tensor([0, 2])
    graph.num_src = 3
    graph.num_dst = 2
    batch = sg.Batch(
        mode="snapshot",
        graph=graph,
        blocks=((graph,), (graph,)),
        features={"x": (torch.ones(3, 1), torch.full((3, 1), 2.0))},
        state={"neighbor_recurrent": torch.tensor([[1.0], [2.0], [3.0]])},
        targets={},
    )
    cell = ToyDelayedCoupledCell()

    scan = run_coupled_window_scan(
        batch,
        input_project=torch.nn.Identity(),
        cell=cell,
        persist_state=True,
        neighbor_state_delay=1,
    )

    assert torch.equal(cell.src_history[0], torch.tensor([[3.0], [2.0], [1.0]]))
    assert torch.equal(cell.dst_history[0], torch.tensor([[1.0], [3.0]]))
    assert torch.equal(cell.src_history[1], torch.tensor([[3.0], [2.0], [1.0]]))
    assert torch.equal(cell.dst_history[1], torch.tensor([[5.0], [6.0]]))
    assert torch.equal(scan.state_embeddings, torch.tensor([[10.0], [2.0], [10.0]]))


def test_coupled_window_scan_can_read_initial_state_from_manager() -> None:
    class ToyCoupledCell(torch.nn.Module):
        reads_neighbor_state = True
        state_key = "neighbor_recurrent"

        def __init__(self) -> None:
            super().__init__()
            self.src_history = []

        def materialize(self, blocks, src):
            return {"x": src["x"], "h_prev": src["h_prev"], "state_like": src["h_prev"]}, blocks[-1]

        def local_forward(self, block, src, dst):
            self.src_history.append(src["h_prev"].detach().clone())
            num_dst = int(dst["h_prev"].shape[0])
            return src["x"][:num_dst] + src["h_prev"][:num_dst] + dst["h_prev"]

    graph = sg.graph_block_from_coo(
        src=torch.tensor([2, 1]),
        dst=torch.tensor([0, 2]),
        edge_ids=torch.tensor([0, 1]),
        num_nodes=3,
        format="coo",
    )
    graph.cache["src_state_rows"] = torch.tensor([2, 1, 0])
    graph.cache["dst_state_rows"] = torch.tensor([0, 2])
    graph.src_nodes = torch.tensor([2, 1, 0])
    graph.dst_nodes = torch.tensor([0, 2])
    graph.num_src = 3
    graph.num_dst = 2
    batch = sg.Batch(
        mode="snapshot",
        graph=graph,
        blocks=((graph,), (graph,)),
        features={"x": (torch.ones(3, 2), torch.full((3, 2), 2.0))},
        targets={},
    )
    manager = sg.StateManager(
        values=torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        kind="neighbor_recurrent",
    )
    cell = ToyCoupledCell()

    scan = run_coupled_window_scan(
        batch,
        input_project=torch.nn.Identity(),
        cell=cell,
        state_reader=manager,
    )

    assert torch.equal(cell.src_history[0], torch.tensor([[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]]))
    assert torch.equal(cell.src_history[1], torch.tensor([[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]]))
    assert torch.equal(scan.state_embeddings, torch.tensor([[6.0, 6.0], [2.0, 2.0], [7.0, 7.0]]))


def test_gconv_gru_state_update_commits_neighbor_recurrent_timestamp() -> None:
    model = sg.GConvGRUModel(in_dim=1, hidden_dim=2, out_dim=1)
    target = sg.TaskTarget(
        target_kind="node",
        target_ids=torch.tensor([2, 0]),
        target_ts=torch.tensor([7.0, 7.0]),
    )
    graph = sg.graph_block_from_coo(
        src=torch.tensor([2, 0]),
        dst=torch.tensor([2, 0]),
        edge_ids=torch.empty(0, dtype=torch.long),
        num_nodes=3,
        format="coo",
    )
    batch = sg.Batch(mode="snapshot", graph=graph, targets={"task": target})
    output = sg.ModelOutput(state_embeddings=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    delta = model.state_update(batch, output)

    assert delta is not None
    assert delta.kind == "neighbor_recurrent"
    assert torch.equal(delta.node_ids, torch.tensor([2, 0]))
    assert torch.equal(delta.timestamps, torch.tensor([7.0, 7.0]))


def test_mpnn_lstm_snapshot_batch_forward_backward_and_state_delta() -> None:
    model = sg.MPNNLSTMModel(in_dim=3, hidden_dim=5, out_dim=2)
    batch = _snapshot_batch()
    batch.state = {"node_recurrent": torch.zeros(3, 20)}

    output = model.encode(batch)
    _backward(output)
    delta = model.state_update(batch, output)

    assert output.embeddings.shape == (3, 5)
    assert output.logits is not None
    assert output.logits.shape == (3, 2)
    assert delta is None


def test_mpnn_lstm_persistent_state_is_explicit() -> None:
    model = sg.MPNNLSTMModel(in_dim=3, hidden_dim=5, out_dim=2, persist_state=True)
    batch = _snapshot_batch()
    batch.state = {"node_recurrent": torch.zeros(3, 20)}

    output = model.encode(batch)
    delta = model.state_update(batch, output)

    assert delta is not None
    assert delta.kind == "node_recurrent"
    assert torch.equal(delta.node_ids, torch.tensor([0, 1, 2]))
    assert delta.values.shape == (3, 20)


def test_evolvegcn_uses_dynahb_model_weight_state() -> None:
    model = sg.EvolveGCNModel(in_dim=3, hidden_dim=5, out_dim=2, persist_state=True)
    batch = _snapshot_batch()
    batch.state = {"model_recurrent": torch.zeros(1, 15)}

    output = model.encode(batch)
    _backward(output)
    delta = model.state_update(batch, output)

    assert output.embeddings.shape == (3, 5)
    assert output.logits is not None
    assert output.logits.shape == (3, 2)
    assert output.state_embeddings is not None
    assert output.state_embeddings.shape == (1, 15)
    assert delta is not None
    assert delta.kind == "model_recurrent"
    assert torch.equal(delta.node_ids, torch.tensor([0]))
    assert delta.values.shape == (1, 15)
    assert delta.metadata["state_layout"] == "model_weight"
    assert delta.metadata["weight_shape"] == (3, 5)


def test_evolvegcn_weight_evolution_depends_on_snapshot_context() -> None:
    torch.manual_seed(37)
    model = sg.EvolveGCNModel(in_dim=3, hidden_dim=5, out_dim=2, persist_state=False)
    first = _snapshot_batch()
    second = _snapshot_batch()
    second.features = {"x": (torch.full((3, 3), 4.0),)}

    first_output = model.encode(first)
    second_output = model.encode(second)

    assert not torch.allclose(first_output.state_embeddings, second_output.state_embeddings)


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires two torchrun ranks")
def test_evolvegcn_context_reduction_is_runtime_scheduled() -> None:
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        torch.manual_seed(43)
        graph = sg.graph_block_from_coo(
            src=torch.empty(0, dtype=torch.long),
            dst=torch.empty(0, dtype=torch.long),
            edge_ids=torch.empty(0, dtype=torch.long),
            num_nodes=1,
            format="coo",
        )
        batch = sg.Batch(
            mode="snapshot",
            graph=graph,
            features={"x": (torch.full((1, 3), 1.0 + 2.0 * rank),)},
        )
        model = sg.EvolveGCNModel(in_dim=3, hidden_dim=5, out_dim=2, persist_state=False)
        scheduler = CommScheduler()

        output = encode_model(model, batch, comm=scheduler)
        gathered = [torch.empty_like(output.state_embeddings) for _ in range(2)]
        dist.all_gather(gathered, output.state_embeddings)

        assert torch.allclose(gathered[0], gathered[1])
    finally:
        if created_group and dist.is_initialized():
            dist.destroy_process_group()


def test_evolvegcn_emits_and_trains_every_snapshot_in_window() -> None:
    model = sg.EvolveGCNModel(in_dim=3, hidden_dim=5, out_dim=2, persist_state=False)
    batch = _two_window_two_layer_batch("snapshot")
    target = batch.targets["task"]
    batch.targets = {**batch.targets, "window_tasks": (target, target)}

    output = model.encode(batch)
    window_logits = output.aux["window_logits"]
    loss = sg.NodePredictionTask(name="node_classification").compute_loss(output, batch)
    expected = torch.stack(
        [torch.nn.functional.cross_entropy(value, target.label) for value in window_logits]
    ).mean()

    assert len(window_logits) == 2
    assert torch.allclose(loss, expected)


def test_evolvegcn_training_state_is_window_local_but_evaluation_state_persists() -> None:
    from starrygl.runtime.loop import _batch_local_snapshot_state

    model = sg.EvolveGCNModel(in_dim=3, hidden_dim=5, out_dim=2)

    assert _batch_local_snapshot_state(model, mode="snapshot", training=True)
    assert _batch_local_snapshot_state(
        model,
        mode="snapshot",
        training=True,
        window_policy="chunk_decay",
    )
    assert not _batch_local_snapshot_state(model, mode="snapshot", training=False)


def test_evolvegcn_log1p_transforms_weight_context_and_spatial_input() -> None:
    torch.manual_seed(41)
    raw_model = sg.EvolveGCNModel(in_dim=3, hidden_dim=5, out_dim=2, input_transform="none")
    log_model = sg.EvolveGCNModel(in_dim=3, hidden_dim=5, out_dim=2, input_transform="log1p")
    log_model.load_state_dict(raw_model.state_dict())
    batch = _snapshot_batch()
    batch.features = {"x": (torch.full((3, 3), 15.0),)}

    raw_output = raw_model.encode(batch)
    log_output = log_model.encode(batch)

    assert not torch.allclose(raw_output.state_embeddings, log_output.state_embeddings)
    assert not torch.allclose(raw_output.embeddings, log_output.embeddings)


def test_gconv_gru_snapshot_batch_forward_backward_and_state_delta() -> None:
    model = sg.GConvGRUModel(in_dim=3, hidden_dim=5, out_dim=2)
    batch = _snapshot_batch()

    output = model.encode(batch)
    _backward(output)
    delta = model.state_update(batch, output)

    assert output.embeddings.shape == (3, 5)
    assert output.logits is not None
    assert output.logits.shape == (3, 2)
    assert delta is not None
    assert delta.kind == "neighbor_recurrent"
    assert torch.equal(delta.node_ids, torch.tensor([0, 1, 2]))
    assert delta.values.shape == (3, 5)
