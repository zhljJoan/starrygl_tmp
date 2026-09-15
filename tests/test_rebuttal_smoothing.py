import pytest
import torch

import starrygl as sg
from starrygl.runtime.builders import build_model_from_config


@pytest.mark.parametrize("name", ["gconv_gru", "dcrnn"])
@pytest.mark.parametrize("extrapolate", [False, True])
def test_unobserved_hot_preserves_local_and_increment_ablation_is_independent(name, extrapolate):
    model = build_model_from_config(
        {"name": name, "in_dim": 1, "hidden_dim": 2, "out_dim": 1,
         "state_extrapolation": extrapolate},
        temporal_state={"consistency": "bounded_stale",
                        "smooth_aggregation": {"enabled": True, "gamma_init": 0.0}},
    )
    nodes = torch.arange(2)
    block = sg.graph_block_from_coo(src=nodes, dst=nodes.flip(0), edge_ids=nodes,
                                   num_nodes=2, format="coo")
    block.cache["snapshot_id"] = 3
    # One unobserved hot row, one observed stale row. State/mean/count/version.
    packet = torch.tensor([[0., 0., 0., 0., 0., 0.], [2., 4., 1., 2., 1., 1.]])
    batch = sg.Batch(mode="snapshot", blocks=((block,),), state={
        "neighbor_recurrent_shared_snapshots": (packet,),
        "neighbor_recurrent_shared_dst_rows": (nodes,),
        "neighbor_recurrent_snapshots": (packet,),
        "neighbor_recurrent_cold_src_rows": (nodes[1:],),
    })
    local = torch.tensor([[10., 12.], [20., 22.]], requires_grad=True)
    result = model.runtime_smooth_state(batch, 0, block, local)
    prediction = torch.tensor([5., 10.]) if extrapolate else torch.tensor([2., 4.])
    torch.testing.assert_close(result[0], local[0])
    torch.testing.assert_close(result[1], (local[1] + prediction) / 2)
    result.sum().backward()
    torch.testing.assert_close(local.grad, torch.tensor([[1., 1.], [.5, .5]]))
    torch.testing.assert_close(model.gamma.grad, (local[1].detach() - prediction).sum() / 4)
    prepared, _ = model.runtime_prepare_scan(batch)
    expected_cold = torch.tensor([4., 8.]) if extrapolate else torch.tensor([2., 4.])
    torch.testing.assert_close(prepared.state["neighbor_recurrent_window_state"][0][1], expected_cold)
    torch.testing.assert_close(packet[1], torch.tensor([2., 4., 1., 2., 1., 1.]))
