import copy

import pytest
import torch
import torch.nn.functional as F

from starrygl.model.graph_conv import GCN
from starrygl.view import GraphBlock


@pytest.mark.parametrize("num_layers", [1, 2, 3])
@pytest.mark.parametrize("layout", ["partial", "full", "zero_pad"])
@pytest.mark.parametrize("use_dgl", [False, True])
def test_gcn_activation_commutes_with_row_materialization(num_layers, layout, use_dgl):
    num_src = 3 if layout == "partial" else 5
    src = torch.arange(num_src)
    dst = torch.tensor([0, 1, 2] if num_src == 3 else [0, 1, 2, 0, 1])
    block = GraphBlock(
        src_nodes=src, dst_nodes=torch.arange(3), edge_ids=src,
        format="coo", row=src, col=dst, num_src=num_src, num_dst=3,
        edata={"gcn_norm": torch.ones(num_src, dtype=torch.float64),
               "self_gcn_norm": torch.ones(3, dtype=torch.float64)},
    )
    block.cache["use_dgl_gcn"] = use_dgl
    model = GCN(3, 3, num_layers=num_layers).double()
    with torch.no_grad():
        for conv in model.convs:
            conv.weight.copy_(torch.tensor([[1., -.2, 0.], [.3, -1., 0.], [0., 0., 1.]]))
            conv.bias.zero_()
    models = [model, copy.deepcopy(model), copy.deepcopy(model)]
    seed = torch.tensor([[-2., 1., 0.], [0., 0., 0.], [2., -1., 0.]], dtype=torch.float64)

    def materialize(_block, value, _layer):
        if layout == "partial":
            return value
        if layout == "full":
            return value.index_select(0, dst)  # Copies include repeated rows.
        return torch.cat((value, value.new_zeros((2, value.shape[1]))))

    inputs = [materialize(block, seed, 0).clone().requires_grad_() for _ in models]
    outputs = []
    for mode, (candidate, value) in enumerate(zip(models, inputs)):
        if mode == 1:
            value = candidate((block,), value, materialize_between_layers=materialize)
        else:
            for layer, conv in enumerate(candidate.convs):
                if mode == 0:
                    value = conv(block, F.relu(value) if layer else value)
                else:
                    value = candidate.forward_layer(layer, block, value)
                    if layer + 1 < num_layers:
                        assert (value >= 0).all()
                if layer + 1 < num_layers:
                    value = materialize(block, value, layer)
        outputs.append(value)
    upstream = torch.tensor([[1., -.5, 2.], [3., -2., 1.], [-1., 2., .5]], dtype=torch.float64)
    for output in outputs:
        (output * upstream).sum().backward()
    for index in (1, 2):
        torch.testing.assert_close(outputs[0], outputs[index], rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(inputs[0].grad, inputs[index].grad, rtol=1e-10, atol=1e-10)
        for before, after in zip(models[0].parameters(), models[index].parameters()):
            torch.testing.assert_close(before.grad, after.grad, rtol=1e-10, atol=1e-10)
