import pytest
import torch

from starrygl.model.evolve_gcn import EvolveGCNOConv
from starrygl.model.graph_conv import GCNConv, _aggregate, edge_rows
from starrygl.view import GraphBlock


def _block(use_dgl, empty=False):
    src = torch.tensor([], dtype=torch.long) if empty else torch.tensor([0, 3, 1, 2])
    dst = torch.tensor([], dtype=torch.long) if empty else torch.tensor([0, 0, 1, 2])
    block = GraphBlock(
        src_nodes=torch.arange(4), dst_nodes=torch.arange(3),
        edge_ids=torch.arange(src.numel()), format="coo", row=src, col=dst,
        num_src=4, num_dst=3,
    )
    block.cache["use_dgl_gcn"] = use_dgl
    return block


@pytest.mark.parametrize("normalize,improved,add_self", [
    (True, False, True), (True, True, True),
    (False, True, True), (True, False, False),
])
@pytest.mark.parametrize("in_dim,out_dim", [(4, 3), (2, 5)])
@pytest.mark.parametrize("edata_kind", ["cached", "weighted", "empty"])
def test_fused_gcn_self_matches_existing_torch_math(
    normalize, improved, add_self, in_dim, out_dim, edata_kind,
):
    torch.manual_seed(29)
    blocks = [_block(False, edata_kind == "empty"), _block(True, edata_kind == "empty")]
    convs = [GCNConv(in_dim, out_dim, normalize=normalize, improved=improved,
                     add_self_loops=add_self).double() for _ in range(2)]
    convs[1].load_state_dict(convs[0].state_dict())
    cached_graph = None
    # Reuse topology with changed normalizers and a fresh autograd graph.
    for step in range(2):
        data = {}
        if edata_kind == "cached":
            data = {
                "gcn_norm": torch.tensor([.5, .25, 1., .75], dtype=torch.float64) + step * .1,
                "self_gcn_norm": torch.tensor([1., .8, .6], dtype=torch.float64) + step * .1,
            }
        elif edata_kind == "weighted":
            data = {"w": torch.tensor([1., .5, 2., .75], dtype=torch.float64) + step * .1}
        # Existing computed self_gcn_norm uses in-place nan_to_num after rsqrt;
        # raw w gradients with normalized self edges already fail in Torch.
        weight_grad = edata_kind == "cached" or not (add_self and normalize)
        for block in blocks:
            block.edata = {key: val.clone().requires_grad_(weight_grad) for key, val in data.items()}
        seed = torch.randn(4, in_dim * 2, dtype=torch.float64)
        inputs = [seed.clone()[:, ::2].requires_grad_() for _ in range(2)]
        outputs = [conv(block, x) for conv, block, x in zip(convs, blocks, inputs)]
        torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-10, atol=1e-10)
        upstream = torch.randn_like(outputs[0])
        for conv, output in zip(convs, outputs):
            conv.zero_grad(set_to_none=True)
            (output * upstream).sum().backward()
        for left, right in ((inputs[0].grad, inputs[1].grad),
                            (convs[0].weight.grad, convs[1].weight.grad),
                            (convs[0].bias.grad, convs[1].bias.grad)):
            torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)
        for key in data:
            torch.testing.assert_close(blocks[0].edata[key].grad, blocks[1].edata[key].grad,
                                       rtol=1e-10, atol=1e-10)
        graphs = [val for key, val in blocks[1].cache.items() if key.startswith("dgl_gcn:")]
        if graphs:
            assert len(graphs) == 1
            graph = graphs[0]
            assert not graph.is_block and graph.idtype == torch.int64
            assert graph.num_edges() == blocks[1].edge_ids.numel() + (3 if add_self else 0)
            assert cached_graph is None or graph is cached_graph
            cached_graph = graph


@pytest.mark.parametrize("add_self", [False, True])
def test_fused_cache_does_not_change_generic_or_evolve_aggregation(add_self):
    torch.manual_seed(31)
    blocks = [_block(False), _block(True)]
    for block in blocks:
        block.edata = {
            "gcn_norm": torch.tensor([.5, .25, 1., .75]),
            "self_gcn_norm": torch.tensor([1., .8, .6]),
        }
    seed = torch.randn(4, 4)
    GCNConv(4, 3)(blocks[1], seed)  # Populate the fused cache first.
    weights = [torch.randn(4, 3, requires_grad=True)]
    weights.append(weights[0].detach().clone().requires_grad_())
    inputs = [seed.clone().requires_grad_() for _ in range(2)]
    outputs = []
    for block, x, weight in zip(blocks, inputs, weights):
        src, dst = edge_rows(block, x.device)
        generic = _aggregate(block, x, src, dst, block.edata["gcn_norm"], 3)
        evolved = EvolveGCNOConv(add_self_loops=add_self)(block, x, weight)
        outputs.append((generic, evolved))
        (generic.square().sum() + evolved.square().sum()).backward()
    torch.testing.assert_close(outputs[0], outputs[1])
    torch.testing.assert_close(inputs[0].grad, inputs[1].grad)
    torch.testing.assert_close(weights[0].grad, weights[1].grad)
    assert len([key for key in blocks[1].cache if key.startswith("dgl_gcn:")]) == 2
